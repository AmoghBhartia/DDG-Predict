import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config_initialisation import CONFIG
from src.embeddings_management.loading_embeddings import EmbeddingsCache
from src.model.ddG_predictor_model import DDGPredictor
from src.repo_paths import PATHS
from src.training_and_validation.checkpoints import (
    EXPERIMENT_DDG_CHECKPOINTS,
    CheckpointKind,
    RunLabel,
    save_checkpoint,
)
from src.training_and_validation.checkpoints import EXPERIMENT_PKA_CHECKPOINTS
from src.training_and_validation.pka_pretraining import train_pka_head
from src.training_and_validation.utils import (
    ddg_data_loader,
    require_finite,
    run_model,
    training_loader,
)
from src.utils.dataframe_processing_tools import read_dataframe
from src.utils.seeding import apply_seed


def train_one_epoch(
    progress: tqdm,
    model: DDGPredictor,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train over one epoch and return the example-weighted mean loss."""
    model.train()
    total_loss = 0.0
    examples = 0

    for batch in progress:
        ddG_true, output = run_model(model, batch)
        ddG_prediction = output["ddG_prediction"]
        require_finite(ddG_prediction, "ddG prediction")

        loss = criterion(ddG_prediction, ddG_true)
        require_finite(loss, "loss")

        optimiser.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        require_finite(gradient_norm, "gradient norm")
        optimiser.step()

        batch_size = ddG_true.shape[0]
        total_loss += loss.item() * batch_size
        examples += batch_size
        progress.set_postfix({
            "batch_loss": f"{loss.item():.3f}",
            "epoch_loss": f"{total_loss / examples:.3f}",
            "prediction": f"{ddG_prediction[0].item():.2f}",
            "true": f"{ddG_true[0].item():.2f}",
        })

    if examples == 0:
        raise ValueError("Training loader produced no batches")
    return total_loss / examples


def model_training_loop(
    model: DDGPredictor,
    loader: DataLoader,
    kind: CheckpointKind = EXPERIMENT_DDG_CHECKPOINTS,
    run: RunLabel | None = None,
    pka_frame: pd.DataFrame | None = None,
    pka_kind: CheckpointKind | None = None,
    generator: torch.Generator | None = None,
) -> DDGPredictor:
    """Run the configured epochs, updating the model and saving checkpoints."""
    num_epochs = CONFIG.num_epochs
    model.to(CONFIG.resolve_torch_device())
    model, optimiser, scheduler, start_epoch = training_loader(model, num_epochs, kind, run)

    if start_epoch == 0:
        # Pretrain the pKa head only for a fresh run. When resuming, the ddG checkpoint already
        # holds the fine-tuned pKa weights, and reloading the PKAD checkpoint would overwrite them.
        model = train_pka_head(
            model,
            pkad_frame=pka_frame,
            kind=pka_kind if pka_kind is not None else EXPERIMENT_PKA_CHECKPOINTS,
            run=run,
            generator=generator,
        )
    else:
        print(f"Resuming ddG training from epoch {start_epoch}; skipping PKAD pretraining.")

    criterion = nn.HuberLoss(delta=CONFIG.ddg_loss_delta)
    for epoch in range(start_epoch, num_epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True)
        average_loss = train_one_epoch(progress, model, optimiser, criterion)
        scheduler.step()

        save_checkpoint(kind, {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, run)
        print(f"ddG epoch {epoch + 1} loss (mean Huber loss over the epoch): {average_loss:.4f}")

    return model


def run_experiment_training() -> None:
    """Train the shipped model on the whole curated dataset.

    There is no testing set here by design: this is the model intended for prediction, and
    its expected performance is the cross-validated figure that --train_for_evaluation and
    --train_for_evaluation produces. PKAD needs no leakage filtering either, since nothing is
    excluded.
    """
    dataset = read_dataframe(PATHS.curated_all_csv)
    print(f"Training the experiment model on all {len(dataset)} curated rows "
          f"({dataset['sequence'].nunique()} wild-type sequences).")

    generator = apply_seed(CONFIG.random_seed)
    with EmbeddingsCache(PATHS.embed_dir) as embedding_db:
        loader = ddg_data_loader(dataset, embedding_db, generator=generator)
        model_training_loop(DDGPredictor.from_config(), loader, generator=generator)
