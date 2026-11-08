"""PKAD pretraining for the shared pKa interaction head."""

import pandas as pd
import torch
from tqdm import tqdm

from src.config_initialisation import CONFIG
from src.data_loading.pka_dataset import PkaSequenceDataset, pka_collate
from src.embeddings_management.loading_embeddings import EmbeddingsCache
from src.model.ddG_predictor_model import DDGPredictor
from src.repo_paths import PATHS
from src.training_and_validation.checkpoints import (
    EXPERIMENT_PKA_CHECKPOINTS,
    CheckpointKind,
    RunLabel,
    load_latest_checkpoint,
    save_checkpoint,
)
from src.training_and_validation.utils import make_loader, move_batch_to_device
from src.utils.dataframe_processing_tools import read_dataframe


def pka_squared_errors(
    predicted_deltas: torch.Tensor,
    observed_pkas: torch.Tensor,
    unfolded_pkas: torch.Tensor,
    titratable: torch.Tensor,
) -> torch.Tensor:
    """Squared errors of the predicted pKa shifts for every observed, titratable residue in the batch.

    Every measured residue counts equally, whichever protein it comes from. The result is
    empty when the batch contains no usable target.
    """
    valid = torch.isfinite(observed_pkas) & titratable
    return (predicted_deltas[valid] - (observed_pkas - unfolded_pkas)[valid]).square()


def _resume(
    model: DDGPredictor,
    optimiser: torch.optim.Optimizer,
    kind: CheckpointKind,
    run: RunLabel | None,
) -> int:
    """Restore the newest pKa checkpoint for this run and return the completed epoch count."""
    state = load_latest_checkpoint(kind, model.pka_pipeline, run)
    if state is None:
        return 0
    model.pka_pipeline.load_state_dict(state["pka_model_state"])
    optimiser.load_state_dict(state["pka_optimizer_state"])
    # Use the configured rate even if the checkpoint was saved with a different one.
    for group in optimiser.param_groups:
        group["lr"] = CONFIG.pka_learning_rate
    return state["epoch"]


def train_pka_head(
    model: DDGPredictor,
    pkad_frame: pd.DataFrame | None = None,
    kind: CheckpointKind = EXPERIMENT_PKA_CHECKPOINTS,
    run: RunLabel | None = None,
    generator: torch.Generator | None = None,
) -> DDGPredictor:
    """Pretrain the pKa head on grouped, leakage-filtered PKAD observations.

    ``pkad_frame`` overrides the default curated PKAD table, which validation training uses to
    supply a PKAD split filtered against that training split's testing-set proteins.
    """
    dataframe = read_dataframe(PATHS.pkad_cleaned_csv) if pkad_frame is None else pkad_frame
    device = CONFIG.resolve_torch_device()
    model.to(device)

    with EmbeddingsCache(PATHS.pka_embeddings_dir) as embedding_db:
        loader = make_loader(
            PkaSequenceDataset(dataframe, embedding_db), pka_collate, shuffle=True, generator=generator
        )
        optimiser = torch.optim.AdamW(
            model.pka_pipeline.parameters(),
            lr=CONFIG.pka_learning_rate,
            weight_decay=CONFIG.weight_decay,
        )
        start_epoch = _resume(model, optimiser, kind, run)
        model.train()

        for epoch in range(start_epoch, CONFIG.pka_num_epochs):
            progress = tqdm(loader, desc=f"pKa epoch {epoch + 1}/{CONFIG.pka_num_epochs}")
            epoch_squared_error, epoch_residues = 0.0, 0
            for batch in progress:
                batch = move_batch_to_device(batch, device)
                predicted = model.predict_pka_deltas(
                    batch["embeddings"], batch["residue_indices"], batch["true_lengths"]
                ).squeeze(-1)
                squared_errors = pka_squared_errors(
                    predicted,
                    batch["targets"],
                    model.default_unfolded_pKas[batch["residue_indices"]],
                    model.titratable_residue[batch["residue_indices"]],
                )
                if squared_errors.numel() == 0:
                    continue
                loss = squared_errors.mean()

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.pka_pipeline.parameters(), 1.0)
                optimiser.step()

                epoch_squared_error += squared_errors.sum().item()
                epoch_residues += squared_errors.numel()
                progress.set_postfix(
                    batch_loss=f"{loss.item():.4f}",
                    epoch_loss=f"{epoch_squared_error / epoch_residues:.4f}",
                )

            if epoch_residues == 0:
                raise ValueError("PKAD loader produced no residues with a usable pKa target")
            print(
                f"pKa epoch {epoch + 1} loss (mean squared error over {epoch_residues} residues): "
                f"{epoch_squared_error / epoch_residues:.4f}"
            )

            save_checkpoint(kind, {
                "epoch": epoch + 1,
                "pka_model_state": model.pka_pipeline.state_dict(),
                "pka_optimizer_state": optimiser.state_dict(),
            }, run)
    return model
