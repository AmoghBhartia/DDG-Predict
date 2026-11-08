from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.config_initialisation import CONFIG
from src.data_loading.making_the_dataset import DatasetMaker, DatasetMode, collate_fn
from src.embeddings_management.loading_embeddings import EmbeddingsCache
from src.model.ddG_predictor_model import DDGPredictor
from src.training_and_validation.checkpoints import (
    EXPERIMENT_DDG_CHECKPOINTS,
    CheckpointKind,
    RunLabel,
    load_latest_checkpoint,
)
from src.utils.seeding import seed_worker


# Tensor and batch utilities.
def require_finite(tensor: torch.Tensor, name: str) -> None:
    """Raise ``FloatingPointError`` when a tensor contains NaN or infinity."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"CRITICAL: {name} contains non-finite (NaN/Inf) values.")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested batches to the requested device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(item, device) for item in batch)
    return batch


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device holding the model's parameters."""
    return next(model.parameters()).device


# Data loaders.
def make_loader(
    dataset: Dataset,
    collate: Callable,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a DataLoader with the configured batch size and worker settings.

    ``generator`` fixes the shuffle order; pass the one returned by ``apply_seed`` so that a
    run with the same seed sees its batches in the same order.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot create a data loader from an empty dataset")
    return DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.resolve_torch_device().type == "cuda",
        generator=generator if shuffle else None,
        worker_init_fn=seed_worker if CONFIG.num_workers > 0 else None,
    )


def ddg_data_loader(
    dataframe: pd.DataFrame,
    embedding_db: EmbeddingsCache,
    shuffle: bool = True,
    mode: DatasetMode = "train/eval",
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a DataLoader over curated mutation rows."""
    return make_loader(DatasetMaker(dataframe, embedding_db, mode=mode), collate_fn, shuffle, generator)


# Optimisation.
def build_optimiser(model: DDGPredictor) -> AdamW:
    """Create AdamW with a separate (smaller) learning rate for the pretrained pKa pipeline."""
    pka_parameters = list(model.pka_pipeline.parameters())
    pka_ids = {id(parameter) for parameter in pka_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in pka_ids]
    return AdamW(
        [
            {"params": other_parameters, "lr": CONFIG.learning_rate},
            {"params": pka_parameters, "lr": CONFIG.pka_finetune_learning_rate},
        ],
        weight_decay=CONFIG.weight_decay,
    )


def training_loader(
    model: DDGPredictor,
    num_epochs: int,
    kind: CheckpointKind = EXPERIMENT_DDG_CHECKPOINTS,
    run: RunLabel | None = None,
) -> tuple[DDGPredictor, AdamW, CosineAnnealingLR, int]:
    """Restore training state from this run's latest ddG checkpoint, or start from scratch."""
    optimiser = build_optimiser(model)
    # Create the scheduler before loading optimiser state: its constructor resets the learning rates.
    scheduler = CosineAnnealingLR(optimiser, T_max=num_epochs)

    state = load_latest_checkpoint(kind, model, run)
    if state is None:
        return model, optimiser, scheduler, 0

    model.load_state_dict(state["model_state"])
    start_epoch = state["epoch"]
    try:
        optimiser.load_state_dict(state["optim_state"])
        scheduler.load_state_dict(state["scheduler_state"])
    except ValueError as error:
        # Checkpoints saved before the separate pKa learning rate have one parameter group.
        print(
            f"Warning: optimiser state in the epoch {start_epoch} checkpoint is incompatible ({error}). "
            "Model weights were restored; the optimiser and learning-rate schedule restart for the "
            "remaining epochs."
        )
        optimiser = build_optimiser(model)
        scheduler = CosineAnnealingLR(optimiser, T_max=max(num_epochs - start_epoch, 1))
    return model, optimiser, scheduler, start_epoch


# Inference helpers.
def load_trained_model() -> DDGPredictor:
    """Build the predictor on the configured device with the latest trained weights."""
    model = DDGPredictor.from_config().to(CONFIG.resolve_torch_device())
    state = load_latest_checkpoint(EXPERIMENT_DDG_CHECKPOINTS, model)
    if state is None:
        raise RuntimeError("No checkpoints found. Train the model before evaluation.")
    model.load_state_dict(state["model_state"])
    return model.eval()


def run_model(
    model: DDGPredictor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Move a collated batch to the model's device and run the ddG predictor."""
    batch = move_batch_to_device(batch, model_device(model))
    output = model(
        batch["wt_esms"],
        batch["mut_esms"],
        batch["wt_residue_indices"],
        batch["mut_residue_indices"],
        batch["pH"],
        batch["experiment_temperature"],
        batch["true_lengths"],
    )
    return batch["ddG"], output
