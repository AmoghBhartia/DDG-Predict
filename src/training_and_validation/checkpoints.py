"""Shared checkpoint saving, validation, pruning and recovery for ddG and pKa training."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.config_initialisation import CONFIG
from src.repo_paths import PATHS
from src.safe_writing import safe_writer

CHECKPOINT_PATTERN = re.compile(r"^checkpoint_epoch(?P<epoch>\d+)(?:_.*)?\.pt$")
RUN_PATTERN = re.compile(
    r"^initial_state_(?P<state>\d+)_training_set_(?P<split>\d+)_epoch_(?P<epoch>\d+)\.pt$"
)


@dataclass(frozen=True)
class RunLabel:
    """Identifies one validation run: an initial state paired with a training split.

    Validation training produces one model per (initial state, training split) pair, so the
    checkpoints of different runs share a folder and have to be told apart by name. Pruning
    and resuming are then done within a single run rather than across the whole folder.
    """

    initial_state: int
    training_set: int

    def __post_init__(self) -> None:
        if self.initial_state < 0 or self.training_set < 0:
            raise ValueError("initial_state and training_set must not be negative")

    @property
    def token(self) -> str:
        return f"initial_state_{self.initial_state}_training_set_{self.training_set}"

    def filename(self, epoch: int) -> str:
        return f"{self.token}_epoch_{epoch}.pt"

    def epoch_of(self, name: str) -> int | None:
        """Return the epoch this file holds for this run, or None if it belongs to another run."""
        match = RUN_PATTERN.match(name)
        if match is None:
            return None
        if int(match.group("state")) != self.initial_state:
            return None
        if int(match.group("split")) != self.training_set:
            return None
        return int(match.group("epoch"))


def runs_present(directory: Path) -> list[RunLabel]:
    """Return every validation run that has at least one checkpoint in ``directory``."""
    if not directory.is_dir():
        return []
    labels = {
        RunLabel(int(match.group("state")), int(match.group("split")))
        for match in (RUN_PATTERN.match(path.name) for path in directory.glob("*.pt"))
        if match is not None
    }
    return sorted(labels, key=lambda label: (label.training_set, label.initial_state))


class CheckpointArchitectureError(RuntimeError):
    """Raised when saved weights do not fit the current model architecture."""


@dataclass(frozen=True)
class CheckpointKind:
    """Where one type of checkpoint lives and which keys it must contain."""

    directory_token: str
    weights_key: str
    required_keys: frozenset[str]

    @property
    def directory(self) -> Path:
        return getattr(PATHS, self.directory_token)


DDG_KEYS = frozenset({"epoch", "model_state", "optim_state", "scheduler_state"})
PKA_KEYS = frozenset({"epoch", "pka_model_state", "pka_optimizer_state"})

# The model that ships: trained on the whole curated dataset.
EXPERIMENT_DDG_CHECKPOINTS = CheckpointKind("checkpoints_experiment_ddg_dir", "model_state", DDG_KEYS)
EXPERIMENT_PKA_CHECKPOINTS = CheckpointKind("checkpoints_experiment_pka_dir", "pka_model_state", PKA_KEYS)

# One model per (initial state, training split), used only to measure performance.
EVALUATION_DDG_CHECKPOINTS = CheckpointKind("checkpoints_evaluation_ddg_dir", "model_state", DDG_KEYS)
EVALUATION_PKA_CHECKPOINTS = CheckpointKind(
    "checkpoints_evaluation_pka_dir", "pka_model_state", PKA_KEYS
)


def list_checkpoints(directory: Path, run: RunLabel | None = None) -> list[tuple[int, Path]]:
    """Return this run's checkpoint files sorted by epoch (oldest first).

    With ``run`` set, only that run's files are listed, so one validation run never sees,
    resumes from or prunes another's.
    """
    if not directory.is_dir():
        return []
    checkpoints = []
    for path in directory.glob("*.pt"):
        if run is not None:
            epoch = run.epoch_of(path.name)
            if epoch is not None:
                checkpoints.append((epoch, path))
            continue
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group("epoch")), path))
    return sorted(checkpoints)


def _summarise(names: list[str], limit: int = 5) -> list[str]:
    return names[:limit] + ["..."] if len(names) > limit else names


def validate_checkpoint_state(state: Any, kind: CheckpointKind, module: nn.Module) -> None:
    """Check required keys, the epoch, and that saved weights match ``module`` exactly."""
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    missing = kind.required_keys.difference(state)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    if not isinstance(state["epoch"], int) or state["epoch"] < 0:
        raise ValueError("Checkpoint epoch must be a nonnegative integer")

    saved = state[kind.weights_key]
    if not isinstance(saved, dict):
        raise TypeError(f"Checkpoint {kind.weights_key} must be a state dictionary")
    expected = module.state_dict()
    incompatible = [
        name for name, tensor in expected.items()
        if name not in saved or saved[name].shape != tensor.shape
    ]
    unexpected = sorted(set(saved).difference(expected))
    if incompatible or unexpected:
        raise CheckpointArchitectureError(
            f"incompatible layers={_summarise(incompatible)}, unexpected entries={_summarise(unexpected)}"
        )


def load_latest_checkpoint(
    kind: CheckpointKind,
    module: nn.Module,
    run: RunLabel | None = None,
) -> dict[str, Any] | None:
    """Load the newest valid checkpoint, skipping unreadable files but never mismatched architectures."""
    for _, path in reversed(list_checkpoints(kind.directory, run)):
        try:
            state = torch.load(path, map_location=CONFIG.resolve_torch_device(), weights_only=True)
            validate_checkpoint_state(state, kind, module)
            return state
        except CheckpointArchitectureError as error:
            raise CheckpointArchitectureError(
                f"Weights stored in {path.name} are incompatible with the current model architecture: {error}"
            ) from error
        except Exception as error:
            print(f"Warning: skipping unreadable or malformed checkpoint {path.name}: {error}")
    return None


def save_checkpoint(kind: CheckpointKind, state: dict[str, Any], run: RunLabel | None = None) -> None:
    """Atomically save a checkpoint and delete the oldest beyond ``max_files_kept``."""
    missing = kind.required_keys.difference(state)
    if missing:
        raise ValueError(f"Cannot save checkpoint; missing essential keys: {sorted(missing)}")

    directory = kind.directory
    directory.mkdir(parents=True, exist_ok=True)
    epoch = state["epoch"]
    name = f"checkpoint_epoch{epoch}.pt" if run is None else run.filename(epoch)
    with safe_writer(directory / name, "wb") as handle:
        torch.save(state, handle)

    checkpoints = list_checkpoints(directory, run)
    for _, old_path in checkpoints[: max(len(checkpoints) - CONFIG.max_files_kept, 0)]:
        try:
            old_path.unlink(missing_ok=True)
        except OSError as error:
            print(f"Warning: could not delete old checkpoint {old_path.name}: {error}")
