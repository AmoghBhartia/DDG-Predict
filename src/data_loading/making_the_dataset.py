import math
from typing import Any, Literal

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.embeddings_management.loading_embeddings import EmbeddingsCache
from src.utils.amino_acid_information_tools import encode_residues

DatasetMode = Literal["train/eval", "find"]


def finite_value(row: pd.Series, column: str, index: int) -> float:
    """Read a numeric field from a dataset row and require it to be finite."""
    try:
        value = float(row[column])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Dataset row {index} has an invalid {column} value") from error
    if not math.isfinite(value):
        raise ValueError(f"Dataset row {index} contains a non-finite {column} value")
    return value


def stack_column(batch: list[dict[str, Any]], key: str) -> torch.Tensor:
    """Stack one scalar tensor per example into a ``[batch, 1]`` tensor."""
    return torch.stack([item[key] for item in batch]).unsqueeze(1)


def pad_column(batch: list[dict[str, Any]], key: str, padding_value: float = 0.0) -> torch.Tensor:
    """Right-pad one variable-length tensor per example into a batch tensor."""
    return pad_sequence([item[key] for item in batch], batch_first=True, padding_value=padding_value)


class DatasetMaker(Dataset):
    """Convert curated mutation rows into model-ready features and embeddings."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        embedding_db: EmbeddingsCache,
        mode: DatasetMode = "train/eval",
    ) -> None:
        if mode not in ("train/eval", "find"):
            raise ValueError(f"Unsupported dataset mode: {mode!r}")
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.embedding_db = embedding_db
        self.mode = mode

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Build one row's scalar features and wild-type/mutant ESM embeddings."""
        row = self.dataframe.iloc[index]
        wildtype = str(row["sequence"])
        mutant = str(row["mutant_sequence"])

        # ddG is unknown (NaN) when predicting new mutants, so it is only required in training/evaluation.
        ddG = finite_value(row, "ddG", index) if self.mode == "train/eval" else float(row["ddG"])

        return {
            "wt_esm": self.embedding_db.get_embedding(wildtype, "wt"),
            "mut_esm": self.embedding_db.get_embedding(mutant, "mut"),
            "wt_residue_indices": encode_residues(wildtype),
            "mut_residue_indices": encode_residues(mutant),
            "ddG": torch.tensor(ddG, dtype=torch.float32),
            "pH": torch.tensor(finite_value(row, "pH", index), dtype=torch.float32),
            "experiment_temperature": torch.tensor(
                finite_value(row, "experiment_temperature", index), dtype=torch.float32
            ),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad wild-type/mutant sequence features and stack scalar features."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    wt_lengths = [item["wt_esm"].shape[0] for item in batch]
    mut_lengths = [item["mut_esm"].shape[0] for item in batch]
    if wt_lengths != mut_lengths:
        raise ValueError("Wild-type and mutant embeddings must have matching sequence lengths")

    return {
        "wt_esms": pad_column(batch, "wt_esm"),
        "mut_esms": pad_column(batch, "mut_esm"),
        "wt_residue_indices": pad_column(batch, "wt_residue_indices"),
        "mut_residue_indices": pad_column(batch, "mut_residue_indices"),
        "true_lengths": torch.tensor(wt_lengths, dtype=torch.long).unsqueeze(1),
        "ddG": stack_column(batch, "ddG"),
        "pH": stack_column(batch, "pH"),
        "experiment_temperature": stack_column(batch, "experiment_temperature"),
    }
