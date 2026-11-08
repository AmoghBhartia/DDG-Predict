"""Dataset utilities for grouped PKAD residue pKa targets."""

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_loading.making_the_dataset import pad_column
from src.embeddings_management.loading_embeddings import EmbeddingsCache
from src.utils.amino_acid_information_tools import encode_residues, is_standard_sequence


class PkaSequenceDataset(Dataset):
    """Return one complete sequence and all observed pKa targets per item."""

    def __init__(self, dataframe: pd.DataFrame, embedding_db: EmbeddingsCache) -> None:
        self.embedding_db = embedding_db
        standard_rows = dataframe["sequence"].astype(str).map(is_standard_sequence)
        self.groups = [
            (sequence, group.reset_index(drop=True))
            for sequence, group in dataframe[standard_rows].groupby("sequence", sort=False)
        ]
        if not self.groups:
            raise ValueError("PKAD dataset contains no sequences composed only of the 20 standard amino acids")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence, group = self.groups[index]
        targets = torch.full((len(sequence),), float("nan"), dtype=torch.float32)
        positions = torch.tensor(group["residue_position"].to_numpy(dtype=int), dtype=torch.long)
        targets[positions] = torch.tensor(group["pKa"].to_numpy(dtype=float), dtype=torch.float32)
        return {
            "embedding": self.embedding_db.get_embedding(sequence, "pka"),
            "residue_indices": encode_residues(sequence),
            "targets": targets,
        }


def pka_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad PKAD sequences; missing pKa targets are NaN so they can be masked out of the loss."""
    if not batch:
        raise ValueError("Cannot collate an empty PKAD batch")
    return {
        "embeddings": pad_column(batch, "embedding"),
        "residue_indices": pad_column(batch, "residue_indices"),
        "targets": pad_column(batch, "targets", padding_value=float("nan")),
        "true_lengths": torch.tensor([item["embedding"].shape[0] for item in batch], dtype=torch.long),
    }
