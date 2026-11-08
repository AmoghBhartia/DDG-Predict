import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from src.repo_paths import PATHS
from src.utils.constants import MIN_SEQUENCE_LENGTH, STANDARD_AMINO_ACIDS


def json_loader(target_file: str | Path) -> Any:
    """Load and return JSON content from a filesystem path."""
    with open(Path(target_file), "r", encoding="utf-8") as info:
        return json.load(info)


@lru_cache(maxsize=1)
def load_aa_info() -> dict[str, dict[str, Any]]:
    """Load amino-acid metadata; its key order defines every residue index in the project."""
    aa_info = json_loader(PATHS.aa_info_json)
    if set(aa_info) != STANDARD_AMINO_ACIDS:
        raise ValueError(f"{PATHS.aa_info_json} must define exactly the 20 standard amino acids")
    for residue, info in aa_info.items():
        if "titratable" not in info:
            raise ValueError(f"Residue {residue!r} must declare \"titratable\" in {PATHS.aa_info_json}")
        if info["titratable"] and "pKa" not in info:
            raise ValueError(f"Titratable residue {residue!r} must define a \"pKa\"")
    return aa_info


@lru_cache(maxsize=1)
def residue_to_index() -> dict[str, int]:
    """Map each residue letter to its index in ``amino_acid_info.json``."""
    return {residue: index for index, residue in enumerate(load_aa_info())}


def encode_residues(sequence: str) -> torch.Tensor:
    """Convert a protein sequence into a ``torch.long`` tensor of residue indices."""
    mapping = residue_to_index()
    try:
        return torch.tensor([mapping[residue] for residue in sequence], dtype=torch.long)
    except KeyError as error:
        raise ValueError(f"Sequence contains non-standard residue {error.args[0]!r}") from error


def titratable_residues() -> torch.Tensor:
    """Return, in residue-index order, which side chains carry a titratable group.

    This is the mask the model uses to decide which residues may take a pKa shift. Residues
    outside it are excluded outright rather than being given a large placeholder pKa.
    """
    return torch.tensor(
        [bool(info["titratable"]) for info in load_aa_info().values()], dtype=torch.bool
    )


def unfolded_pkas() -> torch.Tensor:
    """Return the unfolded (model compound) pKa of each residue in index order.

    Non-titratable residues are given 0.0, which never reaches the linkage equation: their
    predicted shift is masked to zero, so their folded and unfolded terms cancel exactly.
    """
    return torch.tensor(
        [float(info["pKa"]) if info["titratable"] else 0.0 for info in load_aa_info().values()],
        dtype=torch.float32,
    )


def is_standard_sequence(sequence: str) -> bool:
    """Return whether a sequence only contains the 20 standard amino acids."""
    return set(sequence).issubset(STANDARD_AMINO_ACIDS)


def has_supported_length(sequence: str, max_length: int) -> bool:
    """Return whether a sequence length is within the range the model supports."""
    return MIN_SEQUENCE_LENGTH <= len(sequence) <= max_length


def substitute(sequence: str, position: int, residue: str) -> str:
    """Return ``sequence`` with the residue at zero-based ``position`` replaced.

    Positions are not range-checked here so that ``run_verification`` can report every
    out-of-range position together.
    """
    position = int(position)
    return sequence[:position] + residue + sequence[position + 1:]
