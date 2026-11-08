from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from Bio import SeqIO

from src.config_initialisation import CONFIG
from src.data_processing.standardisation import generate_pI, normalise_data_types
from src.data_processing.verification import run_verification
from src.repo_paths import PATHS
from src.utils.amino_acid_information_tools import has_supported_length, substitute
from src.utils.constants import MIN_SEQUENCE_LENGTH, STANDARD_AMINO_ACIDS
from src.utils.dataframe_processing_tools import build_experiment_ids

DEFAULT_PH: Final[float] = 7.0
DEFAULT_TEMPERATURE_K: Final[float] = 298.15
MUTANT_PATTERN: Final[str] = r"^([A-Z])(\d+)([A-Z])$"


# Shared helpers.
def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path.resolve()}")


def _require_columns(dataframe: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{source.name} must contain columns {sorted(required)}; missing {sorted(missing)}")


def _numeric_with_default(dataframe: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Return a numeric column with missing values filled; the whole column defaults if absent."""
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(default)


def _add_mutant_sequences(experiment_df: pd.DataFrame) -> pd.DataFrame:
    experiment_df["mutant_sequence"] = [
        substitute(sequence, position, residue)
        for sequence, position, residue in zip(
            experiment_df["sequence"],
            experiment_df["zero_indexed_mutation_position"],
            experiment_df["mut_residue"],
        )
    ]
    return experiment_df


def _finalise(experiment_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add IDs, validate the rows as unlabelled ("find") data, and add mutant pIs."""
    experiment_df["experiment_id"] = build_experiment_ids(experiment_df["sequence_id"], experiment_df)
    experiment_df["ddG"] = np.nan
    experiment_df = normalise_data_types(experiment_df)
    run_verification(experiment_df, source_name, mode="find")
    return generate_pI(experiment_df)


# Input files.
def load_wt_fasta(wt_path: str | Path = PATHS.experiment_fasta) -> tuple[dict[str, str], Path]:
    """Load wild-type FASTA records and return their ID-to-sequence mapping."""
    file = Path(wt_path)
    _require_file(file, "Wild-type FASTA file (needed to run experiments)")

    sequence_dict: dict[str, str] = {}
    with open(file, "r", encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_id = str(record.id).strip()
            sequence = str(record.seq).upper()
            if not has_supported_length(sequence, CONFIG.max_sequence_length):
                raise ValueError(
                    f"Sequence '{seq_id}' has {len(sequence)} residues; supported lengths are "
                    f"{MIN_SEQUENCE_LENGTH} to {CONFIG.max_sequence_length}. "
                    "Truncate longer sequences for an approximate estimate."
                )
            if seq_id in sequence_dict:
                raise ValueError(f"Sequence ID '{seq_id}' appears more than once in {file.name}")
            sequence_dict[seq_id] = sequence

    if not sequence_dict:
        raise ValueError(f"No FASTA sequences were found in {file}")
    return sequence_dict, file


def load_pH_conditions(pH_file: str | Path = PATHS.experiment_ph_csv) -> tuple[pd.DataFrame, Path]:
    """Load optional per-sequence pH (and temperature) conditions; empty if the file is absent."""
    pH_file = Path(pH_file)
    if not pH_file.is_file():
        return pd.DataFrame(), pH_file

    conditions = pd.read_csv(pH_file)
    _require_columns(conditions, {"sequence_id", "pH"}, pH_file)
    conditions = conditions.dropna(subset=["sequence_id"]).copy()
    conditions["sequence_id"] = conditions["sequence_id"].astype(str).str.strip()
    conditions["pH"] = pd.to_numeric(conditions["pH"], errors="coerce")
    columns = [column for column in ("sequence_id", "pH", "experiment_temperature") if column in conditions]
    return conditions[columns], pH_file


# Experiment sets.
def load_sample_space(pH_file: str | Path = PATHS.experiment_ph_csv) -> pd.DataFrame:
    """Enumerate every single-residue substitution for each FASTA sequence and condition."""
    seq_dict, fasta_file = load_wt_fasta()
    conditions, pH_file = load_pH_conditions(pH_file)
    if conditions.empty:
        conditions = pd.DataFrame({"sequence_id": list(seq_dict), "pH": DEFAULT_PH})

    sequences = pd.DataFrame(list(seq_dict.items()), columns=["sequence_id", "sequence"])
    loaded_df = pd.merge(conditions, sequences, on="sequence_id", how="outer")
    unknown_ids = loaded_df.loc[loaded_df["sequence"].isna(), "sequence_id"].tolist()
    if unknown_ids:
        raise ValueError(f"{pH_file.name} contains sequence IDs {unknown_ids} which are not in {fasta_file.name}.")
    loaded_df["pH"] = _numeric_with_default(loaded_df, "pH", DEFAULT_PH)
    loaded_df["experiment_temperature"] = _numeric_with_default(
        loaded_df, "experiment_temperature", DEFAULT_TEMPERATURE_K
    )

    # One row per (condition, position), then one per possible replacement residue.
    lengths = loaded_df["sequence"].str.len()
    positions = loaded_df.loc[loaded_df.index.repeat(lengths)].copy()
    positions["zero_indexed_mutation_position"] = np.concatenate([np.arange(length) for length in lengths])
    positions["wt_residue"] = [
        sequence[position]
        for sequence, position in zip(positions["sequence"], positions["zero_indexed_mutation_position"])
    ]
    experiment_df = positions.merge(pd.DataFrame({"mut_residue": sorted(STANDARD_AMINO_ACIDS)}), how="cross")
    experiment_df = experiment_df[experiment_df["wt_residue"] != experiment_df["mut_residue"]].copy()

    return _finalise(_add_mutant_sequences(experiment_df), fasta_file.name)


def load_case_studies(cases_path: str | Path = PATHS.experiment_cases_csv) -> pd.DataFrame:
    """Build and validate experiment rows from the mutations listed in experiments.csv."""
    seq_dict, fasta_file = load_wt_fasta()
    cases_path = Path(cases_path)
    _require_file(cases_path, "Case study file experiments.csv")

    case_df = pd.read_csv(cases_path)
    _require_columns(case_df, {"sequence_id", "mutants", "pH"}, cases_path)
    if case_df["mutants"].isna().any():
        raise ValueError(f"{cases_path.name} contains rows without a mutant designation.")

    experiment_df = pd.DataFrame(index=case_df.index)
    experiment_df["sequence_id"] = case_df["sequence_id"].astype(str).str.strip()
    experiment_df["sequence"] = experiment_df["sequence_id"].map(seq_dict)
    experiment_df["pH"] = _numeric_with_default(case_df, "pH", DEFAULT_PH)
    experiment_df["experiment_temperature"] = _numeric_with_default(
        case_df, "experiment_temperature", DEFAULT_TEMPERATURE_K
    )

    unknown_ids = experiment_df.loc[experiment_df["sequence"].isna(), "sequence_id"].unique().tolist()
    if unknown_ids:
        raise ValueError(f"{cases_path.name} contains sequence IDs {unknown_ids} which are not in {fasta_file.name}.")

    mutants = case_df["mutants"].astype(str).str.strip().str.upper()
    extracted = mutants.str.extract(MUTANT_PATTERN)
    invalid = extracted.isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{cases_path.name} contains unrecognised mutation designations {mutants[invalid].tolist()}. "
            "Use one-based notation such as 'A123Q'."
        )
    experiment_df["wt_residue"] = extracted[0]
    experiment_df["zero_indexed_mutation_position"] = extracted[1].astype(int) - 1
    experiment_df["mut_residue"] = extracted[2]

    return _finalise(_add_mutant_sequences(experiment_df), f"{fasta_file.name} and {cases_path.name}")
