import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import Final

REQUIRED_COLUMNS: Final[set[str]] = {
    "sequence", "mutant_sequence", "pH", 
    "zero_indexed_mutation_position",
    "wt_residue", "mut_residue", "ddG", "experiment_temperature"
}

def check_columns_exist(
    df: pd.DataFrame,
    filename: str,
    required_columns: set[str] = REQUIRED_COLUMNS,
) -> None:
    """Raise an error when a source table lacks required mutation columns."""
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataframe created from {filename} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

def normalise_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric fields and normalize sequence/residue text columns."""
    for column in ["zero_indexed_mutation_position", "pH", "ddG", "experiment_temperature"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["sequence", "mutant_sequence", "wt_residue", "mut_residue"]:
        df[column] = df[column].astype(str).str.strip().str.upper()
        df[column] = df[column].replace(["", "NAN", "NONE", "INF", "INFINITY"], np.nan)

    return df

def generate_pI(df: pd.DataFrame) -> pd.DataFrame:
    """Add a calculated mutant isoelectric-point column to the DataFrame."""
    print("Pre-calculating mutant isoelectric points...")
    
    df["mutant_pI"] = [
        float(ProteinAnalysis(seq).isoelectric_point()) 
        for seq in df["mutant_sequence"]
    ]

    return df
