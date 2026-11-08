from typing import Final, Literal

import numpy as np
import pandas as pd

from src.utils.constants import STANDARD_AMINO_ACIDS

INVALID_LETTERS: Final[str] = f"[^{''.join(sorted(STANDARD_AMINO_ACIDS))}]"

# Individual validation rules.
def report_error(
    errors_list: list[Exception],
    criterion: pd.DataFrame,
    group_error_msg: str,
    error_msg: str,
    parameter: str | None = None,
) -> None:
    """Append a summarized validation exception for a nonempty error table."""
    if criterion.empty:
        return
    
    category_errors = [f"{group_error_msg}\n"]

    for idx, row in criterion.head(10).iterrows():
        variable = row["variable"] if parameter is None else parameter
        category_errors.append(f"Row {idx}: {variable} {error_msg}")

    if len(criterion) > 10:
        category_errors.append(f"... and {len(criterion) - 10} more rows with this error were detected.")

    errors_list.append(ValueError("\n".join(category_errors)))

def check_data_is_finite(flat_df: pd.DataFrame, errors_list: list[Exception]) -> None:
    """Record missing, unparseable, or infinite values from melted data."""
    is_number = flat_df["value"].apply(lambda x: not isinstance(x, str))
    numbers = flat_df[is_number]
    non_finite = numbers[~np.isfinite(pd.to_numeric(numbers["value"], errors="coerce"))]

    if not non_finite.empty:
        report_df = non_finite.set_index("index")
        report_error(errors_list, report_df, "Non-finite Information Error:", "contains missing, unparseable, or infinite values.")

def check_pH_range(df: pd.DataFrame, errors_list: list[Exception]) -> None:
    """Record pH values outside the supported 0-to-14 interval."""
    pH_range_error = df[(df["pH"] < 0.0) | (df["pH"]  > 14.0)]
    report_error(errors_list, pH_range_error, "pH Range Error:", "is outside the expected 0.0 to 14.0 range.", "pH")
    
def check_position_type(df: pd.DataFrame, errors_list: list[Exception]) -> None:
    """Record mutation positions that are not integer-valued."""
    position_errors = df[(df["zero_indexed_mutation_position"]  % 1 != 0)]
    report_error(errors_list, position_errors, "Non-Integer Position Error:", "contains values that are not integers.", "Position")

def check_for_unexpected_aa(flat_df: pd.DataFrame, errors_list: list[Exception]) -> None:
    """Record sequence or residue values containing non-standard amino acids."""
    text_columns = ["sequence","mutant_sequence","wt_residue","mut_residue"]
    text_data = flat_df[flat_df["variable"].isin(text_columns)]

    non_valid_aa = text_data[text_data["value"].astype(str).str.contains(INVALID_LETTERS, regex=True)]
    
    if not non_valid_aa.empty:
        report_df = non_valid_aa.set_index("index")
        report_error(errors_list, report_df, "Unexpected Amino Acid Error:", "contains letters which are not standard amino acids.")

def check_for_length_mismatches(df: pd.DataFrame, errors_list: list[Exception]) -> None:
    """Check sequence lengths, mutation bounds, and the single-mutant residue."""

    seq_length = df["sequence"].str.len()
    mutant_seq_length = df["mutant_sequence"].str.len()
    positions = df["zero_indexed_mutation_position"].fillna(-1).astype(int)

    length_mismatches = df[seq_length != mutant_seq_length]
    report_error(errors_list, length_mismatches, "Length Mismatch Error:", "does not match the mutant sequence length.", "Wildtype sequence")

    out_of_range_position = df[((positions < 0) | (positions >= seq_length))]
    report_error(errors_list, out_of_range_position, "Position Out-of-Range Error:", "is outside the length bounds of the sequence.", "Position")

    has_missing_strings = df[["sequence", "mutant_sequence"]].isna().any(axis=1).any()
    if not length_mismatches.empty or not out_of_range_position.empty or has_missing_strings:
        print("Unable to proceed with full verification due to errors in position, wildtype or mutant sequences. Exiting early and reporting errors...")
        return 

    position_array = positions.to_numpy()
    wt_sequences = df["sequence"].tolist()
    mutant_sequences = df["mutant_sequence"].tolist()
    extracted_wt = np.array([
        sequence[int(position)]
        for sequence, position in zip(wt_sequences, position_array)
    ])
    extracted_mut = np.array([
        sequence[int(position)]
        for sequence, position in zip(mutant_sequences, position_array)
    ])
    
    wt_mismatch = df[extracted_wt != df["wt_residue"].to_numpy()]
    report_error(errors_list, wt_mismatch, "Inconsistent WT Sequence Error:", "disagrees with the residue found at that position in the wildtype sequence.", "WT Residue")

    mut_mismatch = df[extracted_mut != df["mut_residue"].to_numpy()]
    report_error(errors_list, mut_mismatch, "Inconsistent MUT Sequence Error:", "disagrees with the residue found at that position in the mutant sequence.", "MUT Residue")

    # Compare only the regions outside the mutation site; sequence lengths may vary by row.
    multiple_mutation_mask = np.array([
        wildtype[:int(position)] != mutant[:int(position)]
        or wildtype[int(position) + 1:] != mutant[int(position) + 1:]
        for wildtype, mutant, position in zip(wt_sequences, mutant_sequences, position_array)
    ])
    
    multiple_mutations = df[multiple_mutation_mask]
    report_error(errors_list, multiple_mutations, "Multiple Mutation Error:", "contains alterations outside the specified single mutation point.", "Sequence structure")


# Validation orchestration.
def run_verification(dataframe: pd.DataFrame, filename: str,  mode: Literal["train/eval", "find"] = "train/eval") -> None:
    """Run all data-integrity checks and raise one ExceptionGroup if any fail."""
    verification_df = dataframe.drop(columns=["ddG"]).copy() if mode == "find" else dataframe.copy()
    flattened_df = verification_df.melt(ignore_index=False).reset_index()
    error_group = []

    check_data_is_finite(flattened_df, error_group)
    check_pH_range(verification_df, error_group)
    check_position_type(verification_df, error_group)
    check_for_unexpected_aa(flattened_df,error_group)
    check_for_length_mismatches(verification_df, error_group)

    if error_group:
        raise ExceptionGroup(f"Dataframe created from {filename} failed checks in {len(error_group)} categories", error_group)

    return 