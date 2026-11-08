from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import median_abs_deviation, sem

from src.config_initialisation import CONFIG

MINIMUM_TOLERANCE = 3e-2  # kcal/mol; values closer than this are always treated as consistent.

STATUS_DICTIONARY = {
    "single": "Single observation accepted.",
    "all": "Consistency checks successful and all values were accepted.",
    "some": "Consistency checks successful but some inconsistent values were removed.",
    "failure": "Measurements were strongly inconsistent and were rejected.",
}

GROUP_KEYS = [
    "sequence",
    "mutant_sequence",
    "pH",
    "experiment_temperature",
    "zero_indexed_mutation_position",
    "wt_residue",
    "mut_residue",
    "mutant_pI",
]


@dataclass(frozen=True)
class AggregationResult:
    aggregate: float | None
    status: str
    accepted_values: NDArray | None
    rejected_values: NDArray


# Robust consistency checks for repeated measurements.
def check_MAD_outliers(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return a mask keeping values within the configured robust (MAD) distance of the median."""
    median = np.median(values)
    mad = median_abs_deviation(values, scale="normal")
    if mad == 0:
        return np.isclose(values, median, atol=MINIMUM_TOLERANCE)
    return np.abs(values - median) <= max(CONFIG.mad_threshold * mad, MINIMUM_TOLERANCE)


def check_standard_error(values: NDArray[np.float64]) -> bool:
    """Return whether the standard error of the mean is within the configured tolerance."""
    return values.size > 0 and float(sem(values)) <= CONFIG.sem_tolerance


def check_array(values: NDArray[np.float64]) -> AggregationResult:
    """Filter inconsistent repeat measurements and average the accepted ones.

    One value is accepted as is. Outliers are only removed (by MAD) when there are more
    than two values. The group is rejected if the remaining values disagree (SEM check).
    """
    if values.size == 0:
        return AggregationResult(None, STATUS_DICTIONARY["failure"], None, values)

    accepted = np.ones(values.size, dtype=bool)
    if values.size == 1:
        status = "single"
    else:
        if values.size > 2:
            accepted = check_MAD_outliers(values)
        if not check_standard_error(values[accepted]):
            return AggregationResult(None, STATUS_DICTIONARY["failure"], None, values)
        status = "all" if accepted.all() else "some"

    return AggregationResult(
        float(values[accepted].mean()),
        STATUS_DICTIONARY[status],
        values[accepted],
        values[~accepted],
    )


# Per-mutation aggregation and dataset curation.
def group_ddG(group: pd.DataFrame) -> pd.Series:
    """Expose a mutation group's aggregate, curation status, and audit arrays."""
    result = check_array(pd.to_numeric(group["ddG"], errors="coerce").to_numpy(dtype=float))
    return pd.Series({
        "ddG": result.aggregate,
        "curation_notes": result.status,
        "accepted_values": result.accepted_values,
        "rejected_values": result.rejected_values,
    })


def curate_data(
    dataframe: pd.DataFrame,
    max_seq_len: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group repeat measurements, curate their ddG values, and return (audit table, clean table)."""
    max_seq_len = CONFIG.max_sequence_length if max_seq_len is None else max_seq_len
    dataframe = dataframe[dataframe["sequence"].str.len() <= max_seq_len]

    print("Resolving inconsistencies across repeat measurements for a particular mutation...")
    audit_df = dataframe.groupby(GROUP_KEYS, sort=False)[["ddG"]].apply(group_ddG).reset_index()
    curated_df = (
        audit_df.drop(columns=["curation_notes", "accepted_values", "rejected_values"])
        .dropna(subset=["ddG"])
        .reset_index(drop=True)
    )
    return audit_df, curated_df
