import re
from pathlib import Path
from typing import IO

import pandas as pd

from src.safe_writing import safe_writer

# Characters Windows does not allow in a file name, plus the path separators.
UNSAFE_IN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def identifier_file_stem(identifier: str) -> str:
    """Return a file-name-safe form of an experiment identifier.

    Identifiers come from FASTA headers, which routinely contain characters Windows forbids
    in file names (``tr|Q308M6|Q308M6_HUMAN``), and a ``/`` or a leading ``..`` would place the
    detail file outside its run folder. Both are replaced with underscores. The identifier
    itself is unchanged in the results table; only the file name is rewritten.
    """
    stem = UNSAFE_IN_FILENAME.sub("_", str(identifier)).strip(" .")
    return stem or "unnamed"


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a CSV file, with a clear error if it does not exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected data file was not found: {path}. Run --prepare first if it is generated.")
    return pd.read_csv(path)


def save_dataframe(target: str | Path | IO[str], dataframe: pd.DataFrame) -> None:
    """Write a DataFrame (without its index) to an open text stream, or atomically to a CSV path."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(dataframe)}")
    if hasattr(target, "write"):
        dataframe.to_csv(target, index=False)
        return
    with safe_writer(target, "w") as handle:
        dataframe.to_csv(handle, index=False)


def build_experiment_ids(prefix: pd.Series, dataframe: pd.DataFrame) -> pd.Series:
    """Build unique, one-based mutation identifiers such as ``P53_A92S``.

    Rows that would otherwise share an identifier (the same mutation measured at
    several pH values or temperatures) receive a condition suffix, e.g.
    ``P53_A92S_pH5.5_T298.15K``, so that per-row detail files are never overwritten.
    """
    one_based_position = (
        pd.to_numeric(dataframe["zero_indexed_mutation_position"]).astype(int) + 1
    ).astype(str)
    identifiers = (
        prefix.astype(str)
        + "_"
        + dataframe["wt_residue"].astype(str)
        + one_based_position
        + dataframe["mut_residue"].astype(str)
    )

    duplicated = identifiers.duplicated(keep=False)
    if duplicated.any():
        condition = "_pH" + dataframe["pH"].map(lambda value: f"{float(value):g}")
        if "experiment_temperature" in dataframe:
            condition = condition + "_T" + dataframe["experiment_temperature"].map(
                lambda value: f"{float(value):g}K"
            )
        identifiers = identifiers.where(~duplicated, identifiers + condition)

    still_duplicated = identifiers.duplicated(keep=False)
    if still_duplicated.any():
        counter = (identifiers.groupby(identifiers).cumcount() + 1).astype(str)
        identifiers = identifiers.where(~still_duplicated, identifiers + "_" + counter)

    return identifiers
