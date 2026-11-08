from pathlib import Path

import pandas as pd

from src.data_processing.pre_cleaning import pre_cleaner
from src.data_processing.standardisation import check_columns_exist, generate_pI, normalise_data_types 
from src.data_processing.verification import run_verification
from src.repo_paths import PATHS


def load_source_data(folder: Path | str = PATHS.source_dir) -> pd.DataFrame:
    """Read and validate all CSV source files in the configured data directory."""
    directory = Path(folder)
    if not directory.is_dir():
        raise FileNotFoundError(f"Target path for loading files is unavailable: {directory}")

    target_files = sorted(directory.glob("*.csv"))
    if not target_files:
        raise FileNotFoundError(
            f"No CSV source files were found in {directory}\n"
            "--prepare trains the model on your own measurements, so it needs source data here. The "
            "curated dataset used for the published results is already in 'data/Curated data/', and "
            "training, evaluation and inference all work without running --prepare at all.\n"
            "To reproduce the published curation instead, export from FireProtDB "
            "(https://loschmidt.chemi.muni.cz/fireprotdb/) filtered to substitutions, a sequence length "
            "under 2000, and entries carrying a ddG and a pH, and save the CSV into this folder. Its "
            "terms of use are why that file is not redistributed here. See PREPARE.md."
        )

    dataframes: list[pd.DataFrame] = []
    for path in target_files:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Source file {path.name!r} contains no rows.")

        df = pre_cleaner(df)
        check_columns_exist(df, path.name)
        df = normalise_data_types(df)
        run_verification(df, path.name)
        dataframes.append(generate_pI(df))

    loaded_data = pd.concat(dataframes, ignore_index=True).drop_duplicates(ignore_index=True)
    if loaded_data.empty:
        raise ValueError(f"No valid source data remained after cleaning files in {directory}")

    return loaded_data
