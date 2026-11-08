"""Curate the source measurements and divide them into training splits.

Splitting happens here, during ``--prepare``, rather than at training time. Two reasons:

* Clustering needs MMseqs2, which is not native to Windows, and ``--prepare`` is already the
  step that has to run under WSL. Training then needs no clustering at all.
* The splits are part of the prepared dataset, so they can be inspected, shipped and reused
  rather than being recomputed — identically, one hopes — by every command that needs them.

``--prepare`` writes the whole curated dataset, one folder per split, and a compact categorisation
table recording which split each sequence belongs to. The split folders are rebuilt from
scratch every time, so they always describe the current curation; the audit log in
``Historical Logs`` keeps the composition of every earlier division.
"""

import json
import shutil
from contextlib import ExitStack
from datetime import datetime

import pandas as pd

from src.config_initialisation import CONFIG
from src.data_loading.loading_source_data import load_source_data
from src.data_processing.clustering import cluster_sequences
from src.data_processing.curation import curate_data
from src.data_processing.verification import run_verification
from src.embeddings_management.adding_embeddings import update_embedding_cache
from src.repo_paths import PATHS
from src.safe_writing import safe_writer
from src.utils.dataframe_processing_tools import read_dataframe, save_dataframe

SPLIT_FOLDER_PREFIX = "split_"
TRAINING_FILE = "training_data.csv"
TESTING_FILE = "testing_data.csv"


def split_folder(number: int):
    """Return the folder holding one split's training and testing-set files."""
    return PATHS.curated_dir / f"{SPLIT_FOLDER_PREFIX}{number}"


def existing_split_numbers() -> list[int]:
    """Return the split numbers currently prepared, in order."""
    numbers = []
    for path in PATHS.curated_dir.glob(f"{SPLIT_FOLDER_PREFIX}*"):
        if path.is_dir() and path.name[len(SPLIT_FOLDER_PREFIX):].isdigit():
            numbers.append(int(path.name[len(SPLIT_FOLDER_PREFIX):]))
    return sorted(numbers)


def assign_splits(dataframe: pd.DataFrame, split_count: int) -> pd.DataFrame:
    """Assign every sequence to a split, balancing the splits by row count.

    Returns the categorisation table: one row per unique sequence with its cluster and split.

    Clusters are placed largest first into whichever split currently holds the fewest rows. A
    cluster cannot be divided, so a protein with more measurements than an average split will
    always make its own split larger; balancing by row count keeps the remaining splits
    comparable, which balancing by cluster count would not.
    """
    if split_count < 2:
        raise ValueError("evaluation_splits must be at least 2")

    clustered = cluster_sequences(dataframe["sequence"])
    cluster_of = dict(zip(clustered["sequence"], clustered["cluster_representative"]))
    missing = set(dataframe["sequence"]) - cluster_of.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} curated sequences were not clustered")

    rows_per_cluster = dataframe["sequence"].map(cluster_of).value_counts()
    if len(rows_per_cluster) < split_count:
        raise ValueError(
            f"Only {len(rows_per_cluster)} sequence clusters remain after curation, which cannot be "
            f"divided into {split_count} splits. Lower evaluation_splits."
        )

    totals = [0] * split_count
    split_of_cluster: dict[str, int] = {}
    for representative, count in rows_per_cluster.items():
        target = min(range(split_count), key=lambda index: (totals[index], index))
        split_of_cluster[str(representative)] = target + 1
        totals[target] += int(count)

    categorisation = pd.DataFrame({"sequence": sorted(cluster_of)})
    categorisation["cluster_representative"] = categorisation["sequence"].map(cluster_of)
    categorisation["split"] = categorisation["cluster_representative"].map(split_of_cluster)
    return categorisation


def write_splits(dataframe: pd.DataFrame, categorisation: pd.DataFrame) -> list[dict]:
    """Rebuild every split folder from scratch and return a description of each split."""
    for number in existing_split_numbers():
        shutil.rmtree(split_folder(number), ignore_errors=True)

    split_of = dict(zip(categorisation["sequence"], categorisation["split"]))
    unassigned = set(dataframe["sequence"]) - split_of.keys()
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} curated sequences are missing from the split categorisation table, so they "
            f"would never be in a testing set. Rerun --prepare to rebuild both together."
        )
    descriptions = []
    for number in sorted(categorisation["split"].unique()):
        in_testing_set = dataframe["sequence"].map(split_of) == number
        training_rows, testing_rows = dataframe[~in_testing_set], dataframe[in_testing_set]
        folder = split_folder(int(number))
        folder.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            save_dataframe(stack.enter_context(safe_writer(folder / TRAINING_FILE)), training_rows)
            save_dataframe(stack.enter_context(safe_writer(folder / TESTING_FILE)), testing_rows)

        run_verification(read_dataframe(folder / TRAINING_FILE), f"{folder.name}/{TRAINING_FILE}")
        run_verification(read_dataframe(folder / TESTING_FILE), f"{folder.name}/{TESTING_FILE}")
        descriptions.append({
            "split": int(number),
            "testing_set_rows": int(len(testing_rows)),
            "training_rows": int(len(training_rows)),
            "testing_set_sequences": sorted(testing_rows["sequence"].unique()),
        })
    return descriptions


def load_split(number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one prepared split's training and testing-set rows, rebuilding it if only the
    categorisation table is present.

    A clone of the repository carries the curated data and the categorisation table but not the
    split folders, so this rebuilds them without needing MMseqs2.
    """
    folder = split_folder(number)
    if not (folder / TRAINING_FILE).is_file():
        rebuild_splits_from_categorisation()
    if not (folder / TRAINING_FILE).is_file():
        raise FileNotFoundError(
            f"Split {number} has not been prepared. Run --prepare, or check that "
            f"{PATHS.split_categorisation_csv.name} lists this split."
        )
    return read_dataframe(folder / TRAINING_FILE), read_dataframe(folder / TESTING_FILE)


def rebuild_splits_from_categorisation() -> None:
    """Recreate the split folders from the curated data and the categorisation table."""
    if not PATHS.split_categorisation_csv.is_file():
        return
    dataframe = read_dataframe(PATHS.curated_all_csv)
    categorisation = read_dataframe(PATHS.split_categorisation_csv)
    print(f"Rebuilding {categorisation['split'].nunique()} split folders from "
          f"{PATHS.split_categorisation_csv.name} ...")
    write_splits(dataframe, categorisation)


def split_pkad_path(number: int):
    """Return the PKAD table prepared for one training split."""
    return PATHS.evaluation_data_dir / f"training_set_{number}_pkad.csv"


def prepare_split_pkad(pkad: pd.DataFrame) -> None:
    """Write a PKAD table per training split, with chains similar to its testing-set proteins removed.

    This runs during --prepare for the same reason the splitting does: it needs MMseqs2, and
    doing it here means training never clusters anything. That is what lets validation training
    run on Windows and on a Kaggle session, neither of which has MMseqs2.
    """
    from src.data_processing.pkad_cleaning import remove_testing_set_similar_sequences

    numbers = existing_split_numbers()
    if not numbers:
        raise RuntimeError("prepare_split_pkad must run after the training splits are written")

    print(f"\nFiltering PKAD for each of the {len(numbers)} training splits ...")
    for number in numbers:
        _, testing_set = load_split(number)
        testing_set_path = PATHS.evaluation_data_dir / f"training_set_{number}_testing_set.csv"
        save_dataframe(testing_set_path, testing_set[["sequence"]].drop_duplicates())
        filtered = remove_testing_set_similar_sequences(pkad, testing_set_path)
        if filtered.empty:
            raise ValueError(
                f"No PKAD records remain for training split {number} after removing chains similar to "
                f"its testing-set proteins, so its pKa head could not be pretrained without leakage."
            )
        save_dataframe(split_pkad_path(number), filtered)
        print(f"  split {number}: {len(pkad)} -> {len(filtered)} PKAD records")


def prepare_curated_dataset() -> None:
    """Curate the source data, embed it, and divide it into training splits."""
    audit_df, dataframe = curate_data(load_source_data())
    unique_sequences = dataframe["sequence"].nunique()
    if unique_sequences < 2:
        raise ValueError(
            "At least 2 unique wild-type sequences are needed to divide the data; "
            f"only {unique_sequences} were found after curation."
        )

    dataframe = update_embedding_cache(dataframe)
    categorisation = assign_splits(dataframe, CONFIG.evaluation_splits)

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    audit_directory = PATHS.curated_dir / "Historical Logs"
    with ExitStack() as stack:
        save_dataframe(stack.enter_context(safe_writer(PATHS.curated_all_csv)), dataframe)
        save_dataframe(stack.enter_context(safe_writer(PATHS.split_categorisation_csv)), categorisation)
        save_dataframe(stack.enter_context(safe_writer(audit_directory / f"curated_data_{run_time}.csv")), dataframe)
        save_dataframe(stack.enter_context(safe_writer(audit_directory / f"audit_data_{run_time}.csv")), audit_df)

    descriptions = write_splits(dataframe, categorisation)

    audit_log = {
        "time_curated_data_was_curated": run_time,
        "seed": CONFIG.random_seed,
        "number_of_splits": CONFIG.evaluation_splits,
        "split_method": "ProtClust/MMseqs2 cluster split, balanced by row count",
        "mmseqs_min_sequence_identity": CONFIG.min_seq_id,
        "mmseqs_coverage": CONFIG.coverage,
        "mmseqs_cov_mode": CONFIG.cov_mode,
        "total_rows": int(len(dataframe)),
        "unique_wild_type_sequences": int(unique_sequences),
        "clusters": int(categorisation["cluster_representative"].nunique()),
        "splits": descriptions,
    }
    with safe_writer(audit_directory / f"split_log_{run_time}.json") as handle:
        json.dump(audit_log, handle, indent=2)

    print(
        f"\nLoading Complete!\n"
        f"Curated rows:      {len(dataframe)} (saved to {PATHS.curated_all_csv})\n"
        f"Wild-type sequences: {unique_sequences} in {audit_log['clusters']} clusters\n"
        f"Training splits:   {CONFIG.evaluation_splits}, each in {PATHS.curated_dir.name}/split_<n>/"
    )
    for entry in descriptions:
        print(f"  split {entry['split']}: {entry['testing_set_rows']} in the testing set, "
              f"{entry['training_rows']} for training")
    print(f"The composition of every split is recorded in "
          f"{audit_directory.name}/split_log_{run_time}.json")
