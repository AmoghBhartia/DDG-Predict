import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config_initialisation import CONFIG
from src.embeddings_management.adding_embeddings import update_embedding_cache
from src.embeddings_management.loading_embeddings import EmbeddingsCache
from src.experiments.load_experiment import load_case_studies, load_sample_space
from src.model.ddG_predictor_model import DDGPredictor
from src.repo_paths import PATHS
from src.safe_writing import safe_writer
from src.training_and_validation.evaluation import collect_predictions, save_details
from src.training_and_validation.utils import ddg_data_loader, load_trained_model

# --find_stable_mutant runs live in their own folder under results/, one subfolder per run, so
# they sit beside the evaluation runs rather than mixed in with them at the top level.
EXPERIMENT_FOLDER = "experiment"
RUN_STATE_FILE = "run_state.json"
PREDICTIONS_FILE = "predictions.csv"
REMOTE_STATE_FILE = "kaggle_state.json"
FINGERPRINT_COLUMNS = ["experiment_id", "mutant_sequence", "pH", "experiment_temperature"]


# Run state: lets a long experiment stop at any time and continue from its last saved chunk.
def _hash_model(digest: "hashlib._Hash", model: DDGPredictor) -> None:
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())


def run_fingerprints(experiment_df: pd.DataFrame, model: DDGPredictor) -> list[str]:
    """Identify an experiment by its exact mutant list, conditions and model weights.

    A stopped run is only continued when all three are unchanged, so one predictions file never
    mixes different inputs or different trained models. The first fingerprint is the current,
    version-independent one (plain text of each row, so it is identical on any computer and any
    pandas version). The second is the format used by runs started before 5 November 2026, so
    those runs can still be continued.
    """
    current = hashlib.sha256()
    for row in experiment_df[FINGERPRINT_COLUMNS].itertuples(index=False):
        current.update(f"{row[0]}|{row[1]}|{float(row[2])!r}|{float(row[3])!r}\n".encode())
    _hash_model(current, model)

    legacy = hashlib.sha256()
    legacy.update(pd.util.hash_pandas_object(experiment_df[FINGERPRINT_COLUMNS], index=False).to_numpy().tobytes())
    _hash_model(legacy, model)
    return [current.hexdigest(), legacy.hexdigest()]


def read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    with safe_writer(path) as handle:
        json.dump(data, handle, indent=2)


def saved_rows_match(saved_ids: list[str], experiment_ids: list[str]) -> bool:
    """True if saved rows are the first rows of the experiment in order, or all of its rows in ranked order."""
    is_prefix = saved_ids == experiment_ids[: len(saved_ids)]
    is_complete_set = len(saved_ids) == len(experiment_ids) and set(saved_ids) == set(experiment_ids)
    return is_prefix or is_complete_set


def read_saved_predictions(directory: Path) -> pd.DataFrame:
    path = directory / PREDICTIONS_FILE
    return pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["experiment_id", "ddG_predictions"])


def candidate_run_directories() -> list[Path]:
    """Every folder that could hold a run, newest name first.

    Runs made before they moved into ``results/experiment/`` sit directly under ``results/``, and
    a Kaggle bundle built by an older version extracts them there too, so both layouts are
    searched. The flat layout is matched only where a run state file is actually present, which
    keeps ``evaluation/`` and ``experiment/`` themselves out of the list.
    """
    if not PATHS.results_dir.is_dir():
        return []
    directories = []
    experiment_root = PATHS.results_dir / EXPERIMENT_FOLDER
    if experiment_root.is_dir():
        directories += [path for path in experiment_root.iterdir() if path.is_dir()]
    directories += [
        path for path in PATHS.results_dir.iterdir()
        if path.is_dir() and (path / RUN_STATE_FILE).is_file()
    ]
    return sorted(directories, key=lambda path: path.name, reverse=True)


def find_unfinished_run(fingerprints: list[str], experiment_ids: list[str]) -> tuple[Path, pd.DataFrame] | None:
    """Return the newest unfinished run of this exact experiment and the predictions it saved."""
    for directory in candidate_run_directories():
        state = read_json(directory / RUN_STATE_FILE)
        if not state or state.get("status") != "running" or state.get("fingerprint") not in fingerprints:
            continue
        saved = read_saved_predictions(directory)
        if not saved_rows_match(saved["experiment_id"].astype(str).tolist(), experiment_ids):
            print(f"WARNING: The unfinished run in '{directory}' does not match its saved predictions; starting a new run.")
            continue
        if state["fingerprint"] != fingerprints[0]:
            # Upgrade an older run to the portable fingerprint so it can also be continued on another computer.
            write_json(directory / RUN_STATE_FILE, {**state, "fingerprint": fingerprints[0]})
        return directory, saved
    return None


def _new_run_directory() -> Path:
    """Create a fresh timestamped folder under results/experiment/, never reusing an existing one."""
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for suffix in ("", *(f"_{number}" for number in range(2, 1000))):
        directory = PATHS.results_dir / EXPERIMENT_FOLDER / f"{base}{suffix}"
        try:
            directory.mkdir(parents=True, exist_ok=False)
            return directory
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a new results folder for {base}")


def load_experiment_set(
    cases_path: str | Path = PATHS.experiment_cases_csv,
    pH_path: str | Path = PATHS.experiment_ph_csv,
) -> pd.DataFrame:
    """The listed case studies if experiments.csv exists, otherwise every single substitution."""
    if Path(cases_path).is_file():
        return load_case_studies(cases_path)
    return load_sample_space(pH_path)


def prepare_run(
    experiment_df: pd.DataFrame,
    model: DDGPredictor,
    for_remote: bool = False,
) -> tuple[pd.DataFrame, Path, int, str]:
    """Continue an unfinished run of this experiment if there is one, otherwise start a new run folder.

    Returns the experiment table (with saved predictions filled in), the run folder, the number of
    rows already predicted, and the run's fingerprint.
    """
    experiment_df = experiment_df.reset_index(drop=True)
    experiment_df["ddG_predictions"] = np.nan
    fingerprints = run_fingerprints(experiment_df, model)
    unfinished = find_unfinished_run(fingerprints, experiment_df["experiment_id"].astype(str).tolist())

    if unfinished is None:
        directory = _new_run_directory()
        write_json(directory / RUN_STATE_FILE, {
            "status": "running",
            "fingerprint": fingerprints[0],
            "total_rows": len(experiment_df),
            "started": datetime.now().isoformat(timespec="seconds"),
        })
        return experiment_df, directory, 0, fingerprints[0]

    directory, saved = unfinished
    remote = read_json(directory / REMOTE_STATE_FILE) or {}
    if remote.get("status") == "submitted" and not for_remote:
        raise RuntimeError(
            f"The unfinished run in '{directory}' is currently being computed on Kaggle. Run "
            "'python run_model.py --find_stable_mutant --remote kaggle' to collect its results, or delete "
            f"'{REMOTE_STATE_FILE}' in that folder to continue it on this computer instead."
        )
    saved_predictions = saved.set_index(saved["experiment_id"].astype(str))["ddG_predictions"]
    experiment_df["ddG_predictions"] = experiment_df["experiment_id"].astype(str).map(saved_predictions)
    completed_rows = len(saved)
    print(
        f"\nWARNING: Found an unfinished run of this experiment in '{directory}'. "
        f"{completed_rows} of {len(experiment_df)} predictions are already saved, so the run continues "
        f"from row {completed_rows}. To start from scratch instead, rename or delete that folder.\n",
        flush=True,
    )
    return experiment_df, directory, completed_rows, fingerprints[0]


def finish_run(experiment_df: pd.DataFrame, result_directory: Path) -> pd.DataFrame:
    """Rank mutants within each sequence and pH, write the final file and mark the run complete."""
    experiment_df["stability_rank"] = (
        experiment_df.groupby(["sequence_id", "pH"], dropna=False)["ddG_predictions"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    experiment_df = experiment_df.sort_values(
        ["sequence_id", "pH", "stability_rank"], kind="stable"
    ).reset_index(drop=True)
    file_path = result_directory / PREDICTIONS_FILE
    with safe_writer(file_path, mode="w") as handle:
        experiment_df.to_csv(handle, index=False)
    write_json(result_directory / RUN_STATE_FILE, {
        **(read_json(result_directory / RUN_STATE_FILE) or {}),
        "status": "complete",
        "finished": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"Results saved to '{file_path}'.")
    return experiment_df


# Prediction.
def _predict_chunk(
    model: DDGPredictor,
    chunk_df: pd.DataFrame,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Embed one chunk in a temporary cache and return its predictions and details."""
    with tempfile.TemporaryDirectory(prefix="ddg_chunk_", ignore_cleanup_errors=True) as temp_dir:
        update_embedding_cache(chunk_df, Path(temp_dir))
        with EmbeddingsCache(temp_dir) as embedding_db:
            loader = ddg_data_loader(chunk_df, embedding_db, shuffle=False, mode="find")
            predictions, _, details = collect_predictions(model, loader)
    return predictions, details


def generate_results(experiment_df: pd.DataFrame) -> pd.DataFrame:
    """Predict experiment rows in chunks, saving each chunk as soon as it completes.

    Progress is saved chunk by chunk, so a run stopped by Ctrl+C or a crash continues from
    its last saved chunk the next time the same experiment is run with the same model.
    """
    if experiment_df.empty:
        raise ValueError("Cannot generate predictions for an empty experiment dataframe")
    chunk_size = CONFIG.experiment_chunk_size
    model = load_trained_model()
    experiment_df, result_directory, completed_rows, _ = prepare_run(experiment_df, model)
    num_rows = len(experiment_df)
    file_path = result_directory / PREDICTIONS_FILE
    print(f"Processing {num_rows - completed_rows} of {num_rows} experiments")

    try:
        for start in range(completed_rows, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            print(f"Processing rows {start} to {end} of {num_rows}...")

            chunk_df = experiment_df.iloc[start:end].copy()
            predictions, details = _predict_chunk(model, chunk_df)
            chunk_df["ddG_predictions"] = predictions.reshape(-1).tolist()
            save_details(chunk_df["experiment_id"], details, result_directory / "details")
            experiment_df.loc[chunk_df.index, "ddG_predictions"] = chunk_df["ddG_predictions"]

            # The predictions file is the record of progress: it only ever gains complete chunks.
            with safe_writer(file_path, mode="w" if start == 0 else "a") as handle:
                chunk_df.to_csv(handle, index=False, header=start == 0)
            completed_rows = end

    except BaseException as error:
        print(
            f"\n{type(error).__name__}: stopped after {completed_rows} of {num_rows} predictions. "
            f"Progress is saved in '{result_directory}'. Run the same command again to continue from there."
        )
        raise

    return finish_run(experiment_df, result_directory)


def run_experiment(
    cases_path: str | Path = PATHS.experiment_cases_csv,
    pH_path: str | Path = PATHS.experiment_ph_csv,
) -> None:
    """Run the listed case studies if experiments.csv exists, otherwise scan every substitution."""
    generate_results(load_experiment_set(cases_path, pH_path))
