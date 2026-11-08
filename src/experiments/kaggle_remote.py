"""Run --find_stable_mutant on a Kaggle GPU and merge the results back into this computer's results folder.

Flow (``python run_model.py --find_stable_mutant --remote kaggle``):

1. Prepare the run exactly as a local run would (same mutant list, same run folder, same resume rules).
2. Upload a small bundle (code, configuration, newest ddG checkpoint, the run's saved progress) as a
   private Kaggle dataset, and start a private GPU script (``kaggle_worker.py``) that uses it.
3. Check on the script every few minutes. When a session ends, download its outputs and merge the new
   predictions and detail files into the local run folder.
4. Repeat with a new session until every mutant is predicted (each session is limited to ~12 hours).

Ctrl+C on this computer only stops the checking: the Kaggle session keeps running, and the same command
later collects its results. Requires ``pip install kaggle`` and a Kaggle API token
(see REMOTE_RUNNING.md).
"""

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config_initialisation import CONFIG
from src.experiments.run_experiment import (
    PREDICTIONS_FILE,
    REMOTE_STATE_FILE,
    RUN_STATE_FILE,
    load_experiment_set,
    prepare_run,
    read_json,
    read_saved_predictions,
    saved_rows_match,
    write_json,
)
from src.repo_paths import PATHS
from src.safe_writing import safe_writer
from src.training_and_validation.checkpoints import EXPERIMENT_DDG_CHECKPOINTS, list_checkpoints
from src.training_and_validation.utils import load_trained_model

WORKER_FILE = Path(__file__).with_name("kaggle_worker.py")
BUNDLE_NAME = "ddg_bundle.zip"
MANIFEST_NAME = "ddg_bundle_manifest.json"
BUNDLED_FOLDERS = ["src", "data/Configuration", "data/Amino acid information", "data/Experiments"]
WORKER_PACKAGES = ["fair-esm", "lmdb", "pydantic-settings", "biopython"]
REMOTE_CONFIG = {"device": "cuda", "batch_size": 16, "esm_batch_size": 16}
DATASET_READY_TIMEOUT_SECONDS = 30 * 60
SETUP_HELP = (
    "To use --remote kaggle:\n"
    "  1. pip install kaggle\n"
    "  2. On kaggle.com, go to Settings > API and create a token. Set it as the KAGGLE_API_TOKEN environment\n"
    "     variable (or save it to ~/.kaggle/access_token). A legacy kaggle.json file also works.\n"
    "  3. Verify your phone number on kaggle.com (needed for GPU and internet access in scripts).\n"
    '  4. Add your username to data/Configuration/config.json, e.g. "kaggle_username": "your-name".'
)


class KaggleError(RuntimeError):
    """A Kaggle command failed or a remote session produced unusable output."""


# Kaggle command-line calls.
# Uploads that Kaggle refuses still exit successfully, printing e.g. "Dataset creation error: ..."
# or "Kernel push error: ...", so their output is checked for errors as well.
REPORTS_ERRORS_IN_OUTPUT = {("datasets", "create"), ("datasets", "version"), ("kernels", "push")}


def kaggle(*arguments: str, check: bool = True) -> str:
    """Run the Kaggle command-line tool and return its output, raising KaggleError if it failed."""
    completed = subprocess.run([sys.executable, "-m", "kaggle", *arguments], capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip()
    reported_error = tuple(arguments[:2]) in REPORTS_ERRORS_IN_OUTPUT and re.search(r"\berror\b", output, re.I)
    if check and (completed.returncode != 0 or reported_error):
        hint = f"\n\n{SETUP_HELP}" if re.search(r"401|unauthori[sz]ed|credential", output, re.I) else ""
        raise KaggleError(f"'kaggle {' '.join(arguments[:2])}' failed:\n{output}{hint}")
    return output


def _status_word(output: str) -> str:
    """Pull 'complete', 'running', 'error', ... out of a `kaggle ... status` message."""
    text = output.lower()
    for word in ("complete", "error", "cancel", "running", "queued", "ready", "pending", "new"):
        if word in text:
            return word
    return text


def dataset_file_names(dataset: str) -> set[str] | None:
    """Names of the files in the dataset's newest version (all pages), or None if Kaggle refused the request."""
    names, token = set(), None
    for _ in range(100):
        arguments = ["datasets", "files", dataset, "--page-size", "200"] + (["--page-token", token] if token else [])
        completed = subprocess.run([sys.executable, "-m", "kaggle", *arguments], capture_output=True, text=True)
        if completed.returncode != 0:
            return None
        for line in completed.stdout.splitlines():
            if line.strip():
                names.add(Path(line.split()[0]).name)
        match = re.search(r"next page token\s*=?\s*(\S+)", completed.stdout, re.I)
        if not match:
            return names
        token = match.group(1)
    return names


def dataset_status(dataset: str) -> tuple[str | None, str]:
    """Return (status word, raw output); the status is None if Kaggle refused the request."""
    completed = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", dataset],
                               capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip()
    return (_status_word(output) if completed.returncode == 0 else None), output


def token_username() -> str | None:
    """The Kaggle username the API token belongs to, as reported by `kaggle config view`."""
    match = re.search(r"username:\s*(\S+)", kaggle("config", "view", check=False))
    return match.group(1) if match and match.group(1).lower() != "none" else None


def check_setup() -> str:
    """Return the Kaggle username to use, checking that it matches the API token."""
    if importlib.util.find_spec("kaggle") is None:
        raise KaggleError(f"The Kaggle package is not installed.\n\n{SETUP_HELP}")
    configured = CONFIG.kaggle_username.strip()
    detected = token_username()
    if configured and detected and configured.lower() != detected.lower():
        raise KaggleError(
            f'config.json has "kaggle_username": "{configured}", but your Kaggle API token belongs to '
            f'"{detected}". Set "kaggle_username": "{detected}" in data/Configuration/config.json.'
        )
    username = configured or detected
    if not username:
        raise KaggleError(f"No Kaggle username is configured.\n\n{SETUP_HELP}")
    return username


# Uploading a session.
def _bundle(run_directory: Path, fingerprint: str, completed_rows: int, staging: Path) -> Path:
    """Zip the code, configuration, newest checkpoint and the run's saved progress."""
    checkpoints = list_checkpoints(EXPERIMENT_DDG_CHECKPOINTS.directory)
    if not checkpoints:
        raise KaggleError("No ddG checkpoint found. Train the model before running on Kaggle.")
    root = PATHS.root
    files = [root / "run_model.py", checkpoints[-1][1]]
    for folder in BUNDLED_FOLDERS:
        files += [p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for name in (RUN_STATE_FILE, PREDICTIONS_FILE):
        if (run_directory / name).is_file():
            files.append(run_directory / name)
    # run_path is the folder's location under results/, which is what the session needs to find
    # the run again; run_name is kept because an older worker reads only that.
    manifest = {"run_name": run_directory.name,
                "run_path": run_directory.relative_to(PATHS.results_dir).as_posix(),
                "fingerprint": fingerprint, "completed_rows": completed_rows,
                "created": datetime.now().isoformat(timespec="seconds")}

    bundle_path = staging / BUNDLE_NAME
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    print(f"Bundle ready ({bundle_path.stat().st_size / 1e6:.0f} MB, {completed_rows} predictions already saved).")
    return bundle_path


def _upload_dataset(username: str, run_directory: Path, fingerprint: str, completed_rows: int) -> tuple[str, str]:
    """Upload the bundle as a new dataset version; return (dataset, marker file name of this upload)."""
    dataset = f"{username}/{CONFIG.kaggle_slug}-data"
    marker = f"ddg_upload_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.txt"
    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        _bundle(run_directory, fingerprint, completed_rows, staging)
        (staging / marker).write_text(f"{run_directory.name}\n", encoding="utf-8")
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": f"{CONFIG.kaggle_slug.replace('-', ' ')} data",
            "id": dataset,
            "licenses": [{"name": "CC0-1.0"}],
        }), encoding="utf-8")
        exists = dataset_status(dataset)[0] is not None
        print(f"Uploading to the private Kaggle dataset {dataset} ...")
        if exists:
            kaggle("datasets", "version", "-p", str(staging), "-m", f"run {run_directory.name}", "-t", "-d")
        else:
            kaggle("datasets", "create", "-p", str(staging), "-t")

    _wait_for_dataset(dataset, marker)
    return dataset, marker


def _wait_for_dataset(dataset: str, marker: str) -> None:
    """Block until this upload's marker file is listed in the dataset.

    Kaggle processes a new version in the background: until then the dataset can answer
    "forbidden", and its status still describes the previous version. Waiting for the marker
    this upload wrote is the only reliable way to know the new version is the live one.
    """
    print("Waiting for Kaggle to finish processing the upload ...")
    waited, output = 0, ""
    while waited <= DATASET_READY_TIMEOUT_SECONDS:
        status, output = dataset_status(dataset)
        if status == "error":
            raise KaggleError(f"Kaggle could not process the dataset {dataset}:\n{output}")
        if status == "ready" and marker in (dataset_file_names(dataset) or set()):
            return
        time.sleep(15)
        waited += 15
    raise KaggleError(
        f"The upload to the Kaggle dataset {dataset} was not ready within 30 minutes. Check the dataset at "
        f"https://www.kaggle.com/datasets/{dataset}. Last status:\n{output}"
    )


def _push_session(username: str, dataset: str, marker: str, run_directory: Path) -> str:
    kernel = f"{username}/{CONFIG.kaggle_slug}-runner"
    settings = {
        "run_name": run_directory.name,
        "run_path": run_directory.relative_to(PATHS.results_dir).as_posix(),
        "dataset": dataset,
        "upload_marker": marker,
        "session_hours": CONFIG.kaggle_session_hours,
        "config": REMOTE_CONFIG,
        "packages": WORKER_PACKAGES,
    }
    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        worker = WORKER_FILE.read_text(encoding="utf-8").replace("__SETTINGS__", json.dumps(settings))
        (staging / "kaggle_worker.py").write_text(worker, encoding="utf-8")
        (staging / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel,
            "title": f"{CONFIG.kaggle_slug.replace('-', ' ')} runner",
            "code_file": "kaggle_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": CONFIG.kaggle_machine_shape,
            "dataset_sources": [dataset],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }), encoding="utf-8")
        accelerator = ["--accelerator", CONFIG.kaggle_machine_shape] if CONFIG.kaggle_machine_shape else []
        kaggle("kernels", "push", "-p", str(staging), *accelerator)
    print(f"Started a Kaggle GPU session: https://www.kaggle.com/code/{kernel}")
    return kernel


# Waiting for and collecting a session.
def _wait_for_session(kernel: str) -> str:
    started = time.monotonic()
    while True:
        status = _status_word(kaggle("kernels", "status", kernel))
        minutes = (time.monotonic() - started) / 60
        print(f"[{datetime.now():%H:%M}] Kaggle session {status} ({minutes:.0f} min since checking began)", flush=True)
        if status in ("complete", "error", "cancel"):
            return status
        time.sleep(CONFIG.kaggle_poll_minutes * 60)


def _session_log_text(path: Path) -> str:
    """Kaggle's own session log is a JSON list of {"stream_name", "data"} entries; fall back to plain text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return "".join(entry.get("data", "") for entry in json.loads(raw))
    except (ValueError, AttributeError, TypeError):
        return raw


def _log_tail(download: Path, kernel: str, lines: int = 25) -> str:
    """Last lines of the worker's log, or of Kaggle's session log when the worker failed before logging."""
    worker_log = download / "worker.log"
    if worker_log.is_file() and worker_log.read_text(encoding="utf-8").strip():
        text = worker_log.read_text(encoding="utf-8")
    else:
        session_logs = sorted(download.glob("*.log"), key=lambda path: path.name != f"{kernel.split('/')[1]}.log")
        text = _session_log_text(session_logs[0]) if session_logs else ""
    tail = "\n".join(text.rstrip().splitlines()[-lines:])
    return tail or f"(no log was returned; open https://www.kaggle.com/code/{kernel} and look at its Logs tab)"


def _collect_session(kernel: str, run_directory: Path, fingerprint: str, experiment_ids: list[str]) -> int:
    """Download a finished session's outputs and merge them into the local run folder; return rows now saved."""
    local_rows = len(read_saved_predictions(run_directory))
    with tempfile.TemporaryDirectory() as download_name:
        download = Path(download_name)
        kaggle("kernels", "output", kernel, "-p", str(download), "-o", "-q")
        log_tail = _log_tail(download, kernel)

        remote_state = read_json(download / RUN_STATE_FILE)
        remote_predictions = download / PREDICTIONS_FILE
        if remote_state is None or not remote_predictions.is_file():
            raise KaggleError(f"The Kaggle session produced no results. Last lines of its log:\n{log_tail}")
        if remote_state.get("fingerprint") != fingerprint:
            raise KaggleError("The Kaggle session's run does not match this experiment and model; results not merged.")
        saved = pd.read_csv(remote_predictions)
        if not saved_rows_match(saved["experiment_id"].astype(str).tolist(), experiment_ids):
            raise KaggleError("The Kaggle session's predictions do not match this experiment; results not merged.")

        if len(saved) >= local_rows:
            with safe_writer(run_directory / PREDICTIONS_FILE, "wb") as handle:
                handle.write(remote_predictions.read_bytes())
            write_json(run_directory / RUN_STATE_FILE, remote_state)
            if (download / "details.zip").is_file():
                with zipfile.ZipFile(download / "details.zip") as archive:
                    archive.extractall(run_directory / "details")
        if len(saved) <= local_rows and remote_state.get("status") != "complete":
            raise KaggleError(f"The Kaggle session made no progress. Last lines of its log:\n{log_tail}")
        return len(saved)


# Entry point.
def run_on_kaggle() -> None:
    username = check_setup()
    experiment_df = load_experiment_set()
    model = load_trained_model()
    experiment_df, run_directory, completed_rows, fingerprint = prepare_run(experiment_df, model, for_remote=True)
    experiment_ids = experiment_df["experiment_id"].astype(str).tolist()
    total = len(experiment_df)
    session = 0

    while (read_json(run_directory / RUN_STATE_FILE) or {}).get("status") != "complete":
        remote = read_json(run_directory / REMOTE_STATE_FILE) or {}
        if remote.get("status") == "submitted":
            kernel = remote["kernel"]
            print(f"Checking on the Kaggle session already running for this run ({kernel}).")
        else:
            session += 1
            completed_rows = len(read_saved_predictions(run_directory))
            print(f"\nKaggle session {session}: {completed_rows} of {total} predictions saved so far.")
            dataset, marker = _upload_dataset(username, run_directory, fingerprint, completed_rows)
            kernel = _push_session(username, dataset, marker, run_directory)
            write_json(run_directory / REMOTE_STATE_FILE, {
                "status": "submitted", "kernel": kernel, "dataset": dataset,
                "submitted": datetime.now().isoformat(timespec="seconds"),
            })

        try:
            status = _wait_for_session(kernel)
        except KeyboardInterrupt:
            print(
                "\nStopped checking. The Kaggle session keeps running; run "
                "'python run_model.py --find_stable_mutant --remote kaggle' again to collect its results."
            )
            raise
        try:
            saved_rows = _collect_session(kernel, run_directory, fingerprint, experiment_ids)
        except KaggleError:
            # Mark the session as failed so the next attempt starts a fresh session instead of re-reading this one.
            write_json(run_directory / REMOTE_STATE_FILE, {
                **(read_json(run_directory / REMOTE_STATE_FILE) or {}), "status": "failed", "last_session_status": status,
            })
            raise
        write_json(run_directory / REMOTE_STATE_FILE, {
            **(read_json(run_directory / REMOTE_STATE_FILE) or {}),
            "status": "collected", "last_session_status": status,
            "collected": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"Collected results from Kaggle: {saved_rows} of {total} predictions saved in '{run_directory}'.")

    print(f"Experiment complete. Results saved to '{run_directory / PREDICTIONS_FILE}'.")

# Remote training.
TRAINING_FOLDERS = BUNDLED_FOLDERS + ["data/Curated data", "data/Pkad_cleaned", "data/Evaluation splits"]
# The audit archive is tens of megabytes of near-duplicate history and nothing reads it remotely.
TRAINING_EXCLUDED = ("Historical Logs",)
TRAINING_OUTPUTS = {
    "evaluation": ("performance.csv", "performance_metrics.json"),
    "experiment": ("experiment_checkpoints.zip",),
}


def _training_bundle(job: str, staging: Path) -> Path:
    """Zip the code and the curated data a training session needs.

    No embeddings are uploaded: the session recomputes them from the curated sequences on the
    GPU, which is faster than sending several gigabytes over a home connection. No checkpoint is
    uploaded either, because training starts from scratch.
    """
    root = PATHS.root
    files = [root / "run_model.py"]
    for folder in TRAINING_FOLDERS:
        directory = root / folder
        if directory.is_dir():
            files += [
                p for p in directory.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
                and not any(excluded in p.parts for excluded in TRAINING_EXCLUDED)
            ]
    manifest = {"job": job, "created": datetime.now().isoformat(timespec="seconds")}
    bundle_path = staging / BUNDLE_NAME
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    print(f"Bundle ready ({bundle_path.stat().st_size / 1e6:.0f} MB).")
    return bundle_path


def _upload_training_dataset(username: str, job: str) -> tuple[str, str]:
    """Upload the training bundle as a private dataset version; return (dataset, marker)."""
    dataset = f"{username}/{CONFIG.kaggle_slug}-train"
    marker = f"ddg_upload_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.txt"
    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        _training_bundle(job, staging)
        (staging / marker).write_text(f"{job}\n", encoding="utf-8")
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": f"{CONFIG.kaggle_slug.replace('-', ' ')} training data",
            "id": dataset,
            "licenses": [{"name": "CC0-1.0"}],
        }), encoding="utf-8")
        exists = dataset_status(dataset)[0] is not None
        print(f"Uploading to the private Kaggle dataset {dataset} ...")
        if exists:
            kaggle("datasets", "version", "-p", str(staging), "-m", f"{job} training", "-t", "-d")
        else:
            kaggle("datasets", "create", "-p", str(staging), "-t")
    _wait_for_dataset(dataset, marker)
    return dataset, marker


def _push_training_session(username: str, dataset: str, marker: str, job: str) -> str:
    """Push the training kernel and return its slug."""
    kernel = f"{username}/{CONFIG.kaggle_slug}-trainer"
    settings = {
        "job": job,
        "dataset": dataset,
        "upload_marker": marker,
        "session_hours": CONFIG.kaggle_session_hours,
        "packages": WORKER_PACKAGES,
        "config": {**REMOTE_CONFIG, "num_epochs": CONFIG.num_epochs,
                   "pka_num_epochs": CONFIG.pka_num_epochs,
                   "random_seed": CONFIG.random_seed},
    }
    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        worker = WORKER_FILE.read_text(encoding="utf-8").replace("__SETTINGS__", json.dumps(settings))
        (staging / "kaggle_worker.py").write_text(worker, encoding="utf-8")
        (staging / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel,
            "title": f"{CONFIG.kaggle_slug.replace('-', ' ')} trainer",
            "code_file": "kaggle_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [dataset],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }), encoding="utf-8")
        print(f"Starting the private Kaggle session {kernel} on a {CONFIG.kaggle_machine_shape} ...")
        kaggle("kernels", "push", "-p", str(staging))
    return kernel


def _collect_training(kernel: str, job: str) -> None:
    """Download a finished training session's outputs into the repository."""
    with tempfile.TemporaryDirectory() as download_name:
        download = Path(download_name)
        kaggle("kernels", "output", kernel, "-p", str(download), check=False)
        log_tail = _log_tail(download, kernel)

        if job == "evaluation":
            bundle = download / "validation_results.zip"
            if not bundle.is_file():
                raise KaggleError(f"The session returned no validation results. Last lines of its log:\n{log_tail}")
            with zipfile.ZipFile(bundle) as archive:
                names = archive.namelist()
                archive.extractall(PATHS.results_dir)
            for name in ("performance.csv", "performance_metrics.json"):
                produced = download / name
                if produced.is_file():
                    with safe_writer(PATHS.performance_dir / name, "wb") as handle:
                        handle.write(produced.read_bytes())
            runs = len({name.split("/")[1] for name in names if name.count("/") >= 2})
            print(f"Collected {runs} validation runs ({len(names)} files) into {PATHS.results_dir}.")
            return

        bundle = download / "experiment_checkpoints.zip"
        if not bundle.is_file():
            raise KaggleError(f"The session returned no checkpoint. Last lines of its log:\n{log_tail}")
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(PATHS.root)
        print(f"Collected the trained checkpoint into {EXPERIMENT_DDG_CHECKPOINTS.directory}.")


def run_training_on_kaggle(job: str) -> None:
    """Train on a free Kaggle GPU and bring the results back.

    One session does the whole job. A validation sweep returns its results files and summary;
    the checkpoints stay on Kaggle, because thirty of them is close to two gigabytes and they
    are only a means to the measurement. An experiment run returns the checkpoint itself.
    """
    if job not in TRAINING_OUTPUTS:
        raise KaggleError(f"Unknown training job {job!r}")
    username = check_setup()

    if job == "evaluation":
        from src.data_loading.saving_validation_and_training_data import existing_split_numbers

        split_count = len(existing_split_numbers())
        if not split_count:
            raise KaggleError(
                "No training splits are prepared, so there is nothing to send. Run --prepare first."
            )
        runs = split_count * CONFIG.evaluation_states
        print(f"\nThis session will train {runs} models ({split_count} prepared training splits x "
              f"{CONFIG.evaluation_states} initial states) of {CONFIG.num_epochs} epochs. A session "
              f"stops itself after {CONFIG.kaggle_session_hours} h; if the sweep does not fit, lower "
              f"evaluation_states in config.json, because a stopped sweep starts again from the "
              f"beginning.")

    dataset, marker = _upload_training_dataset(username, job)
    kernel = _push_training_session(username, dataset, marker, job)
    status = _wait_for_session(kernel)
    if status != "complete":
        raise KaggleError(f"The Kaggle session finished with status {status!r}.")
    _collect_training(kernel, job)
