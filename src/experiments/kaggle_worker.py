"""Runs on Kaggle: continue a DDG-Predict experiment on the GPU and package the new results.

This file is uploaded by ``src/experiments/kaggle_remote.py``; it is not run on your own computer.
The settings line below is filled in when the job is sent. The worker:

1. finds the uploaded bundle (code, configuration, checkpoint and the run's saved progress);
2. installs the few packages Kaggle does not provide;
3. runs the normal ``--find_stable_mutant`` on the GPU, which continues the run from its last saved
   chunk, and stops it safely (as Ctrl+C would) before Kaggle's session limit;
4. writes predictions.csv, run_state.json, the new detail files (details.zip) and a log as outputs.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path

SETTINGS = json.loads(r"""__SETTINGS__""")
MANIFEST_NAME = "ddg_bundle_manifest.json"
BUNDLE_NAME = "ddg_bundle.zip"
# The T4 fragments badly across many differently-sized ESM batches: the first failed session had
# 4.77 GiB reserved but unallocated. Expandable segments let the allocator reuse those blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

INPUT_ROOT = Path(os.environ.get("DDG_KAGGLE_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("DDG_KAGGLE_WORKING", "/kaggle/working"))
REPOSITORY = Path(os.environ.get("DDG_KAGGLE_SCRATCH", "/tmp/ddg_repository"))
STARTED = time.monotonic()
STOP_GRACE_SECONDS = SETTINGS.get("stop_grace_seconds", 15 * 60)  # time allowed to save progress after stopping
MIN_RUN_SECONDS = SETTINGS.get("min_run_seconds", 60)
MOUNT_WAIT_SECONDS = SETTINGS.get("mount_wait_seconds", 2 * 60)  # time allowed for the dataset to appear


def log(message: str) -> None:
    print(message, flush=True)
    with open(WORKING / "worker.log", "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def input_files(root: Path = INPUT_ROOT, errors: list[str] | None = None) -> list[Path]:
    """Every file under ``root``. Kaggle may mount datasets through symlinked folders, which Path.rglob
    does not enter on Python 3.13+, so walk with followlinks=True and record listing errors."""
    def record(error: OSError) -> None:
        if errors is not None:
            errors.append(f"{error.filename}: {error.strerror}")

    return sorted(
        Path(folder) / name
        for folder, _, names in os.walk(root, followlinks=True, onerror=record)
        for name in names
    )


def describe_inputs(root: Path = INPUT_ROOT, depth: int = 5) -> str:
    """Folder tree under ``root`` (with any errors), to show what Kaggle actually attached."""
    lines = []

    def visit(folder: Path, level: int) -> None:
        try:
            entries = sorted(os.scandir(folder), key=lambda entry: entry.name)
        except OSError as error:
            lines.append("  " * level + f"[cannot list {folder}: {error}]")
            return
        for entry in entries:
            kind = "link" if entry.is_symlink() else "dir" if entry.is_dir() else "file"
            lines.append("  " * level + f"{entry.name} ({kind})")
            if entry.is_dir() and level < depth:
                visit(Path(entry.path), level + 1)

    visit(root, 0)
    return "\n".join(lines) or "(empty)"


def find_bundle(files: list[Path]) -> Path | None:
    """Copy or extract the bundle found among ``files`` to a writable folder (Kaggle inputs are read-only).
    Only this upload's files count: an older dataset version lacks its marker file."""
    marker = SETTINGS.get("upload_marker")
    if marker and not any(path.name == marker for path in files):
        return None
    if REPOSITORY.exists():
        shutil.rmtree(REPOSITORY)
    manifests = [path for path in files if path.name == MANIFEST_NAME]
    if manifests:  # Kaggle unzipped the upload
        shutil.copytree(manifests[0].parent, REPOSITORY)
        return REPOSITORY
    archives = [path for path in files if path.name == BUNDLE_NAME]
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(REPOSITORY)
        return REPOSITORY
    return None


def download_dataset() -> list[Path]:
    """Fetch the dataset with kagglehub, which uses the session's own Kaggle sign-in (no token needed)."""
    dataset = SETTINGS.get("dataset")
    if not dataset:
        return []
    try:
        import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
    log(f"Downloading the dataset {dataset} with kagglehub ...")
    folder = Path(kagglehub.dataset_download(dataset))
    log(f"kagglehub returned {folder}:\n{describe_inputs(folder)}")
    return input_files(folder)


def locate_bundle() -> Path:
    """Find the uploaded bundle, waiting briefly for the dataset mount and downloading it if needed."""
    deadline = time.monotonic() + MOUNT_WAIT_SECONDS
    errors: list[str] = []
    while True:
        errors.clear()
        repository = find_bundle(input_files(errors=errors))
        if repository or time.monotonic() >= deadline:
            break
        time.sleep(15)
    if repository:
        return repository
    log(f"The dataset was not visible under {INPUT_ROOT}:\n{describe_inputs()}\nListing errors: {errors or 'none'}")
    try:
        repository = find_bundle(download_dataset())
    except Exception as error:  # report the original problem together with the download failure
        log(f"Downloading the dataset failed: {error}")
    if repository:
        return repository
    raise FileNotFoundError(
        f"This upload (marker {SETTINGS.get('upload_marker')}) was not found under {INPUT_ROOT} or through "
        "kagglehub (see the log above). The session was probably given an older or unfinished dataset version."
    )


def install_packages() -> None:
    packages = SETTINGS.get("packages", [])
    if packages:
        log(f"Installing {' '.join(packages)} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


JOB = SETTINGS.get("job", "experiment")


def job_command() -> list[str]:
    """Return the run_model.py arguments for this session's job."""
    options = json.dumps(SETTINGS["config"])
    if JOB == "evaluation":
        return [sys.executable, "-u", "run_model.py",
                "--train_for_evaluation", "--config", options]
    if JOB == "experiment":
        return [sys.executable, "-u", "run_model.py", "--train_for_experiment", "--config", options]
    return [sys.executable, "-u", "run_model.py", "--find_stable_mutant", "--config", options]


def precompute_embeddings(repository: Path) -> None:
    """Embed the curated sequences in batches before training reads them one at a time.

    Training would otherwise call ESM-2 per example on the first epoch, which wastes most of the
    GPU. Embedding up front in batches takes a few minutes and the cache serves every epoch and
    every run afterwards.
    """
    program = (
        "from src.config_initialisation import initialize_global_config;"
        f"initialize_global_config({SETTINGS['config']!r});"
        "from src.repo_paths import PATHS;"
        "from src.utils.dataframe_processing_tools import read_dataframe;"
        "from src.embeddings_management.adding_embeddings import "
        "update_embedding_cache, update_pka_embedding_cache;"
        "rows = read_dataframe(PATHS.curated_all_csv);"
        "print('Embedding', len(rows), 'curated rows', flush=True);"
        "update_embedding_cache(rows, PATHS.embed_dir);"
        "update_pka_embedding_cache(read_dataframe(PATHS.pkad_cleaned_csv));"
        "print('Embedding cache ready', flush=True)"
    )
    log("Precomputing embeddings on the GPU ...")
    # Streamed line by line, as run_job does: buffering it until the step ends left the log
    # silent for ten minutes and then dumped every progress frame at once.
    process = subprocess.Popen([sys.executable, "-u", "-c", program], cwd=repository,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    reader = threading.Thread(target=lambda: [log(line.rstrip()) for line in process.stdout], daemon=True)
    reader.start()
    process.wait()
    reader.join(timeout=30)
    if process.returncode != 0:
        raise RuntimeError(f"Precomputing embeddings failed with exit code {process.returncode}")


def run_job(repository: Path) -> int:
    """Run this session's job, stopping it safely when the session time budget is used up."""
    budget = SETTINGS["session_hours"] * 3600 - (time.monotonic() - STARTED) - STOP_GRACE_SECONDS
    command = job_command()
    log(f"Running on Kaggle for up to {budget / 3600:.1f} h: {' '.join(command[2:5])}")
    process = subprocess.Popen(command, cwd=repository, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    reader = threading.Thread(target=lambda: [log(line.rstrip()) for line in process.stdout], daemon=True)
    reader.start()
    try:
        process.wait(timeout=max(budget, MIN_RUN_SECONDS))
    except subprocess.TimeoutExpired:
        log("Session time budget reached: stopping safely so progress is saved.")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    reader.join(timeout=30)
    return process.returncode


def package_training_outputs(repository: Path, return_code: int) -> None:
    """Copy out what the local side needs: results for an evaluation sweep, weights otherwise."""
    results = repository / "results"
    if JOB == "evaluation":
        for name in ("performance.csv", "performance_metrics.json"):
            produced = repository / "model performance" / name
            if produced.is_file():
                shutil.copy2(produced, WORKING / name)
        # One archive for the whole evaluation tree, so each run's predictions and its own
        # details stay in the same folder on the way back.
        evaluation = results / "evaluation"
        run_folders = sorted(p for p in evaluation.glob("*") if p.is_dir()) if evaluation.is_dir() else []
        with zipfile.ZipFile(WORKING / "validation_results.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(evaluation.rglob("*")) if evaluation.is_dir() else []:
                if path.is_file():
                    archive.write(path, path.relative_to(results).as_posix())
        packaged = {"return_code": return_code, "run_folders": len(run_folders),
                    "files_archived": sum(1 for p in evaluation.rglob("*") if p.is_file())
                                      if evaluation.is_dir() else 0}
    else:
        with zipfile.ZipFile(WORKING / "experiment_checkpoints.zip", "w",
                             compression=zipfile.ZIP_DEFLATED) as archive:
            for folder in ("checkpoints/experiment/ddg", "checkpoints/experiment/pka"):
                directory = repository / folder
                for path in sorted(directory.glob("*.pt")) if directory.is_dir() else []:
                    archive.write(path, f"{folder}/{path.name}")
        packaged = {"return_code": return_code, "checkpoints": "packaged"}

    (WORKING / "worker_summary.json").write_text(json.dumps(packaged, indent=2), encoding="utf-8")
    log(f"Packaged outputs: {packaged}")


def package_outputs(repository: Path, return_code: int) -> None:
    # run_path locates the folder under results/; run_name is the fallback for an older bundle.
    run_folder = repository / "results" / SETTINGS.get("run_path", SETTINGS["run_name"])
    for name in ("predictions.csv", "run_state.json"):
        if (run_folder / name).is_file():
            shutil.copy2(run_folder / name, WORKING / name)
    details = run_folder / "details"
    detail_files = sorted(details.glob("*.pt")) if details.is_dir() else []
    with zipfile.ZipFile(WORKING / "details.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in detail_files:
            archive.write(path, path.name)
    other_runs = [
        path.name for path in (repository / "results").glob("**/")
        if path.resolve() != run_folder.resolve() and (path / "run_state.json").is_file()
    ]
    summary = {"return_code": return_code, "new_detail_files": len(detail_files), "other_run_folders": other_runs}
    (WORKING / "worker_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Packaged outputs: {summary}")


def main() -> None:
    WORKING.mkdir(parents=True, exist_ok=True)
    log(f"DDG-Predict worker started. Inputs:\n{describe_inputs()}")
    try:
        repository = locate_bundle()
        install_packages()
        if JOB in ("evaluation", "experiment"):
            precompute_embeddings(repository)
            return_code = run_job(repository)
            package_training_outputs(repository, return_code)
        else:
            return_code = run_job(repository)
            package_outputs(repository, return_code)
    except Exception:
        log("The worker failed:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
