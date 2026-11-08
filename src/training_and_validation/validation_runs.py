"""Evaluation training and measurement across repeated training splits and initial states.

A single train/test split gives one performance number, and that number moves between runs for
two separate reasons: which proteins land in the testing set, and which initial weights the
optimiser started from. Reporting one figure hides both.

``--train_for_evaluation`` trains one model per (initial state, training split)
pair, so those two sources of variation can be told apart:

* **Training splits** partition the curated dataset by sequence cluster, so every row falls in
  the testing set of exactly one split, and no testing-set protein has a close relative on its
  own training side. Splits are balanced by row count rather than by cluster count, because a
  handful of heavily measured proteins carry most of the data.
* **Initial states** are starting weights. Initial state *x* is the same set of starting
  weights in every split, so comparing splits within one state isolates the effect of the data,
  and comparing states within one split isolates the effect of the optimisation.

Each training split gets its **own PKAD split**: PKAD chains similar to that split's testing-set
proteins are removed before the pKa head is pretrained, so the electrostatic term cannot have
seen a relative of the protein it is later asked to predict. The pKa head is pretrained
separately for every run rather than shared.

The resulting checkpoints are a measurement, not a product, so they are not version controlled.
The model intended for use is trained by ``--train_for_experiment`` on the whole dataset, and
its expected performance is what these runs estimate.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config_initialisation import CONFIG
from src.data_loading.saving_validation_and_training_data import (
    existing_split_numbers,
    load_split,
    rebuild_splits_from_categorisation,
    split_pkad_path,
)
from src.embeddings_management.loading_embeddings import EmbeddingsCache
from src.model.ddG_predictor_model import DDGPredictor
from src.repo_paths import PATHS
from src.safe_writing import safe_writer
from src.training_and_validation.checkpoints import (
    EVALUATION_DDG_CHECKPOINTS,
    EVALUATION_PKA_CHECKPOINTS,
    RunLabel,
    list_checkpoints,
    load_latest_checkpoint,
    runs_present,
)
from src.training_and_validation.evaluation import (
    METRIC_COLUMNS,
    calculate_metrics,
    collect_predictions,
    averaged_baselines,
    save_details,
)
from src.training_and_validation.training import model_training_loop
from src.training_and_validation.utils import ddg_data_loader
from src.utils.dataframe_processing_tools import build_experiment_ids, read_dataframe, save_dataframe
from src.utils.seeding import apply_seed


# Everything the evaluation writes lives under this one folder, one subfolder per run, beside the
# experiment runs that --find_stable_mutant writes to results/experiment/.
EVALUATION_FOLDER = "evaluation"
PREDICTIONS_FILE = "evaluation_results.csv"
DETAILS_FOLDER = "details"


@dataclass(frozen=True)
class TrainingSplit:
    """One prepared training split, as --prepare wrote it."""

    number: int
    training: pd.DataFrame
    testing_set: pd.DataFrame


def run_directory(run: RunLabel) -> Path:
    """Return the folder holding everything one validation run produced.

    One folder per run keeps a run's predictions and its per-row detail files together, which is
    also what ``--plot_graph`` relies on: given the predictions file it looks for ``details`` in
    the same folder, so no run can be paired with another's detail files.
    """
    return (PATHS.results_dir / EVALUATION_FOLDER
            / f"training_split_{run.training_set}_initial_state_{run.initial_state}")


def predictions_path(run: RunLabel) -> Path:
    return run_directory(run) / PREDICTIONS_FILE


def details_path(run: RunLabel) -> Path:
    return run_directory(run) / DETAILS_FOLDER


def load_training_splits(expected: int | None = None) -> list[TrainingSplit]:
    """Load the prepared training splits, checking there are as many as asked for.

    The splits are made by --prepare, which is also the step that needs MMseqs2, so training
    does no clustering of its own and gives the same division every time it runs.
    """
    numbers = existing_split_numbers()
    if not numbers:
        # A clone carries the curated data and the categorisation table but not the split folders,
        # so rebuild them before giving up. This needs no clustering and so no MMseqs2.
        rebuild_splits_from_categorisation()
        numbers = existing_split_numbers()
    if not numbers:
        raise FileNotFoundError(
            "No training splits are prepared, and there is no split categorisation table to rebuild them "
            "from. Run --prepare first; it divides the curated data into evaluation_splits folders "
            "under data/Curated data/."
        )
    if expected is not None and len(numbers) != expected:
        raise ValueError(
            f"{len(numbers)} training splits are prepared ({numbers}), but {expected} were asked for. "
            f"Run --prepare with --config '{{\"evaluation_splits\": {expected}}}' to divide the data "
            f"that way, or ask for {len(numbers)} splits."
        )
    splits = []
    for number in numbers:
        training, testing_set = load_split(number)
        splits.append(TrainingSplit(number=number, training=training, testing_set=testing_set))
    return splits


def pkad_for_split(split: TrainingSplit) -> pd.DataFrame:
    """Return the PKAD table --prepare wrote for this training split.

    The filtering needs MMseqs2 and so happens during --prepare. Training only reads the result,
    which is why validation training works on Windows and inside a Kaggle session.
    """
    path = split_pkad_path(split.number)
    if not path.is_file():
        raise FileNotFoundError(
            f"No PKAD table has been prepared for training split {split.number} ({path}).\n"
            "Run --prepare, which filters PKAD separately for every split so that the pKa head of "
            "each one never sees a relative of a protein that split holds out."
        )
    return read_dataframe(path)


def run_is_complete(run: RunLabel) -> bool:
    """True when this run already has a checkpoint for the final configured epoch."""
    epochs = [epoch for epoch, _ in list_checkpoints(EVALUATION_DDG_CHECKPOINTS.directory, run)]
    return bool(epochs) and max(epochs) >= CONFIG.num_epochs


def run_evaluation_training() -> None:
    """Train one model for every (initial state, training split) pair.

    Both counts come from the configuration. The number of splits is not a free choice at this
    point: --prepare has already divided the data, and re-dividing it here would need MMseqs2 and
    would invalidate every run already trained. Whatever --prepare wrote is what gets trained.
    """
    state_count = CONFIG.evaluation_states
    splits = load_training_splits()
    split_count = len(splits)
    if split_count != CONFIG.evaluation_splits:
        print(f"\nWARNING: config.json asks for {CONFIG.evaluation_splits} training splits but "
              f"{split_count} are prepared. Training uses the {split_count} that exist. Run "
              f"--prepare again to divide the data differently; that replaces the split folders, "
              f"so runs already trained would no longer match their splits.")
    total_rows = sum(len(split.testing_set) for split in splits)

    print(f"\n{split_count} training splits x {state_count} initial states = "
          f"{split_count * state_count} runs of {CONFIG.num_epochs} epochs, over {total_rows} curated rows")
    for split in splits:
        print(f"  training split {split.number}: {len(split.testing_set)} in the testing set, "
              f"{len(split.training)} for training")

    with EmbeddingsCache(PATHS.embed_dir) as embedding_db:
        for split in splits:
            training_frame = split.training
            pkad_frame = pkad_for_split(split)

            for state in range(1, state_count + 1):
                run = RunLabel(initial_state=state, training_set=split.number)
                if run_is_complete(run):
                    print(f"\n{run.token}: already trained; skipping.")
                    continue

                print(f"\n=== {run.token} ===")
                # The initial state fixes the starting weights, so state x begins from the same
                # weights in every split. The data differs, so the batch order still differs.
                generator = apply_seed(CONFIG.random_seed + state)
                model = DDGPredictor.from_config()
                loader = ddg_data_loader(training_frame, embedding_db, generator=generator)
                model_training_loop(
                    model,
                    loader,
                    kind=EVALUATION_DDG_CHECKPOINTS,
                    run=run,
                    pka_frame=pkad_frame,
                    pka_kind=EVALUATION_PKA_CHECKPOINTS,
                    generator=generator,
                )

    print("\nTraining complete.")


def _evaluate_run(
    run: RunLabel,
    testing_set: pd.DataFrame,
    embedding_db: EmbeddingsCache,
) -> dict[str, int | float] | None:
    """Evaluate one run on its testing set, writing its predictions and details."""
    model = DDGPredictor.from_config().to(CONFIG.resolve_torch_device())
    state = load_latest_checkpoint(EVALUATION_DDG_CHECKPOINTS, model, run)
    if state is None:
        return None
    model.load_state_dict(state["model_state"])
    model.eval()

    loader = ddg_data_loader(testing_set, embedding_db, shuffle=False)
    predictions, targets, details = collect_predictions(model, loader)

    results = testing_set.copy()
    results["experiment_id"] = build_experiment_ids(results["sequence"].str[:10], results)
    results["ddG_predictions"] = predictions.reshape(-1).tolist()
    save_dataframe(predictions_path(run), results)
    save_details(results["experiment_id"], details, details_path(run))

    metrics = calculate_metrics(predictions, targets)
    metrics["epoch"] = int(state["epoch"])
    return metrics


def _spread(values: list[float]) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    return {
        "mean": float(series.mean()),
        "standard_deviation": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "minimum": float(series.min()),
        "maximum": float(series.max()),
    }


def run_performance_evaluation() -> dict:
    """Evaluate every trained validation run and write one summary of all of them."""
    runs = runs_present(EVALUATION_DDG_CHECKPOINTS.directory)
    if not runs:
        raise FileNotFoundError(
            "No evaluation checkpoints were found, so there is nothing to measure.\n"
            "These checkpoints are a measurement rather than part of the repository, so they are "
            "not published and you need to train your own:\n\n"
            "    python run_model.py --train_for_evaluation\n\n"
            "That trains one model for each prepared training split and each of the "
            "evaluation_states initial states in config.json. The model for making predictions is a "
            "different one, trained by --train_for_experiment."
        )

    splits = {split.number: split for split in load_training_splits()}

    print(f"\nEvaluating {len(runs)} validation runs across {len(splits)} training splits")
    per_run: list[dict] = []

    with EmbeddingsCache(PATHS.embed_dir) as embedding_db:
        for run in runs:
            split = splits.get(run.training_set)
            if split is None:
                print(f"  {run.token}: training split {run.training_set} is not prepared; skipping.")
                continue
            training_frame, testing_set = split.training, split.testing_set
            metrics = _evaluate_run(run, testing_set, embedding_db)
            if metrics is None:
                continue

            per_run.append({
                "initial_state": run.initial_state,
                "training_set": run.training_set,
                "testing_set_rows": int(len(testing_set)),
                "training_rows": int(len(training_frame)),
                "metrics": metrics,
                "predictions_file": predictions_path(run).relative_to(PATHS.results_dir).as_posix(),
            })
            print(f"  {run.token}: " + ", ".join(
                f"{name}={metrics[name]:.4f}" for name in METRIC_COLUMNS if name in metrics))

    if not per_run:
        raise RuntimeError("No evaluation run could be evaluated; see the messages above.")

    summary = _summarise(per_run, splits)
    _write_performance(summary)
    _print_summary(summary)
    return summary


def _write_performance(summary: dict) -> None:
    """Write the per-run table and the averaged figures to the model performance folder."""
    rows = [
        {"training_split": run["training_set"], "initial_state": run["initial_state"],
         "testing_set_rows": run["testing_set_rows"], "training_rows": run["training_rows"],
         **{name: run["metrics"][name] for name in METRIC_COLUMNS if name in run["metrics"]}}
        for run in summary["per_run"]
    ]
    save_dataframe(PATHS.performance_csv, pd.DataFrame(rows))
    with safe_writer(PATHS.performance_json) as handle:
        json.dump({key: value for key, value in summary.items() if key != "per_run"}, handle, indent=2)


def _summarise(
    per_run: list[dict],
    splits: dict[int, TrainingSplit],
) -> dict:
    """Average the per-run metrics and describe the spread of the runs."""
    across_runs = {
        name: _spread([run["metrics"][name] for run in per_run if name in run["metrics"]])
        for name in METRIC_COLUMNS
        if any(name in run["metrics"] for run in per_run)
    }

    # Between initial states within one split: the training noise on its own.
    within_split: dict[str, dict] = {}
    for name in METRIC_COLUMNS:
        spreads = []
        for split_number in {run["training_set"] for run in per_run}:
            values = [run["metrics"][name] for run in per_run
                      if run["training_set"] == split_number and name in run["metrics"]]
            if len(values) > 1:
                spreads.append(float(pd.Series(values, dtype=float).std(ddof=1)))
        if spreads:
            within_split[name] = {
                "mean_standard_deviation": float(pd.Series(spreads).mean()),
                "splits_compared": len(spreads),
            }

    # The baselines are measured exactly as the model is: fitted on each split's training rows and
    # scored on that split's testing set, then averaged. Fitting them on the whole curated dataset
    # would score them on rows they had already seen, which flatters them by roughly 0.27 kcal/mol
    # of MAE and makes the comparison with the model meaningless.
    divisions = [(split.training, split.testing_set) for split in splits.values()]
    return {
        "runs_evaluated": len(per_run),
        "epochs_per_run": CONFIG.num_epochs,
        "headline": across_runs,
        "headline_note": (
            "Every reported figure is the mean of the per-run metrics: the metric is computed "
            "inside each run's own testing set and those values are averaged, weighting every run "
            "equally. A correlation measured inside one fold cannot be credited for placing that "
            "fold's overall level correctly, so this is the more conservative of the two ways to "
            "report a cross-validation and it is the one quoted in the documentation."
        ),
        "baselines": averaged_baselines(divisions),
        "baseline_note": (
            "Fitted on each split's training rows, scored on that split's testing set, and then "
            "averaged across the splits exactly as the model is, so the two are comparable. A "
            "predictor that is constant within a fold has no correlation there, reported as null."
        ),
        "spread_between_initial_states_within_a_split": within_split,
        "per_run": sorted(per_run, key=lambda run: (run["training_set"], run["initial_state"])),
    }


def _print_summary(summary: dict) -> None:
    print(f"\n{'=' * 70}\nPerformance over {summary['runs_evaluated']} runs "
          f"({summary['epochs_per_run']} epochs each)\n{'=' * 70}")
    print("\nHeadline: the mean of the per-run metrics, each measured inside its own testing set:")
    for name in ("mae", "rmse", "bias", "pearson", "spearman"):
        spread = summary["headline"].get(name)
        if spread:
            print(f"  {name:<10} {spread['mean']:.4f} +/- {spread['standard_deviation']:.4f}"
                  f"   (range {spread['minimum']:.4f} to {spread['maximum']:.4f}, "
                  f"{spread['runs']} runs)")

    within = summary["spread_between_initial_states_within_a_split"]
    if within:
        print("\nBetween initial states within one training split (training noise alone):")
        for name in ("mae", "rmse", "pearson"):
            if name in within:
                print(f"  {name:<10} mean standard deviation {within[name]['mean_standard_deviation']:.4f}"
                      f"   over {within[name]['splits_compared']} splits")
    print("\nBaselines, fitted on each split's training rows and averaged the same way:")
    for name, metrics in summary["baselines"].items():
        parts = []
        for key in ("mae", "rmse", "pearson", "spearman"):
            entry = metrics.get(key)
            parts.append(f"{key}={entry['mean']:.4f}" if entry else f"{key}=n/a")
        print(f"  {name:<16} " + ", ".join(parts))
    print(f"\nWritten to {PATHS.performance_csv} and {PATHS.performance_json}")
