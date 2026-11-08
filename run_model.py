import os

# On Windows, Intel Fortran/MKL runtimes (used by some NumPy/SciPy/PyTorch builds) install their own
# Ctrl+C handler that kills the process with "forrtl: error (200)" before Python can react. This must
# be set before those libraries are imported so that Ctrl+C reaches the clean exit below instead.
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import argparse
import json
import sys

from src.config_initialisation import initialize_global_config


# Workflows. Imports are deferred so each module reads the configuration after it is initialised.
def prepare_data() -> None:
    from src.data_loading.saving_validation_and_training_data import (
        prepare_curated_dataset,
        prepare_split_pkad,
    )
    from src.data_processing.pkad_cleaning import prepare_pkad_data
    from src.embeddings_management.adding_embeddings import update_pka_embedding_cache

    print(
        "\nWARNING: Please read the README before using --prepare on Windows. "
        "Data cleaning and verification use MMseqs2, which is not native to Windows. "
        "Use Ubuntu through WSL2 and install MMseqs2 before continuing."
    )
    print("\nLoading and curating data from source information...")
    prepare_curated_dataset()
    pkad = prepare_pkad_data()
    prepare_split_pkad(pkad)
    update_pka_embedding_cache(pkad)
    print("\nCurated dataset and training splits created. The split folders are rebuilt every time "
          "--prepare runs; the composition of each one is kept in data/Curated data/Historical Logs.")


def train_for_evaluation(remote: str | None = None) -> None:
    if remote == "kaggle":
        from src.experiments.kaggle_remote import KaggleError, run_training_on_kaggle

        try:
            run_training_on_kaggle("evaluation")
        except KaggleError as error:
            print(f"\nKaggle: {error}")
            sys.exit(1)
        return

    from src.training_and_validation.validation_runs import (
        run_evaluation_training,
        run_performance_evaluation,
    )

    print("\nTraining every prepared training split against evaluation_states initial states. Each "
          "run trains its own pKa head on its own PKAD split and then the full model. Finished "
          "runs are skipped if this is rerun.")
    run_evaluation_training()
    # Training these models is only worth doing for the measurement, so it follows immediately.
    # Rerunning this option once every run is finished therefore just re-measures, since the
    # training itself is skipped.
    print("\nEvaluating every run against its training split's testing set.")
    run_performance_evaluation()


def train_for_experiment(remote: str | None = None) -> None:
    if remote == "kaggle":
        from src.experiments.kaggle_remote import KaggleError, run_training_on_kaggle

        try:
            run_training_on_kaggle("experiment")
        except KaggleError as error:
            print(f"\nKaggle: {error}")
            sys.exit(1)
        return

    from src.training_and_validation.training import run_experiment_training

    print("\nTraining the model intended for prediction on the whole curated dataset. "
          "There is no testing set, so use --train_for_evaluation for performance figures.")
    run_experiment_training()
    print("Model training successful.")


def find_stable_mutant(remote: str | None = None) -> None:
    print("\nPreparing to run experiment...")
    if remote == "kaggle":
        from src.experiments.kaggle_remote import KaggleError, run_on_kaggle

        try:
            run_on_kaggle()
        except KaggleError as error:
            print(f"\nKaggle: {error}")
            sys.exit(1)
    else:
        from src.experiments.run_experiment import run_experiment

        run_experiment()
    print("Experiments complete.")


PLOT_MODES = ("evaluation", "experiment")


def plot_graph(arguments: list[str]) -> None:
    """Plot one saved prediction, locating its file from the mode and what follows it."""
    from src.experiments.plot_graph import plot_ddG_graph

    experiment_id, mode, *rest = arguments
    if mode == "evaluation":
        from src.training_and_validation.checkpoints import RunLabel
        from src.training_and_validation.validation_runs import predictions_path

        split, state = (int(value) for value in rest)
        results_file = predictions_path(RunLabel(initial_state=state, training_set=split))
        print(f"\nReading training split {split}, initial state {state}, and plotting ΔΔG graph...")
    else:
        results_file = _experiment_results_file(rest[0])
        print(f"\nReading {results_file} and plotting ΔΔG graph...")

    plot_ddG_graph(experiment_id, results_file)


def _experiment_results_file(target: str) -> str:
    """Resolve a --find_stable_mutant run to its predictions file, relative to results/.

    ``target`` is normally the run's folder name under ``results/experiment/``. A path to a
    predictions file is accepted too, for a result kept somewhere else, and a run left in the
    flat layout used before experiment runs moved into their own folder is still found.
    """
    from src.experiments.run_experiment import EXPERIMENT_FOLDER, PREDICTIONS_FILE
    from src.repo_paths import PATHS

    if target.lower().endswith(".csv"):
        return target
    for candidate in (f"{EXPERIMENT_FOLDER}/{target}/{PREDICTIONS_FILE}", f"{target}/{PREDICTIONS_FILE}"):
        if (PATHS.results_dir / candidate).is_file():
            return candidate
    # Neither exists; name the expected one so the error says where it looked.
    return f"{EXPERIMENT_FOLDER}/{target}/{PREDICTIONS_FILE}"


# Command line interface.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Computational ML-based toolkit for predicting thermodynamic stability changes (ΔΔG) induced by "
            "single-point mutations. Supports end-to-end model optimization on custom datasets as well as "
            "standalone inference via pre-trained weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "A JSON string containing configuration overrides to apply inline (e.g. '{\"batch_size\": 8}'). "
            "You may find it easier to change the file config.json instead to apply these changes. "
            "Configuration values are validated before the selected stages start."
        ),
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="Curate and verify the source measurements, embed them, and divide them into "
             "evaluation_splits training splits under data/Curated data/split_<n>/. Existing split "
             "folders are replaced.",
    )
    parser.add_argument(
        "--train_for_evaluation", action="store_true",
        help="Train one model for every combination of the training splits --prepare wrote and the "
             "evaluation_states initial states in config.json, to measure performance. Each "
             "training split holds out a share of the sequence clusters and gets its own PKAD "
             "split. Checkpoints are not published, so these runs are yours to train; finished "
             "ones are skipped on a rerun.",
    )
    parser.add_argument(
        "--train_for_experiment", action="store_true",
        help="Train the model intended for prediction on the whole curated dataset. Nothing is held "
             "out, so its expected performance is the figure --train_for_evaluation reports.",
    )
    parser.add_argument(
        "--find_stable_mutant", action="store_true",
        help="Scan all single substitutions or evaluate rows from experiments.csv. "
             "Without this file, all substitutions are scanned. Missing pH values default to 7.0.",
    )
    parser.add_argument(
        "--remote", choices=["kaggle"], default=None,
        help="With --find_stable_mutant or either training option: run the work on a free Kaggle GPU "
             "and bring the results back (see REMOTE_RUNNING.md for the one-off Kaggle setup).",
    )
    parser.add_argument(
        "--plot_graph", nargs="+", metavar="ARGUMENT",
        help="Plot the pH curve of one saved prediction, given its experiment ID and where the "
             "prediction lives. Two forms: 'EXPERIMENT_ID evaluation TRAINING_SPLIT INITIAL_STATE' "
             "for one of the runs --train_for_evaluation produced, or 'EXPERIMENT_ID experiment RUN' for a "
             "--find_stable_mutant scan, where RUN is its timestamped folder under results/ (a "
             "path to a predictions .csv relative to results/ also works).",
    )
    return parser


def _check_plot_arguments(parser: argparse.ArgumentParser, arguments: list[str]) -> None:
    """Reject a --plot_graph call whose shape does not match its mode, before anything is loaded."""
    usage = ("--plot_graph takes 'EXPERIMENT_ID evaluation TRAINING_SPLIT INITIAL_STATE' or "
             "'EXPERIMENT_ID experiment RUN'")
    if len(arguments) < 3:
        parser.error(f"{usage}; got {len(arguments)} argument(s)")
    mode = arguments[1]
    if mode not in PLOT_MODES:
        parser.error(f"--plot_graph mode must be one of {', '.join(PLOT_MODES)}, not {mode!r}. {usage}")
    rest = arguments[2:]
    if mode == "evaluation":
        if len(rest) != 2:
            parser.error(f"--plot_graph evaluation needs a training split and an initial state; {usage}")
        for name, value in zip(("training split", "initial state"), rest):
            if not value.isdigit() or int(value) < 1:
                parser.error(f"--plot_graph {name} must be a positive whole number, not {value!r}")
    elif len(rest) != 1:
        parser.error(f"--plot_graph experiment needs exactly one run; {usage}")


def parse_overrides(parser: argparse.ArgumentParser, raw_config: str | None) -> dict | None:
    """Parse ``--config`` as a JSON object, reporting errors through argparse."""
    if not raw_config:
        return None
    try:
        overrides = json.loads(raw_config)
    except json.JSONDecodeError as error:
        parser.error(f"--config must be valid JSON: {error}")
    if not isinstance(overrides, dict):
        parser.error("--config must contain a JSON object")
    return overrides


def main() -> None:
    """Parse CLI options, initialise configuration, and run the selected workflows in order."""
    parser = build_parser()
    args = parser.parse_args()

    if args.plot_graph:
        _check_plot_arguments(parser, args.plot_graph)

    workflows = [
        (args.prepare, prepare_data),
        (args.train_for_evaluation, lambda: train_for_evaluation(args.remote)),
        (args.train_for_experiment, lambda: train_for_experiment(args.remote)),
        (args.find_stable_mutant, lambda: find_stable_mutant(args.remote)),
    ]
    remote_capable = args.find_stable_mutant or args.train_for_evaluation or args.train_for_experiment
    if args.remote and not remote_capable:
        parser.error("--remote can only be used with --find_stable_mutant, --train_for_evaluation "
                     "or --train_for_experiment")
    if not any(selected for selected, _ in workflows) and not args.plot_graph:
        parser.print_help()
        return

    try:
        initialize_global_config(parse_overrides(parser, args.config))
    except (ValueError, RuntimeError) as error:
        parser.error(f"Invalid configuration: {error}")

    print("\n" + "=" * 60)
    print("        ΔΔG PREDICTOR FOR SINGLE POINT MUTATIONS")
    print("=" * 60)

    for selected, workflow in workflows:
        if selected:
            workflow()
    if args.plot_graph:
        plot_graph(args.plot_graph)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Files are written atomically, so stopping never leaves a half-written file. Training resumes from
        # the last completed epoch; --find_stable_mutant resumes from its last saved chunk.
        print("\n\nStopped by Ctrl+C. Exiting safely; saved progress is kept and will be resumed next run.")
        sys.exit(0)
    except (FileNotFoundError, ValueError) as error:
        # These are the toolkit's own messages about missing inputs or a configuration that cannot
        # work. They are written to be read, so they are printed rather than raised as a traceback.
        print(f"\n{error}")
        sys.exit(1)
