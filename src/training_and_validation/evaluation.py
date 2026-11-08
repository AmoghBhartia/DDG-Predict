from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.ddG_predictor_model import DDGPredictor
from src.safe_writing import safe_writer
from src.training_and_validation.utils import (
    require_finite,
    run_model,
)
from src.utils.dataframe_processing_tools import identifier_file_stem

METRIC_COLUMNS = ["count", "mae", "rmse", "std", "bias", "pearson", "spearman"]


# Prediction collection.
def _example_details(output: dict, batch: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    """Extract the thermodynamic details of one example, as saved in its ``.pt`` file."""
    folded_pKas = output["folded_pKas"]
    return {
        "ddG_ref": output["ddG_ref"][index].cpu(),
        "wildtype_folded_pKas": folded_pKas["wildtype"][index].cpu(),
        "mutant_folded_pKas": folded_pKas["mutant"][index].cpu(),
        "wildtype_residue_indices": batch["wt_residue_indices"][index].cpu(),
        "mutant_residue_indices": batch["mut_residue_indices"][index].cpu(),
        "experiment_temperature": batch["experiment_temperature"][index].cpu(),
        "true_length": batch["true_lengths"][index].cpu(),
    }


def collect_predictions(
    model: DDGPredictor,
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Collect predictions, targets, and per-example thermodynamic details."""
    model.eval()
    predictions, targets, details = [], [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Generating predictions"):
            ddG_true, output = run_model(model, batch)
            predictions.append(output["ddG_prediction"].cpu())
            targets.append(ddG_true.cpu())
            details.extend(_example_details(output, batch, index) for index in range(ddG_true.shape[0]))

    if not predictions:
        raise ValueError("Evaluation loader produced no batches")
    return torch.cat(predictions), torch.cat(targets), details


def save_details(identifiers: pd.Series, details: list[dict[str, torch.Tensor]], directory: Path) -> None:
    """Save one ``<identifier>.pt`` detail file per prediction."""
    if len(identifiers) != len(details):
        raise ValueError("Each detail record needs exactly one identifier")
    directory.mkdir(parents=True, exist_ok=True)
    for identifier, detail in zip(identifiers.astype(str), details):
        with safe_writer(directory / f"{identifier_file_stem(identifier)}.pt", "wb") as handle:
            torch.save(detail, handle)


# Metrics.
def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, int | float]:
    """Calculate error and correlation metrics for predictions."""
    if predictions.numel() == 0:
        raise ValueError("Cannot calculate metrics for empty predictions")
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction and target shapes differ: {predictions.shape} vs {targets.shape}")
    require_finite(predictions, "predictions")
    require_finite(targets, "targets")

    errors = predictions - targets
    prediction_values = predictions.reshape(-1).cpu().numpy()
    target_values = targets.reshape(-1).cpu().numpy()

    def is_constant(values: np.ndarray) -> bool:
        return values.size < 2 or bool(np.all(values == values[0]))

    if is_constant(prediction_values) or is_constant(target_values):
        pearson = spearman = float("nan")
    else:
        pearson = float(pearsonr(prediction_values, target_values).statistic)
        spearman = float(spearmanr(prediction_values, target_values).statistic)

    return {
        "count": int(predictions.numel()),
        "mae": float(errors.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(errors.square()))),
        "std": float(errors.std(unbiased=False)),
        "bias": float(errors.mean()),
        "pearson": pearson,
        "spearman": spearman,
    }


def _as_tensor(values: pd.Series) -> torch.Tensor:
    return torch.tensor(values.to_numpy(dtype=float), dtype=torch.float32)


def _mutation_type(dataframe: pd.DataFrame) -> pd.Series:
    return dataframe["wt_residue"] + "_to_" + dataframe["mut_residue"]


def baseline_predictions(df_training: pd.DataFrame, df_validation: pd.DataFrame) -> dict[str, pd.Series]:
    """Predict ``df_validation`` from ``df_training`` alone, for each reference predictor.

    Returned separately from the metrics so that a cross-validated comparison can fit each split's
    baseline on that split's training rows before scoring it. Both baselines must see exactly the
    data the model saw, or they are not comparable with it.
    """
    if df_training.empty or df_validation.empty:
        raise ValueError("Training and validation data are required for baseline evaluation")
    require_finite(_as_tensor(df_training["ddG"]), "training baseline targets")

    global_mean = float(df_training["ddG"].mean())
    mutation_means = df_training.groupby(_mutation_type(df_training))["ddG"].mean()
    index = df_validation.index
    return {
        "Global mean": pd.Series(global_mean, index=index, dtype=float),
        # A mutation type absent from the training rows falls back to the global mean.
        "Mutation mean": _mutation_type(df_validation).map(mutation_means).fillna(global_mean).astype(float),
    }


def baselines(df_training: pd.DataFrame, df_validation: pd.DataFrame) -> dict[str, dict[str, int | float]]:
    """Evaluate simple reference predictors on one training/validation division."""
    targets = _as_tensor(df_validation["ddG"])
    return {
        name: calculate_metrics(_as_tensor(predictions), targets)
        for name, predictions in baseline_predictions(df_training, df_validation).items()
    }


def average_metrics(runs: list[dict[str, int | float]]) -> dict[str, dict[str, float] | None]:
    """Average a metric across runs, the way the headline figures are reported.

    Each metric is averaged over the runs that defined it. ``calculate_metrics`` returns NaN for a
    correlation it cannot compute, which happens whenever a predictor is constant inside a fold, so
    those runs are skipped and a metric that no run defined is reported as ``None`` rather than as a
    number. Runs are weighted equally regardless of how many rows their fold held.
    """
    if not runs:
        raise ValueError("At least one run is required")
    averaged: dict[str, dict[str, float] | None] = {}
    for name in METRIC_COLUMNS:
        values = [float(run[name]) for run in runs if name in run and not np.isnan(float(run[name]))]
        if not values:
            averaged[name] = None
            continue
        series = pd.Series(values, dtype=float)
        averaged[name] = {
            "mean": float(series.mean()),
            "standard_deviation": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "runs": len(values),
        }
    return averaged


def averaged_baselines(
    divisions: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, dict[str, dict[str, float] | None]]:
    """Evaluate the reference predictors fold by fold and average, as the model is reported.

    Each baseline is still fitted on one split's training rows and scored on that split's testing
    set, so nothing is scored on rows it was fitted on. The metric is computed inside each fold and
    the fold values are averaged, weighting every fold equally, exactly as the model's figures are.
    A fold-constant predictor such as the training-fold mean has no correlation within a fold, so
    its correlations come back as ``None``.
    """
    if not divisions:
        raise ValueError("At least one training/testing-set division is required")
    collected: dict[str, list[dict[str, int | float]]] = {}
    for df_training, df_validation in divisions:
        targets = _as_tensor(df_validation["ddG"])
        for name, predictions in baseline_predictions(df_training, df_validation).items():
            collected.setdefault(name, []).append(calculate_metrics(_as_tensor(predictions), targets))
    return {name: average_metrics(runs) for name, runs in collected.items()}
