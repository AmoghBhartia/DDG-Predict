from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.model.model_infrastructure import WymanTanfordOutput
from src.repo_paths import PATHS
from src.utils.amino_acid_information_tools import titratable_residues, unfolded_pkas
from src.utils.dataframe_processing_tools import identifier_file_stem

CURVE_POINTS = 500


def _find_detail_file(result_path: Path, row: pd.Series, requested_id: str) -> Path:
    """Find the detail file associated with a result row across result layouts."""
    identifiers = [str(requested_id)]
    for column in ("experiment_id", "sequence_id"):
        if column in row and pd.notna(row[column]):
            identifier = str(row[column])
            if identifier not in identifiers:
                identifiers.append(identifier)

    # A validation run writes evaluation_results_<token>.csv beside evaluation_details_<token>/,
    # so the folder is derived from the file's own name before falling back to the shared ones.
    detail_directories = []
    if result_path.stem.startswith("evaluation_results"):
        token = result_path.stem[len("evaluation_results"):]
        detail_directories.append(result_path.parent / f"evaluation_details{token}")
    detail_directories += [
        result_path.parent / "details",
        result_path.parent / "evaluation_details",
    ]
    # Detail files are named with identifier_file_stem; older runs used the raw identifier.
    names = list(dict.fromkeys(
        name for identifier in identifiers
        for name in (identifier_file_stem(identifier), identifier)
    ))
    candidates = [directory / f"{name}.pt" for directory in detail_directories for name in names]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    expected_paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No saved thermodynamic detail file was found for {requested_id!r}. "
        f"Checked: {expected_paths}"
    )


def plot_ddG_graph(
    sequence_id: str,
    result_filename: str,
    result_folder: str | Path = PATHS.results_dir,
) -> None:
    """Plot a saved prediction curve from the main CSV and detail file."""
    result_folder = Path(result_folder)
    result_path = result_folder / result_filename
    results = pd.read_csv(result_path)
    matching_columns = [
        column
        for column in ("experiment_id", "sequence_id")
        if column in results.columns
    ]
    if not matching_columns:
        raise ValueError(
            f"Result file {result_path} contains neither 'experiment_id' nor 'sequence_id'."
        )
    row_mask = pd.Series(False, index=results.index, dtype=bool)
    for column in matching_columns:
        row_mask |= results[column].astype(str) == str(sequence_id)
    rows = results.loc[row_mask]
    if rows.empty:
        raise ValueError(f"No experiment result found for {sequence_id!r}")

    row = rows.iloc[0]
    detail = torch.load(
        _find_detail_file(result_path, row, sequence_id),
        map_location="cpu",
        weights_only=True,
    )
    true_length = int(detail["true_length"].reshape(-1)[0].item())
    unfolded_values = unfolded_pkas()
    titratable = titratable_residues()

    def per_residue(indices_key: str, folded_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pKa delta, unfolded pKa) shaped [CURVE_POINTS, length, 1]."""
        indices = detail[indices_key].long()[:true_length]
        unfolded = unfolded_values[indices].reshape(1, -1, 1)
        folded = detail[folded_key][:true_length].reshape(1, -1, 1)
        # Re-apply the titratable mask so detail files written before the mask existed,
        # which stored a large placeholder pKa for non-titratable residues, still plot correctly.
        delta = (folded - unfolded).masked_fill(~titratable[indices].reshape(1, -1, 1), 0.0)
        return delta.expand(CURVE_POINTS, -1, -1), unfolded.expand(CURVE_POINTS, -1, -1)

    wt_delta, wt_unfolded = per_residue("wildtype_residue_indices", "wildtype_folded_pKas")
    mut_delta, mut_unfolded = per_residue("mutant_residue_indices", "mutant_folded_pKas")
    pH = torch.linspace(0.0, 14.0, CURVE_POINTS)
    curve = WymanTanfordOutput()(
        mut_delta,
        wt_delta,
        mut_unfolded,
        wt_unfolded,
        detail["ddG_ref"].reshape(1).expand(CURVE_POINTS),
        pH,
        detail["experiment_temperature"].reshape(1).expand(CURVE_POINTS),
    )

    plt.figure(figsize=(10, 6))
    plt.plot(pH.numpy(), curve.squeeze(-1).detach().numpy(), label="Predicted ddG curve")
    if "ddG" in rows and "pH" in rows:
        observed = rows[["pH", "ddG"]].dropna()
        if not observed.empty:
            plt.scatter(observed["pH"], observed["ddG"], label="True ddG points")
    plt.xlabel("pH")
    plt.ylabel("ddG")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()
