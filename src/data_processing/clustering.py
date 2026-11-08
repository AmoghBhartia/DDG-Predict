"""Sequence-identity clustering shared by the ddG split and PKAD leakage filtering."""

import pandas as pd

from src.config_initialisation import CONFIG


def cluster_sequences(sequences: pd.Series) -> pd.DataFrame:
    """Cluster unique sequences with ProtClust/MMseqs2 using the configured thresholds.

    Returns one row per unique sequence with ``sequence``, ``sequence_id`` and
    ``cluster_representative`` columns.

    ProtClust is imported here rather than at module level because it is only needed by
    --prepare. Training imports this module through the split helpers but never clusters, and a
    module-level import would make protclust (and MMseqs2) a requirement everywhere, including
    on Windows and on a Kaggle session where neither is installed.
    """
    try:
        from protclust import cluster
    except ImportError as error:
        raise ImportError(
            "Clustering needs the protclust package and the MMseqs2 binary on PATH. Only "
            "--prepare clusters; if you hit this during training, something imported this "
            "function that should not have."
        ) from error

    frame = pd.DataFrame({"sequence": sequences.drop_duplicates().reset_index(drop=True)})
    frame["sequence_id"] = [f"sequence_{index}" for index in frame.index]
    return cluster(
        frame,
        sequence_col="sequence",
        id_col="sequence_id",
        min_seq_id=CONFIG.min_seq_id,
        coverage=CONFIG.coverage,
        cov_mode=CONFIG.cov_mode,
        random_state=CONFIG.random_seed,
    )
