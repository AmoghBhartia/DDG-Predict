import sys
from pathlib import Path

import pandas as pd
import tqdm

from src.config_initialisation import CONFIG
from src.embeddings_management.loading_embeddings import EmbeddingKind, EmbeddingsCache
from src.embeddings_management.running_esm_model import EsmModelRunner
from src.repo_paths import PATHS


def _process_embeddings(
    sequences: list[str],
    kind: EmbeddingKind,
    embeddings_data: EmbeddingsCache,
) -> None:
    """Compute embeddings for sequences that are not cached yet and store them in chunks."""
    missing_sequences = [sequence for sequence in sequences if not embeddings_data.does_seq_exist(sequence, kind)]
    if not missing_sequences:
        print(f"All {kind.upper()} sequences are already cached and validated. Proceeding to next stage.")
        return

    # Sorting by length keeps similar lengths together, which reduces padding in each ESM batch.
    missing_sequences.sort(key=len)
    chunk_size = CONFIG.embedding_chunk_size
    print(f"Vectorising {len(missing_sequences)} missing {kind.upper()} sequences...")

    # Writing to a pipe (a Kaggle session, a redirected log) turns every bar frame into its own
    # line, so update rarely there and normally in a terminal.
    update_interval = 0.1 if sys.stderr.isatty() else 30.0
    with tqdm.tqdm(total=len(missing_sequences), desc=f"Computing {kind.upper()} embeddings",
                   unit="sequence", mininterval=update_interval) as progress:
        for start in range(0, len(missing_sequences), chunk_size):
            chunk = missing_sequences[start : start + chunk_size]
            try:
                embeddings = EsmModelRunner.compute_embeddings(chunk, progress_callback=progress.update)
                for sequence, embedding in embeddings.items():
                    embeddings_data.verify_data_quality(sequence, embedding)
            except Exception as error:
                raise RuntimeError(f"Failed to embed a {kind.upper()} chunk: {error}") from error
            embeddings_data.save_embeddings(embeddings, kind)


def _ensure_cached(sequences: list[str], kind: EmbeddingKind, embeddings_data: EmbeddingsCache) -> None:
    """Embed any missing sequences, then confirm every sequence is present in the cache."""
    _process_embeddings(sequences, kind, embeddings_data)
    missing = [sequence for sequence in sequences if not embeddings_data.does_seq_exist(sequence, kind)]
    if missing:
        raise RuntimeError(
            f"Embedding cache remains incomplete after processing: {len(missing)} {kind.upper()} sequences are missing"
        )


def _unique_sequences(dataframe: pd.DataFrame, column: str) -> list[str]:
    if column not in dataframe.columns:
        raise ValueError(f"Embedding preparation requires a {column!r} column")
    return dataframe[column].dropna().astype(str).unique().tolist()


def update_embedding_cache(dataframe: pd.DataFrame, path: Path = PATHS.embed_dir) -> pd.DataFrame:
    """Ensure wild-type and mutant embeddings for every row are cached."""
    wildtypes = _unique_sequences(dataframe, "sequence")
    mutants = _unique_sequences(dataframe, "mutant_sequence")
    print("Saving new ESM embeddings to the cache...")
    with EmbeddingsCache(path) as embeddings_data:
        _ensure_cached(wildtypes, "wt", embeddings_data)
        _ensure_cached(mutants, "mut", embeddings_data)
    return dataframe


def update_pka_embedding_cache(dataframe: pd.DataFrame, path: Path = PATHS.pka_embeddings_dir) -> pd.DataFrame:
    """Ensure one ESM embedding is cached for every cleaned PKAD sequence."""
    with EmbeddingsCache(path) as embeddings_data:
        _ensure_cached(_unique_sequences(dataframe, "sequence"), "pka", embeddings_data)
    return dataframe
