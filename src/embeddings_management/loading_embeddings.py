import hashlib
import io
import logging
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import lmdb
import torch

from src.embeddings_management.running_esm_model import EsmModelRunner
from src.repo_paths import PATHS
from src.safe_writing import safe_writer
from src.utils.constants import ESM_EMBEDDING_DIM

logger = logging.getLogger("EmbeddingsCache")

EmbeddingKind = Literal["wt", "mut", "pka"]
EMBEDDING_KINDS: tuple[str, ...] = ("wt", "mut", "pka")
# LMDB preallocates this much address space, and on Windows the file is created at this size
# immediately. 15 GB covers the curated training and validation embeddings (about 6 GB) and a
# per-chunk experiment cache, which needs experiment_chunk_size x sequence length x 5.1 KB.
MAP_SIZE_BYTES = 15 * 1024**3


class EmbeddingsCache:
    """Store and validate wild-type, mutant and PKAD ESM embeddings in LMDB."""

    # Environment lifecycle.
    def __init__(self, data_path: Path | str = PATHS.embed_dir) -> None:
        """Store the database path; LMDB is opened lazily so the cache can be sent to worker processes."""
        self.data_path = Path(data_path)
        self._closed = False
        self.lmdb_env = None
        self._databases: dict[str, Any] = {}
        self._max_key_size: int | None = None

    def _init_db(self, write_mode: bool = False) -> None:
        """Open the LMDB environment if needed (read-only unless writing or creating the store)."""
        if self._closed:
            raise RuntimeError("Database environment has been closed. Operations cannot be completed.")
        if self.lmdb_env is not None:
            return

        self.data_path.mkdir(parents=True, exist_ok=True)
        writable = write_mode or not (self.data_path / "data.mdb").exists()
        self.lmdb_env = lmdb.open(
            str(self.data_path),
            map_size=MAP_SIZE_BYTES,
            max_dbs=len(EMBEDDING_KINDS),
            readonly=not writable,
            lock=writable,
        )
        self._max_key_size = self.lmdb_env.max_key_size()

    def _release_env(self) -> None:
        """Close the current environment and drop every database handle that belongs to it."""
        if self.lmdb_env is not None:
            self.lmdb_env.close()
        self.lmdb_env = None
        self._databases = {}
        self._max_key_size = None

    def _ensure_writable(self) -> None:
        """Reopen the environment in write mode if it is currently read-only."""
        if self.lmdb_env is not None and self.lmdb_env.flags().get("readonly", True):
            self._release_env()
        self._init_db(write_mode=True)

    def close(self) -> None:
        """Close the database environment once and mark this cache unusable."""
        if not self._closed:
            self._release_env()
            self._closed = True

    def __enter__(self) -> "EmbeddingsCache":
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed EmbeddingsCache context manager.")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    # Storage helpers.
    def _database(self, kind: EmbeddingKind) -> Any:
        """Return the named LMDB database for ``kind``, creating it on first use."""
        if kind not in EMBEDDING_KINDS:
            raise ValueError(f"Invalid sequence kind: {kind!r}. Must be one of {EMBEDDING_KINDS}.")
        if self._closed or self.lmdb_env is None:
            raise RuntimeError("Database environment is not open. Operations cannot be completed.")
        if kind not in self._databases:
            try:
                self._databases[kind] = self.lmdb_env.open_db(kind.encode())
            except lmdb.Error:
                # A read-only environment cannot create a named database that does not exist yet.
                self._ensure_writable()
                self._databases[kind] = self.lmdb_env.open_db(kind.encode())
        return self._databases[kind]

    def _sequence_key(self, sequence: str) -> bytes:
        """Keep short keys as raw sequences and hash sequences too long for LMDB keys."""
        sequence_bytes = sequence.encode("utf-8")
        if len(sequence_bytes) <= self._max_key_size:
            return sequence_bytes
        return b"sha256:" + hashlib.sha256(sequence_bytes).digest()

    @staticmethod
    def _serialise(tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor.contiguous().cpu(), buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialise(byte_data: bytes) -> torch.Tensor:
        return torch.load(io.BytesIO(byte_data), map_location="cpu", weights_only=True)

    def _read(self, sequence: str, kind: EmbeddingKind) -> bytes | None:
        self._init_db(write_mode=False)
        database = self._database(kind)
        with self.lmdb_env.begin(db=database, write=False) as transaction:
            return transaction.get(self._sequence_key(sequence))

    def _write(self, embeddings: dict[str, torch.Tensor], kind: EmbeddingKind) -> None:
        self._ensure_writable()
        database = self._database(kind)
        with self.lmdb_env.begin(db=database, write=True) as transaction:
            for sequence, embedding in embeddings.items():
                transaction.put(self._sequence_key(sequence), self._serialise(embedding))

    # Public API.
    def does_seq_exist(self, sequence: str, kind: EmbeddingKind) -> bool:
        """Check whether a sequence is stored in the selected database."""
        return self._read(sequence, kind) is not None

    @staticmethod
    def verify_data_quality(sequence: str, embedding: torch.Tensor) -> None:
        """Require an embedding to be finite and shaped ``[sequence_length, ESM_EMBEDDING_DIM]``."""
        if not isinstance(sequence, str) or not isinstance(embedding, torch.Tensor):
            raise TypeError("Sequence is not a string or embedding is not a tensor.")
        expected_shape = (len(sequence), ESM_EMBEDDING_DIM)
        if tuple(embedding.shape) != expected_shape:
            raise ValueError(
                f"Embedding layout is corrupted or length mismatch. "
                f"Expected shape {expected_shape}, got {tuple(embedding.shape)}"
            )
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite (NaN/Inf) ESM embedding calculated for {sequence[:40]}...")

    def heal_sequence_data(self, sequence: str, kind: EmbeddingKind) -> torch.Tensor:
        """Recompute, validate, and store a missing or corrupted embedding."""
        logger.warning(f"Recomputing missing or corrupted {kind.upper()} embedding: {sequence[:40]}...")
        embedding = EsmModelRunner.compute_embeddings(sequence)[sequence]
        try:
            self.verify_data_quality(sequence, embedding)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Recomputed embedding is still invalid, so the sequence itself is likely corrupted: {sequence}"
            ) from error
        self._write({sequence: embedding}, kind)
        return embedding

    def get_embedding(self, sequence: str, kind: EmbeddingKind) -> torch.Tensor:
        """Fetch a validated embedding, recomputing it if it is absent or corrupt."""
        byte_data = self._read(sequence, kind)
        if byte_data is None:
            return self.heal_sequence_data(sequence, kind)

        embedding = self._deserialise(byte_data)
        try:
            self.verify_data_quality(sequence, embedding)
        except (TypeError, ValueError) as error:
            logger.debug(f"Corrupted cache entry detected ({error}); recomputing.")
            return self.heal_sequence_data(sequence, kind)
        return embedding

    def save_embeddings(self, embeddings: dict[str, torch.Tensor], kind: EmbeddingKind) -> None:
        """Write a batch of embeddings to LMDB and to a timestamped ``.pt`` backup."""
        self._write(embeddings, kind)
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            with safe_writer(self.data_path / "backup" / f"{kind}_chunk_{timestamp}.pt", mode="wb") as handle:
                torch.save(embeddings, handle)
        except Exception as backup_error:
            logger.error(f"LMDB save successful, but backup save to .pt file failed: {backup_error}")
