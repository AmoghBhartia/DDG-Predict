"""Project-wide constants shared by data processing, embeddings and the model."""

from typing import Final

STANDARD_AMINO_ACIDS: Final[frozenset[str]] = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_SEQUENCE_LENGTH: Final[int] = 30
ESM_EMBEDDING_DIM: Final[int] = 1280
