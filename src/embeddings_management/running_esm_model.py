import logging
from typing import Any, Callable

import torch

from src.config_initialisation import CONFIG

logger = logging.getLogger("EsmModelRegister")

class EsmModelRunner:
    """Load ESM-2 once per process and compute residue-level embeddings."""

    _model = None
    _tokeniser = None
    _device = None

    @classmethod
    def get_esm_model(
        cls,
        target_device: torch.device | None = None,
    ) -> tuple[torch.nn.Module, Any, torch.device]:
        """Return the cached ESM model, its external batch converter, and device."""
        requested_device = target_device or CONFIG.resolve_torch_device()

        if cls._model is not None:
            if cls._device != requested_device:
                raise RuntimeError(
                    f"ESM is already loaded on {cls._device}; requested {requested_device}. "
                    "Use one ESM device per process."
                )
            return cls._model, cls._tokeniser, cls._device

        cls._device = requested_device

        logger.info(f"Initialising ESM-2 model (esm2_t33_650M_UR50D) onto {cls._device}...")
        
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        cls._model = model.to(cls._device).eval()
        cls._tokeniser = alphabet.get_batch_converter()
        
        logger.info("ESM Model successfully loaded and cached in registry memory.")
        return cls._model, cls._tokeniser, cls._device

    @classmethod
    def _plan_batches(cls, sequences: list[str]) -> list[list[str]]:
        """Group sequences into batches small enough for ESM-2's attention to fit in memory.

        Attention allocates ``batch x heads x length^2`` values, and every sequence in a batch is
        padded to the longest one, so a fixed batch size that suits 150-residue sequences asks for
        gigabytes on 1800-residue ones. Sequences are sorted by length (which also minimises
        padding) and a batch is closed once it would exceed either ``esm_batch_size`` or
        ``esm_attention_budget``.
        """
        batch_size = CONFIG.esm_batch_size
        budget = CONFIG.esm_attention_budget
        batches: list[list[str]] = []
        current: list[str] = []
        for sequence in sorted(sequences, key=len):
            padded_length = len(sequence) + 2  # beginning- and end-of-sequence tokens
            if current and (len(current) + 1 > batch_size
                            or (len(current) + 1) * padded_length ** 2 > budget):
                batches.append(current)
                current = []
            current.append(sequence)
        if current:
            batches.append(current)
        return batches

    @classmethod
    def _embed_batch(cls, model, tokeniser, device, layer: int, batch: list[str]) -> torch.Tensor:
        """Run one batch through ESM-2 and return its representations on the CPU."""
        _, _, tokens = tokeniser([(str(index), sequence) for index, sequence in enumerate(batch)])
        with torch.inference_mode():
            output = model(tokens.to(device, non_blocking=True), repr_layers=[layer])
        return output["representations"][layer].cpu()

    @classmethod
    def compute_embeddings(
        cls,
        sequences: list[str] | str,
        progress_callback: Callable[[float | None], bool | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute CPU ESM-2 layer embeddings shaped ``[sequence_length, ESM_EMBEDDING_DIM]``.

        ``esm_batch_size`` and ``esm_layer`` are range-checked when the configuration is loaded.
        Batches are sized by ``_plan_batches`` so that long sequences do not exhaust GPU memory.
        """
        layer = CONFIG.esm_layer

        if isinstance(sequences, str):
            sequences = [sequences]
        if any(not isinstance(sequence, str) or not sequence for sequence in sequences):
            raise ValueError("ESM inputs must be nonempty protein sequence strings")
        
        unique_sequences = list(dict.fromkeys(sequences))
        if not unique_sequences:
            return {}

        model, tokeniser, device = cls.get_esm_model(CONFIG.resolve_torch_device())
        results: dict[str, torch.Tensor] = {}

        for batched_sequences in cls._plan_batches(unique_sequences):
            try:
                representations = cls._embed_batch(model, tokeniser, device, layer, batched_sequences)
            except torch.OutOfMemoryError:
                # The budget is a heuristic, so fall back to one sequence at a time rather than
                # losing the whole run. A single sequence that still will not fit is a real error.
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if len(batched_sequences) == 1:
                    raise
                logger.warning(
                    f"Out of memory on a batch of {len(batched_sequences)} sequences up to "
                    f"{len(batched_sequences[-1])} residues; retrying them one at a time. "
                    "Lower esm_attention_budget to avoid this."
                )
                representations = None
                for sequence in batched_sequences:
                    single = cls._embed_batch(model, tokeniser, device, layer, [sequence])
                    results[sequence] = single[0, 1 : len(sequence) + 1].contiguous()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if progress_callback is not None:
                    progress_callback(len(batched_sequences))
                continue

            # Position 0 is the beginning-of-sequence token, so residues start at index 1.
            for row_index, sequence in enumerate(batched_sequences):
                results[sequence] = representations[row_index, 1 : len(sequence) + 1].contiguous()

            if progress_callback is not None:
                progress_callback(len(batched_sequences))

        return results

