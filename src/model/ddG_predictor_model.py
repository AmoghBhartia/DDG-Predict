from typing import Final

import torch
import torch.nn as nn

from src.model.model_infrastructure import (
    AminoAcidEmbeddingBank,
    Convoluter,
    SequenceGate,
    WymanTanfordOutput,
    mask_padding,
    validate_scan_layout,
)
from src.utils.amino_acid_information_tools import titratable_residues, unfolded_pkas
from src.utils.constants import ESM_EMBEDDING_DIM, MIN_SEQUENCE_LENGTH

DEFAULT_SCAN_SIZES: Final[tuple[int, ...]] = (1, 2, 3, 4)
DEFAULT_SCAN_DIMENSIONS: Final[tuple[int, ...]] = (256, 64, 16, 4)
PKA_DELTA_LIMIT: Final[float] = 5.0


class InteractionPipeline(nn.Module):
    """Convert per-residue ESM features into one interaction feature per residue."""

    def __init__(
        self,
        embedding_bank: AminoAcidEmbeddingBank,
        interaction_name: str,
        feature_size: int = ESM_EMBEDDING_DIM,
        gate_size: int = ESM_EMBEDDING_DIM,
        scan_dimensions: tuple[int, ...] = DEFAULT_SCAN_DIMENSIONS,
        scan_sizes: tuple[int, ...] = DEFAULT_SCAN_SIZES,
    ) -> None:
        super().__init__()
        validate_scan_layout(scan_sizes, scan_dimensions)
        self.embedding_set = getattr(embedding_bank, interaction_name)
        self.sequence_gate = SequenceGate(feature_size, gate_size, scan_dimensions[0])
        input_dimensions = (scan_dimensions[0],) + tuple(scan_dimensions[:-1])
        self.convoluters = nn.ModuleList(
            Convoluter(input_size, scan_size, output_size)
            for input_size, scan_size, output_size in zip(input_dimensions, scan_sizes, scan_dimensions)
        )
        self.output_projection = nn.Linear(scan_dimensions[-1], 1)

    def _terminal_embeddings(
        self,
        residue_indices: torch.Tensor,
        true_lengths: torch.Tensor,
        scan_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return learned padding for the first and last residue of each sequence."""
        scan_name = f"scan_{scan_size}"
        first_residues = residue_indices[:, 0]
        last_residues = residue_indices.gather(1, true_lengths.reshape(-1, 1) - 1).squeeze(1)
        n_length = (scan_size - 1) // 2
        c_length = scan_size - 1 - n_length
        n_embeddings = self.embedding_set.n_terminus[scan_name][first_residues, :n_length]
        c_embeddings = self.embedding_set.c_terminus[scan_name][last_residues, :c_length]
        return n_embeddings, c_embeddings

    def forward(
        self,
        esm_features: torch.Tensor,
        residue_indices: torch.Tensor,
        true_lengths: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self.sequence_gate(esm_features, residue_indices)
        for convoluter in self.convoluters:
            if convoluter.kernel_size == 1:
                sequence = convoluter(sequence)
            else:
                terminals = self._terminal_embeddings(residue_indices, true_lengths, convoluter.kernel_size)
                sequence = convoluter(sequence, *terminals, true_lengths=true_lengths)
        return mask_padding(self.output_projection(sequence), true_lengths)


class DDGPredictor(nn.Module):
    """Predict ddG from pKa and non-electrostatic WT/mutant representations."""

    def __init__(
        self,
        feature_size: int = ESM_EMBEDDING_DIM,
        gate_size: int = ESM_EMBEDDING_DIM,
        scan_dimensions: tuple[int, ...] = DEFAULT_SCAN_DIMENSIONS,
        scan_sizes: tuple[int, ...] = DEFAULT_SCAN_SIZES,
    ) -> None:
        super().__init__()
        validate_scan_layout(scan_sizes, scan_dimensions)
        self.embedding_bank = AminoAcidEmbeddingBank(scan_sizes, scan_dimensions)
        pipeline_arguments = (feature_size, gate_size, scan_dimensions, scan_sizes)
        self.pka_pipeline = InteractionPipeline(self.embedding_bank, "pka", *pipeline_arguments)
        self.non_electrostatic_pipeline = InteractionPipeline(
            self.embedding_bank, "non_electrostatic", *pipeline_arguments
        )
        self.thermodynamic_output = WymanTanfordOutput()
        # Not persistent: both are looked up from amino_acid_info.json, so they are part of the
        # configuration rather than the learnt weights. Keeping them out of the state dictionary
        # means that file stays the single source of truth and checkpoints survive changes to it.
        self.register_buffer("default_unfolded_pKas", unfolded_pkas(), persistent=False)
        self.register_buffer("titratable_residue", titratable_residues(), persistent=False)

    @classmethod
    def from_config(cls) -> "DDGPredictor":
        """Build the predictor with the convolution layout from ``config.json``."""
        from src.config_initialisation import CONFIG

        return cls(
            scan_dimensions=tuple(CONFIG.convolution_scan_dimensions),
            scan_sizes=tuple(CONFIG.convolution_scan_sizes),
        )

    def unfolded_pkas_for(self, residue_indices: torch.Tensor, true_lengths: torch.Tensor) -> torch.Tensor:
        """Return padded-masked unfolded pKas shaped ``[batch, length, 1]``."""
        return mask_padding(self.default_unfolded_pKas[residue_indices].unsqueeze(-1), true_lengths)

    def titratable_for(self, residue_indices: torch.Tensor) -> torch.Tensor:
        """Return a ``[batch, length, 1]`` boolean mask of residues that carry a titratable group."""
        return self.titratable_residue[residue_indices].unsqueeze(-1)

    def predict_pka_deltas(
        self,
        esm_features: torch.Tensor,
        residue_indices: torch.Tensor,
        true_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Predict bounded folded-minus-unfolded pKa residuals; non-titratable residues stay at zero."""
        raw = self.pka_pipeline(esm_features, residue_indices, true_lengths)
        titratable = self.titratable_for(residue_indices)
        return torch.tanh(raw).mul(PKA_DELTA_LIMIT).masked_fill(~titratable, 0.0)

    @staticmethod
    def _validated_lengths(
        wt_esm: torch.Tensor,
        mut_esm: torch.Tensor,
        wt_residue_indices: torch.Tensor,
        mut_residue_indices: torch.Tensor,
        true_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Check input shapes and return ``true_lengths`` as a ``[batch]`` tensor."""
        if wt_esm.ndim != 3 or wt_esm.shape != mut_esm.shape:
            raise ValueError("WT and mutant ESM embeddings must have matching [batch, length, features] shapes")
        if wt_residue_indices.shape != wt_esm.shape[:2] or mut_residue_indices.shape != wt_esm.shape[:2]:
            raise ValueError("Residue indices must have shape [batch, length]")
        if wt_residue_indices.dtype != torch.long or mut_residue_indices.dtype != torch.long:
            raise TypeError("Residue indices must have dtype torch.long")

        batch_size, sequence_length, _ = wt_esm.shape
        if true_lengths.ndim == 2 and true_lengths.shape[-1] == 1:
            true_lengths = true_lengths.squeeze(-1)
        if true_lengths.ndim != 1 or true_lengths.shape[0] != batch_size:
            raise ValueError("true_lengths must have shape [batch] or [batch, 1]")
        if (
            true_lengths.dtype != torch.long
            or (true_lengths < MIN_SEQUENCE_LENGTH).any()
            or (true_lengths > sequence_length).any()
        ):
            raise ValueError(
                f"true_lengths must be between {MIN_SEQUENCE_LENGTH} and the padded sequence length"
            )
        return true_lengths

    def forward(
        self,
        wt_esm: torch.Tensor,
        mut_esm: torch.Tensor,
        wt_residue_indices: torch.Tensor,
        mut_residue_indices: torch.Tensor,
        pH: torch.Tensor,
        experiment_temperature: torch.Tensor,
        true_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Return ddG, ddG_ref and the folded WT/mutant pKas."""
        true_lengths = self._validated_lengths(
            wt_esm, mut_esm, wt_residue_indices, mut_residue_indices, true_lengths
        )

        wt_pKa_deltas = self.predict_pka_deltas(wt_esm, wt_residue_indices, true_lengths)
        mut_pKa_deltas = self.predict_pka_deltas(mut_esm, mut_residue_indices, true_lengths)
        wt_unfolded_pKas = self.unfolded_pkas_for(wt_residue_indices, true_lengths)
        mut_unfolded_pKas = self.unfolded_pkas_for(mut_residue_indices, true_lengths)

        # Pipeline outputs are already zero at padded positions.
        ddG_ref = (
            self.non_electrostatic_pipeline(wt_esm, wt_residue_indices, true_lengths)
            - self.non_electrostatic_pipeline(mut_esm, mut_residue_indices, true_lengths)
        ).sum(dim=1)

        ddG_prediction = self.thermodynamic_output(
            mut_pKa_deltas,
            wt_pKa_deltas,
            mut_unfolded_pKas,
            wt_unfolded_pKas,
            ddG_ref,
            pH,
            experiment_temperature,
        )
        return {
            "ddG_prediction": ddG_prediction,
            "ddG_ref": ddG_ref,
            "folded_pKas": {
                "wildtype": wt_unfolded_pKas + wt_pKa_deltas,
                "mutant": mut_unfolded_pKas + mut_pKa_deltas,
            },
        }
