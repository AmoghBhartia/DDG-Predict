import math
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import ESM_EMBEDDING_DIM, STANDARD_AMINO_ACIDS

RESIDUE_COUNT: Final[int] = len(STANDARD_AMINO_ACIDS)
LN10: Final[float] = math.log(10.0)


# Shared tensor helpers.
def sequence_mask(true_lengths: torch.Tensor, padded_length: int) -> torch.Tensor:
    """Return a ``[batch, length]`` mask that is True for real (non-padding) residues."""
    positions = torch.arange(padded_length, device=true_lengths.device)
    return positions.unsqueeze(0) < true_lengths.reshape(-1, 1)


def mask_padding(values: torch.Tensor, true_lengths: torch.Tensor) -> torch.Tensor:
    """Zero the padded positions of a ``[batch, length, features]`` tensor."""
    mask = sequence_mask(true_lengths, values.shape[1]).unsqueeze(-1)
    return values.masked_fill(~mask, 0.0)


def log10_one_plus_pow10(x: torch.Tensor) -> torch.Tensor:
    """Compute ``log10(1 + 10**x)`` stably as ``softplus(x * ln 10) / ln 10``.

    The direct form overflows to ``inf`` once ``10**x`` exceeds float32 range (x > ~38), which
    produces NaN gradients. Softplus is linear for large inputs, and its gradient
    ``sigmoid(x * ln 10)`` is always finite and lies in [0, 1].
    """
    return F.softplus(x * LN10) / LN10


def sign_symmetric_silu(values: torch.Tensor) -> torch.Tensor:
    """Apply SiLU symmetrically so equal-magnitude inputs have equal output magnitudes."""
    return torch.sign(values) * F.silu(values.abs())


def validate_scan_layout(scan_sizes: tuple[int, ...], scan_dimensions: tuple[int, ...]) -> None:
    """Check that convolution kernel sizes and feature dimensions describe valid stages."""
    if not scan_sizes or len(scan_sizes) != len(scan_dimensions):
        raise ValueError("scan_sizes and scan_dimensions must be non-empty and of equal length")
    if len(set(scan_sizes)) != len(scan_sizes):
        raise ValueError("scan_sizes must be unique; each size names its own terminal embedding table")
    if min(scan_sizes) < 1 or min(scan_dimensions) < 1:
        raise ValueError("scan_sizes and scan_dimensions must be positive")


# Learned terminal padding.
class _AminoAcidEmbeddingSet(nn.Module):
    """Learned N- and C-terminal padding embeddings for every residue and convolution stage."""

    def __init__(self, scan_sizes: tuple[int, ...], scan_dimensions: tuple[int, ...]) -> None:
        super().__init__()
        self.c_terminus = self._terminal_table(scan_sizes, scan_dimensions)
        self.n_terminus = self._terminal_table(scan_sizes, scan_dimensions)

    @staticmethod
    def _terminal_table(scan_sizes: tuple[int, ...], scan_dimensions: tuple[int, ...]) -> nn.ParameterDict:
        return nn.ParameterDict({
            f"scan_{size}": nn.Parameter(torch.randn(RESIDUE_COUNT, size, dimension))
            for size, dimension in zip(scan_sizes, scan_dimensions)
        })


class AminoAcidEmbeddingBank(nn.Module):
    """Separate learned terminal embeddings for the pKa and non-electrostatic pipelines."""

    def __init__(self, scan_sizes: tuple[int, ...], scan_dimensions: tuple[int, ...]) -> None:
        super().__init__()
        self.pka = _AminoAcidEmbeddingSet(scan_sizes, scan_dimensions)
        self.non_electrostatic = _AminoAcidEmbeddingSet(scan_sizes, scan_dimensions)


# Network blocks.
class SequenceGate(nn.Module):
    """Project sequence features, apply residue-specific gates, and use SwiGLU."""

    def __init__(self, feature_size: int, gate_size: int = ESM_EMBEDDING_DIM, compressed_size: int = 256) -> None:
        super().__init__()
        if min(feature_size, gate_size, compressed_size) < 1:
            raise ValueError("feature_size, gate_size, and compressed_size must be positive")

        self.sequence_projection = nn.Linear(feature_size, gate_size)
        self.learned_gate = nn.Parameter(torch.randn(RESIDUE_COUNT, 1, gate_size))
        self.swiglu_projection = nn.Linear(gate_size, compressed_size * 2)

    def forward(self, sequence: torch.Tensor, residue_indices: torch.Tensor) -> torch.Tensor:
        """Return compressed sequence features shaped ``[batch, length, compressed_size]``."""
        gate = self.learned_gate[residue_indices].squeeze(2)
        gated = self.sequence_projection(sequence) * torch.sigmoid(gate)
        activation, value = self.swiglu_projection(gated).chunk(2, dim=-1)
        return F.silu(activation) * value


class Convoluter(nn.Module):
    """Apply normalized SiLU features, terminal padding, convolution, and a residual."""

    def __init__(self, feature_size: int, kernel_size: int, output_size: int | None = None) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        output_size = feature_size if output_size is None else output_size
        self.kernel_size = kernel_size
        self.normalization = nn.LayerNorm(feature_size)
        self.linear = nn.Linear(feature_size, output_size)
        self.convolution = nn.Conv1d(output_size, output_size, kernel_size=kernel_size, padding=0)
        self.residual_projection = nn.Linear(feature_size, output_size)

    def _check_terminals(
        self,
        sequence: torch.Tensor,
        n_terminus: torch.Tensor | None,
        c_terminus: torch.Tensor | None,
    ) -> None:
        if n_terminus is None or c_terminus is None:
            raise ValueError("n_terminus and c_terminus are required when kernel_size is greater than 1")
        if n_terminus.ndim != 3 or c_terminus.ndim != 3:
            raise ValueError("Terminal embeddings must have shape [batch, scans, feature_size]")
        if n_terminus.shape[0] != sequence.shape[0] or c_terminus.shape[0] != sequence.shape[0]:
            raise ValueError("Terminal embeddings must have the same batch size as sequence")
        if n_terminus.shape[1] + c_terminus.shape[1] != self.kernel_size - 1:
            raise ValueError("Terminal scan lengths must sum to kernel_size - 1")

    @staticmethod
    def _add_terminals(
        values: torch.Tensor,
        n_terminus: torch.Tensor,
        c_terminus: torch.Tensor,
        true_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Extend each sequence with its terminal embeddings, at its own C-terminus.

        Sequences in a batch are padded to a common length, so the C-terminus of a shorter
        sequence is not the end of the row. The learned C-terminal embedding is written
        immediately after that sequence's last real residue and everything beyond it is zeroed,
        so a residue near a terminus is read in the same learned context whatever else shares
        its batch. Without this, a short sequence would see padding where its terminal
        embedding should be, and its prediction would depend on the batch it landed in.
        """
        batch, padded_length, features = values.shape
        n_length, c_length = n_terminus.shape[1], c_terminus.shape[1]
        extended = torch.cat([n_terminus, values, values.new_zeros(batch, c_length, features)], dim=1)
        if c_length == 0:
            return extended

        lengths = (
            torch.full((batch,), padded_length, device=values.device, dtype=torch.long)
            if true_lengths is None
            else true_lengths.reshape(-1).to(values.device)
        )
        positions = torch.arange(extended.shape[1], device=values.device).unsqueeze(0)
        offset = positions - (lengths + n_length).unsqueeze(1)  # 0 at each sequence's C-terminus
        inside = (offset >= 0) & (offset < c_length)
        terminal = c_terminus.gather(
            1, offset.clamp(0, c_length - 1).unsqueeze(-1).expand(-1, -1, features)
        )
        extended = torch.where(inside.unsqueeze(-1), terminal, extended)
        return extended.masked_fill((offset >= c_length).unsqueeze(-1), 0.0)

    def forward(
        self,
        sequence: torch.Tensor,
        n_terminus: torch.Tensor | None = None,
        c_terminus: torch.Tensor | None = None,
        true_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform ``[batch, length, features]`` and preserve its sequence length."""
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, length, feature_size]")

        residual = self.residual_projection(sequence)
        transformed = sign_symmetric_silu(self.linear(self.normalization(sequence)))

        if self.kernel_size > 1:
            self._check_terminals(sequence, n_terminus, c_terminus)
            transformed = self._add_terminals(transformed, n_terminus, c_terminus, true_lengths)
        transformed = self.convolution(transformed.transpose(1, 2)).transpose(1, 2)
        return transformed + residual


class WymanTanfordOutput(nn.Module):
    """Calculate pH-dependent ddG from folded and unfolded pKa values."""

    def __init__(self, gas_constant: float = 1.987e-3) -> None:
        """Create the thermodynamic linkage output using kcal/(mol K) units."""
        super().__init__()
        if gas_constant <= 0:
            raise ValueError("gas_constant must be positive")
        self.gas_constant = gas_constant

    def forward(
        self,
        mutant_pKa_deltas: torch.Tensor,
        wildtype_pKa_deltas: torch.Tensor,
        mutant_unfolded_pKas: torch.Tensor,
        wildtype_unfolded_pKas: torch.Tensor,
        ddG_ref: torch.Tensor,
        pH: torch.Tensor,
        experiment_temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``ddG(pH)`` from folded-minus-unfolded pKa deltas."""
        pKa_inputs = (mutant_pKa_deltas, wildtype_pKa_deltas, mutant_unfolded_pKas, wildtype_unfolded_pKas)
        if any(values.ndim != 3 or values.shape[-1] != 1 for values in pKa_inputs):
            raise ValueError("All pKa inputs must have shape [batch, length, 1]")
        if any(values.shape != mutant_pKa_deltas.shape for values in pKa_inputs[1:]):
            raise ValueError("All pKa inputs must have the same shape")

        scalar_inputs = (ddG_ref, pH, experiment_temperature)
        if any(values.ndim not in {1, 2} for values in scalar_inputs):
            raise ValueError("ddG_ref, pH, and experiment_temperature must be [batch] or [batch, 1]")

        batch_size = mutant_pKa_deltas.shape[0]
        ddG_ref, pH, experiment_temperature = (values.reshape(batch_size, 1) for values in scalar_inputs)
        if not torch.isfinite(torch.cat((ddG_ref, pH, experiment_temperature), dim=1)).all():
            raise ValueError("ddG_ref, pH, and experiment_temperature must be finite")
        if (experiment_temperature <= 0).any():
            raise ValueError("experiment_temperature must be in Kelvin and greater than zero")

        def linkage_term(pKa_deltas: torch.Tensor, unfolded_pKas: torch.Tensor) -> torch.Tensor:
            folded_pKas = unfolded_pKas + pKa_deltas
            pH_column = pH.unsqueeze(1)
            folded = log10_one_plus_pow10(pH_column - folded_pKas)
            unfolded = log10_one_plus_pow10(pH_column - unfolded_pKas)
            return (folded - unfolded).sum(dim=1)

        mutant_term = linkage_term(mutant_pKa_deltas, mutant_unfolded_pKas)
        wildtype_term = linkage_term(wildtype_pKa_deltas, wildtype_unfolded_pKas)
        # ln(10) converts the base-10 linkage terms to natural-log free energies (exact, not 2.303).
        return ddG_ref - LN10 * self.gas_constant * experiment_temperature * (mutant_term - wildtype_term)
