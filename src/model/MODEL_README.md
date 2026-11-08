# Machine Learning Architecture

[← DDG-Predict main README](../../README.md)

![DDG-Predict model architecture](<DDG-Predict Model Architecture.png>)

`DDGPredictor` (`ddG_predictor_model.py`) predicts the change in folding free energy caused by a single point mutation, as a function of pH and temperature. It does this by splitting ΔΔG into two parts that are learnt separately: a pH-independent term, and an electrostatic term that is reconstructed from per-residue pKa values through the Wyman–Tanford linkage relation.

---

## 1. Inputs

| Input | Shape | Meaning |
| ----- | ----- | ------- |
| `wt_esm`, `mut_esm` | `[batch, length, 1280]` | per-residue ESM-2 embeddings |
| `wt_residue_indices`, `mut_residue_indices` | `[batch, length]` | amino-acid identity at each position |
| `true_lengths` | `[batch]` | real sequence length before padding |
| `pH`, `experiment_temperature` | `[batch]` | conditions; temperature in Kelvin |

Sequences in a batch are padded to a common length. Every per-residue tensor is passed through `mask_padding`, which zeroes padded positions, so padding contributes nothing to any sum.

---

## 2. The two interaction pipelines

Wild type and mutant are processed through a series of layers. The layers use identical weights for wildtype and mutant, ensuring the sequences are treated equally. Two instances of the `InteractionPipeline` are used. 

- The first is a **pKa head** that is trained to predict the difference in pKa between the folded and unfolded state of charged residues within a sequence.
- The second is a **non-electrostatic** head designed to predict the local non-electrostatic interactions at one particular point. 

The pipeline is built from the following standard blocks: 

1) **`SequenceGate`.** Compresses 1280 ESM features to 256 through a standard linear layer followed by a specialised gate with a SwiGLU feed-forward block. The specialised gate is constructed by merging (1, 256) embeddings stored by the model (for each of the 20 amino acids) into a full sequence tensor of size (L,256) and then passing this tensor through a sigmoid function. 

2) **`Convoluter`.** Four pre-norm residual stages of 1-D convolution then widen the receptive field while narrowing the feature count. Kernel sizes and widths come from `convolution_scan_sizes` and `convolution_scan_dimensions` in `config.json`; by default kernels are 1, 2, 3, 4 and widths run 256 → 64 → 16 → 4.

Within the **`Convoluter`**, two details are non-standard: 

- Firstly, the activation is a **sign-symmetric SiLU**, `sign(x)·SiLU(|x|)`. Symmetry matters here because at the later stages of the model, these numbers either represent pKa shifts or energy contributions. Applying a non-symmetric activation results in the negative shifts being actively compressed against the positive shifts, when there is no physical reason of doing so. Thus it was decided to make this activation completely symmetric. 

- Secondly, the convolution padding is specialised. A normal convolution padding may add artificial 0s at the end or reflect the start values backwards. However, for a protein, the end of a sequence signifies either a C-terminus or an N-terminus. As a result, the model stores C and N-terminus embeddings for each amino acid, which are added onto the ends of the sequence to add useful context that this is the end of the protein sequence. 

A final linear map takes the 4 remaining features to one number per residue. 
---

## 3. Interpreting the results from the head 

The (L,1) output from the pKa head and the non-electrostatic head must then be processed: 

1) **pKa head:** Two (L,1) tensors for wildtype and mutant are then passed through a sign preserving tanh function `ADD EQUATION` that bounds the maximum change to ±5 pKa units, which keeps early training stable and matches the expected physical range. This is then added to the unfolded pKa values, which are **approximated by free amino-acid side-chain pKas**. 

     The unfolded values are looked up per amino acid from [`amino_acid_info.json`](<../../data/Amino acid information/amino_acid_info.json>) and residues flagged there as non-titratable are masked, ensuring that they complete cancel out and do not impact downstream layers. 

2) **Non-electrostatic head:** Two (L,1) tensors are summed across L to give a (1,1) total non-electrostatic contribution for the wildtype and for the mutant. The difference between the mutant and wildtype is used to caclulate the pH-independent reference energy $\Delta\Delta G_{\mathrm{ref}}$. This is the term that carries packing, hydrophobicity and everything else the model infers from sequence context but cannot attribute to a titratable group.

---

## 4. Wyman–Tanford linkage (`WymanTanfordOutput`)

These terms then provide the input of the Wyman-Tanford equation. The electrostatic contribution comes from thermodynamic linkage: if a residue's pKa differs between the folded and unfolded states, the free energy of folding depends on pH. For one sequence of length $L$, define the sum:

$$\Theta(\mathrm{pH}) = \sum_{i=1}^{L}\left[\,\Lambda\left(\mathrm{pH}-\mathrm{p}K^{\mathrm{fold}}_{a,i}\right) - \Lambda\left(\mathrm{pH}-\mathrm{p}K^{\mathrm{unf}}_{a,i}\right)\right]$$

The prediction is then:

$$\Delta\Delta G(\mathrm{pH}) = \Delta\Delta G_{\mathrm{ref}} - RT\ln(10)\left[\Theta_{\mathrm{mut}}(\mathrm{pH}) - \Theta_{\mathrm{wt}}(\mathrm{pH})\right]$$

with $R = 1.987\times10^{-3}\ \mathrm{kcal\ mol^{-1}\ K^{-1}}$ and $T$ in Kelvin. Because $\Delta\Delta G_{\mathrm{ref}}$, the folded pKas and the unfolded pKas are all independent of pH, this enables ΔΔG to be found at **any** pH. This is what `--plot_graph` uses to draw a full titration curve from one saved result.

**Numerical form.** The linkage function $\Lambda$ is

$$\Lambda(x) = \log_{10}\left(1 + 10^{x}\right) = \frac{\operatorname{softplus}\left(x\ln 10\right)}{\ln 10}$$

The two expressions are identical, but the direct form can diverge to infinity and produces NaN gradients. Softplus is linear for large inputs and its derivative stays in $[0, 1]$, preventing this divergence. 

---

## 5. Training

Training runs in two stages, because the pKa pipeline has a supervision signal of its own that the ΔΔG data cannot provide.

**Stage 1 — pKa pre-training.** On a fresh run the pKa pipeline is trained directly on measured residue pKas from PKAD, under a mean squared error taken over titratable residues only. PKAD proteins similar to the ΔΔG validation set are excluded, and their embeddings are cached separately, so this stage cannot leak into validation.

**Stage 2 — ΔΔG training.** All weights are then trained on measured ΔΔG under a Huber loss, which behaves quadratically for small residuals and linearly beyond the threshold, so the strongly destabilising outliers common in FireProtDB do not dominate the gradient.

The optimiser used is AdamW. Although the pKa head can be fine-tuned during ΔΔG training, the learning rate was kept at 1e-7 so as to not wash out any learning from the PKAD dataset. This was also found to significantly improve model performance. 

---

## 6. What the model returns

`forward` returns a dictionary:

| Key | Shape | Meaning |
| --- | ----- | ------- |
| `ddG_prediction` | `[batch, 1]` | ΔΔG at the requested pH and temperature |
| `ddG_ref` | `[batch, 1]` | the pH-independent term |
| `folded_pKas["wildtype"]`, `folded_pKas["mutant"]` | `[batch, length, 1]` | per-residue folded pKas |

Evaluation and experiments write a predictions CSV plus one `.pt` detail file per result holding the folded pKas, residue indices, `ddG_ref` and the temperature. `--plot_graph` reconstructs the full pH curve from that file without re-running the network.

For validation performance and known limitations, see [results/RESULTS_README.md](../../results/RESULTS_README.md).
