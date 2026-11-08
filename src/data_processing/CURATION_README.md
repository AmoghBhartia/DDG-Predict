# Curation Process

[← DDG-Predict main README](../../README.md)

This section describes how raw measurements are split into the training and validation sets, and where the model's scientific inputs come from. The code for every step can be found in this folder (`src/data_processing/`) and runs as part of `python run_model.py --prepare`.

Curation thresholds are configurable in `config.json` and should be reviewed when preparing a dataset for a new experimental run. By keeping the random seeds and configuration information, this whole process is completely repeatable, so results should be reproducible. 

---

## 1. ΔΔG data curation

The preparation pipeline combines source-curated and non-source-curated measurements, groups repeat experiemnts, and rejects rows that fail the data-consistency checks. A single measurement is retained without statistical filtering as it is not possible to check with other experimental results for consistency. 

For three or more readings of the same experiment, it is possible to remove outliers. Outliers are removed when a reading x_i is 1.5 times further than the MAD (normal-scaled). 

All results are then checked for consistency by calculating the standard error of the mean over the retained values,

$$\mathrm{SEM}(x) = \frac{s(x)}{\sqrt{n}}$$.

The group of readings is accepted when the SEM is at most the configured tolerance (default 0.25 kcal mol⁻¹), and the data is combined by taking the mean of that group of readings. A group that fails is excluded from the curated training data. Every decision is recorded in a timestamped audit file in `data/Curated data/Historical Logs`.

## 2. Dividing the data into training splits

Wild-type sequences are clustered with MMseqs2 at 30% sequence identity (`clustering.py`), and the division is made **by cluster, not by datapoint**, so a testing-set protein never has a close relative on its own training side. Coverage defaults to 0.8 with `cov_mode` 0.

Each split is written to `data/Curated data/split_<n>/` as `training_data.csv` and `testing_data.csv`, with `split_categorisation.csv` recording which split each sequence belongs to. The folders are rebuilt every time `--prepare` runs, and the composition of every division is recorded in `Historical Logs/split_log_<timestamp>.json`.

## 3. Amino-acid information

Each amino acid is marked as titratable or not, and the seven that titrate (C, D, E, H, K, R, Y) carry an unfolded side-chain pKa, approximated by the free amino-acid value. The flag determines whether the model masks the side chain pKa, ensuring that only titratable pKa inputs are used in the Wyman–Tanford equation.

The sources and reasoning behind the values in [`amino_acid_info.json`](<../../data/Amino acid information/amino_acid_info.json>) are set out in [CITATIONS.md](../../CITATIONS.md#sources-for-amino-acid-pka-values).

## 4. PKAD curation

A PDB residue number ("Res ID" in PKAD) is **not** necessarily the position in the sequence. It is thus necessary to compare to the entry's own residue-numbering table (`_pdbx_poly_seq_scheme` in the mmCIF file), and the residue letter is checked at that position. 

Records that do not match fall back to a local aligment where close neighbours resiues are checked. If this shift results in virtually a full set of Res IDs aligning for a particular sequence, then only is that shift accepted. Records that neither step can confirm are dropped rather than guessed.

Repeat measurements of the same residue are averaged, and the cleaning summary reports how many records were placed exactly, placed by the fallback, or dropped.

```
