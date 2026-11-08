# DDG-Predict

# Project Overview and Introduction

DDG-Predict is a machine‑learning toolkit for predicting changes in protein thermodynamic stability. Given a wild‑type sequence and a single substitution, it predicts the associated change in Gibbs free energy, ΔΔG, as a function of pH and temperature. The pH at which the prediction is made is optional, and if it is not provided, it defaults to 7.0; the temperature defaults to 298.15 K (25 °C).

This toolkit uses ESM‑2, a protein language model, to obtain residue‑level representations for the wild‑type and mutant sequences. Two learned pipelines process both sequences: a pKa pipeline, which predicts how the folded protein shifts the pKa of every titratable residue, and a non-electrostatic pipeline, which predicts a pH-independent reference ΔΔG. The predicted folded pKas are combined with unfolded residue pKas — approximated by free amino-acid side-chain pKas — in the Wyman–Tanford linkage equation, so the model predicts a complete physically-shaped ΔΔG curve over pH rather than a single number.

Because the model uses sequence embeddings rather than a full tertiary structure provided by atomic coordinates, it is designed to support comparatively rapid screening of single substitutions. Such predictions can help identify the most stabilising single‑point mutations. However, the model is unable to assess functional impact, so further screening is still necessary to exclude any mutations that adversely affect protein functionality, followed by experimental verification of the results.

This model is trained on two datasets:

- **FireProtDB** supplies the experimental ΔΔG measurements the model is trained and scored on.
- **PKAD** supplies experimentally measured residue pKas, used to pretrain the pKa pipeline so the pH-dependent term starts from real electrostatics rather than being learnt from ΔΔG alone.

# Model Architecture

![DDG-Predict model architecture](<src/model/DDG-Predict Model Architecture.png>)

The wild-type and mutant ESM-2 embeddings are passed through stacked convolution layers, which reduce 1280 features per residue to a single value per residue. Two such pipelines run with separate weights: one predicts a bounded pKa shift for every titratable residue, the other a pH-independent reference value. The two are combined through the Wyman–Tanford linkage equation, which turns the predicted folded and unfolded pKas into the pH-dependent part of ΔΔG, so one forward pass gives ΔΔG at any pH and temperature.

The gating, the convolution stages, the learned terminal embeddings and the full linkage equation are set out in **[src/model/MODEL_README.md](src/model/MODEL_README.md#machine-learning-architecture)**.

# Curation Process

During curation, duplicate measurements are removed, and data is cleaned and filtered to remove outliers. It is then split by MMseqs2 sequence clustering so no testing-set protein has a close relative in training, and PKAD residues are also filtered accordingly to ensure no data leakage occurs.

Further details can be found in **[src/data_processing/CURATION_README.md](src/data_processing/CURATION_README.md)**.

# Results and Limitations

Under k-fold cross-validation, the model reaches a **Spearman correlation of 0.48** and a **Pearson correlation of 0.43**, with a mean absolute error of 1.24 kcal mol⁻¹. The best reference predictor reaches 0.31 and 0.25 on the same rows.

The full metrics, baselines and chosen settings are in the **[model performance README](<model performance/PERFORMANCE_README.md>)** and the results and limitations of the model are discussed in **[results/RESULTS_README.md](results/RESULTS_README.md)**.

# Starting the Toolkit
The toolkit is launched from `run_model.py`.

```bash
git clone https://github.com/AmoghBhartia/DDG-Predict.git
cd DDG-Predict
```
Use Python 3.11 or newer. From the repository root, install the dependencies (PyTorch among them) and confirm that the command-line options are available:

```bash
python -m pip install -r requirements.txt
python run_model.py --help
```

**On Windows with an NVIDIA graphics card**, install PyTorch first from the CUDA index, because the default Windows package is CPU-only and will silently ignore the card:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```
See the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the command matching your CUDA version. On Linux the default package already includes CUDA support, and on a machine with no NVIDIA card nothing extra is needed.

See [How to Use This Toolkit](#how-to-use-this-toolkit) for the other modes and their required input files. The default device setting is `auto`, which uses CUDA when available and otherwise falls back to CPU. ESM model weights are downloaded the first time embeddings are generated, so that initial run requires an internet connection.

# How to Use This Toolkit

DDG-Predict is operated through `run_model.py`. The available options are `--prepare`, `--train_for_evaluation`, `--train_for_experiment`, `--find_stable_mutant`, `--plot_graph`, and `--config`.

### Run on a free Kaggle GPU (`--remote kaggle`)

ESM‑2 is slow without a graphics card, so the three training and experiment options (`--train_for_evaluation`, `--train_for_experiment`, `--find_stable_mutant`) can each be sent to a free Kaggle GPU with `--remote kaggle`, which brings the results back to this computer. The setup and how resuming works are in **[REMOTE_RUNNING.md](REMOTE_RUNNING.md)**.

### Configure the toolkit (`--config`)

Settings are supplied as a JSON object through `--config` (for example `--config '{"batch_size": 8}'`) or loaded from `data/Configuration/config.json`. A full table of changeable parameters is listed in **[CONFIG.md](CONFIG.md)**.

### Prepare data (`--prepare`)

`--prepare` is for training the model on **your own measurements**. It cleans and verifies source data, then divides it into training and testing splits by sequence cluster so that no protein appears on both sides. Nothing needs to be prepared to use the model as published: the curated dataset it produced is already in the repository, and the split folders rebuild themselves from it.

It relies on MMseqs2 through ProtClust, which is not native to Windows, so it must be run in Ubuntu through WSL2. The setup, the input format and the FireProtDB export used here are in **[PREPARE.md](PREPARE.md)**.

### How training works

Both training options share the same two-stage procedure, described in **[src/training_and_validation/TRAINING_README.md](src/training_and_validation/TRAINING_README.md)**.

### Train for evaluation (`--train_for_evaluation`)

```bash
python run_model.py --train_for_evaluation
```
This trains one model for every combination of **training splits** (different splits of training and testing data) and **initial states** (different starting weights for training), based on the splits `--prepare` wrote. This training step is designed for k-fold cross-validation to measure model performance. This will not be needed unless someone is retraining the model on their own data.

Each training split also uses its **own PKAD table**, prepared by the `--prepare` step, ensuring that no data leakage from PKAD occurs into the testing sets. Checkpoints are written to `checkpoints/evaluation/ddg/` and `checkpoints/evaluation/pka/` as `initial_state_x_training_set_y_epoch_z.pt`.

If the local device doesn't have a GPU, it is recommended to use `--remote kaggle` to complete this step (the alternative requires over 40 compute hours). In this setting, the **checkpoints will not be returned locally** and only the model performance metrics are returned (use `--train_for_experiment` for checkpoint weights to be returned). **Note that the kaggle session will disconnect automatically after 11 hours and reset, so attempting validation with more than 70 total runs (training splits × initial states) is not recommended.**

### Train for use in experiments (`--train_for_experiment`)

```bash
python run_model.py --train_for_experiment
```
This trains a single model on the whole curated dataset and writes it to `checkpoints/experiment/ddg/`. This is the model `--find_stable_mutant` uses for experiments, and it uses the full dataset as it is designed for experimental exploration rather than to test model performance.

### Run experiments (`--find_stable_mutant`)

Provide wild-type sequences in `data/Experiments/wt_sequences.fasta`, with a sequence ID in each FASTA header. Each sequence must contain between 30 and `max_sequence_length` (default 2000) residues.

To analyse the full sample space of possible substitutions, no further files are required. The system will automatically evaluate all sequences at pH 7.0. To evaluate at a pH other than 7.0, provide `data/Experiments/pH_conditions.csv` with `sequence_id` and `pH` columns. For each sequence–pH row with sequence length $L$, this mode evaluates $19L$ non-identical single substitutions.

For selected mutations, provide `data/Experiments/experiments.csv` with `sequence_id`, `mutants`, and `pH` columns. A mutation is written using one-based notation, such as `A123Q`; missing pH values default to 7.0. Either CSV may also contain an optional `experiment_temperature` column in Kelvin; missing temperatures default to 298.15 K (25 °C).

Every result receives an `experiment_id` of the form `<sequence_id>_<wt><one-based position><mut>`, for example `lysozyme_N64D`. If the same mutation is evaluated under several conditions, the pH and temperature are appended (for example `lysozyme_N64D_pH3.5_T298.15K`) so that each result keeps its own detail file.

Results are written to `results/experiment/<timestamp>/predictions.csv`, ranked from most stabilising to least stabilising within each sequence and pH. The matching `results/experiment/<timestamp>/details/` directory contains one `.pt` file per result with folded pKas and `ddG_ref`.

### Plot results (`--plot_graph`)

Plots a saved prediction's pH curve. This operates under two modes:

1) Evaluation (plotting a graph from `--train_for_evaluation` results):
   This picks a particular training split and initial state from an evaluation run and plots the result from there.

```bash
python run_model.py --plot_graph <EXPERIMENT_ID> evaluation <TRAINING_SPLIT> <INITIAL_STATE>
```

2) Experiment (plotting a graph from `--find_stable_mutant` results):
   This picks a scan by the timestamp of its folder under `results/experiment/` and plots the result from there.

```bash
python run_model.py --plot_graph <EXPERIMENT_ID> experiment <TIME_OF_EXPERIMENT>
```

# AI Assistance
Generative AI was used to help structure the documentation, to improve the efficiency of parts of the code (for example in developing the remote Kaggle threading system), and to format a couple of the graphs.

# Citations
See [CITATIONS.md](CITATIONS.md#references) for references used for this toolkit.

# Using this toolkit
If you wish to use the architecture or model developed in this GitHub repository, please cite as follows:

Bhartia, A. (2026). DDG‑Predict: A sequence‑embedding‑based toolkit for predicting pH‑dependent protein stability changes (ΔΔG). GitHub Repository.
https://github.com/AmoghBhartia/DDG-Predict
