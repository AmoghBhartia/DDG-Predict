# Configuring the toolkit (`--config`)

[← DDG-Predict main README](README.md)

Settings may be supplied as a JSON object through `--config` (for example `--config '{"batch_size": 8}'`) or loaded from `data/Configuration/config.json`. Configuration values are validated during startup, and unknown keys are rejected. Repository paths are configured separately in `data/Configuration/paths_config.json`. The most commonly changed settings are:

| Setting | Default | Purpose |
| ------- | ------- | ------- |
| `num_epochs` / `pka_num_epochs` | 5 / 20 | ΔΔG training epochs / PKAD pretraining epochs |
| `learning_rate` | 1e-4 | Learning rate of the network, except the pKa pipeline |
| `pka_learning_rate` / `pka_finetune_learning_rate` | 1e-4 / 1e-7 | Learning rate of the pKa pipeline during PKAD pretraining / during ΔΔG training |
| `random_seed` | 42 | Fixes the data split, the weight initialisation and the batch order |
| `batch_size` | 4 | Sequences per batch |
| `evaluation_splits` | 10 | Training splits `--prepare` divides the curated data into |
| `evaluation_states` | 3 | Initial states `--train_for_evaluation` trains per split |
| `experiment_chunk_size` | 500 | Mutants predicted (and saved) per chunk in `--find_stable_mutant` |
| `esm_batch_size` / `esm_attention_budget` | 16 / 4,000,000 | Sequences per ESM batch, and the cap on `batch x padded_length²` that shrinks a batch of long sequences so attention fits in GPU memory |
| `device` | `auto` | `auto`, `cpu` or `cuda` |
| `convolution_scan_sizes` / `convolution_scan_dimensions` | `[1,2,3,4]` / `[256,64,16,4]` | Kernel size and output width of each convolution stage |

Changing `convolution_scan_sizes` or `convolution_scan_dimensions` changes the architecture, so existing checkpoints will no longer load.
