# Running on a free Kaggle GPU (`--remote kaggle`)

[← DDG-Predict main README](README.md)

ESM‑2 is slow without a graphics card: a full scan of a 1,100-residue protein is over 20,000 mutants, and thirty validation runs is thirty full trainings. Adding `--remote kaggle` sends the work to a free Kaggle GPU and brings the results back to this computer. It works with the experiment and with either kind of training:

```bash
python run_model.py --find_stable_mutant --remote kaggle
python run_model.py --train_for_evaluation --remote kaggle
python run_model.py --train_for_experiment --remote kaggle
```
The toolkit uploads a small bundle (the code, your configuration and experiment files, the newest ΔΔG checkpoint and the run's saved progress) as a **private** Kaggle dataset, and starts a **private** Kaggle script that runs `--find_stable_mutant` on a GPU. It checks on the script every few minutes (`kaggle_poll_minutes`). When the session finishes, the new predictions and detail files are downloaded and merged into the usual `results/experiment/<timestamp>/` folder, so the results look exactly like a local run's.

Kaggle sessions are limited to 12 hours, so each session stops itself safely after `kaggle_session_hours` (default 11), keeping all of its progress. The toolkit then automatically starts the next session from where the last one stopped, until every mutant is predicted. Because it uses the same run folder and resume rules as a local run, an unfinished run started on this computer is continued on Kaggle, and the other way round.

**One-off setup:**

1. Install the Kaggle tool: `python -m pip install kaggle`.
2. On kaggle.com, open **Settings → API** and create a token. Set it as the `KAGGLE_API_TOKEN` environment variable.
3. Verify your phone number on kaggle.com. Kaggle requires this before scripts can use a GPU or download the ESM‑2 weights.
4. Optionally add your Kaggle username to `data/Configuration/config.json`, e.g. `"kaggle_username": "your-name"`.
   If you leave it blank, the username is read from your API token, so nothing identifying is stored in the repository.
   If you do set it, it must match the account the token belongs to, and the toolkit says so if it does not.

The remaining settings rarely need changing: `kaggle_slug` names the private datasets and scripts (`<slug>-data` and `<slug>-runner` for experiments, `<slug>-train` and `<slug>-trainer` for training), and `kaggle_machine_shape` selects the GPU (default `NvidiaTeslaT4`).
