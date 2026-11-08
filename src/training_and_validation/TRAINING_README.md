# Training

`--train_for_evaluation` and `--train_for_experiment` both use the code in this folder, and both run the same two stages:

1. **PKAD pretraining.** The pKa pipeline alone is trained to reproduce experimental pKa shifts, at `pka_learning_rate` (default 1e-4). Every measured residue counts equally in the loss.
2. **ΔΔG training.** All weights are then trained with a Huber loss on the experimental ΔΔG values. The rest of the network uses `learning_rate` (default 1e-4) while the pKa pipeline uses `pka_finetune_learning_rate` (default 1e-7), which is small enough that the pKa head keeps most of the calibration PKAD gave it. Set that to 0 to freeze the pKa head entirely during this stage.

The progress bar shows the current batch loss and the running loss for the epoch, and the mean loss of every completed epoch is printed. A resumed run continues from its latest ΔΔG checkpoint without repeating PKAD pretraining; checkpoints are kept for the last `max_files_kept` epochs (default 3).

**Runs are reproducible.** `random_seed` fixes weight initialisation and batch order as well as the data split, so the same configuration gives the same model. That is what makes it possible to tell whether a change to the code changed the result, rather than the run simply landing somewhere else.
