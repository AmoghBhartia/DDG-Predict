# Preparing the data (`--prepare`)

[← DDG-Predict main README](README.md)

`--prepare` is the only step that needs MMseqs2, and therefore the only step Windows users must run inside WSL2. It cleans and verifies the source measurements, divides them into training splits by sequence cluster, writes each split its own PKAD table, and computes the ESM-2 embedding caches.

## Windows/WSL2 setup

Install Ubuntu from an elevated PowerShell terminal:

```powershell
wsl --install -d Ubuntu
```

Restart Ubuntu after installation:

1. Close all Ubuntu/WSL terminal windows.
2. From PowerShell, run `wsl --shutdown`.
3. Open Ubuntu again from the Start menu, or launch it from PowerShell:

   ```powershell
   wsl -d Ubuntu
   ```

   If the distribution has a different name, list installed distributions
   with `wsl --list --verbose` and replace `Ubuntu` with that name.

Install or enable Conda inside Ubuntu. Windows Anaconda is separate from WSL2, so a Windows-only Conda installation cannot provide the Linux MMseqs2 executable. In Ubuntu, create the MMseqs2 environment:

```bash
conda create -n ddG_predict -c conda-forge -c bioconda mmseqs2 -y
conda install -n ddG_predict -c conda-forge python=3.14 -y
conda activate ddG_predict
```
Use `conda run` if `conda activate` does not update `PATH`. From the Ubuntu terminal, access the repository on the Windows drive and run preparation in that environment:

```bash
python --version
python -m pip install -r requirements.txt
python run_model.py --prepare
```
## Source data

**You only need `--prepare` to train on your own data.** The curated dataset this step produced is already in the repository as `data/Curated data/curated_data.csv`, and the split folders rebuild themselves from it and `split_categorisation.csv` without MMseqs2, so training, evaluation, plotting and inference all work on a fresh clone with nothing prepared. To reproduce the published curation from its raw input instead, fetch the same FireProtDB export described below.

Three inputs are needed for `--prepare` to run:

1) `data/Source data/fireprotdb_V2.0.csv` — the experimental ΔΔG measurements. **Not included here**: FireProtDB's [terms of use](https://loschmidt.chemi.muni.cz/fireprotdb/terms-of-use/) do not permit its data to be passed on to third parties. Download it from [FireProtDB](https://loschmidt.chemi.muni.cz/fireprotdb/) and put it in `data/Source data/`.

   The file used for the results in this repository is an export filtered to **substitutions**, a **sequence length under 2,000** residues, and entries that carry **a ΔΔG value and a pH**. That returned 7,956 rows across 187 UniProt accessions. Note that `--prepare` discards multi-point substitutions. Re-exporting later may return a different number of rows if the database has been updated, in which case the curated data, and so the measured performance, may not match what is published here.

2) `data/pKa Data/PKAD_pKa_data.csv` — the experimental residue pKas, from [PKAD](https://compbio.clemson.edu/pkad/), which is already included in this repository.

3) `data/Uniprot to sequence/uniprot_sprot_subset.fasta` — the wild-type sequences. The full Swiss-Prot release is 275 MB, too large for this repository, so a subset holding only the 173 required entries has been added to the public repository. It produces exactly the same curated data as the full file.

The mmCIF numbering tables PKAD needs are downloaded from the RCSB Protein Data Bank as they are required, and the ESM-2 embeddings are computed on the first run, so neither is stored here.

Curated rows are divided by wild-type sequence cluster (MMseqs2, 30% identity), so a protein never appears on both sides of a split, and each split gets its own PKAD table with chains similar to its testing-set proteins removed. The thresholds, the sequence reconstruction, the repeat-measurement consolidation and the PKAD residue numbering are all documented in **[src/data_processing/CURATION_README.md](src/data_processing/CURATION_README.md)**.

## Preparing your own data

To prepare your own dataset, put CSV files in `data/Source data` with the columns `sequence`, `mutant_sequence`, `pH`, `zero_indexed_mutation_position`, `wt_residue`, `mut_residue`, `ddG` and `experiment_temperature`; mutation positions in these files are **zero-indexed**. Every `.csv` in that folder is read and concatenated, whatever it is called, so keep only the files you mean to train on.

Also, it would be necessary to download the full Swiss-Prot release, which is `uniprot_sprot.fasta` from the [UniProt FTP site](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/), and put it at `data/Uniprot to sequence/uniprot_sprot.fasta`.