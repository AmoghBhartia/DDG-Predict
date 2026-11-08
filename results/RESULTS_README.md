# DDG-Predict — Results and Limitations

[← DDG-Predict main README](../README.md)

## Model Performance

The model performance on testing data can be found in the **[model performance README](<../model performance/PERFORMANCE_README.md>)**.

## Model Behaviour across pH

Five prediction detail files ship with the repository so that `--plot_graph` can be tried straight after cloning, without training anything first: thioredoxin 1 D27A and lysozyme C V128F, which are the two curves in Figure 1 below, lysozyme C C95A and the H65A mutant of split 5, which both titrate clearly, and polymerase E671W from the whole-protein scan for the experiment mode. Every other detail file is left out, as there is one per prediction and about 13 GB of them in total.

Predicting a folded pKa for every residue means one forward pass gives ΔΔG at any pH, by producing the necessary parameters for a full graph of ΔΔG as a function of pH. To test if the model can meaningfully interpret the impact electrostatic interactions have, a comparison can be made between two mutations. In the first case the mutation is a switch between two uncharged residues. In the second case, the mutation is a switch between a charged and uncharged residue. If the model can indeed identify the residues that result in electrostatic changes, it should show a clear titration curve for the latter case, and a virtually flat line for the former.

Two such cases have been shown in the Figure below:

![Predicted ΔΔG against pH for a mutation that removes a titratable side chain and one that does not](<pH dependence by mutation type.png>)

Figure 1: predicted ΔΔG against pH for two validation mutations.

Thioredoxin 1 D27A removes a buried aspartate, and the prediction falls by 5.8 kcal mol⁻¹ between pH 2 and pH 12 as the residue titrates. Lysozyme C V128F exchanges a valine for a phenylalanine, neither of which titrates, and its curve moves by only 0.33 kcal mol⁻¹ over the same range. The model has learnt which mutations should respond to pH, rather than applying a pH term indiscriminately.

## Scanning the full sample space of possible mutations

Every single substitution of the human DNA polymerase δ catalytic subunit (UniProt Q308M6, unreviewed, 1,133 residues) was scanned at pH 7.0 — 1,133 × 19 = **21,527 mutants** — in one `--remote kaggle` session on a single T4, taking about four and a half hours.

Table 2: the six most destabilising and six most stabilising mutations from a complete sample space run of the human DNA polymerase δ catalytic subunit (positions are **one-based**; negative means stabilising)
| Most destabilising | Predicted ΔΔG / kcal mol⁻¹ | | Most stabilising | Predicted ΔΔG / kcal mol⁻¹ |
| ------------------ | -------------------------- |-| ---------------- | -------------------------- |
| F789S | +7.93 | | E671W | −5.39 |
| F789G | +7.69 | | E671L | −4.76 |
| F789A | +7.26 | | E671F | −4.69 |
| F789T | +6.86 | | E671Y | −4.64 |
| F789C | +6.29 | | E671I | −4.53 |
| I141R | +6.22 | | Y1084C | −4.36 |

Let's first consider the destabilising mutations. Five of the six worst predictions are substitutions at F789, which sits five residues after the YGDTDS motif found commonly in DNA polymerase, placing them inside a strongly conserved catalytic core. This is actually an interesting condition where the model has implicitly learnt functionality through ESM-2. A possible theory is that because this region is highly conserved in many proteins, ESM-2 amplifies the effect of mutations in this region.

If the ΔΔG of this mutation were really measured, it would most likely not show such a highly destabilising value. Instead the model has learnt that this is a conserved region that should not be touched. In some ways, this is beneficial because it means that the stabilising mutations the model predicts are less likely to directly impact the functionality of the protein. However, this evidently does **not** solve the functionality issue, because the model is simply guessing based on the upstream input from ESM-2. Instead, it means that for a purely numerical standpoint the model is inherently biased.

Despite the current issue that the model predicts stabilising mutations in regions that affect protein functionality, `--find_stable_mutant` is nonetheless a valuable resource to shortlist potential mutations that could be stabilising, assisting with protein design for extreme temperatures and pH values. Then some entries can be manually screened out if they would adversely affect protein functionality. Future versions of this model will hope to address this particular issue by training the model on domains which are typically conserved so that it can screen those positions out prior to evaluating ΔΔG.

## Limitations

This section highlights the key limitations that the model results show, and potential ways future versions of DDG Predict will address this issue.

**Functionality is not checked:** The model has not been trained to recognise active sites, binding interfaces or any other functional motif, so it will happily propose stabilising mutations that would adversely affect protein functionality. Whilst some functionality is inherently hidden inside ESM-2 embeddings, this is unpredictable and can lead to the model choosing when functionality makes a difference or doesn't. Future versions will add a functionality head to ensure strongly conserved residues are filtered out at possible locations for mutation. 

**Limited tertiary structure input:** Training uses sequence-derived information only. The model can infer secondary structure and which titratable groups are affected, but struggles with hydrogen-bonding networks and inter-molecular interactions. This can often lead to the model attempting to overfit to the training data, given that it has such little information about the overall 3D structure. 

**The database is biased toward extremes:** Experimental ΔΔG measurements are rarely made at random. Measurement are usually taken for protein design, or to explain the adverse effect of a particular mutation in disease. In the case of the former, one would look for highly stabilising mutations, whilst in the case of the latter, one would expect to find a highly destabilising mutation. However, in reality the vast majority of mutations have virtually no effect at all. The dataset is inherently biased because the scientific community is looking for the most *interesting results* and so the model learns that most mutations lead to big changes. This causes quite a few problems in training, because the model's standard deviation grows considerably per additional epoch of training, as it attempts to explain some of those incredibly destabilising and stabilising mutations in the database. This is a potential explanation as to why training for more than 5 epochs often leads to a significant deterioration in model performance.

**The model collapses to 0 when it is unsure about the size:** To avoid large errors, if the model is unsure about sign, the model can end up collapsing values down to the mean. This ensures the model minimises loss because predicting 0 when the true value is 5 is much better than predicting -5 when the true value is 5. This factor directly conflicts with the skew towards extremes described above, and this can results in the order from most stabilising to destabilising getting distorted. This conflict is probably one of the largest hindrances to the spearman and pearson values. 