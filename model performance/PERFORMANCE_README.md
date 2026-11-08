# Model performance

[← DDG-Predict main README](../README.md)

## Validation

MMseqs2 clusters the 153 wild-type sequences at 30% identity into 132 clusters, and whole clusters are distributed across 10 splits, so no testing-set protein has a close relative on its own training side. After 10 train-test splits and 3 initial seeds for each, the mean absolute error, the RMSE, the bias and the pearson and spearman correlation was calculated. These were then compared to two dummy models. One takes a global mean of the training set, and the second takes a mean for each possible substitution (e.g. mutating an A to a Q). 

These results were then averages across all 30 runs, and final values have been shown below:

| Predictor | Mean absolute error | Root mean squared error | Bias | Pearson | Spearman |
| --------- | ------------------- | ----------------------- | ---- | ------- | -------- |
| **DDG-Predict** | **1.24** | **1.84** | −0.24 | **0.43** | **0.48** |
| Mean ΔΔG of each substitution type (e.g. A -> Q) | 1.41 | 2.02 | −0.02 | 0.25 | 0.31 |
| Global Mean ΔΔG | 1.44 | 2.01 | +0.04 | – | – |

All errors are in kcal mol⁻¹. Across the runs, the trained model outcompetes the reference model every time. However, each run showed significant variation with single runs having a Pearson correlation ranging from 0.24 to 0.59. The large change in values is due to the testing set being dominated by one wildtype protein, which can heavily skew results one way. Considering uncertainty the model performs with a:

- Pearson correlation of 0.43 ± 0.10 
- Spearman rank of 0.48 ± 0.09

**Sign accuracy:**  Destabilising mutations are called correctly 87.8% of the time, stabilising ones 46.5%, giving an average accuracy of **67.2%**. The model is conservative about stabilising mutations, missing more than half, so the ones it ranks at the very top are more likely to be genuinely stabilising. 

**Epochs and learning rate:** (full records in **[lr_fine_tuning/](../lr_fine_tuning/)**). The optimum training parameters were found to be 5 ΔΔG epochs at a learning rate of 1e-4. This was fine-tuned by completing a detailed sweep at different learning rates and epoch numbers. The pKa head was kept virtually frozen after it was trained on PKAD data as shifting this later during training significantly worsened the pearson and spearman values.

## An Example Case

The results below show some predicted values during evaluation, and how they compare to the true value. The rows below are taken from lysozyme C measurements from one run (initial state 1 of training split 4). 

Table 1: Out-of-fold predictions for Lysozyme C, hen egg white, taken from that cross-validation (negative means stabilising; positions use UniProt numbering)
| Mutation |  pH |   Predicted ΔΔG / kcal mol⁻¹  |  True ΔΔG / kcal mol⁻¹  |
| -------- | --- | ----------------------------- | ----------------------- |
|  A49V    | 6.4 |              -0.58            |           -1.20         |
|  A49L    | 6.4 |              -1.41            |           -1.80         |
|  D119P   | 5.5 |               0.28            |            2.40         |
|  G120P   | 5.5 |              -1.47            |            0.10         |
|  N64D    | 3.5 |               0.86            |           -0.46         |
|  N64D    | 4.5 |               0.79            |            0.69         |
|  N64D    | 5.5 |               0.77            |            0.84         |
|  E53Q    | 3.5 |              -0.52            |            0.53         |
|  E53A    | 3.5 |              -1.09            |           -1.46         |
|  E53H    | 3.5 |              -0.04            |            3.43         |
|  E53A    | 4.0 |              -0.98            |           -1.14         |

Table 1 is typical of how the model behaves, including its failures. The stabilising mutations A49V, A49L and E53A come out with the correct sign and roughly the right size, and *E53A* at two pH values is close on both. The model does show a clear pH dependence and generally follows the right trend, though *N64D* shows a systematic offset and gets the sign wrong at pH 3.5.

Some reponses are however innacurate. For instance *G120P* has a predicted value of −1.47 against a measured +0.10. At the same time, *E53H* has a predicted value of -0.04 compared to a measured value of +3.43. These results show opposite direction responses. In some cases, the model badly underpredicts, whilst in other cases the model padbly overpredicts. 

This is due to two competiting effects. One of the limitations discussed in the results readme `` ADD LINK `` is that the dataset is inherently biased towards more extreme values. This means that the model get skewed (especially after training for more than 5 epochs) and overpredicts numbers in a bid to reduce the large errors caused by those heavily stabilising and destabilising mutations.

On the other hand, the model badly underpredicts in some cases. This is a consequence of mean collapse. When the model is not sure about the **sign of the answer**, it is always safer to collapse to the mean, because predicting a large positive number when the true answer is a large negative number leads to a huge error. The model decides it is simply better to predict a value near 0 to avoid the risk of a really large error. 