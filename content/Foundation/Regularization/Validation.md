---
title: Validation
date: 2025-10-13
---
## Training Set v.s. Model Complexity
Two modes of behavior in fitting data are identified depending on the size of the training set
1. *Nonasymptotic mode*: $N < 30W$
   In the use of cross-validation to stop training, the optimal value of parameter $r$ that determines the split of the training data between estimation and validation subsets is defined by
$$
r_{opt}\ =\ 1\ -\ \frac{\sqrt{2W\,-\,1}\,-\,1}{2(W\,-\,1)}
$$
2. *Asymptotic mode*: $N > 30W$
   In this mode, exhaustive learning is satisfactory.

