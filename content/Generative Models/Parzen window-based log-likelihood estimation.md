---
title: Parzen window-based log-likelihood estimation
properties:
  - hide
image: 0011.jpeg
---
## Overview
- the early method to estimate the performance of implicit generative models.

> [!warning] Obsolescence
> This evaluation method does not used in modern models!

## Method
1. Generate some samples $\{\tilde{x}_i\}_{i = 1}^N$.
2. Apply Gaussian Parzen window density estimation to these samples.
$$
\hat{p}_g(x)\ =\ \frac{1}{N}\sum_{i = 1}^N\,\mathcal{N}(x;\,\tilde{x}_i\,\sigma^2\,I)
$$
3. Tune the hyperparameter $\sigma$ using cross-validation.
4. Compute the log-likelihood of each test sample $x_j$ under the estimated distribution. The resulting values are used as the evaluation scores.
$$
\text{log-likelihood}\ =\ \frac{1}{M}\,\sum_{j = 1}^M\,\log\,\hat{p}_g(x_j)
$$
## Discussion
#### Advantages
- provides a way to compute evaluation scores for models with implicit generative distributions.

#### Disadvantages:
- The evaluation is highly dependent on the kernel bandwidth $\sigma$, which must be carefully tuned.
- The estimated log-likelihood becomes unreliable in high-dimensional data due to the curse of dimensionality.
- The resulting scores do not necessarily correlate with perceptual or semantic quality of the generated samples.