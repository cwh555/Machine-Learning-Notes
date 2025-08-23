---
title: Maximum A Posteriori Estimation
date: 2025-08-21
order: "10"
---
## Introduction
According to **Bayesian statistics**,
$$
P(\theta|\,x)\ =\ \frac{P(x|\,\theta)P(\theta)}{P(x)}
$$

where $P(\theta| x)$ is the posteriori probability, $P(\theta)$ is the priori probability, $P(x|\,\theta)$ is the likelihood function and $P(x)$ is marginal probability.

The MAP estimate chooses the point of maximal posterior probability
$$
\theta_{MAP}\ =\ \arg \max_\theta\, p(\theta|\,x)\ =\ \arg\max_\theta\,\log p(\,x\,|\,\theta)\ +\ \log\,p(\theta).
$$
## Discussion
MAP builds upon the foundation of MLE by incorporating a prior on the model parameter $\theta$, effectively adding a constraint that acts like regularization.

For example, consider a linear model
$$
y_i = x_i^\top \theta + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$
The corresponding likelihood is
$$
P(\text{data} \mid \theta) = \prod_i \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\Big(-\frac{(y_i - x_i^\top \theta)^2}{2\sigma^2}\Big)
$$
Suppose we place a Gaussian prior on $\theta$ : 
$$
\theta \sim \mathcal{N}(0, \tau^2 I)\quad \Rightarrow \quad
P(\theta) = \frac{1}{(2 \pi \tau^2)^{d/2}} \exp\Big(-\frac{\|\theta\|^2}{2 \tau^2}\Big)
$$
The MAP estimate is then
$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\text{data}\mid \theta) P(\theta)
$$
Taking the logarithm:
$$
\log P(\text{data}\mid \theta) + \log P(\theta) = - \sum_i \frac{(y_i - x_i^\top \theta)^2}{2\sigma^2} - \frac{\|\theta\|^2}{2\tau^2} + \text{const}
$$
Maximizing the log posterior is equivalent to minimizing the negative log posterior:
$$
\hat{\theta}_{\text{MAP}} = \arg\min_\theta \sum_i \frac{(y_i - x_i^\top \theta)^2}{2\sigma^2} + \frac{\|\theta\|^2}{2\tau^2}
$$
Comparing with **ridge regression** :
$$
\arg\min_\theta \sum_i (y_i - x_i^\top \theta)^2 + \lambda \|\theta\|^2, \quad \lambda = \frac{\sigma^2}{\tau^2}
$$
We see that applying MAP with a Gaussian prior on $\theta$ is equivalent to adding an $\ell_2$ regularization term—the prior effectively acts as a regularizer.