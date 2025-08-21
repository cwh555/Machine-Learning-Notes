---
title: Maximum Likelihood Estimation
date: 2025-08-21
order: "10"
---
## Introduction
Consider a set of $m$ examples $\mathbb{X} = \{x^{(1)}, \dots, x^{(m)}\}$ drawn independently from the true but unknown data generating distribution $p_{data}(x)$.

Let $p_{model}(x; \theta)$ be a parametric family of probability distributions over the same space indexed by $\theta$. In other words, $p_{model}(x; \theta)$ maps any configuration $x$ to a real number estimating the true probability $p_{data}(x)$.

The maximum likelihood estimator for $\theta$ is defined as
$$
\begin{aligned}
\theta_{ML}\ &= \arg\max_{\theta}\,p_{model}(\mathbb{X};\ \theta)\\
&=\ \arg\max_{\theta}\,\prod_{i = 1}^m\,p_{model}(x^{(i)}; \theta)
\end{aligned}
$$
Let $\hat{p}_{data}$ denote the empirical distribution.
$$
\theta_{ML}\ =\ \arg\max_\theta\ \mathbb{E}_{x\, \sim\,\hat{p}_{data} }\log p_{model}\,(x;\,\theta)
$$
We may view it as minimizing the dissimilarity between the empirical distribution and the model distribution.
$$
D_{KL}(\hat{p}_{model}\|p_{model})\ =\ \mathbb{E}_{x\, \sim\, \hat{p}_{data}}[\,\log \hat{p}_{model}(x)\, -\, \log p_{model}(x) \,]
$$
## Properties
- Under the following conditions, the maximum likelihood estimator has the property of *consistency*.
	- The true distribution $p_{data}$ must lie within the model family $p_{model}(\cdot\,;\,\theta)$. Otherwise, no estimator can recover $p_{data}$.
	- The true distribution $p_{data}$ must correspond to exactly one value of $\theta$. Otherwise, maximum likelihood can recover the correct $p_{data}$, but will not be able to determine which value of $\theta$ was used by the data generating processing.
- For $m$ large, the ***Cramer-Rao lower bound*** shows that no consistent estimator has a lower mean squared error than the MLE.
