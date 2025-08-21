---
title: Maximum A Posteriori Estimation
date: 2025-08-21
order: "10"
---

According to **Bayesian statistics**,
$$
P(\theta|\,x)\ =\ \frac{P(x|\,\theta)P(\theta)}{P(x)}
$$

where $P(\theta| x)$ is the posteriori probability, $P(\theta)$ is the priori probability, $P(x|\,\theta)$ is the likelihood function and $P(x)$ is marginal probability.

The MAP estimate chooses the point of maximal posterior probability
$$
\theta_{MAP}\ =\ \arg \max_\theta\, p(\theta|\,x)\ =\ \arg\max_\theta\,\log p(\,x\,|\,\theta)\ +\ \log\,p(\theta).
$$
