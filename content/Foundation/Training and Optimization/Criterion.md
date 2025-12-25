---
title: Criterion
date: 2025-10-21
properties:
  - hide
tags:
  - test
---
## Classification
### Multi-class SVM score
The score of the correct class should be higher than all other scores.

Given an example $(x_i, y_i)$
Let $s = f(x_i, W)$ be scores.
Then the SVM loss has the form $L_i = \sum_{j \neq y_i}\max(0, s_j - s_{y_i} + 1)$

If all scores were random, the expected loss is $C - 1$, where $C$ is the number of categories.