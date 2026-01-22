---
title: "TENT: Fully Test-Time Adaption By Entropy Minimization"
properties:
  - hide
tags:
  - idea
image: 0003.jpeg
---
## Information
- *title*: TENT: Fully Test-Time Adaption By Entropy Minimization
- *authors*: Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T.
- *conference*: arXiv (2020)
- *task*: domain adaption

## Overview
### Background
1. Transfer Learning / Fine-Tuning
    - Method: Perform supervised fine-tuning using target labels.
    - Limitation: Cannot be applied when target labels are unavailable.
2. Domain Adaptation
    - Method: Train on source and target data using a cross-domain loss $L(x_s, x_t)$
    - Limitation: Cannot be applied when target data are unavailable.
3. Test-Time Training (TTT)
    - Method: Adjust the model at test-time using an unsupervised loss $L(x_t​)$.
    - Limitation: Requires designing a specific loss; not easily generalizable.

### Breakthrough
efficiency, unsupervised test time adaption, simple

### Abstract
- At test time, adjust the model parameters to **minimize entropy**, making the model more confident in its predictions.
- Only the affine parameters are updated (for faster inference), e.g., the $\beta$ and $\gamma$ of batch normalization.

## Methods
### Objective
The test-time objective of Tent is to minimize the model’s prediction entropy on the test data:
$$
L(x_t) = H(\hat{y}) = -\sum_c p(\hat{y}_c) \log p(\hat{y}_c)
$$
where $p(\hat{y}_c)$ is the predicted probability for class $c$.

> [!lime]
> To avoid trivial solutions from single-point predictions, Tent optimizes the shared parameters across the entire batch, ensuring stable adaptation.
### Optimization Parameter
If updating the entire model parameters:
- Computation is expensive
- Optimization can be unstable → may diverge at test time

Therefore, only the affine transformations are updated:
$$
\bar{x} = \frac{x - \mu}{\sigma}, \quad x' = \gamma \bar{x} + \beta
$$
- $\mu, \sigma$: batch-wise mean and standard deviation
- $\gamma, \beta$: affine parameters to be optimized (scale & shift)

*Procedure*:
1. Use the model’s existing normalization layers (e.g., BatchNorm)
2. Update normalization statistics $\mu, \sigma$
3. Update affine parameters $\gamma, \beta$ to reduce prediction entropy

### Algorithm
*Initialization*
- Collect all affine parameters of the normalization layers:
$$
\{\gamma_{l,k}, \beta_{l,k}\} \quad \forall l \text{ layer}, k \text{ channel}
$$
- Fix all other model parameters $\theta \setminus \{\gamma, \beta\}$
- Discard source data normalization statistics and compute them from the test batch

*Iteration*
- Forward pass:
    - Compute batch mean $\mu_{l,k}$​ and standard deviation $\sigma_{l,k}$​ for each layer
    - Normalize inputs: $\bar{x} = (x - \mu)/\sigma$
- Backward pass:
    - Compute gradient of prediction entropy w.r.t. affine parameters: $\nabla_{\gamma,\beta} H(\hat{y})$
    - Update $\gamma, \beta$

*Termination*
- Online adaptation: Continue updating as long as test data is available
- Offline adaptation: First update affine parameters, then perform a complete forward pass for inference

## Experiments
### Baseline
| Method                                     | Description                                                                                                                               | Features / Limitations                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Source                                     | Apply the trained model directly on test data without any adaptation                                                                      | Model is not updated; performance affected by dataset shift                           |
| Adversarial Domain Adaptation (RG)         | Use a domain classifier with gradient reversal to make feature distribution invariant to source/target (Ganin & Lempitsky, 2015)          | Requires both source and target data for adversarial training                         |
| Self-Supervised Domain Adaptation (UDA-SS) | Jointly train self-supervised rotation and position tasks on source and target to learn shared representation (Sun et al., 2019a)         | Requires source and target data; designing proxy tasks may affect supervised task     |
| Test-Time Training (TTT)                   | Jointly train supervised and self-supervised tasks on source; during testing, update only the self-supervised task (Sun et al., 2019b)    | Requires joint loss design during training; updates many parameters at test-time      |
| Test-Time Normalization (BN)               | Update batch normalization statistics (mean / std) on test data (Schneider et al., 2020; Nado et al., 2020)                               | Does not require source/target labels; only updates statistics, not affine parameters |
| Pseudo-Labeling (PL)                       | Set a confidence threshold, assign predictions above the threshold as pseudo-labels, and optimize the model with these labels (Lee, 2013) | Requires manual threshold; may propagate incorrect labels                             |
Note: TENT does not determine the label until final inference. Therefore, it has less risk than PL.

## Reference
```apa
Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2020). Tent: Fully test-time adaptation by entropy minimization. _arXiv preprint arXiv:2006.10726_.
```
