---
title: Test-Time Training with Self-Supervision for Generalization under Distribution Shifts
properties:
  - hide
  - image
tags:
  - idea
  - experiments
---
## Information
- *title*: Test-Time Training with Self-Supervision for Generalization under Distribution Shifts
- *authors*: Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, Moritz Hardt
- *conference*: ICML 2020
- *task*: domain adpation

## Overview
### Abstract
- modern meta-learning
During testing, first use the test data to perform self-supervised learning to update the model before making predictions.

This is compared to models trained with supervised learning and self-supervised + supervised learning, but with a fixed model at test time 
> [!lime]
> (some studies suggest that using self-supervised learning during training results in more robustness).

## Methods
### Model
The model has three components (one shared block, two branches)

*shared feature extractor*
$$
\theta_e = (\theta_1, \dots, \theta_\kappa)
$$
*main task branch*
$$
\theta_m = (\theta_{\kappa+1}, \dots, \theta_K)
$$
*self-supervised task branch*
$$
\theta_s = (\theta'_{\kappa+1}, \dots, \theta'_K)
$$
### Training
training objective: (include two branches, train two task together)
$$
\min_{\theta_e, \theta_m, \theta_s} \frac{1}{n} \sum_{i=1}^n \big[ l_m(x_i, y_i; \theta_m, \theta_e) + l_s(x_i; \theta_s, \theta_e) \big]
$$

### Testing
#### TTT for Single Data
Given a test image $x$,
$$
\min_{\theta_e} l_s(x; \theta_s, \theta_e)
$$
only the shared feature extractor is updated.

The paper’s augmentation uses rotations of 0, 90, 180, and 270 degrees. The self-supervised task is rotation prediction (predicting the angle by which the image has been rotated).

After updating the parameters for each sample, they are discarded; when a new sample arrives, the update is performed starting from the old parameters (only trained during training). 

This is because the update is optimized for a single sample (assuming the test data are i.i.d.) and prevents the model from gradually biasing toward the most recent test samples.

#### Online
The difference is that the result of the previous update is retained for further updating.

For the $t$-th sample:
$$
\theta_e(x_t) \leftarrow \arg\min_{\theta_e} l_s(x_t; \theta_s, \theta_e) \quad\text{initialization:}\quad \theta_e(x_{t-1})
$$

## Theoretical Results
*Goal*: Prove that TTT can really decrease the main task loss

First, we consider a toy model with two linear layers

### Setup
1. Input and labels
$$
x \in \mathbb{R}^d, \quad y_1 \in \mathbb{R} \quad \text{(main task label)}, \quad y_2 \in \mathbb{R} \quad \text{(self-supervised label)}
$$

2. Two-layer linear network
$$
\hat{y} = v^\top A x, \quad A \in \mathbb{R}^{h \times d}, \ v \in \mathbb{R}^h
$$
	- Shared feature extractor: $A$
	- main branch: $v$
	- self-supervised branch: $w$
3. Main task loss
$$
l_m(x, y_1; A, v) = \frac{1}{2} \|y_1 - v^\top A x\|^2
$$
4. Self-supervised task loss
$$
l_s(x, y_2; A, w) = \frac{1}{2} \|y_2 - w^\top A x\|^2
$$
### TTT Update
Perform one gradient step on the self-supervised loss to update $A$:
$$
A' \leftarrow A - \eta \nabla_A l_s​
$$
Compute the gradient:
$$
\nabla_A l_s = -(y_2 - w^\top A x) w x^\top
$$
so the update formula is
$$
A_0 \leftarrow A - \eta (y_2 - w^\top A x) w x^\top
$$

If we let the step be
$$
\eta^* = \frac{y_1 - v^\top A x}{(y_2 - w^\top A x) v^\top w x^\top x} 
$$
then, we obtain the optimal solution for main task
$$
l_m(x, y_1; A_0, v) = 0
$$
However, this $\eta^*$ requires knowledge of $y_1$, which is not available for test data. Intuitively, we can use the TTT update to move $A$ toward the optimal solution of the main task. To achieve this, the following two conditions must be satisfied:
1. The error directions are aligned.
$$
\text{sign}(y_1 - v^\top A x) = \text{sign}(y_2 - w^\top A x) 
$$
2. The decision boundary directions are correlated
$$
v^\top w > 0
$$
we can prove that these two conditions are satisfied  $\iff \langle \nabla l_m(A), \nabla l_s(A) \rangle > 0$

### Theorem
Just a generalization of the discussion before.

Let $l_m(x, y; \theta)$ denote the main task loss on test instance $x, y$ with parameters $\theta$, and $l_s(x; \theta)$the self-supervised task loss that only depends on $x$. 
Assume that for all $x, y, l_m(x, y; \theta)$ is differentiable, convex and $\beta$-smooth in $\theta$, and both $\|\nabla l_m(x, y; \theta)\|, \|\nabla l_s(x, \theta)\| \le G$ for all $\theta$. With a fixed learning rate
$$
\eta = \frac{\epsilon}{\beta G^2},
$$
for every $x, y$ such that
$$
\langle \nabla l_m(x, y; \theta), \nabla l_s(x; \theta) \rangle > \epsilon
$$
we have
$$
l_m(x, y; \theta) > l_m(x, y; \theta(x))
$$
where $\theta(x) = \theta - \eta \nabla l_s(x; \theta)$, i.e. Test-Time Training with one step of gradient descent.

## Results
### Image
On static images, the method performs better than both jointly trained models (trained with self-supervised + supervised learning but using a fixed model at test time) and the pure baseline, and it is able to resist corruptions.

### Objection Detection on Video Frame
The VID-Robust dataset is used (it has been shown that models trained on ImageNet fail on this dataset). It is based on the ImageNet Video Detection Dataset.

Because a model trained on ImageNet (1000 classes) needs to be applied to VID-Robust (30 classes), the authors used the **max-conversion function in Shankar et al. (2019)**:
1. Each VID-Robust class corresponds to multiple ImageNet classes.
2. The maximum value is taken over the logits of these corresponding ImageNet classes.
$$
y_{\text{VID class}} = \max_{c \in \text{corresponding ImageNet classes}} \text{logit}_c
$$
This can let 1000-class logits → 30-class logits

## Reference
```apa
Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., & Hardt, M. (2020, November). Test-time training with self-supervision for generalization under distribution shifts. In _International conference on machine learning_ (pp. 9229-9248). PMLR.
```