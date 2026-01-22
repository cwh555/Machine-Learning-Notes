---
title: Wasserstein Generative Adversarial Networks
properties:
  - hide
image: 0016.jpeg
---
## Information
- *title*: Wasserstein Generative Adversarial Networks
- *authors*: Martin Arjovsky, Soumith Chintala, Leon Bottou
- *conference*: ICML 2017
- *task*: image generation

## Overview
### Background
Let $P_r$ denote the real distribution and $P_\theta$ denote the model distribution.

The original problem of GANs is that when $P_r$​ and $P_\theta$ do not overlap, the JS distance remains $\log 2$ and gradients cannot be used for updates.

### Breakthrough
This work proposes a new distance function $\rho$, with the training objective of minimizing $\rho(P_\theta, P_r)$. The model, WGAN, can solve GAN’s mode collapse and mode dropping issues.

### Abstract
A new distance is proposed.

The GAN discriminator is replaced with a critic, which is updated several times before each generator update to provide an accurate measure.

This approach improves training stability, gives a meaningful loss, and prevents mode collapse. Using a Mixture of Gaussians to visualize mode collapse shows that WGAN can correctly approximate the distribution, first learning the low-dimensional structure (circle) and then focusing on local peaks.

## Methods
### Probability Measure
- [[Probability Measure|Probability Measure Comparisons]]

*Example*

Suppose that $Z \sim U[0, 1]$ is uniform distribution and $P_0 = (0, Z) \in \mathbb{R}^2$.
Let the generator be $g_\theta(z) = (\theta, z)$

Under this condition, we may find that
![[截圖 2025-12-27 下午1.53.20.png]]
This shows that only $W$ can let the parameter $\theta$ update under this situation.

### WGAN
> [!danger] Danger
> There are many mathematical derivations in the paper regarding the properties of the Wasserstein distance. However, these are mostly basic analytical results, so this note will not cover them.

According to the _Kantorovich-Rubinstein duality_, the Wasserstein-1 distance has a dual form.
![[截圖 2025-12-27 下午1.57.39.png]]

The above constraint requires the function to be 1-Lipschitz, but it can be relaxed to K-Lipschitz, resulting in $K \cdot W(\mathbb{P}_r, \mathbb{P}_\theta)$. Therefore, in practice, if there exists a family of parameterized functions $\{f_w\}_{w \in \mathcal{W}}$ that are all K-Lipschitz, it is equivalent to solving the following optimization problem:
![[截圖 2025-12-27 下午1.57.57.png]]

*Algorithm*
![[截圖 2025-12-27 下午1.58.21.png]]

## Discussion
### Retrospective
This Wasserstein distance has been applied to many different tasks.

## Reference
```apa
Arjovsky, M., Chintala, S., & Bottou, L. (2017, July). Wasserstein generative adversarial networks. In _International conference on machine learning_ (pp. 214-223). PMLR.
```
