---
title: Conditional Variational Autoencoder
properties:
  - hide
tags:
  - experiments
image: 0005.jpeg
---
## Information
- _title_: Learning Structured Output Representation using Deep Conditional Generative Models  
- _authors_: Sohn, K., Lee, H., & Yan, X.
- _conference_: NeurIPS 2015
- *task*: structured prediction

## Overview
### Background
When we use supervised learning for structured output, the deterministic model can only learn
$$
f(x) \approx \mathbb{E}[y \mid x]
$$
This leads to
- Averaging over modes
- Blurry or unrealistic predictions
- No uncertainty modeling

However, the actually thing we want should be $p(y|\,x)$.
### Breakthrough
- Propose a deep conditional generative model with Gaussian latent variables.
	- The input $x$ determines the prior distribution of the latent variable $z$, and the output $y$ is generated conditioned on $z$.  
	- The training objective is to maximize $\log_\theta p(y \mid x)$
### Abstract
Training technique:
- Train the model using stochastic gradient variational Bayes (SGVB), combining the reparameterization trick with variational inference.
- At inference time, use stochastic feed-forward inference to efficiently generate diverse predictions.
New strategies:
- **Input noise injection**: add noise to the input during training to improve robustness.
- **Multi-scale prediction objective**: apply losses at multiple scales to help the model capture structural features at different levels of granularity.

## Method
### Model
A deep conditional generative model has three variables:
- *input variables*: $x$
- *output variables*: $y$
- *latent variables*: $z$

![[截圖 2025-12-28 下午2.31.58.png]]

In (a), this is a deterministic model mapping input $x$ to output $y$, which fails to learn structured outputs for high dimensions.

In (b), we introduce a latent variable $z$:
$$
z \sim p_\theta(z \mid x), \quad y \sim p_\theta(y \mid x, z)
$$
This gives the conditional distribution:
$$
p_\theta(y \mid x) = \int p_\theta(y \mid x, z)\, p_\theta(z \mid x)\, dz
$$
$z$ is a low-dimensional latent variable that captures the diversity and different modes of the output. Therefore, conditional generation becomes more controllable (although the same $x$, the different $z$ will leads to different $y$).

However, the distribution $p(z|\,x)$ is intractable. Thus, we use the concept of VAE, which introduce $q_\phi$.

In (c), we introduce a variational posterior $q_\phi(z \mid x, y)$ and use SGVB with the reparameterization trick as in VAE.

Figure (d) implements the CVAE in a practical CNN architecture. The key modification is that the prior network now receives both the input $x$ and the CNN’s initial prediction $\hat{y}$​, enabling a recurrent refinement of the output.

In conclusion, the empirical lower bound (ELBO) for the CVAE is written as:
$$
\mathcal{L}_{\text{CVAE}}(x, y; \theta, \phi) = - \mathrm{KL}\big(q_\phi(z|x, y) \parallel p_\theta(z|x)\big) + \frac{1}{L} \sum_{l=1}^{L} \log p_\theta(y|x, z^{(l)})
$$
where
$$
z^{(l)} = g_\phi(x, y, \epsilon^{(l)}), \quad \epsilon^{(l)} \sim \mathcal{N}(0, I)
$$

### Training Technique
#### Aligning Test time and Training Procedure
The problem of CVAE is that
- During training, the recognition network $q_\phi(z|x,y)$ sees the ground-truth $y$.
- During testing, the prior network $p_\theta(z|x)$ is used, which does not see $y$.
Consequently, the pipeline of training is different from testing.

The paper proposed a solution: <span style="color: pink; font-weight: bold;">Gaussian stochastic neural network (GSNN)</span>.

Setting $q_\phi(z|\,x, y)\ =\ p_\theta(z|\,x)$, we have the following training objective
$$
\mathcal{L}_{\text{GSNN}}(x,y;\theta,\phi) = \frac{1}{L} \sum_{l=1}^L \log p_\theta(y|x, z^{(l)}), \quad z^{(l)} = g_\theta(x,\epsilon^{(l)}), \epsilon^{(l)} \sim N(0,I)
$$
##### Hybrid
- CVAE advantage: Uses $y$ during training to learn the latent space, capturing multi-modal outputs.
- CVAE disadvantage: Training and testing pipelines are inconsistent; at test time, it may generate suboptimal $y$.
- GSNN advantage: Training and testing pipelines are consistent, leading to more stable predictions.
- GSNN disadvantage: The latent space learning is weaker, so multi-modal outputs may be insufficient.
The authors use the hybrid objective:
$$
\mathcal{L}_{\text{hybrid}} = \alpha \mathcal{L}_{\text{CVAE}} + (1-\alpha) \mathcal{L}_{\text{GSNN}}
$$

#### For Image Segmentation and Labeling​
The challenge is to predict fine-grained pixel-level labels for high-resolution images, where each pixel belongs to a semantic class.

##### Multi-scale Prediction Objective
For high-resolution images, pixel-level prediction is difficult to generate accurately.  
Training at a single scale tends to ignore either global structure or fine details, resulting in degraded reconstruction or semantic segmentation quality.

*Solution*: Produce predictions at multiple resolutions and sum the losses across all scales.

![[截圖 2025-12-28 下午3.03.49.png#center|300]]

##### Input Noise
The paper proposes a simple regularization technique for semantic segmentation: corrupt the input data x into $\tilde{x}$ according to noise process and optimize the network with the following objective: $\tilde{L}(\tilde{x}, y)$

> [!lime]
> For semantic image segmentation, we consider random block omission noise.

### Inference
#### Deterministic prediction
At test time, instead of sampling $z$, one can use a deterministic inference:
$$
y^* = \arg\max_y p_\theta(y \mid x, z^*), \quad z^* = \mathbb{E}[z \mid x]
$$

#### Monte Carlo estimation
To evaluate how well the model captures the full conditional distribution, sample $z$ from the prior $p_\theta(z|x)$ multiple times:
$$
p_\theta(y|x) \approx \frac{1}{S}\sum_{s=1}^S p_\theta(y|x,z^{(s)}), \quad z^{(s)} \sim p_\theta(z|x)
$$
#### Importance Sampling
Monte Carlo requires many samples for accurate estimation. We can use the recognition network $q_\phi(z|x,y)$ to sample $z$ instead, correcting for the sampling difference via importance weights:    
$$
p_\theta(y|x) \approx \frac{1}{S} \sum_{s=1}^S p_\theta(y|x,z^{(s)}) \frac{p_\theta(z^{(s)}|x)}{q_\phi(z^{(s)}|x,y)}, \quad z^{(s)} \sim q_\phi(z|x,y)
$$

## Discussion
### My Idea
- a application of reparametrization to other task.

## Reference
```apa
Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation using deep conditional generative models. _Advances in neural information processing systems_, _28_.
```