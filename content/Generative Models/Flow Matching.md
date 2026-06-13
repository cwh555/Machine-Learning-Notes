---
title: Flow Matching
properties:
  - hide
  - image
---

## Information
- *title*: Flow Matching for generative model
- *author*: Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M.
- *conference*: arXiv 2022

## Overview
### Background
- Diffusion models use simple diffusion processes, which limits their expressiveness and performance.
- Continuous Normalizing Flows (CNFs) can represent arbitrary probability paths; however:
    - Under maximum likelihood estimation (MLE), they require expensive ODE simulations.
    - With other simulation-free training methods, the resulting gradients are either intractable or biased.
### Abstract
- Generative modeling built on Continuous Normalizing Flows (CNFs)  
  is efficient, simulation-free, and not restricted to diffusion processes.    
- It allows the use of non-diffusion probability paths, such as Optimal Transport (OT) displacement interpolation, which directly defines the shortest path from noise → data.
- Flow Matching (FM) learns the vector field that directly generates a target probability path.  
  Compared to score matching, FM training is more stable and achieves better performance.

## Methods
### Continuous Normalizing Flows
- *Data space*: $\mathbb{R}^d$, with data points $x = (x_1, \dots, x_d) \in \mathbb{R}^d$
- *Probability density path*:  $p : [0,1] \times \mathbb{R}^d \to \mathbb{R}_{>0}$​: a time-evolving probability density such that for each $t$, $\int p_t(x)\, dx = 1 \quad \text{(total probability equals 1)}$.
- *Time-dependent vector field* : $v : [0,1] \times \mathbb{R}^d \to \mathbb{R}^dv$
  for each time $t$ and each location $x$, it specifies a velocity vector.

A vector field $v$ can be used to construct a time-dependent diffeomorphic map, called a *flow* $\phi : [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$,
$$
\begin{aligned}
\frac{d}{dt}\phi_t(x) &= v_t(\phi_t(x)), \\\\
\phi_0(x) &= x
\end{aligned}
$$
Here, $\phi_t(x)$ denotes the position at time $t$ of a particle that starts from the initial position $x$.

**Continuous Normalizing Flow (CNF):**  
A CNF uses a neural network $v_t(x; \theta)$ to approximate the vector field that defines $\phi_t$.

A CNF can be used to transport distributions, pushing a simple prior density $p_0$​ (e.g., noise) into a complex distribution $p_1$​:
$$
p_t = [\phi_t] * p_0
$$
where the *pushforward operator* ∗ is defined as
$$
[\phi_t] * p_0(x) = p_0(\phi_t^{-1}(x)) \left| \det \frac{\partial \phi_t^{-1}}{\partial x}(x) \right|
$$
- $\phi_t^{-1}(x)$ finds which initial point flows to $x$.

If the flow $\phi_t$ induced by a vector field $v_t$​ satisfies the above pushforward relation, then $v_t$ is said to generate the probability path $p_t$.

### Flow Matching
#### Training Objective
- $x_1 \sim q(x_1)$: an unknown data distribution. We can only sample from it; the density cannot be evaluated explicitly.
- $p_t$: a probability path, where $p_0 = p$ is a simple distribution and $p_1 \approx q$.
- *Goal*: design a probability path and its corresponding vector field that transports noise $p_0$ to the data distribution $p_1$.

Assume that we are given:
1. a target probability path $p_t(x)$,
2. a vector field $u_t(x)$ that generates this path.

The Flow Matching (FM) objective is:
$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], x \sim p_t} \|v_t(x; \theta) - u_t(x)\|^2
$$

A neural network $v_t$ is trained to approximate the true vector field $u_t$.

When the loss is zero, the CNF induced by $v_t$ generates exactly the target probability path $p_t$.

*Challenges*
- It is unclear which probability path $p_t$ is optimal, and in most cases the corresponding $u_t$ does not have a closed form.
- Many different probability paths can lead to the same final distribution.

Therefore, we model the problem using **conditional probability paths**, defining a simple path separately for each data point $x_1$.

#### Conditional Probability Path
Assume we are given a data point $x_1$.

We define $p_t(x \mid x_1)$ as a conditional probability path, i.e., a continuous stochastic process that goes from noise $\to$ the neighborhood of $x_1$.

This path must satisfy the following boundary conditions:
- Initial distribution is a noise prior:
$$
p_0(x \mid x_1) = p(x), \qquad \text{e.g., } \mathcal{N}(0, I).
$$
- Terminal distribution is a small Gaussian centered at $x_1$:
$$
p_1(x \mid x_1) = \mathcal{N}(x \mid x_1, \sigma^2 I).
$$

We then mix all conditional paths by first sampling $x_1$ and then sampling from its corresponding path:
$$
p_t(x) = \int p_t(x \mid x_1) q(x_1) dx_1. \tag{6}
$$

At $t = 1$:
$$
p_1(x) = \int p_1(x \mid x_1) q(x_1) dx_1 \approx q(x).
$$

The mixture recovers the target data distribution, since each component distribution is concentrated around its corresponding data point.

#### Marginal Vector Field
We define the marginal vector field as
$$
u_t(x) = \int u_t(x \mid x_1) \underbrace{\frac{p_t(x \mid x_1) q(x_1)}{p_t(x)}}_{\text{posterior weight}} dx_1. \tag{8}
$$

Note that
$$
\frac{p_t(x \mid x_1) q(x_1)}{p_t(x)} = q(x_1 \mid x_t = x),
$$
which is exactly the Bayesian posterior.

Therefore, the marginal vector field can be written as
$$
u_t(x) = \mathbb{E}_{x_1 \sim q(\cdot \mid x_t = x)} [u_t(x \mid x_1)].
$$

*Observation:   
The marginal vector field $u_t$ generates the marginal probability path $p_t$.

> [!gray] Theorem 1
> Given vector fields $u_t(x \mid x_1)$ that generate conditional probability paths $p_t(x \mid x_1)$, for any distribution $q(x_1)$, the marginal vector field $u_t$ defined in equation (8) generates the marginal probability path $p_t$ defined in equation (6). That is, $u_t$ and $p_t$ satisfy the continuity equation:
> $$
> \partial_t p_t + \nabla \cdot (p_t u_t) = 0.
> $$

In other words, we can compose simple, tractable per-sample vector fields to obtain a correct but intractable global vector field.

#### Conditional Flow Matching
Previously, the training objective involved the marginal quantities $u_t(x)$ and $p_t(x)$, both of which require integration and are therefore intractable. The paper proposes a simpler formulation.

> [!gray] Theorem 2
> Assuming that $p_t(x) > 0$ for all $x \in \mathbb{R}^d$ and all $t \in [0,1]$, then, up to a constant independent of $\theta$, the Conditional Flow Matching loss ($\mathcal{L}_{\mathrm{CFM}}$) is equal to the Flow Matching loss ($\mathcal{L}_{\mathrm{FM}}$).
> Consequently,
> $$
> \nabla_\theta \mathcal{L}_{\mathrm{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\mathrm{CFM}}(\theta).
> $$

In other words, the two losses have identical gradients, even though one is defined using intractable marginal quantities and the other is fully tractable.

#### Construction of Conditional Probability Path
Consider Gaussian conditional probability paths
$$
p_t(x \mid x_1) = \mathcal{N}\big(x \mid \mu_t(x_1), \sigma_t(x_1)^2 I\big)
$$
- $\mu_t(x_1): [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$: the mean of the conditional Gaussian.
- $\sigma_t(x_1): [0,1] \times \mathbb{R}^d \to \mathbb{R}_{>0}$: the scalar standard deviation of the conditional Gaussian.

*Boundary conditions*    
At $t=0$:
$$
\mu_0(x_1) = 0, \qquad \sigma_0(x_1) = 1.
$$
All conditional paths collapse to the same Gaussian at $t=0$.

At $t=1$:
$$
\mu_1(x_1) = x_1, \qquad \sigma_1(x_1) = \sigma_{\min}.
$$

Hence, $p_1(x \mid x_1) \approx \delta(x - x_1)$. Each conditional path converges to its own data point $x_1$ at $t=1$.

There are infinitely many vector fields that can generate the same probability path. The authors choose the simplest one, namely the canonical transformation for Gaussian distributions:
$$
\psi_t(x) = \sigma_t(x_1)x + \mu_t(x_1).
$$

If $x \sim \mathcal{N}(0,I)$, then $\psi_t(x) \sim \mathcal{N}\big(\mu_t(x_1), \sigma_t(x_1)^2 I\big)$.

Thus, $\psi_t$ pushes the initial noise distribution $p(x)$ to the conditional distribution $p_t(x \mid x_1)$.

The vector field corresponding to this flow satisfies:
$$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x) \mid x_1).
$$

Substituting this into the CFM loss gives:
$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \left\| v_t(\psi_t(x_0); \theta) - \frac{d}{dt}\psi_t(x_0) \right\|^2,
$$
where $x_0 \sim p(x) = \mathcal{N}(0, I)$, $x_1 \sim q(x_1)$ is drawn from the data distribution, and $v_t(\cdot; \theta)$ is the neural network being trained with regression target $\frac{d}{dt}\psi_t(x_0)$.

> [!gray] Theorem 3
> Let $p_t(x \mid x_1)$ be a Gaussian probability path as in equation (10), and let $\psi_t$ be its corresponding flow map as in equation (11). Then, the unique vector field that defines $\psi_t$ is:
> $$
> u_t(x \mid x_1) = \frac{\sigma'_t(x_1)}{\sigma_t(x_1)}\big(x - \mu_t(x_1)\big) + \mu'_t(x_1). \tag{15}
> $$
- The first term $\frac{\sigma'_t}{\sigma_t}(x - \mu_t)$: radial velocity induced by the change in scale.
- The second term $\mu'_t$: velocity induced by translation of the mean.

## Examples
The functions $\mu_t(x_1)$ and $\sigma_t(x_1)$ can be chosen arbitrarily.

### Diffusion conditional VFs
#### Variance Exploding (VE)
In diffusion models, the forward process goes from data to noise:
$$
x_t = x_1 + \sigma_t \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I).
$$

Hence, $p(x_t \mid x_1) = \mathcal{N}(x_1, \sigma_t^2 I)$.

Flow Matching, however, is formulated from the noise $\to$ data perspective, so we rewrite this as the reversed Variance Exploding (VE) path:
$$
p_t(x \mid x_1) = \mathcal{N}(x \mid x_1, \sigma_{1-t}^2 I),
$$
where $\sigma_t$ is an increasing function, $\sigma_0 > 0$, and $\sigma_1 \gg 1$.

Comparing with the general Gaussian conditional path, we have:
- mean: $\mu_t(x_1) = x_1$
- std: $\sigma_t(x_1) = \sigma_{1-t}$

Plugging these into Theorem 3 yields:
$$
u_t(x \mid x_1) = - \frac{\sigma'_{1-t}}{\sigma_{1-t}} (x - x_1).
$$
#### Variance Preserving (VP)
For the VP diffusion, the conditional distribution is:
$$
p_t(x \mid x_1) = \mathcal{N} \Big( x \mid \alpha_{1-t}x_1, (1-\alpha_{1-t}^2)I \Big),
$$
where $\alpha_t = e^{-\frac{1}{2} T(t)}$ and $T(t) = \int_0^t \beta(s) ds$.

Comparing with the general form:
- mean: $\mu_t(x_1) = \alpha_{1-t}x_1$
- std: $\sigma_t(x_1) = \sqrt{1-\alpha_{1-t}^2}$

Substituting into Theorem 3 gives:
$$
\begin{aligned}
u_t(x \mid x_1) &= \frac{\dot{\alpha}_{1-t}}{1-\alpha_{1-t}^2} (\alpha_{1-t}x - x_1)\\\\
&= - \frac{T'(1-t)}{2} \left[ \frac{ e^{-T(1-t)}x - e^{-\frac{1}{2}T(1-t)}x_1 }{ 1-e^{-T(1-t)} } \right].
\end{aligned}
$$

This $u_t(x \mid x_1)$ coincides exactly with the deterministic probability flow of Song et al. (2020b).

Originally, diffusion models are formulated as SDEs trained via score matching. Here, they are reformulated as conditional vector fields, which can instead be learned via Flow Matching regression.

### Optimal Transport (OT) conditional VFs
We can also directly design a linear and minimal probability path:
$$
\mu_t(x_1) = t x_1, \quad \sigma_t(x_1) = 1 - (1 - \sigma_{\min}) t.
$$
- $\mu_t(x_1)$: the mean moves linearly from $0$ to $x_1$. 
- $\sigma_t(x_1)$: the standard deviation shrinks linearly from $1$ to $\sigma_{\min}$.
- With $t \in [0,1]$, this defines a path from a standard Gaussian to a small Gaussian concentrated at $x_1$.

Applying Theorem 3:
$$
u_t(x \mid x_1) = \frac{x_1 - (1-\sigma_{\min}) x}{1 - (1-\sigma_{\min}) t}.
$$

The corresponding conditional flow is:
$$
\psi_t(x) = (1 - (1-\sigma_{\min}) t)x + t x_1.
$$

The CFM loss becomes:
$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \Big\| v_t(\psi_t(x_0); \theta) - [x_1 - (1-\sigma_{\min}) x_0] \Big\|^2.
$$

Here, $\psi_t(x)$ moves each particle from the initial Gaussian to the target Gaussian along the shortest path, i.e., the Optimal Transport displacement interpolation under the Wasserstein-2 distance.

## Reference
```apa
Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. _arXiv preprint arXiv:2210.02747_.
```