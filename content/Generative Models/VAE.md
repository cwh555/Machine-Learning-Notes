---
title: Variational Autoencoder
properties:
  - hide
  - image
---
## Information
- *Title*: Auto-Encoding Variational Bayes  
- _authors_: Kingma, D. P., & Welling, M.  
- _conference_: ICLR 2014
- *task*: image generation

## Overview
### Background
- Many directed probabilistic models with continuous latent variables have intractable posterior distributions.
- The variational Bayesian (VB) approach involves the optimization of an approximation to the intractable posterior. However, the common mean-field approach requires analytical solutions of expectations w.r.t. the approximate posterior, which are also intractable in the general case.

### Breakthrough
Propose the following method that enabling efficient, differentiable approximate posterior inference and learning for such models.

- reparameterization trick
- Stochastic Gradient Variational Bayes (SGVB) estimator
- auto-encoding VB (AEVB algorithm) algorithm

## Method
### Background
#### Setup
We have a dataset $x^{(1)}, \dots, x^{(N)}$ and a generative model $p_\theta(x, z)$ with latent variable $z$:
- $p_\theta(z)$: prior over latent variables, often standard Gaussian.
- $p_\theta(x|z)$: likelihood (decoder), the generative model from $z \to x$.

The goal is to maximize the marginal likelihood of the observed data:
$$
\log p_\theta(x^{(1)}, \dots, x^{(N)}) = \sum_{i=1}^N \log p_\theta(x^{(i)})
$$
However, the true posterior $p_\theta(z|x)$ is generally intractable. Therefore, we introduce an encoder $q_\phi(z|x)$ to approximate the true posterior $p_\theta(z|x)$.

So the VAE consists of
- Encoder $q_\phi(z|x)$: maps data $x$ to a distribution over latent codes $z$.
- Decoder $p_\theta(x|z)$: maps latent codes $z$ back to a distribution over data $x$.
#### Variational Bound
Following [[ELBO|Evident Lower Bound]], we want to differentiate and optimize the lower bound
$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \| p_\theta(z))
$$
w.r.t to $\phi$ and $\theta$.

Naive Monte Carlo gradient for $\phi$ has high variance:
$$
\begin{aligned}
\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] &= \mathbb{E}_{q_\phi(z|x)}[f(z) \nabla_\phi \log q_\phi(z|x)] \\\\
&\approx \frac{1}{L} \sum_{l=1}^{L} f(z^{(l)}) \, \nabla_\phi \log q_\phi(z^{(l)}), \quad \text{where } z^{(l)} \sim q_\phi(z \mid x^{(i)})
\end{aligned}
$$
The main problem is that computing the gradient w.r.t $\phi$ is hard because $z \sim q_\phi(z|x)$ depends on $\phi$.
### Reparametrization Trick
for a chosen approximate posterior $q_\phi(z|x)$, we can reparameterize the random variable $\tilde{z} \sim q_\phi(z|x)$ using a differentiable transformation $g_\phi(\epsilon, x)$ of an (auxiliary) noise variable
$$
\tilde{z} \sim q_\phi(z|x)\qquad \text{with}\quad \epsilon\sim p(\mathbf{\epsilon})
$$
We can now form Monte Carlo estimates of expectations of some function $f(z)$ w.r.t. $q_\phi(z|x)$ as follows:
$$
\mathbb{E}_{q_\phi(z|x)}[f(z)] \approx \frac{1}{L} \sum_{l=1}^L f(g_\phi(\epsilon^{(l)}, x)), \quad \epsilon^{(l)} \sim p(\epsilon)
$$

### SGVB estimator
Applying the reparametrization technique to the variational lower bound
#### Generic Form
$$
\hat{\mathcal{L}}_A(\theta, \phi; x^{(i)}) = \frac{1}{L} \sum_{l=1}^{L} \Big[ \log p_\theta(x^{(i)}, z^{(i,l)}) - \log q_\phi(z^{(i,l)}|x^{(i)}) \Big], \quad z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})
$$
where $\epsilon^{(l)} \sim p(\epsilon)$.
#### KL Form
Recall the KL form equation of EBLO. We have the following KL form, which has lower variance than the generic estimator.
$$
\hat{\mathcal{L}}_B(\theta, \phi; x^{(i)}) = -\mathrm{KL}(q_\phi(z|x^{(i)}) \| p_\theta(z)) + \frac{1}{L} \sum_{l=1}^{L} \log p_\theta(x^{(i)}|z^{(i,l)})
$$
---
Given multiple datapoints from a dataset $X$ with $N$ datapoints, we can construct an estimator of the marginal likelihood lower bound of the full dataset, based on minibatches:
$$
\mathcal{L}(\theta,\, \phi;\, X)\ \approx\ \mathcal{\tilde{L}}_M(\theta,\, \phi;\, X^M)\ =\ \frac{N}{M} \sum_{i=1}^M\, \mathcal{\tilde{L}}(\theta,\, \phi;\, x^{(i)} )
$$
where the minibatch $X^M = \{x^{(i)}\}^M_{i=1}$ is a randomly drawn sample of $M$ datapoints from the full dataset $X$ with $N$ datapoint.

### AEVB Algorithm
![[截圖 2025-12-27 晚上9.05.39.png]]

### Example: VAE 
#### Setup
1. Prior over latent variables:
$$
p_\theta(z) = \mathcal{N}(z; 0, I)
$$
2. Generative model / decoder:
$$
p_\theta(x|z) = \begin{cases} \text{Multivariate Gaussian} & \text{for real-valued data} \\\\ \text{Bernoulli} & \text{for binary data} \end{cases}
$$
	- The parameters of $p_\theta(x|z)$ (mean for Gaussian, probability for Bernoulli) are outputs of a neural network (MLP) taking $z$ as input.
3. Recognition model / encoder:
$$
q_\phi(z|x) = \mathcal{N}(z; \mu(x), \mathrm{diag}(\sigma^2(x)))
$$
- Multivariate Gaussian with diagonal covariance
- $\mu(x)$ and $\sigma(x)$ are outputs of a neural network taking $x$ as input

#### Sampling
we sample from the posterior $z^{(i,l)} \sim q_\phi(z|x^{(i)})$ using
$$
z^{(i,l)} = g_\phi(x^{(i)},\,\epsilon^{(l)}) = \mu^{(i)} + \sigma^{(i)} \odot \epsilon^{(l)}\qquad \text{where}\quad \epsilon^{(l)} \sim \mathcal{N}(0, I)
$$

#### SGVB estimator
Because both $p_\theta(z)$ and $q_\phi(z|x)$ are Gaussian with diagonal covariance:
- The KL divergence can be computed analytically, no sampling needed.
- Only the reconstruction term $\log p_\theta(x|z)$ needs Monte Carlo sampling.
$$
\mathcal{L}(\theta, \phi; x^{(i)}) \approx \underbrace{\frac{1}{2} \sum_{j=1}^J \Big[1 + \log(\sigma_j^2) - (\mu_j)^2 - (\sigma_j)^2 \Big]}_{\text{KL term (analytic)}} + \underbrace{\frac{1}{L} \sum_{l=1}^L \log p_\theta(x^{(i)}|z^{(i,l)})}_{\text{reconstruction term (sampled)}}
$$
- $J$ = dimensionality of latent space $z$
- $z^{(i,l)} = \mu^{(i)} + \sigma^{(i)} \odot \epsilon^{(l)}$

## Discussion
### Future Works
The SGVB estimator and the AEVB algorithm can be applied to any model with continuous latent variables, including:
- Hierarchical generative architectures
- Time-series models / Dynamic Bayesian networks
- Applying SGVB to global parameters
- Supervised models with latent variables

## Reference
```apa
Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_.
```