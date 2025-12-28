---
title: Evidence Lower Bound
---
## Setup
Consider a probabilistic model with observed data $x$ and latent variables $z$, parameterized by $\theta$:
$$
p_\theta(x, z) = p_\theta(x|z)\, p_\theta(z)
$$

- *Goal*: maximize the marginal likelihood $\log p_\theta(x) = \log \int p_\theta(x, z)\, dz$
- *Problem*: the true posterior $p_\theta(z|x) = p_\theta(x, z)/p_\theta(x)$ is generally intractable.

## ELBO
We introduce a variational distribution $q_\phi(z|x)$ to approximate the true posterior. Then we can decompose the log marginal likelihood as:
$$
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + \mathrm{KL}(q_\phi(z|x) \| p_\theta(z|x))
$$

where
$$
\mathcal{L}(\theta, \phi; x) \equiv \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x, z) - \log q_\phi(z|x)]
$$

Since KL divergence is non-negative, maximizing ELBO simultaneously increases the marginal likelihood; meanwhile, minimizes divergence between approximate posterior $q_\phi$​ and true posterior $p_\theta(z|x)$.

## Equivalent Form
Split the joint:
$$
p_\theta(x, z) = p_\theta(x|z) p_\theta(z)
$$
Then ELBO can be rewritten as:
$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \| p_\theta(z))
$$
- Expected log-likelihood (reconstruction term): measures how well latent $z$ explains the data.
- KL regularization term: encourages the approximate posterior to be close to the prior
