---
title: Hyper-Parameter Optimization
date: 2025-08-24
---

## Bayesian Optimization
treating the generalization performance of models and hyper-parameter settings $\lambda$ as black-box function $f(\lambda)$.

consists of:
- a surrogate probabilistic model: approximates the objective function based on observed data.
- an acquisition model: determine the optimal candidate hyper-parameters given the probabilistic model.

At each iteration,
1. probabilistic model is trained to fitted all historical observations.
2. the acquisition function determines each candidate configuration, and chooses one for the next evaluation.
### Sequential Model-Based Optimization
<div style="text-align:center;">
<img src="https://i.postimg.cc/XvzzkSpq/2025-07-30-9-36-52.png" alt="Example Image" style="height: 300px;">
</div>

### Surrogate Models
- **Gaussian process**
A Gaussian process $\mathcal{G}_\lambda$  is characterized by a mean $m(\lambda)$ and a covariance function $k(\lambda, \lambda')$. The prior of the mean function $m(\lambda)$ is often assumed to be constant.

Now, suppose we have previously observed the function values of some configurations, then for any given configuration $\lambda$, the predicted mean and variance of $f(\lambda)$ can given in close-forms, i.e.
$$
\mu(\lambda)\ =\ k_\lambda^T\,K^{-1}y,\qquad\sigma^2(\lambda)\ =\ k(\lambda, \lambda)\ +\ k_\lambda^TK^{-1}k_\lambda.
$$
where $k_\lambda$ is the vector of covariances between $\lambda$ and all previously examined configurations, $K$ is the covariance matrix of all these evaluated configurations, and $y$ is the observed values of these examined configurations.

The covariance function $k$ has many choice such as *Matern 5/2 kernel*.
- **Neural Network**
- **Random Forest**
- **Tree Parzen Estimator (TPE)**
estimates the density functions $p(\lambda|\,y < \alpha)$ and $p(\lambda|\,y \geq \alpha)$, where $\alpha$ is the percentile that distinguishes the "good" configurations from the "bad" ones.

### Acquisition Functions
- **Expected improvement**
measures the expected gain compared to the historical minimum.
$$
EI(\lambda,\,M)\ =\ \mathbb{E}_{y\,\sim\,p_M(y|\,\lambda)}[\,\max\{ f_{min}\,-\,y,\,0\}\ =\ \int_{\mathbb{R}}\{ f_{min} - y,\,0 \}p_M(y|\,\lambda)\,dy
$$
where $p_M(y|\,\lambda)$ is the probability distribution of the loss $y$ w.r.t the candidate configuration $\lambda$ given the surrogate model $M$, and $f_{min}$ is the minimum of the observed losses.

In Gaussian process,
$$
EI(\lambda,\,M)\ =\ (f_{min}\,-\,\mu(\lambda))\Phi(\frac{f_{min}\,-\,\mu(\lambda)}{\sigma})\ +\ \sigma\phi(\frac{f_{min}\,-\,\mu(\lambda)}{\sigma})
$$
where $\phi$ and $\Phi$ represents the standard normal density and distribution function, and $f_{min}$ is the best observation.

- **Entropy search**
selects candidate configurations based on the predicted information gain w.r.t the optimum.
$$
ES(\lambda,\,M)\ =\ \mathbb{E}_{y\,\sim\,p_M(y|\,\lambda)}[\, KL(p_{min}(\cdot|\,\mathcal{H}\,|\,\cup\,\{ (\lambda,\,y) \});\ \mu(\lambda)) \,]
$$
## Bandit Variants
### Successive Halving Algorithm (SH)
uniformly allocates the budget to a set of arms for a predefined number of iterations before discarding the worst half, then repeats process until only one arm left.

<div style="text-align:center;">
<img src="https://i.postimg.cc/XY7B6XgZ/2025-07-30-10-02-42.png" alt="Example Image" style="height: 300px;">
</div>

### Hyperband
it runs SH as the inner loop, where the outer loop iterates over different numbers of samples under the same budget.

<div style="text-align:center;">
<img src="https://i.postimg.cc/Ghz9Kd1b/2025-07-30-10-03-24.png" alt="Example Image" style="height: 330px;">
</div>

