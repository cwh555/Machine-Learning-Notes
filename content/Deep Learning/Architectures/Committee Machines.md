---
title: Committee Machines
date: 2025-08-24
tags:
  - EM
---

## Static Structures
the responses of several predictors are combined by means of a mechanism that does not involve the input signal.
### Ensemble averaging
Experts share a common input and whose individual outputs are somehow combined to produce an overall output.

<div style="text-align:center;">
<img src="https://i.postimg.cc/KY042CH2/2025-07-13-20-34-1.jpg" alt="Example Image" style="height: 220px;">
</div>

Let $F_I$ denotes the average of the input-output functions of the export networks.
Let $\mathscr{D}'$ denote the remnant space. Then,
$$
E_{\mathscr{D}}[F(x)^2]\ \geq\ E_{\mathscr{D'}}[F_I(x)^2]
$$
$$
V_{\mathscr{D'}}(F_I(x))\ \leq\ V_{\mathscr{D}}(F(x))
$$
These points to a training strategy for reducing the overall error produced by a committee machine due to varying initial conditions.
### Boosting
experts are trained on data sets with entirely different distributions.   
three fundamental different ways:
1. **Boosting by filtering**: filtering the training examples according to performance of the experts training before.
2. **Boosting by subsampling**: The training sample is of fixed sized. The examples are resampled according to a given probability distribution during training.    Misclassified samples will have higher sampling probability.
3. **Boosting by reweighting**: The training sample is of fixed sized; however, each training sample has a "weight". When the sample is misclassified, its weight will increase; otherwise, its weight will decrease in the next round.

#### AdaBoost
<div style="text-align:center;">
<img src="https://i.postimg.cc/k55RBWzK/2025-07-13-20-34-2.jpg" alt="Example Image" style="height: 350px;">
</div>

**Theoretical property of error**   
Suppose that the weak learning model, when called by AdaBoost, generates hypothesis with errors $\epsilon_1,\epsilon_2,\dots,\epsilon_T$, where the error $\epsilon_n$ on iteration $n$ of the AdaBoost algorithm is defined by
$$
\epsilon_n\ =\ \sum_{i:\, \mathscr{F_n(x_i)\,\neq\, d_i}} \mathcal{D}_n(i)
$$
Assume that $\epsilon_n\leq\frac{1}{2}$, and let $\gamma_n\ =\ 1/2\,-\,\epsilon_n$. Then the following upper bound holds on the error of the final hypothesis:
$$
\frac{1}{N}|\{i:\,\mathscr{F}_{fin}(x_i)\neq d_i |\}\leq\ \prod_{n = 1}^T\,\sqrt{1 - 4\gamma_n^2}\leq\ \exp{\left(-2\sum_{n = 1}^T \gamma_n^2\right)}
$$

## Dynamic Structures
the input signal is directly involved in actuating the mechanism that integrates the outputs of the individual experts into an overall output.

### Mixture of Experts Model
<div style="text-align:center;">
<img src="https://i.postimg.cc/zvgSf43y/2025-07-13-20-34-3.jpg" alt="Example Image" style="height: 350px;">
</div>
Each expert network consists of a linear filter.
$$
y_k\ =\ w_k^\top x_k,\qquad k = 1, 2, \dots,K
$$
The gating network consists of a single layer of $K$ neurons, with each neurons assigned to a specific expert. The neurons of the gating network are nonlinear, with their activation functions defined by
$$
g_k = \frac{\exp(u_k)}{\sum_{j = 1}^K\exp(u_j)}\qquad k = 1, 2, \dots, K
$$
where $u_k$ is the inner product of the input vector $x_k$ and synaptic weight vector $a_k$.   
The gating network are ensured to satisfies the requirements:
$$
0\leq g_k\leq 1\quad \text{for all } k
$$
and
$$
\sum_{k = 1}^K g_k\ =\ 1
$$
Assuming the prediction error for each expert follows a normal distribution:
$$
f_D(d \mid x, k, \theta) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} (d - y_k)^2 \right)
$$

The overall probability distribution is called an *associative Gaussian mixture model*
$$
f_D(d \mid x, \theta) = \sum_{k=1}^K g_k \cdot \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} (d - y_k)^2 \right)
$$

### Hierarchical Mixture of Experts Model
<div style="text-align:center;">
<img src="https://i.postimg.cc/SKfPS23B/2025-07-13-20-34-4.jpg" alt="Example Image" style="height: 350px;">
</div>
Initialization of the HME
1. Apply **classification and regression tree (CART)** algorithm to the training data.
2. Set the synaptic weight vectors of the experts in the HME model equal to the least-squares estimates of the parameter vectors at the corresponding terminal nodes of the binary tree resulting from the application of CART.
3. For the gating networks:
	1. set the synaptic weight vectors to point in directions that are orthogonal to the corresponding splits in the binary tree obtained from CART.
	2. set the lengths of the synaptic weight vectors equal to small random vectors.

Training goal: [[Maximum Likelihood Estimation]]
maximum the following log-likelihood function
$$
L(\theta)\ =\ \log[\,f_D(d\, |\, x, \theta)]
$$
where
$$
f_D(d\,|\,x, \theta)\ =\ \frac{1}{2\pi}\,\sum_{k = 1}^2g_k\,\sum_{j = 1}^2 g_{j|k}\,\exp\left(-\frac{1}{2}(d - y_{jk})^2\right)
$$
Learning strategies
1. Stochastic gradient approach
2. **Exception-maximization approach**   
	- *E-step*, which use observed data set of an incomplete data problem and the current value of the parameter vector to manufacture data so as to postulate an augmented or so-called complete data set.    
	  In HME, the "incomplete" part refers to which expert is responsible for generating each observed data point.
	- *M-step*, which consists of deriving a new estimate of the parameter vector by maximizing the log-likelihood function of the complete data manufactured in the E-step.

Accordingly, the M-step of the algorithm reduces to the following three optimizations problems for an HME with two levels of the hierarchy:
$$
w_{jk}(n+1)\ =\ \arg\min_{w_{jk}}\,\sum_{i = 1}^{N} h_{jk}^{(i)}\,\left(d_i - y_{jk}^{(i)}\right)^2
$$
$$
a_j(n + 1)\ =\ \arg\max_{a_j}\,\sum_{i = 1}^N \sum_{k = 1}^2\,h_{jk}^{(i)} \log g_k^{(i)}
$$
$$
a_{jk}(n + 1)\ =\ \arg\max_{a_{jk}}\,\sum_{i = 1}^{N}\,\sum_{l = 1}^2 h_l^{(i)}\,\sum_{m = 1}^2\,h_{m | l}^{(i)} \log g_{m|l}^{(i)}
$$
