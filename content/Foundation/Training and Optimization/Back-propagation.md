---
title: Multi-layer Perceptrons
date: 2025-10-14
---
## Back-propagation Algorithm
1. *Forward computation*
$$
v_j^{(j)}(n)\ =\ \sum_{i = 0}^{m_0}\,w_{ji}^{(l)}(n)\,y_i^{(l - 1)}(n)
$$
- $v_j^{(l)}(n)$: the induced local field for neuron $j$ in layer $l$
- $y^{(l - 1)}_i(n)$: the output signal of neuron $i$ in the previous layer $l - 1$ at iteration $n$
- $w_{ji}^{(l)}(n)$: the synaptic weight of neuron $j$ in layer $l$
The output signal of neuron $j$ in layer $l$ is 
$$
y_j^{(l)}\ =\ \varphi_j(v_j(n))
$$
If neuron $j$ is in the form first hidden layer, set
$$
y_j^{(0)}(n)\ =\ x_j(n)
$$
If neuron $j$ is in the output layer, let
$$
y_j^{(L)}\ =\ o_j(n)
$$
Compute the error signal
$$
e_j(n)\ =\ d_j(n)\ -\ o_j(n)
$$
2. *Backward Computation*
$$
\delta_j^{(l)}(n)\ =\ \begin{cases}
e_j^{(L)}(n)\,\varphi_j'(v_j^{(L)}(n))\qquad &\text{for neuron } j \text{ in output layer } L\\\\
\varphi'(v_j^{(L)}(n))\,\sum_{k}\delta_k^{(l + 1)}(n)\,w_{kj}^{(l + 1)}(n) &\text{for neuron } j \text{ in hidden layer }l
\end{cases}
$$
where $\varphi'$ denotes the differentiation with respect to the argument.

Adjusting the synaptic weights.
$$
w_{ji}^{(l)}(n + 1)\ =\ w_{ji}^{(l)}(n)\ +\ \alpha[w_{ji}^{(l)}(n - 1)]\ +\ \eta\,\delta_j^{(l)}(n)\,y_i^{(l - 1)}(n)
$$
### Heuristics for Optimization
- Sequential versus batch update
- Maximizing information content
	- the use of an example that results in the largest training error
	- the use of an example that is radically different from all those pervious used
	- emphasizing scheme
- Activation function
	- antisymmetric $\varphi(-v)\ =\ -\varphi(v)$
	- $\varphi(v)\ =\ 1.7159\tanh(\frac{2}{3}v)$
	  in order to have the following properties
	  1. $\varphi(1) = 1$ and $\varphi(-1) = -1$
	  2. the slope at the origin is close to $1$ ($\varphi'(0) = 1.1424$)
	  3. the second derivative attains its maximum at $v = 1$
- Target values should be offset by some amount $\epsilon$ away from the limiting value of the activation function. The offset is to prevent the derivative of the activation function goes to infinity.
- Input preprocess
	- Normalize
	- The input variables should be **uncorrelated**
	- The decorrelated input variables should be scaled so that their **covariances** are approximately equal.
- Initializing the synaptic weights so that the standard deviation of the induced local field of a neuron lies in the transition area between the linear and saturated parts of its sigmoid activation function.
Consider the induced local field of neuron $j$ as
$$
v_j\ =\ \sum_{i = 1}^m\,w_{ji}\,y_i
$$
Assume that the inputs applied to each neuron in the network have zero mean and unit variance and the inputs are correlated.
$$
\mu_y\ =\ E[\,y_i\,]\ =\ 0\qquad \text{for all }i
$$
$$
\sigma_y^2\ =\ E[\, (y_i\ -\ \mu_i)^2 \,]\ =\ E[\, y_i^2 \,]\ =\ 1\qquad \text{for all }i
$$
$$
E[\,y_i\,y_k\,]\ =\ \begin{cases}
1,\qquad \text{for }k\ =\ i\\
0,\qquad \text{for }k\ \neq i
\end{cases}
$$
As for the synaptic weights, suppose they are drawn from a uniformly distributed set of numbers with zero mean
$$
\mu_w\ =\ E[\,w_{ji}\,]\ =\ 0\qquad \text{for all } (j,\,i) \text{ pairs}
$$
and variance
$$
\sigma^2_w\ =\ E[\,w_{ji}^2\,] \qquad \text{for all } (j,\,i) \text{ pairs}
$$
Then, we may express the mean and variance of the induced local field as
$$
\mu_v\ =\ 0
$$
$$
\sigma_v^2\ =\ m\,\sigma^2_w
$$

A good choice is to let $\sigma_v\ =\ 1$, as we can set $\sigma_w\ =\ m^{-1/2}$
- Learning from hints