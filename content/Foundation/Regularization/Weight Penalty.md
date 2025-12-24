---
title: Weight Penalty
date: 2025-10-15
---
## Approximate Smoother
This method is designed for a multilayer perceptron with a single hidden layer and a single neuron in the output layer.
$$
\mathscr{E}_c(w)\ =\ \sum_{i = 1}^M\,w_{oj}^2\,\|w_j\|^p
$$
where
- $w_{oj}$ are the weights in the output layer
- $w_j$ is the weight vector for the $j$ th neuron in the hidden layer
the power $p$ is defined by
$$
p\ =\ \begin{cases}
2k\ -\ 1\qquad &\text{for a global smoother}\\\\
2k\ &\text{for a local smoother}
\end{cases}
$$
where $k$ is the order of differentiation of $F(x, w)$ with respect to $x$.
