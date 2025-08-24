---
title: Early Stopping
date: 2025-08-23
---
## Introduction
At the ith step, we start at weights $w_{i-1}$ and take a step of size $\eta$ to $w_i\ =\ w_{i-1}\ -\ \eta \frac{g_{i-1}}{\|g_{i-1}\|}$ .
In fact, we indirectly searched entire hypothesis set $\mathcal{H}\ =\ \{w:\ \|w\,-\,w_{i-1}\|\ \leq\ \eta\}$ .

Stopping early can constrain the learning to the smaller hypothesis set.

## Discussion
In the case of a simple linear model with a quadratic error function and simple gradient descent, *early stopping is equivalent to $L^2$ regularization*.

Suppose that the learning algorithm takes $\tau$ optimization steps and with learning rate $\epsilon$.
The cost function $J$ with a quadratic approximation in the neighborhood of the empirically optimal value of the weights $w^*$:
$$
\hat{J}(\theta)\ =\ J(w^*)\ +\ \frac{1}{2}(w\,-\,w^*)^\top\,H\,(w\,-\,w^*).
$$

Apply the gradient descent on $J$,
$$
w^{(t)}\ -\ w^*\ =\ (I\, -\, \epsilon H)(w^{(t - 1)}\,-\,w^*) 
$$
Let $H\ =\ Q\Lambda Q^T$, where $\Lambda$ is a diagonal matrix and $Q$ is an orthonormal basis of eigenvectors.
$$
\implies Q^\top(w^{(t)}\,-\,w^*)\ =\ (I\,-\,\epsilon \Lambda)Q^\top(w^{(t - 1)}\,-\,w^*)
$$
Assuming that $w^{(0)} = 0$ and that $\epsilon$ is chosen to be small enough to guarantee $|1 - \epsilon \lambda_i| < 1$.
$$
Q^\top w^{(\tau)}\ =\ [I - (I - \epsilon\Lambda)^{\tau}]Q^\top w^*
$$
Compare to the result of $L^2$ regularization
$$
Q^\top\tilde{w}\ =\ [I\, -\, (\Lambda + \alpha I)^{-1}\alpha]Q^\top w^*
$$
If the hyperparameters $\epsilon, \alpha, \tau$ are chosen such that
$$
(I\,-\,\epsilon \Lambda)^{\tau}\ =\ (\Lambda\,+\,\alpha I)^{-1}\alpha
$$
then $L^2$ regularization and early stopping can be seen to be equivalent.
If all $\lambda_i$ are small (that is, $\epsilon \lambda_i\ll 1$ and $\lambda_i / \alpha \ll 1$),
$$
\begin{aligned}
\tau\ \approx\ \frac{1}{\epsilon \alpha}\\\\
\alpha\ \approx\ \frac{1}{\tau \epsilon}
\end{aligned}
$$