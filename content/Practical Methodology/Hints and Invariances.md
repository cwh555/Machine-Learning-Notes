---
title: Hints and Invariances
date: 2025-08-22
---
## Examples
- $\textbf{Symmetry or Anti-symmetry hints}$
- $\textbf{Rotational Invariance hints}$
	$f(x)$ depends only on $\|x\|$ .
- $\textbf{General Invariance hints}$ <br>
	For some transformation $\mathcal{T}$ , $f(x)\ =\ f(\mathcal{T}x)$ . Invariance to scale, shift and rotation of an image are common in vision applications.
- $\textbf{Monotonicity hints}$ <br>
	$f(x\,+\,\Delta x)\ \geq\ f(x)$ if $\Delta x\,\geq\,0$ .
- $\textbf{Convexity hint}$ <br>
	$f(\eta x\,+\,(1\,-\,\eta)\,x')\ \leq\ \eta f(x)\, +\, (1\,-\,\eta)f(x')$ for $0\,\leq\,\eta\,\leq\,1$ .
- $\textbf{Perturbation hint}$ <br>
	$f$ is closer to a known function $g$, so $f\,=\,g\,+\,\delta f$ .

## Virtual Examples
### Invariance Hint
A general invariance partitions the input space into disjoint regions
$$
\mathcal{X}\ =\ \cup_\alpha\mathcal{X}_{\alpha}
$$
For $x,\,x'\ in\ \mathcal{X}_{\alpha}$ , $f(x)\ =\ f(x')$ .

We may define the hint error
$$
E_{hint}(h)\ =\ \frac{1}{N} \sum_{n = 1}^{N}\,(h(x_n)\,-\,h(x_n'))^2\ .
$$
### Monotonicity Hint
$$
E_{hint}(h)\ =\ \frac{1}{N}\,\sum_{n = 1}^N\,(h(x_n)\,-\,h(x_n'))^2\,\left[\!\left[ h(x_n')\,<\,h(x_n) \right]\!\right]\ .
$$

## Hint Versus Regularization
$$
E_{aug}(h)\ =\ E_{in}(h)\ +\ \frac{\lambda_1}{N}\Omega(h)\ +\ \lambda_2\,E_{hint}(h)\ .
$$