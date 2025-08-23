---
title: Principal Components Analysis (PCA)
date: 2025-08-22
---
## Basic Form
This method attempts to get rid of redundancy or less informative dimensions to help generalization.
<span class = 'lime'>There is no use doing PCA after whitening, since every direction will be on equal footing after whitening.</span>

#### Idea
Consider the standard Euclidean coordinate system in $d$ dimensions. Given the vector $x$ and some other coordinate system $v_1,\,v_2,\,\dots,\,v_n$ , we can define the transformed feature vector whose components are the coordinates $z_1,\,\dots,\,z_d$ in this new coordinate system. 
Suppose that only the first $k < d$ of these transformed coordinates are the informative ones, so we throw the rest of them to arrive at <span style="color:pink; font-weight:bold;">dimension-reduced</span> feature vector

$$
z\ =\ \begin{bmatrix}
x^Tv_1\\
\vdots\\
x^Tv_k
\end{bmatrix}
\ = \ \Phi(x)\ .
$$
So, $\hat{x}\ =\ \sum_{i = 1}^{k}z_i\,v_i$  and we want $\sum_{n = 1}^{N}\,\|x_n\,-\,\hat{x}_n\|^2$ to be small.

#### Algorithm

> [!gray] Theorem (Eckart and Young, 1936)
> For any $k,\,v_1,\,\dots,\,v_k$ (the top $k$ right singular vectors of the data matrix $X$) are a set of top-k principal component directions and the optimal reconstruction error is $\sum_{i = k + 1}^{d} \gamma_i^2$ .
> 
> The components of the dimensionally reduced feature vector are $z_i\ =\ x^T v_i$ and the reconstructed vector is $\hat{x}\ =\ \sum_{i = 1}^k\,z_i\,v_i$ , which in matrix form is
> $$\hat{X}\ =\ XV_kV_k^T$$
> where $V_k\ =\ [v_1,\,\dots,\,v_k]$ is the matrix of top-k right singular vectors of $X$ .

<div style="text-align:center;">
<img src="https://i.imgur.com/SCWWDjH.jpeg" alt="Example Image" style="height: 200px;">
</div>

#### When PCA Works?
1. Do you expect the data have linear structure, for example does the data lie in a linear subspace of lower dimension ?
2. Do the bottom principal components contain primarily small random fluctuations that correspond to noise and should be thrown away ?
3. Does the target function depend primarily on the top principal components, or are the small fluctuations in the bottom principal components key in determining the value ?

## Nonlinear Form
<div style="text-align:center;">
<img src="https://i.imgur.com/XpgxrS0.jpeg" alt="Example Image" style="height: 300px;">
</div>

