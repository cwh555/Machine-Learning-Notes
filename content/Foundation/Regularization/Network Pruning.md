---
title: Network Pruning
date: 2025-10-15
---
#### Hessian-based network pruning
By *Taylor series*, for a perturbation $\Delta w$,
$$
\mathscr{E}_{av}(w\,+\,\Delta w)\ =\ \mathscr{E}_{av}(w)\ +\ g^\top(w)\Delta w\ +\ \frac{1}{2}\Delta w^\top H \Delta w\ +\ \mathcal{O}(\|\Delta w\|^3)
$$
where $g(w)$ is the gradient at $w$.

We assume that parameters are deleted only after the training process have converged. Therefore, $g^\top(w)\ =\ 0$.
Hence, 
$$
\Delta \mathscr{E}_{av}\ \approx\ \frac{1}{2}\Delta w^\top H w
$$
There are two strategies solving this problem.
- **optimal brain damage (OBD) procedure**: suppose that hessian matrix is a diagonal matrix for simplicity, which is contained in the following **OBS**.
- **optimal brain surgeon (OBS) procedure**

The elimination of a synaptic weight $w_i(n)$ is equivalent to the condition
$$
\mathbf{1}_i^\top\,\Delta w\ +\ w_i\ =\ 0
$$
where $\mathbf{1}_i$ is the unit vector whose elements are all zero, except for the $i$-th element.

> [!abstract] Goal of OBS
> Minimize the quadratic form $\frac{1}{2}\Delta w^\top Hw$ with respect to the incremental change in the weight vector, $\Delta w$, subject to the constraint that $\mathbf{1}_i^\top \Delta w\ +\ w_i$ is zero, and then minimize the result with respect to the index $i$.

To solve this problem, we use the *Lagrangian*
$$
S\ =\ \frac{1}{2}\,\Delta w^\top H w\ -\ \lambda(\mathbf{1}_i^\top \Delta w\,+\,w)
$$
Taking derivatives with respect to $\Delta w$ and substituting the result into the constraint equation, we obtain
$$
\Delta w\ =\ -\frac{w_i}{[H^{-1}]_{i, i}}H^{-1}\mathbf{1}_i
$$
and the corresponding optimum value of the Lagrangian $S$ for element $w_i$ is
$$
S_i\ =\ \frac{w_i^2}{2[H^{-1}]_{i, i}}
$$
where $[H^{-1}]_{i, i}$ is the $i\, i$-th element in $H^{-1}$.

When the network is large, computing the Hessian matrix is expensive. Hence, we introduce a method for computing it.

Recall that the error signal is as follows.
$$
\mathscr{E}_{av}(w)\ =\ \frac{1}{2N}\,\sum_{i = 1}^N\,(d(n)\ -\ o(n))^2
$$
where $o(n)$ is the actual output of the $n$-th data point, so $o(n)\ =\ F(w, x)$.

Therefore, the Hessian matrix is
![[截圖 2025-10-15 中午12.36.32.png]]

Since it is supposed that the network is converged, the error signal in the second term is small enough that we may ignore it.

To simplify the notation, define a $W$-by-$1$ vector
![[截圖 2025-10-15 中午12.37.53.png]]

We may rewrite the Hessian matrix as
![[截圖 2025-10-15 中午12.38.38.png]]

This recursion formula is in the form of *matrix inversion lemma*, also known as *Woodbury's equality*.

> [!abstract] Matrix Inversion Lemma
> Let $A$ and $B$ denote two positive definite matrices related by
> $$
> A\ =\ B^{-1}\ +\ CDC^\top
> $$
> where $C$ and $D$ are two other matrices.
> The the inverse of $A$ is defined by
> $$
> A^{-1}\ =\ B\ -\ BC(D\ +\ C^\top BC)^{-1}C^\top B
> $$

Applying the matrix inversion lemma, we obtain the inverse of the Hessian matrix.
![[截圖 2025-10-15 中午12.41.51.png]]

 To initialize the algorithm we need to make $H^{-1}(0)$ very large, since it is being constantly reduced during the process. A good choice is
$$
H\ =\ \delta^{-1}I 
$$
where $\delta$ is a small positive number.

The summary of the Optimal Brain Surgeon Algorithm is as follows.
![[截圖 2025-10-15 中午12.46.41.png]]
