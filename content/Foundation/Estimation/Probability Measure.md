---
title: Probability Measure
---
Probability measures are primarily used to quantify and compare the model distribution and the real distribution in generative models.

Let $\mathcal{X}$ be a compact metric space, say the space of images $[0, 1]^d$.
Let $\Sigma$ denote the set of all Borel subsets of $\mathcal{X}$.
Let Prob($\mathcal{X}$) denote the space of probability measures defined on $\mathcal{X}$.
We define the elementary distances and divergences between two distributions $\mathbb{P}_r,\,\mathbb{P}_g \in$ Prob($\mathcal{X}$)

## Types
### Total Variation (TV) distance
![[截圖 2025-12-27 下午1.48.40.png]]
### Kullback-Leibler (KL) divergence
![[截圖 2025-12-27 下午1.48.53.png]]
where both $\mathbb{P}_r$ and $\mathbb{P}_g$ are assumed to admit densities with respect to a same measure $\mu$ defined on $\mathcal{X}$.

Property: asymmetric and unbounded (can grow to infinity).

### Jensen-Shannon (JS) divergence
![[截圖 2025-12-27 下午1.49.09.png]]
where $\mathbb{P}_m = (\mathbb{P}_r + \mathbb{P}_g) / 2$.

### Earth-Mover (EM) distance (Wasserstein-1)
![[截圖 2025-12-27 下午1.49.26.png]]
where $\Pi(P_r,\,P_g)\ =\ \{ \gamma \in \text{Prob}(X \times X)\,|\,\gamma(A\times X)\ =\ P_r(A),\,\gamma(X\times B)\ =\ P_g(B) \}$


