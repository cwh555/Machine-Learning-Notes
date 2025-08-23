---
title: Data Preprocessing
date: 2025-08-22
---
## Input Centering
Centering deals with bias.
Consider the transformation
$$
Z\ =\ X\ -\ 1\bar{x}^\top
$$
After centering, $Z^\top \textbf{1} =\ 0$ .

## Input Normalization
Normalization deals with scale.
Consider the transformation
$$
z_n\ =\ D\,x_n
$$
where $D$ is a diagonal matrix with entries $D_{ii}\,=\,1/\sigma_i$ .  
After normalization, $\sigma_i^2(z)\ = 1$ .

## Input Whitening
Whitening deals with correlations since strongly correlated input variables can have unexpected outcome.
Let $\Sigma\ =\ \frac{1}{N}X^\top X$ be the covariance matrix. Note that the data had been centered.
Consider the transformation
$$
z_n\ =\ \Sigma^{-\frac{1}{2}}\,x_n\ .
$$
After whitening, $\frac{1}{N} Z^\top Z\ =\ I$ .

## Data Cleaning
### Use a Simpler Model
Use a simpler first to identify the noisy data point.
- A common choice is linear mode.
- It is generally better to have several instances of the simpler hypothesis.

### Validation Leverage Score
Consider the data set $\mathcal{D}$ and use the data set $\mathcal{D}_n$ which contains all the data in $\mathcal{D}$ except $(x_n,\,y_n)$ . We denoted by $g_n^-$ the final hypothesis that gets from $\mathcal{D}_n$ . 
We denote the <span style="color:pink; font-weight:bold;">leverage score</span> of data point $(x_n,\,y_n)$ by
$$
\ell_m\ =\ E_{out}(g)\ -\ E_{out}(g^-)\ .
$$
If $\ell_n$ is large and positive, $(x_n,\,y_n)$ is detrimental and should be discarded.
