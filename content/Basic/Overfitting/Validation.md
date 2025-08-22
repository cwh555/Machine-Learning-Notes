---
title: Validation
date: 2025-08-21
---
## Description
We partition the data set $\mathcal{D}$ into a training set $\mathcal{D}_{train}$ of size $(N\,-\,K)$ and a validation set $\mathcal{D}_{val}$ of size $K$ . We run the learning algorithm on $\mathcal{D}_{train}$ to obtain a final hypothesis $g^- \in \mathcal{H}$. We compute the validation error for $g^-$ 
$$
E_{val}(g^-)\ =\ \frac{1}{K} \sum_{x_n\,\in\,\mathcal{D}_{val}} e(g^-(x_n),\,y_n)
$$
## Generalization Bound
$$
\mathbb{E}[E_{val}(g^-)]\ = E_{out}(g^-)
$$
 > [!green] VC Bound
 > For binary target functions, with high probability,
 > $$
 > E_{out}(g^-)\ \leq\ E_{val}(g^-)\ +\ \mathcal{O}\left( \frac{1}{\sqrt{K}} \right)
 > $$
 > For regression,
 > $$
 > E_{out}(g^-)\ \leq\ E_{val}(g^-)\ +\ \mathcal{O}\left( \frac{\sigma(g^-)}{\sqrt{K}} \right)
 > $$
 > where $\sigma(g^-)$ is bounded by a constant in the case of classification.

> [!blue] Discussion
> Fix $g^-$ (learned from $\mathcal{D}_{train}$) and define $\sigma_{val}^{2}\ :=\ Var_{\mathcal{D}_{val}}[E_{val}(g^-)]$ . Let 
> $$
> \sigma^2(g^-)\ =\ Var_x[e(g^-(x_n),\,y)]
> $$
> be the pointwise variance in the out-of-sample error of $g$ .
>
> 1. $\sigma^2\ =\ \frac{1}{K} \sigma^2(g^-)$ .
> 2.  In a classification problem, where $e(g(x),\,y)\ =\ \left[\!\left[ g^-(x_n)\,\neq\,y\right]\!\right]$ ,
> $$
> \sigma_{var}^2\ =\ \mathbb{P}\left[ g^-(x_n)\,\neq\,y \right] (1\,-\,\mathbb{P}\left[ g^-(x_n)\,\neq\,y \right])
> $$ 
> 3.  For any $g^-$ in a classification problem, $\sigma^{2}_{var}\ \leq\ \frac{1}{4K}$ .
> 4.  In the case of regression with squared error, since the squared error is unbounded, the variance $\sigma(g^-)$ is also unbounded.
> $$
> \implies Var[E_{val}(g^-)]\ \text{is also unbounded}
> $$
> 5. For regression with squared error, if we train using fewer points to get $g$, then the model will be worse, resulting $\mathbb{E}\left[ e(g^-(x),\,y)\right]$ becomes higher. For continuous, non-negative random variables, higher mean often implies higher variance, thus $\sigma^2(g^-)$ is higher.
> 6. For regression, increasing the size of the validation set, the estimate of $E_{out}$ depends on which of $\sigma(g^-)$ or $K$ grows faster, thus the estimate can be better or worse.

> [!gray] Note:
> $$
> \begin{align} 
> \sigma_{val}^2\ &= Var_{\mathcal{D}_{val}} \left[ \frac{1}{K} \sum_{x_n\,\in\,\mathcal{D}_{val}} e(g^-(x_n)\,y_n)\right] \\
> &= \frac{1}{K^2}\left[  \sum_{x_n\,\in\,\mathcal{D}_{val}} Var_{x}\left[ e(g^-(x_n)\,y_n)\right]\right] \\
> &= \frac{1}{K^2}\left[ \sum_{x_n\,\in\,\mathcal{D}_{val}}\sigma^2(g^-)\right] \\
> &= \frac{1}{K} \sigma^2(g^-)
> \end{align}
> $$
> <span class = 'lime'>A rule of thumb is to set</span>  $K\ =\ \frac{N}{5}$ .

## Application : Model Selection
Consider a new model $\mathcal{H}_{val}$ which consisting of the final hypotheses learned from the training data using each mode $\mathcal{H}_1,\,\dots\,,\mathcal{H}_M$ :
$$
\mathcal{H}_{val}\ =\ \{g^-_1,\,\dots\,,g^-_M\}
$$
> [!green] VC Bound
> We apply the VC bound for finite hypothesis sets, with $|\mathcal{H}_{val}|\ =\ M$ :
> $$E_{out}(g_m*)\ \leq\ E_{out}(g^-_{m*})\ \leq\ E_{val}(g^-_{m*})\ +\ \mathcal{O}\left(\sqrt{\frac{\ln{M}}{K}}\right)\ .$$
> no matter which final model $m*$ we choose.

<div style="text-align:center;">
<img src="https://i.imgur.com/kSUL6zx.jpeg" alt="Example Image" style="width: 250px; height: 200px;">
</div>

### Cross Validation
To illustrate easily, we focus on the <span style="color:pink; font-weight:bold;">leave-one-out</span> version which corresponds to a validation set of size $K\,=\,1$ . 
There are $N$ ways to partition the data into a training set of size $N\,-\,1$ and a validation set of size $1$ . Specifically, let
$$
\mathcal{D}_n\,=\,(x_1,\,y_1)\ \dots\ (x_{n-1},\,y_{n-1}),\, \cancel{(x_n,\,y_n)},\ \dots\ (x_N,\,y_N)
$$
The cross validation estimate is the average
$$
E_{cv}\ =\ \frac{1}{N} \sum_{n=1}^{N}e_n
$$
In fact, the expectation of the cross validation is
$$
\mathbb{E}_{\mathcal{D}}[e_n]\ =\ \bar{E}_{out}(N\,-\,1)
$$

<div style="text-align:center;">
<img src="https://i.imgur.com/5SKOIKZ.jpeg" alt="Example Image" style="width: 250px; height: 200px;">
</div>

In general, we use the <span style="color:pink; font-weight:bold;">V-folder cross validation</span>.  In $\text{V-}$folder cross validation, the data are partitioned into $V$ disjoint sets $\mathcal{D}_1,\, \dots \,\mathcal{D}_V$ , each of size approximately $\frac{N}{V}$ .

<span class = 'lime'>A common choose in practice is 10-folder cross validation.</span>

### Examples : Cross Validation for selecting $\lambda$

<div style="text-align:center;">
<img src="https://i.postimg.cc/SKj7xkwD/2025-08-21-6-10-47.png" alt="Example Image">
</div>

**Analytic computation of $E_{cv}$ for linear models**
$$
E_{cv}\ =\ \frac{1}{N} \sum_{n=1}^N \left( \frac{\hat{y}_n\ -\ y_n}{1\ -\ H_{nn}(\lambda)} \right)^2 .
$$

