---
title: Regularization
date: 2025-08-21
---
## Soft Constraint
Consider the polynomial models in a space $\mathcal{Z}$ , under a nonlinear transformation $\Phi :\ \mathcal{X}\,\rightarrow\,\mathcal{Z}$ . Here, for the $Q$-th order polynomial mode, $\Phi$ transforms $x$ into a vector $z$ of Legendre polynomials.
$$
z\ =\ \begin{bmatrix}
1 \\
L_1(x) \\
\vdots \\
L_Q(x)
\end{bmatrix}
$$
We will use the <span style="color:pink; font-weight:bold;">Legendre polynomials</span> for the sake of its orthogonal property.

Our hypothesis set 
$$
H_Q\ =\ \left\{ h\ :\ h(x)\,=\,w^T z\, =\, \sum_{q = 0}^{Q}\,w_q L_q(x) \right\}_{w\ \in\ \mathbb{R}^{Q + 1}}
$$
We would like to minimize the squared error
$$
E_{in}(w)\ =\ \frac{1}{N} \sum_{n=1}^{N} \left( w^Tz_n\,-\,y_n \right)^2
$$

Define the <span style="color:pink; font-weight:bold;">soft-order-constrainted</span> hypothesis set $\mathcal{H}(C)$ by
$$
\mathcal{H}(C)\ :=\ \left\{ h\ :\ h(x)\, =\, w^T z,\, w^T w\, \leq\,C \right\}
$$
We can deduce that for some $\lambda_C\ \gt\ 0$ , $w_{reg}$ locally minimizes 
$$
E_{in}(w)\ +\ \lambda_C\, w^T w
$$

> [!note] Discussion
> A more general soft constraint is the <span style="color:pink; font-weight:bold;">Tikhonov</span> regularization constraint
> $$
> w^T\,\Gamma^T\,\Gamma\,w\ \leq\ C
> $$
> 
> For example, taking $\Gamma\ =\ I_Q$, we obtain the constraint  $\sum_{q=0}^{Q}\ \leq\ C$ .
> 

## Augmented Error
In general, we will use the <span style="color:pink; font-weight:bold;">augmented error</span> for a hypothesis $h\,\in\,\mathcal{H}$ is 
$$
E_{aug}(h,\,\lambda,\,\Omega)\ =\ E_{in}(h)\ +\ \frac{\lambda}{N}\Omega(h)\ .
$$
$\frac{\lambda}{N}\Omega(h)$ : is called the penalty term
- The regularizer $\Omega(h)$ penalizes a particular property of $h$.
- The <span style="color:pink; font-weight:bold;">regularization parameter</span> $\lambda$ controls the strength of the penalty.

Since the need for regularization goes down as the number of data points goes up, we factored out $\frac{1}{N}$ .

For the <span style="color:pink; font-weight:bold;">weight decay</span>, $\Omega(h)\ =\ w^T w$ .
The penalty term $\lambda w^T w$ is a form of <span style="color:pink; font-weight:bold;">ridge regression</span>.

### Weight Based Complexity Penalties
$\textbf{\textcolor{pink}{Squared weight decay regularizer}}$
$$E_{aug}(w)\ =\ E_{in}(w)\ +\ \frac{\lambda}{N}\,\sum_{l,i,j} \left( w_{ij}^{(l)}\right)^2$$
$\textbf{\textcolor{pink}{Weight Elimination}}$
$$E_{aug}(w\,\lambda)\ =\ E_{in}(w)\ +\ \frac{\lambda}{N}\,\sum_{l,i,j}\, \frac{\left( w_{ij}^{(l)}\right)^2}{1\,+\, \left(w_{ij}^{(l)}\right)^2}$$

## Empirical Perspectives
<ul>
<li class="lime">The best way to constrain the learning is in the direction of the target function.</li>
<li class="lime">The more noise, the more constraint is needed.</li>
<li class="lime">Constraining the learning towards smoother hypotheses "hurts" out ability to overfit the noise more than it hurts out ability to fit the useful information.</li>
</ul>

## Example : Linear Model
### Description
Let $Z\ =\ [z_1\ \dots\ z_N]^T$ be the data matrix and assume that $Z$ has the full column rank.
Let $w_{lin}\ =\ (Z^TZ)^{-1}Z^Ty$, and let $H\ =\ Z(ZZ^T)^{-1}Z^T$ .
Then, the error 
$$
E_{in}\ =\ \frac{(w\,-\,w_{lin})^T Z Z^T (w\,-\,w_{lin})\ +\ y^T(I\,-\,H)y}{N}
$$
### Weight Decay
$$
E_{aug}\ =\ \frac{(w\,-\,w_{lin})^T Z Z^T (w\,-\,w_{lin})\ +\ \lambda w^Tw\ +\ y^T(I\,-\,H)y}{N}
$$
In order to have $\nabla_w E_{aug}\ =\ 0$, we obtain
$$
w_{reg}\ =\ (Z^T Z\, +\, \lambda I)^{-1} Z^Ty\ .
$$
The predictions on the in-sample data are given 
$$
\hat{y}\ =\ Zw_{reg}\ =\ H(\lambda)y,\quad \quad where\ H(\lambda)\ =\ Z(Z^T Z\, +\, \lambda I)^{-1} Z^T .
$$
The in-sample error 
$$
E_{in}(w_{reg})\ =\ \frac{1}{N}y^T (I\,-\,H(\lambda)\,)^2y\ .
$$