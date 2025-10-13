---
title: Regularization Theory
date: 2025-10-01
---

Tikhonov proposed a new method called *regularization* for solving ill-posed problems.

Let the set of input-output data (i.e. training sample) available for approximation be described by
$$
\begin{aligned}
\text{Input signal:}&\qquad x_i \in \mathbb{R}^{m_0}, \qquad i\ =\ 1, 2, \dots, N\\\\
\text{Desired response:}&\qquad d_i \in \mathbb{R}^{1}, \qquad i\ =\ 1, 2, \dots, N
\end{aligned}
$$

Basically, Tikhonov's regularization theory involves two terms:
1. *Standard Error Term*
$$
\begin{aligned}
\mathscr{E}_s(F)\ &=\ \frac{1}{2}\,\sum_{i = 1}^N\,(d_i\ -\ y_i)^2\\\\
\ &=\ \frac{1}{2}\,\sum_{i = 1}^N\,[d_i\ -\ F(x_i)]^2
\end{aligned}
$$
2. *Regularization Term*
$$
\mathscr{E}_c(F)\ =\ \frac{1}{2}\,\| \mathbf{D}F \|^2
$$
where $\mathbf{D}$ is a linear differential operator. We also refer to $\mathbf{D}$ as the stabilizer because it stabilizes the solution to the regularization problem. The symbol $\|\cdot\|$ denotes a norm imposed on the function space. We use a $L^2$ norm here.

> [!quote] Principle of Regularization
> Find the function $F_\lambda(x)$ that minimizes the *Tikhonov functional* $\mathscr{E}(F)$, defined by
> $$
> \mathscr{E}(F)\ =\ \mathscr{E}_s(F)\ +\ \mathscr{E}_c(F)
> $$
> where $\mathscr{E}_s(F)$ is the standard error term, $\mathscr{E}_c(F)$ is the regularizing term, and $\lambda$ is the regularization parameter.

## Frechet Differential
To proceed with the minimization of the cost functional $\mathscr{E}(F)$, we have to evaluate the differential. We can take care of this matter by using the *Frechet differenital*.

> [!abstract] Definition: Frechet Differential
> The Frechet differential of the functional $\mathscr{E}(F)$ is formally defined by
> $$
> d\mathscr{E}(F,\,h)\ =\ \left[ \frac{d}{d\beta}\,\mathscr{E}(F\ +\ \beta h) \right]_{\beta\ =\ 0}
> $$
> where $h(x)$ is a fixed function of the vector $x$.

For the function $F(x)$ to be a relative extremum of the functional $\mathscr{E}(F)$ is that Frechet differential must be zero at $F(x)$ for all $h \in \mathscr{H}$.
![[截圖 2025-10-01 晚上9.03.52.png]]

1. Evaluating $d\mathscr{E}_s(F,\,h)$
![[截圖 2025-10-01 晚上9.04.31.png]]

Recall the **Riesz representation theorem**
> [!quote] Riesz Representation Theorem
> ![[截圖 2025-10-01 晚上9.05.44.png]]
> The symbol $(\cdot, \cdot)_{\mathscr{H}}$ used here stands for the inner (scalar) product of two functions in $\mathscr{H}$ space.

Hence, we may rewrite `Equation (5.26)` as
![[截圖 2025-10-01 晚上9.08.09.png]]

2. Evaluating $d\mathscr{E}_c(F,\,h)$
![[截圖 2025-10-01 晚上9.12.39.png]]

## Euler-Lagrange Equation
> [!abstract] Definition: Green's identity
> Given a linear differential operator $\mathbf{D}$, we can find a uniquely determined **adjoint operator**, denoted by $\tilde{D}$, such that for any pair of functions $u(x)$ and $v(x)$ which are sufficiently differentiable and which satisfy proper boundary conditions, they satisfy the *Green's identity*
> $$
> \int_{\mathbb{R}^m}\,u(x)\,\mathbf{D}v(x)\,dx\ =\ \int_{\mathbb{R}^m}\,v(x)\,\mathbf{\tilde{D}}u(x)\,dx
> $$

Consider the following identifications:
$$
\begin{aligned}
u(x)\ &=\ \mathbf{D}F(x)\\\\
\mathbf{D}v(x)\ &=\ \mathbf{D}h(x)
\end{aligned}
$$
Using the Green's identity, we may rewrite
![[截圖 2025-10-01 晚上9.21.08.png]]

Returning to the extremum condition, we have
![[截圖 2025-10-01 晚上9.21.33.png]]

If the Frechet differential is zero for every $h(x)$ in $\mathscr{H}$ space, then it must satisfied
![[截圖 2025-10-01 晚上9.22.45.png]]
This equation is the **Euler-Lagrange equation** for the Tikhonov functional, which defines a necessary condition for the Tikhonov functional.

## Green's Function
To solve the partial differential equation in the `Equation (5.33)`, we introduce another tool: **Green's Function**

> [!abstract] Definition: Green's Function
> Let $G(x, \xi)$ denote a function in which both vectors $x$ and $\xi$ appear on equal footing but for different purposes: $x$ as a parameter and $\xi$ as an argument.
> For a given differential operator $\mathbf{L}$, we stipulate the function $G(x,\,\xi)$ satisfies the following conditions
> 1. For a fixed $\xi$, $G(x,\,\xi)$ is a function of $x$ and satisfies the prescribed boundary conditions (that is, the original problem constraints on $x$).
> 2. Except at the point $x = \xi$, the derivatives of $G(x, \xi)$ with respect to $x$ are all continuous; the number of derivatives is determined by the order of the operator $\mathbf{L}$.
> 3. The function $G(x, \xi)$ satisfies the following partial differential equation
> $$
> \mathbf{L}G(x, \xi)\ =\ \delta(x\ -\ \xi)
> $$

> [!example] Proposition
> Let $\varphi(x)$ denote a continuous or piecewise continuous function of $x \in \mathbb{R}^{m_0}$. Then, the function
> $$
> F(x)\ =\ \int_{\mathbb{R}^{m_0}}G(x,\,\xi)\varphi(\xi)\,d\xi
> $$
> is a solution of the differential equation
> $$
> \mathbf{L}F(x)\ =\ \varphi(x)
> $$

## Solution
Setting
$$
\begin{aligned}
\mathbf{L}\ &=\ \mathbf{\tilde{D}D}\\\\
\varphi(\xi)\ &=\ \frac{1}{\lambda}\,\sum_{i = 1}^N\,[d_i\ -\ F(x_i)]\,\delta(\xi\ -\ x_i)
\end{aligned}
$$
and using the proposition, we obtain a solution
![[截圖 2025-10-01 晚上9.43.29.png]]

Using the sifting property of the Dirac delta function,
![[截圖 2025-10-01 晚上9.43.39.png]]

This equation states that the solution to the regularization problem is a linear superposition of $N$ Green's function.
The $x_i$ represents the *centers* of the *expansion*, and the weights $[d_i - F(x_i)]/\lambda$ represent the *coefficients* of the expansion.

## Regularization Parameter
Consider a nonlinear regression problem
$$
y_i\ =\ f(x_i)\ +\ \epsilon_i,\qquad i\ =\ 1, 2,\dots, N
$$
where $f$ is a smooth curve, and $\epsilon_i$ is a sample drawn from a white noise process zero mean and variance $\sigma^2$. That is
$$
E[\epsilon_i]\ =\ 0\qquad \text{for all } i
$$
and
$$
E[\epsilon_i\,\epsilon_k]\ =\ 
\begin{cases}
\sigma^2\qquad &\text{for }k\ =\ i\\\\
0\qquad &\text{otherwise}
\end{cases}
$$
The problem is to reconstruct the function $f$, given the training sample $\{ (x_i,\,y_i) \}_{i = 1}^N$.

Let $F_\lambda(x)$ be the regularized estimate of $f(x)$ for some value of the regularization parameter $\lambda$. That is, $F_\lambda$ is the minimizer of the Tikhonov functional.
$$
\mathscr{E}(F)\ =\ \frac{1}{2}\,\sum_{i = 1}^N\,[y_i\ -\ F(x_i)]^2\ +| \frac{\lambda}{2}\|\mathbf{D}F(x)\|^2
$$

Let $R(\lambda)$ denote the average squared error over a given data set between two functions: $f$ and $F_\lambda$. That is,
$$
R(\lambda)\ =\ \frac{1}{N}\,\sum_{i = 1}^N\,[f(x_i)\ -\ F_\lambda(x_i)]^2
$$
The optimum $\lambda$ is the particular value of $\lambda$ that minimizes $R(\lambda)$.

Let $F_\lambda(x_k)$ be expressed as a linear combination of the given set of observable as follows:
$$
F_\lambda{x_k}\ =\ \sum_{i = 1}^N\,a_{ki}(\lambda)y_i
$$
In matrix form,
$$
\mathbf{F}_\lambda\ =\ \mathbf{A}(\lambda)\mathbf{y}
$$
We may rewrite the equation of $R(\lambda)$ as
![[截圖 2025-10-13 上午9.18.39.png]]

Substituting $\mathbf{y} = \mathbf{f}\ +\ \mathbf{\epsilon}$, 
![[截圖 2025-10-13 上午9.19.29.png]]
- The first term is a constant.
- The expectation of the second term is zero
- The expectation of the third term is
$$
\begin{aligned}
E[\| \mathbf{A}(\lambda)\,\mathbf{\epsilon} \|^2]\ &=\ tr\{ E[\,\epsilon^\top A^\top(\lambda)\,A(\lambda)\,\epsilon  \,] \}\\\\
&=\ E[\, tr[\epsilon^\top A^\top(\lambda)\,A(\lambda)\,\epsilon] \,]\\\\
&=\ E[\, tr[A^\top(\lambda) A(\lambda)\, \epsilon\,\epsilon^\top ] \,]\\\\
&=\ \sigma^2\,tr[A^\top(\lambda)A(\lambda)]
\end{aligned}
$$
In conclusion,
![[截圖 2025-10-13 上午9.25.00.png]]

But this required the knowledge of the regression function $f$, which is known. Consequently, we will use the following estimation in practice.
![[截圖 2025-10-13 上午9.26.00.png]]

We may show that
$$
E[\hat{R}(\lambda)]\ =\ E[R(\lambda)]
$$
Accordingly, the minimizer of the estimator $\hat{R}(\lambda)$ can be a good choice for the regularization parameter.