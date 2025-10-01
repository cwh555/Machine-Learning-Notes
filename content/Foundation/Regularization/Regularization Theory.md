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

## Solution to the Regularization Problem
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
