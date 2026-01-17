---
title: Error estimation in the neural network solution of ordinary differential equations
properties:
  - hide
---
## Information
- *title*: Error estimation in the neural network solution of ordinary differential equations
- *author*: Filici, C.
- *journal*: Neural Networks 2010

## Overview
### Abstract
Use a neural network to approximate the ODE solution, and employ the neighboring method to estimate the error of this approximate solution.

## Methods
### ODE Problem
Let $f : \mathbb{R}^{m+1} \to \mathbb{R}^m$ and $z : \mathbb{R} \to \mathbb{R}^m$ and define the **initial value problem** in the interval $[0, t_f]$:
$$
\dot{z} = f(t, z) \tag{1}
$$
$$
z(0) = z_0 \tag{2}
$$
where $t \in \mathbb{R}$ represents the independent variable and $z_0 \in \mathbb{R}^m$ is the initial condition.

### Neighboring Problem Method
Use some discretization algorithm to iteratively compute an approximate solution $z_n$​ at time points $t_n$​ (the true solution being $z(t_n)$).

Define the error as
$$
e_n\ =\ z_n\ -\ z(t_n)
$$
Interpolate the discrete points $\{z_n\}$ (e.g., using polynomials) to form a continuous approximation:
$$
P(t_n)=z_n
$$
$P(t)$ is now a fully known, differentiable approximate trajectory.

Construct a _neighboring problem_ using $P$
$$
\dot{x} = f(t, x) + D_P(t) \tag{4}
$$
$$
x(0) = x_0, \tag{5}
$$
where $D_P(t) = \dot{P} - f(t, P)$.

We observe that $P$ is an exact solution of this new ODE. By solving this neighboring problem using the same method, we can compute the estimated error:
$$
\bar{e}_n = x_n - P(t_n) \tag{6}
$$
This error $\bar{e}_n$​ can then be used to estimate the original true error $e_n$.

### Neural ODE Solver
Instead of using numerical methods to approximate the ODE solution, we use a single-layer, multiple-output perceptron $\mathcal N : \mathbb{R} \to \mathbb{R}^m$ to approximate the solution.

Define the loss function as
$$
F_f(W) = \left\| \frac{z_0 - \mathcal N(t_0, W)}{\sqrt{2}} \right\|^2 + \sum_{\mu=1}^N \left\| \frac{\dot{\mathcal N}(t_\mu, W) - f(\mathcal N(t_\mu, W), t_\mu)}{\sqrt{2\mathcal N}} \right\|^2 \tag{7}
$$
- The first term enforces the initial condition so that the network solution matches the original ODE at $t_0$.
- The second term enforces that $\mathcal N$ satisfies the ODE at the sampled time points.
- Each term is normalized to have comparable influence.

### Neural Neighboring Problem
Apply the same neighboring problem method to the Neural ODE solver.  
(substitute $P$ → $\mathcal N$)

The original error is
$$
e(t)\ =\ z(t)\ -\ \mathcal N(t)
$$
he network $\mathcal N$ is the exact solution of the following ODE:
$$
\dot{y} = f(t, y) + D(t) \triangleq G(t, y) \tag{9}
$$
$$
y(0) = \mathcal N(t_0) \tag{10}
$$
where $D = \dot{\mathcal N}(t) - f(t, \mathcal N)$.

Using another network to minimize the loss function $F_G(W)$, we obtain another solution $\mathcal M$. Its error can be directly computed as:
$$
\bar{e}(t) = \mathcal N - \mathcal M \tag{11}
$$
The neural neighboring problem is defined as
$$
\dot{w} = G(t, w) + \bar{D}(t) \tag{12}
$$
$$
w(0) = \mathcal M(t_0) \tag{13}
$$
where $\bar{D}(t) = \dot{\mathcal M} - G(t,\mathcal M)$.

### Error Estimation
We summarize the two error terms and rewrite them in terms of $f$:
$$
\dot e(t)=f(t,z)-f(t,y)-D(t) \tag{14}
$$
$$
e(0)=z_0-\mathcal N(t_0)\tag{15}
$$
$$
\dot{\bar e}(t)=f(t,y)-f(t,w)-\bar D(t) \tag{16}
$$
$$
\bar e(0)=\mathcal N(t_0)- \mathcal M(t_0) \tag{17}
$$

Perform a Taylor expansion for the $i$-th component:
$$
f_i(t,z)-f_i(t,y) = \nabla f_i(t,\,y+\theta_i e)^T e
$$
where $\theta_i(t)\in[0,1]$ and similarly,
$$
f_i(t,y)-f_i(t,w) = \nabla f_i(t,\,w+\hat\theta_i\bar e)^T \bar e
$$
Then the $i$-th components of $\dot e_i$ and $\dot{\bar e}_i$ are
$$
\dot e_i = \nabla f_i(t,y+\theta_i e)^T e - D_i(t) \tag{18}
$$
$$
\dot{\bar e}_i = \nabla f_i(t,w+\hat\theta_i\bar e)^T \bar e - \bar D_i(t) \tag{19}
$$
with $\theta_i(t)$ and $\hat{\theta}_i(t) \in [0, 1]$.

Also, noting that $y = N$ and that $w = N - \bar{e}$, the gradients in (18) and (19) are packed in the matrices:
$$
A(t,\mathcal N,\theta,e) = \begin{bmatrix} \nabla f_1(t,\mathcal N+\theta_1 e)^T\\ \vdots\\ \nabla f_m(t,\mathcal N+\theta_m e)^T \end{bmatrix}
$$

$$
\bar A(t,\mathcal N,\bar\theta,\bar e) = \begin{bmatrix} \nabla f_1(t,\mathcal N+\bar\theta_1 \bar e)^T\\ \vdots\\ \nabla f_m(t,\mathcal N+\bar\theta_m \bar e)^T \end{bmatrix}
$$
​​where $\forall i, \bar{\theta}_i = \hat{\theta}_i - 1$.

Unified form:
$$
\dot{e} = A(t,\mathcal N, \theta, e)e - D(t)
$$
$$
\dot{\bar{e}} = \bar{A}(t,\mathcal N, \bar{\theta}, \bar{e})\bar{e} - \bar{D}(t),
$$
Define:
$$
\begin{aligned}
h(t, e) &= A(t, N, \theta, e)e - D(t)\\\\
g(t, \bar{e}) &= \bar{A}(t, N, \bar{\theta}, \bar{e})\bar{e} - A(t, N, \theta, \bar{e})\bar{e} - \bar{D}(t) + D(t)
\end{aligned}
$$

Hence, we obtain
$$
\dot{e} = h(t, e) \tag{20}
$$
$$
\dot{\bar{e}} = h(t, \bar{e}) + g(t, \bar{e}) \tag{21}
$$
Now the authors apply the following theorem:

> [!gray] Khalil (1996), Theorem 2.5
> Let $\dot{x} = f(t, x)$ be the nominal system and $\dot{y} = f(t, y) + g(t, y)$ be the perturbed system.
> If:
> 1. *Lipschitz Condition:* The function $f$ is Lipschitz continuous with constant $L$ (i.e., $\|f(t, x) - f(t, y)\| \leq L \|x - y\|$).    
> 2. *Bounded Perturbation:* The perturbation term $g$ is uniformly bounded such that $\|g(t, y)\| \leq \mu$ for all $t$.
> 3. *Initial Bound:* The initial states are close such that $\|x(t_0) - y(t_0)\| \leq \gamma$.
> 
> Then, for all $t \geq t_0$, the difference between the solutions is bounded by:
> $$
> \|x(t) - y(t)\| \leq \gamma e^{L(t-t_0)} + \frac{\mu}{L} \left( e^{L(t-t_0)} - 1 \right)
> $$

By adding the required conditions, the authors derive the following proposition (the proof consists of simply deriving to satisfy the theorem’s assumptions).

![[截圖 2026-01-07 下午2.00.39.png#center|500]]