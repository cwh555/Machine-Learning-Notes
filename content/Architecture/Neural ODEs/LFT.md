---
title: "LFT: Neural Ordinary Differential Equations With Learnable Final-Time"
properties:
  - hide
---
## Information
- *title*: LFT: Neural Ordinary Differential Equations With Learnable Final-Time
- authors: Pang, D., Le, X., Guan, X., & Wang, J.
- *journal*: IEEE Transactions on Neural Networks and Learning Systems 2022

## Overview
### Abstract
- Existing NODEs methods fix the final time $T$, which restricts the model’s expressive capacity. 
  → The authors propose **learnable final time**, treating $T$ as a learnable parameter.
- Due to numerical computation, gradients obtained via ODE solvers contain estimation errors.  
  → To address this, a **discrete-form optimization scheme** is adopted in the backward pass.

## Preliminary: Neural ODEs
A residual block is given by
$$
z_{\ell+1} = z_\ell + f(z_\ell, \theta) \tag{1}
$$
which can be viewed as an _Euler approximation_ of a data transformation.

The above formulation is discrete. If we instead consider a continuous transformation, we obtain **Neural Ordinary Differential Equations (NODEs)**:
$$
\frac{dz(t)}{dt} = f(z(t), \theta, t) \tag{2}
$$
- $z$ denotes the latent state vector (feature map)
- $\theta$ denotes the parameter vector
- $f$ is an activation function
- $t$ is time

### Forward Mode
In the forward pass, NODEs solve an **Initial Value Problem (IVP)**:
$$
z(t_1) = z(t_0) + \int_{t_0}^{t_1} f(z(t), \theta, t)\, dt. \tag{3}
$$

To guarantee the existence and uniqueness of the ODE solution, the activation function $f$ must be *Lipschitz continuous*, such as ReLU or Tanh. 

Under this condition, *Picard’s Existence Theorem* holds, and numerical ODE solvers such as the _Euler_ or _Runge–Kutta_ methods can be used to solve the ODE.

### Backward Mode
The training objective of NODEs can be formulated as the following **constrained optimization problem**:
$$
\begin{aligned}
\min_{\theta}&\ \mathcal{L}\left(z(t_0) + \int_{t_0}^{t_1} f(z(t), \theta, t) dt\right) \\\\
\text{s.t. }& \frac{dz(t)}{dt} = f(z(t), \theta, t)
\end{aligned}
$$

where $\mathcal{L}(\cdot)$ is the loss function of $z(t_1)$. The gradients $\nabla_\theta \mathcal{L}$ can be estimated by the adjoint method.

## Methods: Learnable Final Time
In the preliminary NODE formulation, the final time $T$ is fixed. In this paper, the authors propose to make $T$ **learnable**, denoted as $t_f$​. With this modification, the constrained optimization problem becomes
$$
\min_{\theta ,t_f} \mathcal{L}(z(t_f)) \quad \text{s.t. } \dot{z}(t) = f (z(t), \theta ,t) \tag{5}
$$
where the final state $z(t_f)$ can be obtained by solving the IVP
$$
z(t_f) = z(t_0) + \int_{t_0}^{t_f} f (z(t), \theta ,t)dt. \tag{6}
$$

Introduce the *Lagrange Multiplier Method*, we obtain the following equivalent form
$$
\min_{\theta ,t_f} \tilde L = \mathcal{L}(z(t_f)) + \int_{t_0}^{t_f} \lambda(t)^T [ f (z(t), \theta ,t) - \dot{z}(t)]dt \tag{7}
$$

where $\tilde L$ is the Lagrangian function, $\lambda(t)$ is the Lagrange multiplier vector.

For simplicity, we introduce Hamiltonian into the Lagrangian function as follows:
$$
\tilde L = \mathcal{L}(z(t_f)) + \int_{t_0}^{t_f} \{H(z(t), \theta, \lambda(t),t) - \lambda(t)\dot{z}(t)\}dt \tag{9}
$$
where $H$ is the Hamiltonian written as
$$
H(z(t), \theta, \lambda(t),t) = \lambda(t)^T f (z(t), \theta ,t). \tag{10}
$$
By considering small perturbations of each term, we obtain
$$
\begin{align*}
\delta \tilde L &= \frac{\partial \mathcal{L}}{\partial z}(z(t_f)) \cdot \delta z_f \\\\
&+ \int_{t_0}^{t_f} \left[  \frac{\partial H}{\partial z} \cdot \delta z(t) + \frac{\partial H}{\partial \theta} \cdot \delta \theta + \frac{\partial H}{\partial \lambda} \cdot \delta \lambda(t) - \dot{z}(t) \cdot \delta \lambda(t) - \lambda(t) \cdot \delta \dot{z}(t) \right] dt 
\\\\
&+ \left[ H(z(t_f), \theta, \lambda(t_f), t_f) - \lambda(t_f)^T \dot{z}(t_f) \right] \cdot \delta t_f \tag{11}
\end{align*}
$$
By integration by parts
$$
\begin{align*}
\int_{t_0}^{t_f} -\lambda(t)^T \delta \dot{z}(t) dt &= -\lambda(t_f)^T \delta z(t_f) + \lambda(t_0)^T \delta z(t_0) + \int_{t_0}^{t_f} \dot{\lambda}(t)^T \delta z(t) dt \\\\
& = -\lambda(t_f)^T \delta z(t_f) + \int_{t_0}^{t_f} \dot{\lambda}(t)^T \delta z(t) dt
\tag{12}
\end{align*}
$$
where $\delta z(t_0) = 0$ since the initial state is fixed.

Using the first-order Taylor expansion of $\delta z_f$:
$$
\delta z_f \approx \delta z(t_f) + \dot{z}(t_f)\delta t_f. \tag{13}
$$
and substituting (12) and (13) into (11), we obtain
$$
\begin{aligned}
\delta \tilde L &= \left[ \frac{\partial \mathcal{L}}{\partial z}(z(t_f)) - \lambda(t_f)^T \right] \cdot \delta z_f \\\\
&+ H(z(t_f), \lambda(t_f), \theta ,t_f) \cdot \delta t_f \\\\
&+ \int_{t_0}^{t_f} \left[ \left( \frac{\partial H}{\partial z} + \dot{\lambda}(t)^T \right) \cdot \delta z(t) + \left( \frac{\partial H}{\partial \lambda} - \dot{z}(t)^T \right) \cdot \delta \lambda(t) \right] dt \\\\
&+ \left[ \int_{t_0}^{t_f} \frac{\partial H}{\partial \theta} dt \right] \delta \theta
\end{aligned}
$$
At the optimum, $\delta \tilde L = 0$ for arbitrary perturbations. This leads to the following result.

> [!gray] Theorem 1
> The optimality conditions derived from the calculus of variations to optimization problem is
> $$
> 0 = \frac{\partial L}{\partial\theta} = \int_{t_0}^{t_f} \lambda(t)^T \frac{\partial f (z(t), \theta ,t)}{\partial\theta} dt \tag{8a}
> $$
> $$
> 0 = \lambda(t_f)^T f (z(t_f), \theta ,t_f) \tag{8b}
> $$
> $$
> \frac{dz(t)}{dt} = f (z(t), \theta ,t) \tag{8c}
> $$
> $$
> \frac{d\lambda(t)}{dt} = -\lambda(t)^T \frac{\partial f (z(t), \theta ,t)}{\partial z(t)} \tag{8d}
> $$
> $$
> \lambda(t_f) = \frac{\partial \mathcal{L}(z(t_f))}{\partial z(t_f)} \tag{8e}
> $$

- (8a) is the stationarity condition
- (8b) is boundary condition corresponding to final-time-free
- (8c) is the state equation
- (8d) is the costate equation
- (8e) is boundary conditions corresponding to final-state-free

### Backward Pass Gradient Computation
The gradient computation in the backward pass can be summarized as follows:
1. Determine the initial values $z(t_f)$ and determine $\lambda(t_f)$ using given initial values $z(t_f)$
2. Obtain $z(t)$ and $\lambda(t)$ in time $[t_f, t_0]$ by solving the following equations in reverse time:
$$
z(t_0) = z(t_f) + \int_{t_f}^{t_0} f (z(t), \theta )dt \tag{16a}
$$
$$
\lambda(t_0) = \lambda(t_f) - \int_{t_f}^{t_0} \lambda(t)^T \frac{\partial f (z(t), \theta )}{\partial z(t)} dt \tag{16b}
$$
3. Compute the gradients using
$$
\frac{\partial \mathcal{L}}{\partial \theta} = - \int_{t_f}^{t_0} \lambda(t)^T \frac{\partial f (z(t), \theta ,t)}{\partial \theta} dt \tag{16c}
$$
$$
\frac{\partial \mathcal{L}}{\partial t_f} = \lambda(t_f)^T f (z(t_f), \theta ,t_f) \tag{16d}
$$

### Algorithm
![[截圖 2026-01-06 晚上8.32.27.png#center|400]]

## Analysis: Gradient Error
Forward Euler formula:
$$
z(t_{i+1}) = z(t_i) + f_{\theta}(z(t_i)) \cdot h. \tag{17}
$$
Backward Euler formula:
$$
\hat{z}(t_i) = \hat{z}(t_{i+1}) - f_{\theta}(\hat{z}(t_{i+1})) \cdot h. \tag{18}
$$

It can be observed that during the backward pass, the states stored in the forward pass are not reused.   
As a result,
$$
\hat{z}(t_i) \neq z(t_i)
$$
and the accumulated discrepancy leads to an error, referred to as the intermediate state truncated error.

Therefore, some checkpoint-based methods have been proposed to reduce this error, but other sources of error still remain.

---
Theoretically, the desired gradient should be
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \theta} &= \int_{t_0}^{t_f} \lambda(t)^T \frac{\partial f_{\theta}(z(t))}{\partial \theta} dt \approx \sum_{i=0}^{N} \lambda(t_i)^T \frac{\partial f_{\theta}(z(t_i))}{\partial \theta} \cdot h\\\\
&= \sum_{i=0}^{N} \left( \frac{\partial \mathcal{L}}{\partial z(t_i)} \right)^T \frac{\partial f_{\theta}(z(t_i))}{\partial \theta} \cdot h
\end{aligned}
$$
where $\lambda(t_i) = \partial \mathcal{L} / \partial z(t_i)$ denotes the adjoint state.

However, in practice, the gradient computed via backpropagation is
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \theta} &= \frac{\partial \mathcal{L}}{\partial z(t_N)}^T \frac{\partial f(z(t_{N-1}))}{\partial \theta} \cdot h \\\\
&+ \frac{\partial \mathcal{L}}{\partial z(t_{N-1})}^T \frac{\partial f(z(t_{N-2}))}{\partial \theta} \cdot h\\\\
&+ \dots + \frac{\partial \mathcal{L}}{\partial z(t_1)}^T \frac{\partial f(z(t_0))}{\partial \theta} \cdot h\\\\
&= \sum_{i=1}^{N} \left( \frac{\partial \mathcal{L}}{\partial z(t_i)} \right)^T \frac{\partial f_{\theta}(z(t_{i-1}))}{\partial \theta} \cdot h
\end{aligned}
$$
By using a first-order Taylor expansion to estimate the discrepancy between the two gradients, the following theorem is obtained:

> [!gray] Theorem 2
> Assume that a NODE is solved using the Euler method in the backward pass. The estimation error associated with the gradient of the loss function is derived as follows:
> $$
> E(t_N \to 0) = \frac{\partial \mathcal{L}}{\partial \theta} - \frac{\partial \overline{\mathcal{L}}}{\partial \theta} = \left( \frac{\partial \mathcal{L}}{\partial z(t_0)} \right)^T \frac{\partial f_{\theta}(z(t_0))}{\partial \theta} \cdot h + \sum_{i=1}^{N-1} R(t_i) \tag{24}
> $$
> where $R(t_i)$ denotes the truncated estimation error at $t_i$.
> $$
> R(t_i) = \left( \frac{\partial \mathcal{L}}{\partial z(t_i)} \right)^T \nabla_{\theta} \left( \frac{\partial f_{\theta}(z(t_{i-1}))}{\partial z(t_{i-1})}^T f_{\theta}(z(t_{i-1})) \right) \cdot h^2. \tag{25}
> $$

## Applications: NODEs-based Generative Models
Generative models require tracking how the probability density $p(z(t))$ changes as $z(t)$ evolves.

For continuous transformations, the change in log density can be computed using the trace of the Jacobian:
$$
\frac{\partial \log p(z(t))}{\partial t} = -\text{Tr}\left( \frac{\partial f}{\partial z(t)} \right) \tag{31}
$$
Then, we can compute the change of log probability from the sample distribution to the prior distribution
$$
\log p(z_1) - \log p(z_0) = \int_{t_0}^{t_1} -\text{Tr}\left( \frac{\partial f}{\partial z(t)} \right) dt. \tag{32}
$$

### Training Mode
FFJORD solves the following IVPs
$$
\begin{bmatrix} z_1 \\ \log p(z_1) - \log p(z_0) \end{bmatrix} = \int_{t_0}^{t_1} \begin{bmatrix} f(z(t), t; \theta) \\ -\text{Tr}\left( \frac{\partial f}{\partial z(t)} \right) \end{bmatrix} dt + \begin{bmatrix} z_0 \\ 0 \end{bmatrix} \tag{33}
$$
where $z_0$ is the sample data, $z_1$ is the prior data.
- The first row of the integral computes $z_1$.
- The second row of the integral computes the change in log density.

*Loss function*:
$$
\mathcal L\ =\ -\log p(z_0)
$$

### Generative Mode
Reverse integration:
$$
\begin{bmatrix} z_0 \\ \log p(z_0) \end{bmatrix} = \int_{t_1}^{t_0} \begin{bmatrix} f(z(t), t; \theta) \\ -\text{Tr}\left( \frac{\partial f}{\partial z(t)} \right) \end{bmatrix} dt + \begin{bmatrix} z_1 \\ \log p(z_1) \end{bmatrix} \tag{34}
$$
This integrates from the prior $z_1$ to generate a sample $z_0$​, while simultaneously computing the corresponding log probability $\log p(z_0)$.

## Discussion
### Future Works
- investigate the connections between the final time and the representation capability of NODEs models
- solving optimization problems with the constraints of physical dynamics

## Reference
```apa
Pang, D., Le, X., Guan, X., & Wang, J. (2022). LFT: Neural ordinary differential equations with learnable final-time. _IEEE Transactions on Neural Networks and Learning Systems_, _35_(5), 6918-6927.
```