---
title: Neural Ordinary Differential Equations
properties:
  - hide
image: 0002.jpeg
---
## Information
- *title*: Neural Ordinary Differential Equations
- *authors*: Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K.
- *journal*: Advances in neural information processing systems 2018

## Overview
### Abstract
Inspired by ResNet, this work proposes letting a neural network directly predict $dz/dt$, so that state evolution is continuous.
- Using an ODE solver enables fast computation with low numerical error.
- The model size is independent of depth; the effective depth is controlled by the time horizon $T$.
- The framework can be applied to continuous normalizing flows.
- The proposed method enables backpropagation without increasing memory consumption.
## Methods

Forward pass 的 ODE:
$$
z(t_1) = z(t_0) + \int_{t_0}^{t_1} f(z(t), t, θ) dt
$$
Consider optimizing a scalar-valued loss function $\mathcal{L}(\cdot)$, whose input is the result of an ODE solver:
$$
\mathcal{L}(z(t_1)) = \mathcal{L} \left( z(t_0) + \int_{t_0}^{t_1} f(z(t), t, \theta) dt \right) = \mathcal{L}(\text{ODESolve}(z(t_0), f, t_0, t_1, \theta)) \tag{3}
$$
To perform gradient descent, we need $\partial L/\partial \theta$ 
The first step is to understand how $\mathcal{L}$ depends on the hidden state $z(t)$.
Define the *adjoint* as
$$
a(t) = \frac{\partial L}{\partial z(t)}
$$

Differentiating with respect to $t$, the adjoint satisfies the following ODE:
​
$$
\frac{da(t)}{dt} = -a(t)^\top \frac{\partial f(z(t), t, \theta)}{\partial z}
$$


Thus, we can run an ODE solver backward in time, starting from  
$a(t_1) = \partial \mathcal{L} / \partial z(t_1)a(t1​)$ and integrating back to $t_0$.  
This avoids storing the entire forward trajectory.

Now, we can compute the gradients with respect to the parameters $\theta$
$$
\frac{dL}{dθ} = \int_{t_0}^{t_1} a(t)^\top \frac{\partial f(z(t), t, \theta)}{\partial θ} dt
$$

The summary of the algorithm is as follows:

![[截圖 2026-01-07 下午2.43.32.png]]

## Applications
### Continuous Normalizing Flows

![[截圖 2026-01-07 下午2.48.47.png]]

For example, discrete planar normalizing flow:
$$
z(t+1) = z(t) + u h(w^T z(t) + b)
$$
The change in log probability is
$$
\log p(z(t+1)) = \log p(z(t)) - \log|1 + u^T \frac{\partial h}{\partial z}
$$
Now, we can examine the continuous analog of the planar flow
$$
\frac{dz(t)}{dt} = u h(w^T z(t) + b), \quad \frac{\partial \log p(z(t))}{\partial t} = - u^T \frac{\partial h}{\partial z(t)}
$$

Since $tr(\cdot)$ is linear, the computation would be easy even when using multiple hidden units. (linear cost)
$$
\frac{dz}{dt} = \sum_{n=1}^{M} f_n(z), \quad \frac{d \log p}{dt} = \sum_{n=1}^{M} \text{tr} \frac{\partial f_n}{\partial z}
$$

### Generative Latent Function Time-series Model
This framework can also be applied to generative models for irregularly sampled time series, such as medical data, where time bins have varying sizes. Modeling the latent state as a continuous trajectory naturally resolves this issue.

## Reference
```apa
Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. _Advances in neural information processing systems_, _31_.
```
