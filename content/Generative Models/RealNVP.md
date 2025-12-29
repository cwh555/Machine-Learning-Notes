---
title: Density Estimation Using Real NVP
properties:
  - hide
---
## Information
- _title_: Density Estimation Using Real NVP  
- _authors_: Dinh, L., Sohl-Dickstein, J., & Bengio, S.  
- _conference_: ICLR 2017  
- *task*: image generation
## Overview
### Background
Unsupervised probabilistic modeling is challenging, requiring tractable learning, sampling, inference, and evaluation.

### Abstract
Improvement of NICE: remove volume-preserving constraint in the coupling layer

## Methods
### Affine Coupling Layer
Given a $D$ dimensional input $x$ and $d < D$, the output y of an affine coupling layer follows the equations
$$
\begin{aligned}
y_{1:d} &= x_{1:d}  \\\\
y_{d+1:D} &= x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})
\end{aligned}
$$
- $s(\cdot)$：scale function $\mathbb{R}^d \to \mathbb{R}^{D-d}$
- $t(\cdot)$：translation function $\mathbb{R}^d \to \mathbb{R}^{D-d}$
- $\odot$：element-wise product

#### Jacobian
$$
\frac{\partial y}{\partial x^T} = \begin{bmatrix} I_d & 0 \\ \frac{\partial y_{d+1:D}}{\partial x_{1:d}^T} & \mathrm{diag}(\exp(s(x_{1:d}))) \end{bmatrix}
$$

*determinant*
$$
\det\left(\frac{\partial y}{\partial x^T}\right) = \exp\left(\sum_j s(x_{1:d})_j\right)
$$
#### Inverse
$$
x_{1:d} = y_{1:d},\quad x_{d+1:D} = (y_{d+1:D} - t(y_{1:d})) \odot \exp(-s(y_{1:d}))
$$

### Partitioning Input: Masked Convolution
Implement input partitioning to exploit local image correlations.

General formulation:
$$
y = b \odot x + (1-b) \odot \left( x \odot \exp(s(b \odot x)) + t(b \odot x) \right)
$$
- $b$ is binary mask
- $s$ and $t$ are implemented as ReLU convolutional networks.

![[截圖 2025-12-29 中午12.33.49.png#center|400]]
- left: spatial checkerboard mask
- right: channel-wise mask
#### Spatial checkerboard mask
Exploit local spatial correlation in images.

Let each pixel have coordinates $(i, j)$. Define $b(i,j)$ as:
$$
b(i,j) = \begin{cases} 1 & \text{if } (i+j) \text{ is odd} \\\\ 0 & \text{if } (i+j) \text{ is even} \end{cases}
$$
- Pixels with mask = 1 are kept unchanged; pixels with mask = 0 are transformed by the coupling layer.
- Alternating pattern ensures that in subsequent layers, previously unchanged pixels can be updated.
#### Channel-wise mask
Exploit correlations along channels (feature maps).

For an input with $C$ channels, define:
$$
b(c) = \begin{cases} 1 & \text{for } c \leq C/2 \\\\ 0 & \text{for } c > C/2 \end{cases}
$$
- First half of channels remain unchanged, second half are transformed.
- Alternating masks in subsequent layers ensures all channels eventually get updated.​

### Combining Coupling Layer
Jacobian determinant of composite functions
$$
\det\left(\frac{\partial (f_b \circ f_a)}{\partial x^T}\right) = \det\left(\frac{\partial f_a}{\partial x_a^T}\right) \cdot \det\left(\frac{\partial f_b}{\partial x_b^T}\right)
$$

Inverse
$$
(f_b \circ f_a)^{-1} = f_a^{-1} \circ f_b^{-1}
$$

### Multi-scale
This can reduce computation and memory cost while capturing multi-scale features.

Define the **squeezing** operation as reshaping $s \times s \times c \to s/2 \times s/2 \times 4c$

At each scale, 
- Apply 3 coupling layers with checkerboard masks
- Apply squeezing
- Apply 3 coupling layers with channel-wise masks

Recursive definition:
$$
\begin{aligned}
h^{(0)} &= x \\\\
(z^{(i+1)}, h^{(i+1)}) &= f^{(i+1)}(h^{(i)}) \\\\
z^{(L)} &= f^{(L)}(h^{(L-1)}) \\\\
z &= (z^{(1)}, ..., z^{(L)}) \\\\
\end{aligned}
$$

### BatchNormalization
To improve training signal propagation and stability in deep stacks of coupling layers, the paper use batch normalization.

Batch normalization is easily incorporated into the Jacobian determinant:
$$
x \mapsto \frac{x - \tilde{\mu}}{\sqrt{\tilde{\sigma}^2 + \epsilon}}
$$
$$
\det = \prod_i (\tilde{\sigma}_i^2 + \epsilon)^{-1/2}
$$

## Discussion
### Advantages
- Defines a class of invertible functions with tractable Jacobian determinant 
  → enables exact log-likelihood evaluation, inference, and sampling.
- Learns a semantically meaningful latent space of the same dimension as input
  → potentially useful for semi-supervised learning.

### Future Works
- *Architectural improvements*: incorporate dilated convolutions or residual networks to enhance transformations.
- *Structure Prediction*: RealNVP can be conditioned on additional variables (e.g., class labels) to create structured output models.
- *Other applications*:
    - In reinforcement learning, invertible functions can expand tractable functions for continuous Q-learning.
    - Facilitate representations where local linear Gaussian approximations are appropriate.

## Reference
```apa
Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using real nvp. _arXiv preprint arXiv:1605.08803_.
```