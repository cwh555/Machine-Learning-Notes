---
title: "NICE: Non-linear Independent Components Estimation"
properties:
  - hide
  - image
tags:
  - experiments
---
## Information
- _title_: NICE: Non-linear Independent Components Estimation  
- _authors_: Dinh, L., Krueger, D., & Bengio, Y.  
- _conference_: ICLR 2015  
- *task*: image generation

## Overview
### Abstract
- *Intuition*: a good representation is one in which the data has a distribution that is easy to model
- *Method*: learn a non-linear transformation to map the data to a latent space and make each dimension independent.

## Method
### Concept
In this paper, we want to find a transformation $h = f(x)$ of the data into a new space such that the resulting distribution factorizes, i.e., the components $h_d$ are independent:
$$
p_H(h) = \prod_d p_{H_d}(h_d)
$$
Then, using change of variable, we can obtain the relationship between two distribution:
$$
p_X(x) = p_H(f(x)) \left|\det \frac{\partial f(x)}{\partial x}\right|
$$
The sampling process would be
$$
h \sim p_H(h), \quad x = f^{-1}(h)
$$
Hence, we want the transformation has two properties
- trivial Jacobian matrix
- trivial inverse

The core idea is to split $x$ into two blocks $(x_1, x_2)$ and apply as building block a transformation from $(x_1, x_2)$ to $(y_1, y_2)$ of the form:
$$
\begin{aligned} y_1 &= x_1 \\ y_2 &= x_2 + m(x_1) \end{aligned}​
$$
where $m$ is an arbitrarily complex function. (ReLU in the paper)

This building block has a unit Jacobian determinant for any m and is trivially invertible since:
$$
\begin{aligned} x_1 &= y_1 \\ x_2 &= y_2 - m(y_1) \end{aligned}​
$$

### Learning
The objective is to maximum the likelihood
$$
\log p_X(x) = \log p_H(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|​​
$$
where $p_H(h)$ is a predefined prior, e.g., isotropic Gaussian.

By hypothesis, $p_H$​ is factorial, so the NICE objective becomes:
$$
\log p_X(x) = \sum_{d=1}^{D} \log p_{H_d}(f_d(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|
$$
> [!lime]
> prevent trivial solutions

This design can prevent trivial solutions (e.g., contracting all points) because the Jacobian determinant penalizes contraction and encourages expansion in high-density regions.

### Architecture
The remaining problem is to find the transformation that is
- invertible
- easy to compute Jacobian determinant
- complex enough to represent the distribution
#### Jacobian Determinant
For simplicity, we would like to use several composition functions $f = f_1 \circ f_2 \circ \dots \circ f_N$.
Each $f_i$ is affine transformation or triangular matrix, so the determinant of $f$ is the product of those functions.

However, restricting each layer to be affine or triangular is too limiting. Instead, we only require that the **Jacobian matrix** of the overall transformation be triangular.

#### Coupling Layer
##### General Definition
Let $x \in \mathbb{R}^D$, and partition indices into $I_1, I_2$ with $|I_1| = d$. Let $m: \mathbb{R}^d \to \mathbb{R}^{D-d}$ be any function. Define
$$
y_{I_1} = x_{I_1}, \quad y_{I_2} = g(x_{I_2}; m(x_{I_1}))
$$
where $g$ is an invertible map w.r.t. its first argument given the second. Then
$$
\det \frac{\partial y}{\partial x} = \det \frac{\partial y_{I_2}}{\partial x_{I_2}}
$$
and the inverse is
$$
x_{I_1} = y_{I_1}, \quad x_{I_2} = g^{-1}(y_{I_2}; m(y_{I_1}))
$$
##### Stacking
Each layer only modifies part of the input → alternate $I_1$​ and $I_2$​ between layers

At least 3 layers needed for all dimensions to influence each other; typically 4 layers are used.

#### Rescaling
As the additive coupling layer above has $\det = 1$, we add a diagonal scaling matrix $S$ as the top layer, which multiplies the $i$-th ouput value by $S_{ii}$
$$
x_i \mapsto S_{ii}
$$
To prevent $S_{ii}$ goto infinity, the criterion of NICE is designed as
$$
\log p_X(x) = \sum_{i=1}^D \Big[ \log p_{H_i}(f_i(x)) + \log |S_{ii}| \Big]
$$
​(this scaling works like PCA, showing how much variation is present in each of the latent dimensions)

#### Prior
Some choice of distribution:
- **Gaussian**
$$
\log p_{H_d}(h_d) = -\frac{1}{2}(h_d^2 + \log 2\pi)
$$
- **Logistic**
$$  
\log p_{H_d}(h_d) = -\log(1 + \exp(h_d)) - \log(1 + \exp(-h_d))
$$
We tend to use the logistic distribution as it tends to provide a better behaved gradient.

## Application
### Inpainting
Naive iterative procedure using projected gradient ascent with Gaussian noise:
$$
x_{H,i+1} = x_{H,i} + \alpha_i \Big( \frac{\partial \log p_X((x_O, x_{H,i}))}{\partial x_{H,i}} + \epsilon \Big), \quad \epsilon \sim \mathcal{N}(0, I)
$$
- Step size: $\alpha_i = \frac{10}{100+i}$ (decreasing with iteration)
- Projection ensures $x_H$​ remains within original value range
*Observation*: The model is not specifically trained for inpainting, yet it can produce reasonable qualitative results.

## Discussion
### Advantage
- Learns a highly non-linear bijective transformation mapping data to a factorized latent space.
- Supports efficient, unbiased ancestral sampling.

### Future Works
- The architecture can be trained using other inductive principles, e.g., toroidal subspace analysis (TSA).
- Connection with variational auto-encoders (VAE): NICE can enable more powerful approximate inference, allowing:
	- More complex approximate posterior distributions
    - Richer family of priors

## Reference
```apa
Dinh, L., Krueger, D., & Bengio, Y. (2014). Nice: Non-linear independent components estimation. _arXiv preprint arXiv:1410.8516_.
```