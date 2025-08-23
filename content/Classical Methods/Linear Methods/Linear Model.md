---
title: Linear Model
date: 2025-08-21
---
We present core supervised learning methods—linear classification, linear regression, and logistic regression—and discuss how nonlinear feature transforms can enhance model flexibility.
## Linear Classification
Linear classification aims to separate data points using a linear decision boundary. Here, we introduce the linear model, discuss when it works well, and present algorithms like the Perceptron and Pocket Algorithm to find effective classifiers.
### Model
We use a hypothesis set of linear classifiers, where each h has the form
$$
h(x)\ =\ sign(w^\top x)
$$
for some $w\ \in R^{d+1}$, where $d$ is the dimensionality of the input space.
Note that we add coordinate $x_0\ =\ 1$ corresponds to the bias 'weight' $w_0$

### Feasibility
1. Generalization Bound  
   With high probability
>[!green] VC generalization bound 
>$$
>E_{out}(g)\ =\ E_{in}(g)\, +\, \mathcal{O} \left( \sqrt{\frac{d}{N}\ln N}\ \right)
>$$

2. $E_{in}$ can be small  
   If there is a linear hypothesis that has small $E_{in}$ , then PLA algorithm can help us find it.

### Algorithm
Because of noise, rather to solve $E_{in}\ =\ 0$, we would like to find a hypothesis with the minimum $E_{in}$ , that is, we need to solve the combinatorial optimization problem:
$$
\min_{w\, \in\, \mathbb{R^{d+1}}} \frac{1}{N}\ \sum_{n\, =\, 1}^{N} \left[\!\left[ sign(w^\top x_n)\ \neq\ y_n \right]\!\right] 
$$
The pocket algorithm can deal with this problem.
> [!abstract] **The Pocket Algorithm**
> <div style="text-align:center;">
> <img src="https://i.postimg.cc/0y0jByCr/2025-08-21-8-10-32.png" alt="Example Image">
> </div>



**Remark**  
The Perceptron Learning Algorithm can only handle linearly separable data, in which case it will eventually converge. The Pocket Algorithm can be viewed as its extension to deal with data that are not linearly separable.
> [!abstract] **Perceptron Learning Algorithm**
> 
> <div style="text-align:center;">
> <img src="https://i.postimg.cc/x11V9j2X/2025-08-21-8-13-45.png" alt="Example Image">
> </div>

## Linear Regression
Linear regression models the relationship between inputs and a continuous output using a linear function. In this section, we introduce the linear model, discuss conditions for a feasible solution, and present algorithms to compute the optimal weights, including both closed-form and pseudo-inverse methods.

### Model
We consider the hypothesis set where each model has the following form.
$$
h(x)\ =\ \sum_{i\,=\,0}^d w_ix_i\ =\ w^\top x
$$
where $x_0\ =\ 1$ and $x\ \in\ {1} \times\ \mathbb{R}^{d}$, $w\ \in\ \mathbb{R}^{d+1}$

### Feasibility
>[!green] VC generalization bound 
>$$
>E_{out}(g)\ =\ E_{in}(g)\, +\, \mathcal{O} \left( \frac{d}{N} \right)
> $$


### Algorithm
#### Analysis
$$
\begin{aligned}
	&E_{out}(h)\ =\ \mathbb{E} \left[ (h(\mathbf{x})\ -\ y)^2\right] \\
	\\
	&E_{in}\ =\ \frac{1}{N} \sum_{n\,=\,1}^{N} \left( h(\mathbf{x_n}\ -\ y_n)^2 \right)
\end{aligned}
$$

Define the data matrix $X \in\ \mathbb{R}^{N\,\times\,(d+1)}$  whose rows are the inputs $x_n$ as row vectors.
Define the target vector $y\ \in\ \mathbb{R}^N$.
$$
E_{in}\ =\ \frac{1}{N}\left( w^\top X^\top Xw\,-\,2w^\top X^\top y\,+\,y^\top y \right)
$$
We want to solve the optimization problem:
$$
w_{lin}\ =\ \underset{w\,\in\,\mathbb{R}^{d+1}}{\arg\min}\,E_{in}(w)
$$
To get $\nabla E_{in}(w)\ =\ 0$, one should solve for $w$ satisfies
$$
X^\top Xw\ =\ X^\top y
$$
If $X^\top X$ is invertible, the solution is easy to obtain.
>[!gray] Solution for invertible matrix
>$$
>w\ =\ X^†y,\ \text{where}\ X^†\ =\ (X^\top X)^{-1}X^\top \ \text{is the}\ pseudo-inverse\ \text{of X}
>$$

Otherwise, if $X^\top X$ is not invertible, we have to resort to Singular Value Decomposition (SVD).
Let $\rho$ be the rank of $X$.
Assume that the SVD of $X$ is $X\ =\ U\Gamma V^\top$, where
$$
\begin{aligned}
U\ \in\ \mathbb{R}^{N \times \rho}\quad &\text{satisfies}\quad UU^\top\ =\ I_{\rho} \\
V\ \in\ \mathbb{R}^{(d+1)\,\times\,\rho}\quad &\text{satisfies}\quad V^\top V\ =\ I_{\rho} \\
\Gamma\ \in\ \mathbb{R}^{\rho \times \rho}\quad &\text{is a positive diagonal matrix}
\end{aligned}
$$
Then,
> [!gray] Solution for non-invertible matrix
> $$
> w_{lin}\ =\ V\,\Gamma^{-1}U^\top\mathbf{y}\quad\quad \text{is a solution}
> $$

#### Pseudo Code
<div style="text-align:center;">
<img src="https://i.postimg.cc/G3FWYT3B/2025-08-21-8-15-17.png" alt="Example Image">
</div>

#### Others : hat matrix
Here, we introduce the definition of <span style="color:pink; font-weight:bold;">hat matrix</span>.
$$
H\ =\ X\left( X^\top X\right)^{-1}X^\top 
$$
This matrix has the properties that $H^2\ =\ H$, which can facilitate the analysis of error.

## Logistic Regression
Logistic regression is a fundamental method for binary classification, estimating the probability that a given input belongs to a particular class. In this section, we introduce the logistic model, discuss how to measure prediction error using maximum likelihood and cross-entropy, and present algorithms such as batch and stochastic gradient descent for learning the model parameters.

### Description
Given a data set $\mathcal{X}$  with data points $(x,\,y)$ , where $x$ is a vector and $y\ \in\ \{-1,\,1\}$. The logistic regression model is designed to estimate the probability $\mathbb{P}[y\ =\ 1\,|\,x]$ for a given input $x$.

### Model
We consider the hypothesis set where each model has the following form.
$$
h(\mathbf{x})\ =\ \theta(w^\top x)
$$
where $\theta$ is the so-called <span style="color:pink; font-weight:bold;">logistic</span> function $\theta(s)\ =\ \frac{e^s}{1\,+\,e^s}$ .

Another popular soft threshold is the <span style="color:pink; font-weight:bold;">hyperbolic tangent</span>
$$
\tanh{s}\ =\ \frac{e^s\,-\,e^{-s}}{e^s\,+\,e^{-s}}
$$

### Algorithm
#### Error measure
We are trying to learn the target function
$$
f(\mathbf{x})\ =\ \mathbb{P}[y\ =\ +1\,|\,\mathbf{x}]
$$
However, the given data set is generated by a noisy target function
$$
P(y\,|\,x)\ =\ 
\begin{cases}
f(x)\quad,\ &if\ y\ =\ +1;\\ \\
1\ -\ f(x), &if\ y\ =\ -1.
\end{cases}
$$
Hence, the target distribution captured by our hypothesis $h(x)$ is
$$
P(y\,|\,x)\ =\ 
\begin{cases}
h(x)\quad,\ &if\ y\ =\ +1;\\ \\
1\ -\ h(x), &if\ y\ =\ -1.
\end{cases}
$$
We would like to maximum the product
$$
\prod_{n\,=\,1}^{N} P(y_n\,|\,x_n)
$$
The method of [[Maximum Likelihood Estimation|maximum likelihood]] selects the hypothesis $h$ which maximizes this probability. Below, there are two equivalent error measurement.

1.  For the reason of simplicity, we solve an equivalent problem: 
$$
E_{in}(w)\ =\ \frac{1}{N}\sum_{n\,=\,1}^{N}\ln{\left( \frac{1}{P(y_n\,|\,x_n)}\right)}\ =\ \frac{1}{N}\sum_{n\,=\,1}^{N} \ln{\left(1\ +\ e^{-y_nw^\top x_n}\right)}
$$
2. <span style="color:pink; font-weight:bold;">Cross-entropy error measure</span>
<br>The maximum likelihood method reduces to the task of finding $h$ that minimizes
$$
E_{in}(\mathbf{w})\ =\ \sum_{n\,=\,1}^{N}\left[\!\left[ y_n\ =\ +1\right]\!\right]\ln{\frac{1}{h(x_n)}}\ +\ \left[\!\left[ y_n\ =\ -1\right]\!\right]\ln{\frac{1}{1\ -\ h(x_n)}}
$$

>[!blue] Discussion
> $$
> \nabla E_{in}(w)\ =\ -\frac{1}{N} \sum_{n\,=\,1}^{N} \frac{y_nx_n}{1\ +\ e^{y_xw^\top x_n}}
> $$
> A misclassified example contributes more to the gradient than a correctly classified one.


#### Gradient descent
Gradient descent is a method used to minimize $E_{in}(w)$. It may not reach the global minimum, but the local minimum it converges to is usually small enough.
1. <span style="color:pink; font-weight:bold;">Batch gradient descent</span>
<div style="text-align:center;">
<img src="https://i.postimg.cc/gjg4sWB7/2025-08-21-8-16-58.png" alt="Example Image">
</div>

Note that the <span style="color:pink; font-weight:bold;">learning rate</span> have to be specified. $\textcolor{lime}{\text{A good choice for }\eta\text{ is around 0.1}\,.}$

2. <span style="color:pink; font-weight:bold;">Stochastic gradient descent (SGD)</span>
This is a more efficient method for performing gradient descent.
<div style="text-align:center;">
<img src="https://i.postimg.cc/v80vpG3q/2025-08-21-8-17-42.png" alt="Example Image">
</div>

- $\textbf{Explanation}$
	Consider the expected weight change, we may find that this is exactly the same as the deterministic weight.
- $\textbf{Termination}$
	A good stopping criterion should consider the total error on all the data.
#### Pseudo Code
<div style="text-align:center;">
<img src="https://i.postimg.cc/pT2D6S8M/2025-08-21-8-18-12.png" alt="Example Image" style="height: 220px;">
</div>

- **Initialization**
<span class = 'lime' style="display: block; margin-left: 20px;">Choosing each weight independently from a Normal distribution with zero mean and small variance usually works well in practice.</span>

- **Termination**
<span class = 'lime' style="display: block; margin-left: 20px;">A maximum number of iterations, marginal error improvement, coupled with small value for the error itself works reasonably well.</span>
	
## Nonlinear Transformation

We can view the hypothesis as a linear one after applying a nonlinear transformation on $\mathbf{x}$ .
The transform $\Phi$ that takes us from $\mathcal{X}$ to $\mathcal{Z}$ (<span style="color:pink; font-weight:bold;">feature space</span>) is called a <span style="color:pink; font-weight:bold;">feature transform</span>.

For example, the feature transform $\Phi_Q$ for degree-Q curves in $\mathcal{X}$ is called the <span style="color:pink; font-weight:bold;">Q-th order polynomial transform</span>.

> [!green] VC dimension
> If the transform $\Phi_Q$ maps a two-dimensional vector $\mathbf{x}$ to $\tilde{d}\ =\ \frac{Q(Q\,+\,3)}{2}$ dimensions in $\mathcal{Z}$. 
> $$
> d_{VC}\ =\ \frac{Q(Q\,+\,3)}{2}\ +\ 1
> $$

> [!blue] Discussion
> Consider the following feature transform, which maps a $d$-dimensional $\mathbf{x}$ to a one-dimensional $\mathbf{z}$, keeping only the k-th coordinate of $\mathbf{x}$ .
> $$
> \Phi_k(x)\ =\ (1,\,x_k)
> $$
> The hypothesis set $\mathcal{H}$ is called the <span style="color:pink; font-weight:bold;">Decision stump model</span> on dimensional k.








