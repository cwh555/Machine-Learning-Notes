---
title: Support Vector Machine
date: 2025-08-23
---
## The Optimal Hyperplane
### The Fattest Separating Hyperplane
#### Separating hyperplanes
The hyperplane $h$ separates the data if and only if it can be represented by weights $(b, w)$ that satisfy
$$
\min_{n=1,\dots,N}{\ y_n(w^Tx_n\,+\,b)}\ =\ 1
$$
#### Margin of a hyperplane
The distance from $x$ to $h$ is
$$
\begin{aligned}
dist(x,h)\ =\ \frac{|w^Tx_n\,+\,b|}{\|w\|}\ =\ \frac{y_n(w^Tx_n\,+\,b)}{\|w\|}\\\\
\implies \min_{n=1\dots N}{dist(x_n, h)}\ =\ \frac{1}{\|w\|}\ .
\end{aligned}
$$

#### Solving the fattest separating hyperplane
We need to solve the following optimization problem.
$$
\begin{aligned}
\underset{b, w}{minimize}&:\qquad \tiny\frac{1}{2}\normalsize w^Tw\\\\
subject\ to&:\qquad \min_{n=1\dots N}{\ y_n\,(w^Tx_n\,+\,b)\ =\ 1}.
\end{aligned}
$$
This is equivalent to solve the optimization problem as follows at the optimal solution.
$$
\begin{aligned}
\underset{b, w}{minimize}&:\qquad \tiny\frac{1}{2}\normalsize w^Tw\\\\
subject\ to&:\qquad \min_{n=1\dots N}{\ y_n\,(w^Tx_n\,+\,b)\ \geq\ 1}.
\end{aligned}
$$

- $\textbf{QP-Problem}$
	Standard form:
$$
\begin{aligned}
\underset{u\in \mathbb{R}^L}{minimize}&:\qquad \tiny\frac{1}{2}\normalsize u^TQ\,u\ +\ p^T u\\\\
subject\ to&:\qquad a_m^Tu\ \geq\ c_m \quad (m = 1,\,\dots,\,M).
\end{aligned}
$$
For the QP-problem to be convex, the matrix $Q$ must to be positive semi-definite.  
Specify $a_m$ as the rows of an $M\,\times\,L$ matrix $A$ and the $c_m$ as components of an $M\,\times\,1$ vector $c$ .  
The subject can be written as $Au\ \geq\ c$ .

We will write 
$$
u^*\quad\leftarrow\quad QP(Q,\,p,\,A,\,c)
$$
to denote the process of running a QP-solver to get an optimal solution $u^*$ .

<span style="color:pink; font-weight:bold;">Linear hard-margin support vector machine</span>

<div style="text-align:center;">
<img src="https://i.imgur.com/nR1DoOT.jpeg" alt="Example Image" style="height: 300px;">
</div>

### Advantages of Fat Separator
1. Large margin is better
2. Fat hyperplanes shatter fewer points
> [!green] VC dimension
> Suppose the input space is the ball of radius $R$ in $\mathbb{R}^d$ , so $\|x\|\,\leq\,R$. Then, 
> $$
> d_{VC}(\rho)\ \leq\ \lceil R^2\,/\,\rho^2 \rceil\ +\ 1\ .
> $$
> where $\rho$ is the margin of the hyperplane.

3. Bounding the cross validation error   
	If the data points other than the support vectors are removed, the resulting separator produced by the SVM is unchanged.
$$
E_{cv}(SVM)\ \leq\ \frac{\#\ support\ vectors}{N}\ .
$$

## Dual Formulation of The SVM
The dual SVM problem is equivalent to the original primal problem. However, the primal one is a QP-problem with $\tilde{d}$ variables $(\tilde{b}, \tilde{w})$ and $N$ constraints. The dual problem will also be a QP-problem, but with $N$ variables and $N + 1$ constraints.
### Lagrange Dual for a QP-Problem
<span style="color:pink; font-weight:bold;">Karush-K\"uhn-Tucker</span>
For a feasible convex QP-problem in primal form, 
$$
\begin{aligned}
\underset{u\in \mathbb{R}^L}{minimize}&:\qquad \tiny\frac{1}{2}\normalsize u^TQ\,u\ +\ p^T u\\\\
subject\ to&:\qquad a_m^Tu\ \geq\ c_m \quad (m = 1,\,\dots,\,M).
\end{aligned}
$$
define the Lagrange function
$$
\mathcal{L}(u, \alpha)\ =\ \frac{1}{2} u^T Q u\ +\ p^T u\ +\ \sum_{m=1}^{M}\,\alpha_m (c_m\,-\,a_m^T u)\ .
$$
The solution $u^*$ is optimal for the primal iff. $(u^*, \alpha^*)$ is a solution to the dual optimization problem
$$
\max_{\alpha\,\geq\,0} \min_{u}\ \mathcal{L}(u, \alpha)
$$
The optimal $(u^*, \alpha^*)$ satisfies the **Karush-Kuhn-Tucker** conditions :
$(i)\ \textit{Primal and dual constraints:}$
$$
a_m^T u^*\ \geq\ c_m\quad \text{and} \quad \alpha_m\ \geq\ 0.
$$
$(ii)\ \textit{Complementary slackness}:$
$$
\alpha^*_m\,(a_m^T\,u^*\ -\ c_m)\ =\ 0\ .
$$
$\textit{(iii) Stationarity with respect to }u :$
$$
\nabla_u\, \mathcal{L}(u,\,\alpha)|_{u = u^*,\,\alpha = \alpha^*}\ =\ 0\ .
$$

### Dual of the Hard-Margin SVM
Apply the KKT theorem to the convex QP-problem for hard-margin SVM, we have the following optimization problem to solve.
$$
\begin{aligned}
\underset{\alpha \in \mathbb{R}^{N}}{minimize}:&\qquad \frac{1}{2} \sum_{m = 1}^{N} \sum_{n = 1}^{N}\ y_n\,y_m\,\alpha_n\,\alpha_m\,x_n^T\,x_m\ -\ \sum_{n = 1}^N\ \alpha_n\\\\

\text{subject to :}&\qquad \sum_{n = 1}^N\ y_n\,\alpha_n
\ =\ 0\\\\
&\qquad \alpha_n\ \geq\ 0 \qquad\qquad(n\ =\ 1,\dots\,,N\,)\ .
\end{aligned}
$$

### SVM from the Dual Solution
By solving the dual problem, 
$$
w^*\ =\ \sum_{n = 1}^N\ y_n\,\alpha^*_n\,x_n\ .
$$
Assume that the data contains at least one positive and one negative example. Then, at least one of the $\alpha^*_s$ must be strictly positive. As a result, $y_s\,(w^{*T}x_s\ +\ b^*)\ =\ 1$ .
We can solve $b^*\, =\, y_s\, -\, w^{*T}x_s$ .

Note that the optimal hypothesis is $g(x)\ =\ sign\left( w^{*T}x\ +\ b^* \right)$ .
<div style="text-align:center;">
<img src="https://i.imgur.com/7FNDv7j.jpeg" alt="Example Image" style="height: 500px;">
</div>

We get a much better bound on $E_{cv}$
$$
E_{cv}\ \leq\ \frac{\text{number of }\alpha^*_m\ \gt\ 0}{N}\ .
$$
## Kernel Trick for SVM
### Introduction
Consider a linear transform $\Phi :\ \mathcal{X}\ \rightarrow\ \mathcal{Z}$ , which can be done by replacing $x$ by $z\ =\ \Phi(x)$ in the algorithm before. Throughout the procedure, the only step related to $\Phi$ is $z^Tz'$ in the final hypothesis. As a result, we may define a function that both combines the transform and the inner product. 
$$
K_{\Phi}(x,\,x')\ \equiv\ \Phi(x)^T\,\Phi(x')\ .
$$
This function is called a <span style="color:pink; font-weight:bold;">kernel function</span> . Finding a kernel function is just like using a transformation $\Phi$ . Moreover, the efficiency of kernel would be related to $d$ , rather than the dimension $\tilde{d}$ when we transform to $\mathcal{Z}$ explicitly.
<div style="text-align:center;">
<img src="https://i.imgur.com/7q1TDkA.jpeg" alt="Example Image" style="height: 500px;">
</div>

### Polynomial Kernel
For example, consider the second-order polynomial transform:
$$
\Phi_2(x)\ =\ (1, x_1, x_2,\,\dots,\,x_d,\, x_1x_1,\, x_1x_2,\,\dots,\,x_dx_d)\ .
$$
We may calculate $\Phi_2(x)^T\,\Phi_2(x')$ by an equivalent function
$$
K(x,\,x')\ =\ 1\ +\ (x^Tx')\ +\ (x^Tx')^2\ .
$$
- <span style="color:pink; font-weight:bold;">Degree-Q Polynomial Kernel</span>
$$
K(x,\,x')\ =\ (\zeta\ +\ \gamma x^T x')^Q\ ,
$$
where $\gamma\ >\ 0,\ \zeta\ >\ 0,\ \text{and}\ Q\,\in\,\mathbb{N}\ .$

<span class = 'lime'>The kernel is typically used only with</span> $Q\ \leq\ 10$ .

- <span style="color:pink; font-weight:bold;">Linear Kernel</span>
This is the special case of a polynomial kernel with $Q\,=\,1$, $\gamma = 1$, and $\zeta = 0$ .

**Advantages** : 
the value of $w^*$ can carry some explanation on how the prediction is made.

**Disadvantages**: 
Inability to produce a sophisticated boundary.
### Gaussian-RBF Kernel
$$
K(x,\,x')\ =\ \exp{\left(\ -\gamma \|x\ -\ x'\|^2\ \right)}\ .
$$
which is equivalent to an inner product in a feature space defined by the nonlinear, infinite-dimensional transformation 
$$
\Phi(x)\ =\ \exp{(-x^2)}\ \cdot\ \left( 1,\,\sqrt{\frac{2^1}{1!}}\,x,\,\sqrt{\frac{2^2}{2!}}\,x^2,\,\sqrt{\frac{2^3}{3!}}\,x^3,\,\dots \right)
$$

<span class='lime'>The width</span> $\gamma$ <span class = 'lime'>is universally chosen in the interval [0, 1].</span>

**Disadvantages**    
The resulting hypothesis can only be expressed by the support vectors rather than the actual hyperplane. 

### Kernel Choices
One may define the <span style="color:pink; font-weight:bold;">Gram matrix</span> by
$$
K\ =\ \begin{bmatrix}
K(x_1,\,x_1) & K(x_1,\,x_2) & \dots & K(x_1,\,x_N) \\
K(x_2,\,x_1) & K(x_2,\,x_2) & \dots & K(x_2,\,x_N) \\
\dots & \dots & \dots & \dots\\
K(x_N,\,x_1) & K(x_N,\,x_2) & \dots & K(x_N,\,x_N)
\end{bmatrix}
$$
Note that the matrix should be positive semi-definite. The matrix should satisfies <span style="color:pink; font-weight:bold;">Mercer's condition</span>
> [!gray] Mercer's condition
> $K(x,\, x')$ is a valid kernel function iff. the kernel matrix $K$ is always symmetric PSD for any given $\{x1_1,\,\dots,\,x_N\}$.

## Soft-Margin SVM
### Concept
Introduce an amount of margin violation $\xi_n\ \geq\ 0$ for each data point $(x_n, y_n)$ and require that $y_n (w^Tx_n\ +\ b)\ \geq\ 1\ -\ \xi\ .$  

The soft-margin optimization problem is :
$$
\begin{aligned}
\min_{w, b, \xi}\qquad &\frac{1}{2}w^T w\ +\ C\,\sum_{n = 1}^N\,\xi_n\\
\text{subject to}\qquad &y_n(w^Tx_n\ +\ b)\ \geq\ 1\ -\ \xi_n\ for\ n\,=\,1,\,2,\,\dots,\,N\,;\\
&\xi_n\ \geq\ 0\ for\ n\,=\,1,\,2,\,\dots,\,N\ .
\end{aligned}
$$
### Dual Problem
The Lagrange function is 
$$
\mathcal{L}(b,\,w,\,\xi,\,\alpha,\,\beta)\ =\ \frac{1}{2} w^T w\ +\ C\sum_{n = 1}^N\,\xi_n\ +\ \sum_{n = 1}^N\,\alpha_n(1\, -\, \xi_n\, -\, y_n(w^T x_n\,+\,b))\ -\ \sum_{n=1}^N\beta_n\xi_n
$$

Using the $\textbf{KKT condition}$ , the Lagrange dual problem simplifies to 
$$
\max_{\substack{\alpha\,\geq\,0\ \beta\,\geq\,0\\ \alpha_n\ +\ \beta_n\ =\ C}}\quad \min_{b, w, \xi} \quad \mathcal{L}(b, w, \xi, \alpha)
$$
where $\mathcal{L}\ =\ \frac{1}{2} w^T w\ +\ \sum_{n = 1}^N\,\alpha_n(1\,-\,y_n(w^Tx_n\,+\,b))$ .

The dual problem is
$$
\begin{aligned}
\min_{\alpha} \qquad &\frac{1}{2} \alpha^T Q_D\, \alpha\ -\ 1^T \alpha\\\\
\text{subject to } \qquad &y^T\,\alpha\ =\ 0\,;\\
&0\ \leq\ \alpha\ \leq\ C \cdot 1\ .
\end{aligned}
$$
The support vector with $0\ <\ \alpha_n^*\ <\ C$ are called the free support vectors, which are guaranteed to be on the boundary of the fat-hyperplane.
The support vector with $\alpha^*_n\ =\ C$ are called the bounded support vectors. Anything can happen to them.

### Viewpoint in Regularization
$$
E_{SVM}(b,\,w)\ =\ \frac{1}{N} \sum_{n = 1}^N\,max\{\ 1 - y_n(w^Tx_n\ +\ b),\, 0\ \}\ .
$$
Minimize the soft-margin SVM can be re-written as the following optimization problem 
$$
\min_{b,\,w}\ \lambda w^T w\ +\ E_{SVM}(b,\,w)
$$
subjects to the constraints, and where $\lambda\ =\ \frac{1}{2}CN$ .

### Summarize
1. Deliver a large-margin hyperplane, and in so doing it can control the effective model complexity.
2. Deal with high or infinite-dimensional transforms using the kernel trick.
3. Express the final hypothesis using only a few support vectors, their corresponding Lagrange multipliers, and the kernel.
4. Control the sensitivity to outliers, the regularize the solution through setting C appropriately.
