---
title: Multilayer Neural Networks
date: 2025-08-23
---
## Forward propagation
<div style="text-align:center;">
<img src="https://i.imgur.com/n27vfPs.jpeg" alt="Example Image" style="height: 400px;">
</div>

## Backpropagation Algorithm
We want to compute the gradient $\nabla E_{in}(w)$, the weight vector $w$ contains all the weight matrices $W^{(1)},\,\dots,\,W^{(L)}$. Let $e_n\, =\, e\left(h(x_n),\,y_n \right)$.
$$
E_{in}(w)\ =\ \frac{1}{N}\sum_{n=1}^{N}e_n\ .
$$
$$
\frac{\partial E_{in}}{\partial W^{(l)}}\ =\ \frac{1}{N} \sum_{n=1}^{N}\,\frac{\partial e_n}{\partial W^{(l)}}\ .
$$
Define the <span style="color:pink; font-weight:bold;">sensitivity vector</span> for layer $l$, which is the sensitivity (gradient) of the error $e$ with respect to the input signal $s^{(l)}$ that goes into the layer $l$. 
$$
\delta^{(l)}\ =\ \frac{\partial e}{\partial s^{(l)}}\ \implies\ \frac{\partial e}{\partial W^{(l)}}\ =\ x^{(l - 1)}\left( \delta ^{(l)}\right) ^\top\ .
$$
<div style="text-align:center;">
<img src="https://i.imgur.com/wBWX41N.jpeg" alt="Example Image" style="height: 300px;">
</div>


$$
\delta^{(l)}\ =\ \theta'\left( s^{(l)} \right)\otimes \left[ W^{(l+1)} \delta^{(l+1)} \right]_{1}^{d^{(l)}}
$$
where the vector $\left[ W^{(l+1)} \delta^{(l+1)} \right]_{1}^{d^{(l)}}$ contains components $1,\,\dots,\,d^{(l)}$ of the vector $W^{(l+1)}\delta^{(l+1)}$ (excluding the bias component which has index 0).
$\otimes$ denotes the component-wise multiplication.

<div style="text-align:center;">
<img src="https://i.imgur.com/Jwek6Y0.jpeg" alt="Example Image" style="height: 400px;">
</div>

### Compute $E_{in}(w)$ and $\nabla E_{in}(w)$ 

<div style="text-align:center;">
<img src="https://i.imgur.com/RvF66pw.jpeg" alt="Example Image" style="height: 300px;">
</div>

$G^{(l)}(x_n)$ is the gradient on data point $x_n$. The weight update for a single iteration of fixed learning rate gradient descent is $W^{(l)}\ \leftarrow\ W^{(l)}\,-\,\eta \,G^{(l)}$ , for $l\ =\ 1,\dots,\,L$ .

### Initialization and Termination
One good choice is to initialize using <span class = 'lime'>Gaussian random weights</span>, $w\ ~\ N(0, \sigma^2_w)$, where $\sigma_w^2$ is small. Since $\mathbb{E}[|w^Tx_n|^2]\ =\ \sigma_w^2\|w_x\|^2$ , we should choose $\sigma_w^2\, \cdot\, max_n\|x_n\|^2 \ll 1$ .

<span class = 'lime'>Stopping only when there is marginal error improvement coupled with small error plus an upper bound on the number of iterations.</span>

## A Greedy Deep Learning Algorithm
<div style="text-align:center;">
<img src="https://i.imgur.com/nEPAO8b.jpeg" alt="Example Image" style="height: 300px;">
</div>

We greedily pick the first layer weights, fix them, and then move on to the second layer weights.

- <span style="color:pink; font-weight:bold;">Unsupervised Auto-encoder</span> 
	To reconstruct the input itself: the output wi. l be $\hat{x}_n$ , a prediction of the input $x_n$ , and the error is the difference between the two.
	We dissect the input-space itself into pieces that are representative of the learning problem.
- <span style="color:pink; font-weight:bold;">Supervised Deep Network</span>  
	A more direct approach is to train the two-layer network on the targets. In this case, the output is the predicted target $\hat{y}_n$ and the error measure $e_n(y_n, \hat{y}_n)$ would be computed in the usual way.

<span class = 'lime'>A common tactic is to use the unsupervised auto-encode first to set the weights and then fine tune the whole network using supervised learning.</span>


## Hidden Units
### Rectified Linear Unit (ReLU)
$$
g(z)\ =\ \max\{ 0,\,z \}
$$
One drawback is that they cannot learn via gradient-based methods on examples for which their activation is zero.

Three generalizations are based on using a non-zero slope $\alpha_i$ when $z_i < 0$
$$
h_i\ =\ g(x,\,\alpha)_i\ =\ \max(0,\,z_i)\ +\ \alpha_i\min(0,\,z)
$$
- **Absolute value rectification**: $\alpha_i\ =\ -1\ \implies\ g(z)\ =\ |z|$.
- **Leaky ReLU**: fixes $\alpha_i$ to small value, like $\alpha_i\ =\ 0.01$
- **Parametric ReLU**: treat $\alpha_i$ as a learnable parameter

#### Maxout Units
Maxout units divide $z$ into groups of $k$ values.
$$
g(z)_i\ =\ \max_{j\,\in\,\mathbb{G}^{(i)}}z_j
$$
where $\mathbb{G}^{(i)}$ is the set of indices into the inputs for group $i$.
> For example, $z=[1.2,−0.5,3.0,0.9,2.1,−1.1]$ and we divide to three group.
> $$
> G^{(1)}=\{0,1\},\ G^{(2)}=\{2,3\},\ G^{(3)}=\{4,5\}
> $$
> Then 
> $$
> g(z)\,=\,[\max(1.2,−0.5),\,\max(3.0,0.9),\,\max(2.1,−1.1)]\,=\,[1.2,3.0,2.1]
> $$

### Softplus
$$
g(a)\ =\ \zeta(a)\ =\ \log(1\,+\,e^a)
$$
However, it does not work well empirically.

### Hard tanh
$$
g(a)\ =\ \max(-1,\,\min(1, a))
$$

## Universal Approximation Properties
[[Theory#Universal Approximation Theorem]]

*Montufar et al.*:   
The number of linear regions carved out by a deep rectifier network with $d$ inputs, depth $l$, and $n$ units per hidden layer, is
$$
\mathcal{O}\left( \binom{n}{d}^{d(l - 1)}n^d \right)
$$
In the case of maxout networks with $k$ filters per unit, the number of linear regions is
$$
\mathcal{O}\left( k^{(l - 1) + d} \right)
$$
