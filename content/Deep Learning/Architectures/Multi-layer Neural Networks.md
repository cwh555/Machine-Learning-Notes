---
title: Multi-layer Neural Networks
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


