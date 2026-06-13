---
title: Generative Adversarial Nets
properties:
  - hide
  - image
---

## Information
- *title*: Generative Adversarial Nets
- *author*: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y.
- *conference*: NeurIPS 2016
- *task*: image generation

## Overview
### Background
At that time, deep generative models have several challenges:
- *difficulty of probability estimation*: maximum likelihood estimation and related strategies, which need to approximated inference methods.
- *difficulty of leveraging the benefits of piecewise linear units* (activation functions)

### Breakthrough
GAN proposes a new frameworks:
- No approximate inference or Markov chains

### Abstract
The whole model has two components:
- *Generative Model*: generates samples by passing random noise through a multilayer perceptron.
- *Discriminative Model*: multilayer perceptrons to distinguish whether an object is generated or real.

The entire training process is like a minimax game: the generative model tries to fool the discriminative model, while the discriminative model tries to correctly classify.

After completing the training, the generative model can reconstruct the training distribution and The discriminative model assigns a probability of $\frac{1}{2}$ to being real or fake for any image.

## Methods
### Frameworks
Suppose that we want to learn the distribution $p_G(x)$ from data $x$. First, we define the prior noise distribution $p_z(z)$ (usually, researcher chooses to use Gaussian distribution).

We have two components:
- $G(z;\,\theta_g)$: map $z\ \rightarrow\ x$ 
- $D(x;\,\theta_d)$: discriminate the data point is from $data$ or $G$

The training objective is
![[截圖 2025-12-25 晚上9.40.28.png]]

In practice, since $G$ is poor in the beginning, $D$ can reject samples with high confidence because they are clearly different from the training data. This leads to $log(1 - D(G(z))) \rightarrow 0$ $\Rightarrow$ no gradient!!! Hence, we turn to let $G$ maximize $\log(D(G(z))$, which also converges to the same point.

### Training Process
![[截圖 2025-12-25 晚上9.43.14.png]]

### Analysis
#### Optimal Solution
![[截圖 2025-12-25 晚上9.44.08.png]]
> Proof
> the function $y \rightarrow a\log{y}\,+\,b\log{1 - y}$ achieves its maximum in $[0, 1]$ at $\frac{a}{a +b}$.

Using this *proposition*, we can rewrite the training objective as:
![[截圖 2025-12-25 晚上9.47.40.png]]

Therefore,
$$
\begin{aligned}
C(G)\ &=\ -\log{4}\ +\ KL\left( p_{data}\left\lVert \frac{p_{data}\ +\ p_g}{2}\right . \right)\ +\  KL\left( p_{g}\left\lVert \frac{p_{data}\ +\ p_g}{2}\right . \right)\ \\\\
&=\ -\log{4}\ +\ 2\ \cdot\ JSD(p_{data}\|\ p_g)
\end{aligned}
$$
> [!pink]
> This implies that the global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p_g = p_{data}$. At that point, $C(G)$ achieves the value $-\log 4$.
#### Convergence
![[截圖 2025-12-26 中午12.36.50.png]]

Fix $D$. Consider $V(G, D) = U(p_g, D)$ as a function of $p_g$.
Note that 
$$
\begin{aligned}
U(p_g,\,D)\ &=\ \mathbb{E}_{x\sim p_x}[\log D(x)]\ +\ \mathbb{E}_{x\sim p_g}[\log(1\, −\, D(x))] \\\\
&=\ C\ +\ \int p_g(x)\log{( 1\,-\,D(x) )}\,dx
\end{aligned}
$$
Thus, $U(p_g,\,D)$ is a convex function.
Recall the property of convex function.
> [!Abstract] Property
> Let $\{f_\alpha(x):\ \alpha \in A\}$  be a set of functions, where each function is convex. If $f(x)\ =\ \sup_{\alpha \in A}f_\alpha(x)$, then
> $$
> \partial f_\beta(x)\ \in\ \partial f(x)
> $$
> where $\beta\ =\ \arg\sup_{\alpha \in A}f_\alpha(x)$.

We return to the discussion of GAN convergence.
The objective of GAN is
$$
\sup_DU(p_g,\, D)
$$
By the above property, finding the solution is equivalent to computing a gradient descent update for $p_g$ at the optimal $D$. Therefore with sufficiently small updates of $p_g$, $p_g$ converges to $p_x$

## Experiments
use [[Parzen window-based log-likelihood estimation|Gaussian Parzen window-based log-likelihood estimation]] to evaluate the performance. 

## Discussion
### Advantages
- Does not require Markov chains or explicit inference.
- Input features are not directly copied into the generator’s parameters, preventing overfitting and memorization of training samples.
- Capable of learning very sharp distributions, including degenerate distributions.
### Disadvantages
- implicit distribution
- need to train $G, D$ together.

### Future Works
- *Conditional GAN*: By providing additional information $c$ as input to both the generator $G$ and discriminator $D$, the model can learn the conditional distribution $p(x \mid c)$.
- *Learned Approximate Inference*: Train an auxiliary inference network to predict the latent code $z$, i.e. $z \approx f_\phi(x)$.
- *Modeling all conditionals*: For any subset $S$ of the vector $x$, build a conditional model $p(x_S \mid x_{\backslash S})$.
- *Semi-supervised Learning*: Extend GANs to leverage both labeled and unlabeled data for training.

### Retrospective
#### Summary
- GANs are notoriously difficult to train.
- The training objective may cause the model to fail to converge if the generated distribution does not overlap with the real data distribution.
- Training can easily result in mode collapse.

#### My Opinion
- The value of this paper lies not in its generative performance, but in the idea of the _min-max game_.
## Reference
```apa
Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. _Advances in neural information processing systems_, _27_.
```
