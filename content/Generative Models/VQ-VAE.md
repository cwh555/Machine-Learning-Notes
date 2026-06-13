---
title: VQ-VAE
properties:
  - hide
  - image
---
## Information
- _title_: VQ-VAE: Neural Discrete Representation Learning  
- _authors_: van den Oord, A., Vinyals, O., & Kavukcuoglu, K.  
- _conference_: NeurIPS 2017  
- *task*: Representation Learning

## Overview
### Background
- Learning useful representations without supervision remained a key challenge in machine learning.
- Traditional VAEs suffer from posterior collapse, where latent variables are ignored when paired with a powerful decoder.
### Breakthrough
- Learns a prior instead of using a static one, mitigating collapse and enabling effective representation learning.

### Abstract
*Methods*
- The encoder network outputs discrete codes, implemented via vector quantization (VQ).  
  The discrete latent space effectively captures the main features of the data.
- The prior is learned rather than static.

*Applications*
- Generative tasks
- Unsupervised representation learning
- Speaker conversion
- Reinforcement learning, e.g., environment modeling or planning/prediction

## Methods
![[截圖 2025-12-28 下午4.06.13.png#center|600]]

### Discrete Latent Variables
Define a latent embedding space $e \in \mathbb{R}^{K \times D}$
- $K$ = number of embeddings (categorical size)
- $D$ = dimension of each embedding vector $e_i$
Each embedding vector $e_i$ is learnable.

*Forward*:   
1. Input $x$  is passed through an encoder producing output $z_e(x)$ (continuous representation).
2. The discrete latent variables $z$ are then calculated by a nearest neighbor.
![[截圖 2025-12-28 下午4.13.24.png]]
3. The posterior categorical distribution $q(z|\,x)$ probabilities are defined as one-hot as follows:
![[截圖 2025-12-28 下午4.13.57.png]]

In this way, the KL divergence $KL[q(z|\,x)\|\,p(z)]\ =\ \log K$ is a constant.

### Training 
Because the discrete mapping prevents standard gradient descent, one approach is to copy the decoder’s gradients to the encoder.

This way, the gradients carry information about how to adjust the encoder’s output to reduce the reconstruction loss.

We define the loss function as:
$$
\begin{aligned} L &= \underbrace{\log p(x \mid z_q(x))}_{\text{Reconstruction Loss}} + \underbrace{\| \text{sg}[z_e(x)] - e_k \|^2}_{\text{Vector Quantization Loss / Embedding update }} + \underbrace{\beta \| z_e(x) - \text{sg}[e_k] \|^2}_{\text{Commitment Loss}} \end{aligned}
$$
- **Reconstruction Loss:** optimizes decoder and indirectly encoder (via straight-through estimator).
- **Vector Quantization (VQ) Loss:** updates embeddings $e_k$​ toward encoder outputs $z_e(x)$.
- **Commitment Loss:** ensures encoder outputs commit to an embedding and prevents uncontrolled growth of encoder outputs.   

(paper choose $\beta = 0.25$)

The optimization of each parameters are via:
- Decoder → reconstruction loss only
- Encoder → reconstruction + commitment loss
- Embeddings → VQ loss only

#### Multiple Latent
In practice, $N$ discrete latents are used. (e.g., 32×32 for ImageNet, 8×8×10 for CIFAR10)

Note that the latent variable we described before is single latent variable. For multiple latent variable, it uses a grid of such discrete latents
$$
z \in \mathbb{R}^{H \times W \times D} \quad \text{with each position taking one of } K \text{ embeddings}
$$
- Here $H \times W$ = number of discrete latent positions.

Each latent has its own embedding and participates in the same VQ-VAE training with the three-part loss. The total loss is averaged over all latent positions.

The exact log-likelihood would be:
$$
\log p(x) = \log \sum_k p(x \mid z_k) p(z_k)
$$
where the sum is over all possible latent assignments $z_k$.

Because the decoder is trained with MAP inference, it effectively assigns almost all probability mass to the discrete latents $z_q(x)$ obtained from the encoder.

Therefore, the log-likelihood can be approximated by:
$$
\log p(x) \approx \log p(x \mid z_q(x))\, p(z_q(x))
$$
Jensen’s inequality ensures this is a lower bound:
$$
\log p(x) \ge \log p(x \mid z_q(x))\, p(z_q(x))
$$
### Prior
During training: a uniform distribution is used for $p(z)$.
During data generation: $p(z)$ can be designed as autoregressive:
- First, fit an autoregressive prior $p(z)$ over the trained VQ latents $z$
    - Images → PixelCNN
    - Raw audio → WaveNet
- Use ancestral sampling to sequentially generate latent $z$
- Finally, decode $z_q(x)$ back into data xxx using the VQ-VAE decoder

## Discussion
### Conclusion
this is the first discrete latent variable model that can successfully model long range sequences and fully unsupervisedly learn high-level speech descriptors that are closely related to phonemes.

### My Idea
a learnable $k$-nn

## Reference
```apa
Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. _Advances in neural information processing systems_, _30_.
```