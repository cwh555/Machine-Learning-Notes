---
title: "VQ-VAE-2: Generating Diverse High-Fidelity Images"
properties:
  - hide
---
## Information
- _title_: VQ-VAE-2: Generating Diverse High-Fidelity Images  
- _authors_: Razavi, A., van den Oord, A., & Vinyals, O.  
- _conference_: NeurIPS 2019  
- *task*: Representation Learning

## Overview
### Background
Existing image generation methods either suffer from slow sampling in pixel space or instability and mode collapse in GANs, especially at large scale.
### Breakthrough
Introduce a multi-scale hierarchical VQ-VAE with powerful latent priors to enable fast, high-fidelity large-scale image generation.

### Abstract
Improving VQ-VAE
- multi-scale hierarchical organization: multilayer codebook
- stronger autoregressive prior (PixelCNN + self-attention)

## Methods
This paper use the exponential moving average updates for the codebook, so the loss function does not have the second term.

### Stage 1: Learning Hierarchical Latent Codes
![[截圖 2025-12-29 凌晨12.34.32.png#center|400]]

Well, just hierarchical latent codes.

### Stage 2: Priors over Latent Codes
After Stage 1, we have
- encoder：image → hierarchical discrete latent
- decoder：latent → image
Therefore, the whole model can only do reconstruction and compression. However, we want the model learn to generation. This requires the priors. Thus, in stage 2, we let the model learn priors.

![[截圖 2025-12-29 凌晨12.37.20.png#center|400]]

The goal is to learn $p(z_\text{top},\,z_\text{bottom})$. This paper used a hierarchical prior factorization:
$$
p(z_{\text{top}}, z_{\text{bottom}}) = p(z_{\text{top}}) \cdot p(z_{\text{bottom}} \mid z_{\text{top}})
$$
#### Top-level prior
We want the model learn global structure.

The author $p(z_{\text{top}})$ using PixelCNN to learn the top-level prior. To let the model capture correlations that are far part in the mage, we uses self-attention to capture long-range spatial dependencies. This is feasible since it only operates on 32×32 latent grid.

#### Bottom-level prior
We want the model learn local details.

The authors use deep residual conditioning from top-level latents, operating on 64×64 latent grid. Thus, no self-attention is in the model (too expensive at this resolution).

### Classifier Based Rejection Sampling
This is a technique to reject "bad" sampling.

Since the models trained with maximum likelihood are forced to model all of the training data distribution, thus the sampling will cover all modes makes learning harder and autoregressive sampling accumulates errors.   
$\Rightarrow$ Some generated samples are low quality but still valid.

The authors proposed an automated method for trading off diversity and quality of samples. The intuition is that the closer our samples are to the true data manifold, the more likely they are classified to the correct class labels by a pre-trained classifier.

## Discussion
### Limitations
- VQ‑VAE‑2 still **relies on autoregressive priors**, which can be **slow to sample** relative to diffusion models or GANs.
- Evaluation still relies on visual inspection due to metric shortcomings.
- Latent sampling quality depends on good priors → prior modeling remains challenging.

## Reference
```apa
Razavi, A., Van den Oord, A., & Vinyals, O. (2019). Generating diverse high-fidelity images with vq-vae-2. _Advances in neural information processing systems_, _32_.
```