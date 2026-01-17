---
title: Models
---
## AutoEncoding Generative Models
### Variational Autoencoder Family
#### Variational AutoEncoder (VAE)
- *title*: (2014) **Auto-Encoding Variational Bayes**
- *note*: [[VAE]]

Auto-Encoding Variational Bayes learns latent-variable generative models by maximizing the ELBO using a neural encoder–decoder architecture and the reparameterization trick for low-variance gradient estimation. This enables scalable approximate inference for continuous latent variables with stochastic optimization, but is limited by the need for reparameterizable posteriors and simple prior assumptions.
#### Conditional Variational AutoEncoder (CVAE)
- *title*: (2015) **Learning Structured Output Representation using Deep Conditional Generative Models**
- *note*: [[CVAE]]

Deep Conditional Generative Models (CVAE) introduce Gaussian latent variables to capture multi-modal structured outputs, overcoming deterministic supervised learning’s mode-averaging. While CVAE learns diverse outputs using yyy during training, GSNN and hybrid objectives fix the training–testing mismatch, and multi-scale prediction with input noise improves pixel-level tasks, inspiring reparameterization for other structured predictions.
### Discrete Latent Autoencoders
#### VQ-VAE
- *title*: (2017) **VQ-VAE: Neural Discrete Representation Learning**
- *note*: [[VQ-VAE]]

VQ-VAE maps inputs to a discrete latent space via vector quantization and learns a decoder to reconstruct the data, enabling effective unsupervised representation learning while avoiding posterior collapse. The learned latents can also be paired with autoregressive priors for generative tasks such as image, audio, or video synthesis.
#### VQ-VAE-2
- *title*: (2019) **VQ-VAE-2: Generating Diverse High-Fidelity Images**
- *note*: [[VQ-VAE-2]]

VQ-VAE-2 extends VQ-VAE by using a hierarchical multi-scale latent space and powerful autoregressive priors to model both global structure and local details, enabling high-fidelity and diverse image generation. Classifier-based rejection sampling can be applied to trade off diversity and quality, while the model remains fully likelihood-trained and avoids GAN-like mode collapse.

## GAN Series
### GAN
- *title*: (2014) **Generative Adversarial Nets**
- *note*: [[GAN]]

GANs use a generator to produce samples from random noise and a discriminator to distinguish real from generated data in a minimax game. This enables learning complex, implicit distributions without approximate inference, but training is difficult and prone to mode collapse or convergence failure.
### DCGAN
- *title*: (2016) **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**
- *note*: [[DCGAN]]

DCGAN uses a fully convolutional generator and discriminator with batch normalization and specific activations to learn meaningful features from unlabeled data in a stable adversarial training framework. This enables semantic and vector-structured image generation, but training can still suffer from mode collapse and slower convergence on certain datasets.
### WGAN
- *title*: (2017) **Wasserstein Generative Adversarial Networks**
- *note*: [[WGAN]]

WGAN replaces the GAN discriminator with a critic and optimizes the Wasserstein distance to provide meaningful gradients even when the real and model distributions do not overlap. This improves training stability, prevents mode collapse, and allows the generator to better approximate the true distribution, though enforcing the Lipschitz constraint can introduce additional implementation challenges.
### CycleGAN
- *title*: (2017) **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**
- *note*: [[CycleGAN]]

CycleGAN uses generators to learn mappings between two unpaired image domains and discriminators to distinguish real from translated images. This enables unpaired image-to-image translation without paired data, but training can be unstable and struggles with large geometric changes or semantic ambiguities.
## Explicit Density Generative Models

### Normalizing Flow Models
Normalizing Flow (NF) models are a class of generative models that transform a simple probability distribution (e.g., Gaussian) into a complex data distribution using a sequence of invertible and differentiable transformations.

#### NICE
- *title*: (2015) **NICE: Non-linear Independent Components Estimation**
- *note*: [[NICE]]

NICE learns a highly non-linear bijective transformation that maps data to a latent space with independent dimensions, enabling exact log-likelihood computation and efficient ancestral sampling. Its coupling layer architecture ensures invertibility and tractable Jacobian determinants, allowing complex transformations while maintaining stable training and potential applications like inpainting.
#### RealNVP
- *title*: (2017) **Density Estimation Using Real NVP**
- *note*: [[RealNVP]]

RealNVP extends NICE by introducing non-volume-preserving affine coupling layers, enabling more flexible transformations while maintaining invertibility and tractable Jacobian determinants for exact log-likelihood evaluation, inference, and sampling. Its multi-scale architecture with masked convolutions and batch normalization allows efficient modeling of high-dimensional data, learning a semantically meaningful latent space suitable for structured output and semi-supervised tasks.
#### Glow
- *title*: (2018) **Glow: Generative Flow with Invertible 1×1 Convolutions**

### Diffusion-Based Generative Models
#### DDPM
- *title*: (2020) **Denoising Diffusion Probabilistic Models**

#### DDIM
- *title*: (2020) **Denoising Diffusion Implicit Models**

#### Latent Diffusion Model
- *title*: (2022) **High-Resolution Image Synthesis with Latent Diffusion Models**

### Continuous Normalizing Flows
#### Flow Matching
- *title*: (2023) **Flow Matching for Generative Modeling**
- *note*: [[Flow Matching]]

Flow Matching trains a neural network to directly learn vector fields that transport noise distributions to data distributions along continuous probability paths, enabling efficient, simulation-free generative modeling beyond standard diffusion processes. By using conditional paths per data point, it can construct exact marginal flows from simple per-sample flows, and allows flexible choices such as Optimal Transport paths for minimal, straight-line particle transport.
### Consistency-Based Generative Models
#### Consistency Models
- *title*: (2023) **Consistency Models**
