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

#### VQ-VAE-2
- *title*: (2019) **VQ-VAE-2: Generating Diverse High-Fidelity Images**


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
#### NICE
- *title*: (2015) **NICE: Non-linear Independent Components Estimation**

#### RealNVP
- *title*: (2017) **Density Estimation Using Real NVP**

#### Glow
- *title*: (2018) **Glow: Generative Flow with Invertible 1×1 Convolutions**

### Diffusion-Based Generative Models
- *title*: (2023) **Flow Matching for Generative Modeling**
#### DDPM
- *title*: (2020) **Denoising Diffusion Probabilistic Models**

#### DDIM
- *title*: (2020) **Denoising Diffusion Implicit Models**

#### Latent Diffusion Model
- *title*: (2022) **High-Resolution Image Synthesis with Latent Diffusion Models**

### Continuous Deterministic Generative Dynamics
#### Consistency Models
- *title*: (2023) **Consistency Models**
