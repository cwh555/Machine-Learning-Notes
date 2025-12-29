---
title: Task Classification
---
Here, we classify the paper notes according to the tasks they focus on.

## Computer Vision
### Image Generation
#### Pure Generation
- VAE: [[VAE]]
- GAN: [[GAN]], [[DCGAN]], [[WGAN]]
- Normalizing Flows: [[NICE]]

#### Image-to-Image Translation
The goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs.

For example
![[截圖 2025-12-27 下午4.39.25.png]]

- [[CycleGAN]]

#### Inpainting
Given partially observed data $x_O$​, infer missing dimensions $x_H$ by maximizing the log-likelihood:
$$
\max_{x_H} \log p_X(x_O, x_H)
$$
- [[NICE]]

### Structured Prediction
Structured prediction models learn a mapping from an input $x$ to a complex, interdependent output $y$, where the components of $y$ are statistically dependent.

For example, the typical structures are sequences, graphs and images.
In structured prediction, models are $p(y|\,x)$
- There is a meaningful input $x$
- Output $y$ is structured
- Task = prediction conditioned on input
while the pure generating models such as VAE, GAN are $x \sim p_\theta​(x)$

- [[CVAE]]


## Representation Learning
Representation Learning is the task of automatically learning informative and compact features from raw data, such that these learned representations can be effectively used for downstream tasks like classification, generation, or retrieval. The goal is to transform high-dimensional, complex, or noisy input into a structured, low-dimensional latent space that captures essential information.

- [[VQ-VAE]], [[VQ-VAE-2]]