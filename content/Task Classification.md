---
title: Task Classification
---
Here, we classify the paper notes according to the tasks they focus on.

## Computer Vision
### Image Generation
#### Pure Generation
- [[dir_model|Generative Models]]: VAE, GAN, Diffusion Models...

#### Image-to-Image Translation
The goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs.

For example
![[截圖 2025-12-27 下午4.39.25.png]]

- [[CycleGAN]]

### Structured Prediction
Structured prediction models learn a mapping from an input $x$ to a complex, interdependent output $y$, where the components of $y$ are statistically dependent.

For example, the typical structures are sequences, graphs and images.
In structured prediction, models are $p(y|\,x)$
- There is a meaningful input $x$
- Output $y$ is structured
- Task = prediction conditioned on input
while the pure generating models such as VAE, GAN are $x \sim p_\theta​(x)$

- [[CVAE]]
