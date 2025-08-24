---
title: Meta Learning
---
## Introduction
### Model-Based Methods
a model that can quickly update its parameter within just a few training steps.

#### Memory-Augmented Neural Networks (MANN)
- a neural network with augmented memory capabilities that can memorize information about previous and leverage that to learn a learner $l_{new}$.


### Metric-Based Methods
transfer information from the most similar task to the new task, where the task similarity can be measured by a distance function or a kernel function.
$$
P_\theta(y|\,X,\,\mathcal{S})\ =\ \sum_{(x_i,\,y_i)\ in\ \mathcal{S}}\,k_\theta(x,\,x_i)\,y_i
$$
where $k_\theta$ is a kernel function measuring the similarity between two data samples.

#### Matching Networks
aims to solve a *k*-shot classification problem.

$$
c_S(\hat{x})\ =\ P(\hat{y}|\,\hat{x},\,\mathcal{S})\ =\ \sum_{i = 1}^k\,a(\hat{x},\,x_i)\,y_i
$$
The attention kernel depends on two embedding functions, $f$ and $g$.
$$
a(\hat{x},\,x_i)\ =\ \frac{\exp(\cos(f(\hat{x}),\ g(x_i)))}{\sum_{j = 1}^k\,\exp( \cos(\, f(\hat{x}),\ g(x_j) \,) )}
$$
#### Prototypical Networks
use an embedding function $f_\phi\ :\ \mathbb{R}^D\ \rightarrow\ \mathbb{R}^M$ to encode each input $x_i \in \mathbb{R}^D$ into a *M*-dimensional feature vector, $c_k\ \in\ \mathbb{R}^M$, which is also called prototype.

A prototype feature vector is defined for every class $c \in \mathcal{C}$, as the mean vector of the embedded support points belonging to its class:
$$
c_k\ =\ \frac{1}{|S_k|}\ \sum_{(x_i,\,y_i)\in S_k}\ f_\phi(x_i).
$$

The distribution over classes for a given query input point $x$ is:
$$
P_\phi(y\ =\ k|\,x)\ =\ \text{softmax}(-d(f_\phi(x),\ c_k))\ =\ \frac{\exp(-d(f_\phi(x),\ c_k))}{\sum_{k'}\ \exp(-d(f_\phi(x),\ c_k'))}
$$
where $d\ :\ \mathbb{R}^M\ \times\ \mathbb{R}^M\ \rightarrow\ [0,\,+\infty)$ can be any distance function as long as $d$ is differentiable.

### Optimization-Based Methods
the inner-level task is solved as an optimization problem, and the outer-level process focuses on extracting meta-knowledge $w$ required to improve optimization performance.

#### The Model Agnostic Meta-Learning (MAML) Algorithm
the goal is to learn a model parameter initialization that generalizes better to similar tasks. It aims to optimize the model parameters such that one or a small number steps of gradient descent can produce maximally effective behavior on an new task.

Pick several tasks $T_1,\,\dots,\,T_n$ from the distribution.
- Inner-loop learning: for each task $T_i$, update the parameter on support set
$$
\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}^{\text{train}}(\theta)
$$
- Outer-loop learning: 
$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{T_i}^{\text{val}}(\theta_i')
$$

