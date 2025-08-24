---
title: Neural Architecture Search
date: 2025-08-24
---
## Search Space
- **Global search space**: search the whole architecture.
- **Cell-based space**: the whole architecture consists of some identical cells.
Typically, there are two kinds of cells:
1. *Normal cell*: maintain the size of feature map.
2. *Reduction cell*: decrease the size of feature map.

## Search Strategy
### Problems
A mapping $\Lambda$ is defined as follows:
$$
\Lambda\ :\ D\ \times\ A\ \rightarrow\ M
$$
Here, $D$ denotes the space of all datasets, $A$ denotes the architecture search space, and $M$ denotes the space of all deep learning models.

Given a dataset $d$, the general deep learning algorithm $\Lambda$ estimates the model $m_{\alpha,\,\theta}\in M_\alpha$, where $\alpha$ denotes the architecture and $\theta$ denotes the weight. The model is estimated by minimizing the sum of a loss function $\mathcal{L}$ and a regularization term $\mathcal{R}$ with respect to the training data. That is,
$$
\Lambda(\alpha, d)\ =\ \underset{m_{\alpha,\,\theta\ \in\ M_{\alpha}}}{\arg\min}\ \mathcal{L}(m_{\alpha,\,\theta},\,d_{train})\ +\ \mathcal{R}(\theta)
$$
Given a dataset $d$, the neural architecture search task aims to finding the architecture $\alpha^*$ which maximizes an objective function $\mathcal{O}$ on the validation data $d_{valid}$. Formally,
$$
NAS(d)\ =\ \alpha^*\ =\ \underset{\alpha\,\in\,A}{\arg\max}\mathcal{O}(\, \Lambda(\alpha,\,d_{train}),\,d_{valid} \,)\ =\ \underset{\alpha\,\in\,A}{\arg\max}f(\alpha)
$$
### Performance Estimation Strategy
- **Lower fidelity estimates**: try to reduce the parameters related to evaluation time, including training epochs, data size, etc. This kind of estimation method is based on the assumption that the ranks of candidates are consistent before and after reducing fidelity so that the effectiveness of any search strategy only relying on ranks instead of absolute performance will not be disrupted.
- **Performance predictor**: learns a performance estimator based on the history of a full evaluation of previous architectures, to obtain performance estimates of candidates.
- **Weight inheritance**: initializing newly generated candidates with the same weights as previous similar one.
- **Weight sharing**: first creates a supernet, and all candidates can be drawn as subgraphs of it, including structures and weights. Then, whole structure can train together.