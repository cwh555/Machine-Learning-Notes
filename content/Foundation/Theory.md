## Cover's Theorem on the separability of patterns

> [!abstract] Statement
> A complex pattern-classification problem cast in a high-dimensional space nonlinearly is more likely to be linearly separable than in a low-dimensional space.

Suppose that the activation patterns $x_1, \dots, x_N$ are chosen independently, according to a probability measure imposed on the input space.   
Suppose also that all the possible dichotomies of $\mathcal{H} = \{x_i\}_{i = 1}^N$ are equiprobable.   
Let $P(N, m_1)$ denote the probability that a particular dichotomy picked at random is $\varphi$-separable, where the class of separating surfaces chosen has $m$ degrees of freedom.

Following Cover (1965), we may then state that
$$
P(N, m_1)\ =\ \frac{1}{2^{N - 1}}\,\sum_{m = 0}^{m_1 - 1}\binom{N - 1}{m}
$$