---
title: Similarity-Based Methods
date: 2025-08-23
---
## Similarity
### Similarity measures
- **Euclidean distance**
$$
d(x,\,x')\ =\ \|x\ -\ x' \|
$$	   
For boolean features, the Euclidean distance is the square root of the <span style="color:pink; font-weight:bold;">Hamming distance</span>.
- **General distance measure**
$$
d(x,\,x')\ =\ (x\,-\,x')^T\,Q\,(x\,-\,x')
$$
where $Q$ is an arbitrary positive semi-definite matrix.  
A useful special case, known as <span style="color:pink; font-weight:bold;">Mahalanobis distance</span> is to set
$$
Q\ =\ \Sigma^{-1}\quad,\text{where }\Sigma\text{ is the covariance matrix ,}\quad \Sigma\ =\ \frac{1}{N}\sum_{i=1}^{N}x_ix^T\,-\,\bar{x} \bar{x}^T
$$
- <span style="color:pink; font-weight:bold;">Cosine similarity</span> 
Especially useful for Boolean vectors
$$
CosSim(x,\,x')\ =\ \frac{x\, \cdot\, x'}{\|x\|\,\|x'\|}
$$
- <span style="color:pink; font-weight:bold;">Jaccard coefficient</span>  
This measure is used when the objects represent sets.
$$
J(S_1,\,S_2)\ =\ \frac{|S_1\,\cap\,S_2|}{|S_1\,\cup\,S_2|}
$$
## Nearest Neighbor
### Model
Consider the classification problem.  
Suppose that the data set is $\mathcal{D}\ =\ (x_1,\,y_1),\,\dots\,(x_N,\,y_N)$, where $y_n\,=\,\pm1$ .

To classify the test point $x$, we reorder the data according to distance from $x$. We write $(x_{[n]},\,y_{[n]})$ for the $\textit{n}$th such reordered data point with respect to $x$.   
The final hypothesis is 
$$
g(x)\ =\ y_{[1]}(x)
$$
In the final hypothesis $g(x)$, each data point $x_n$ owns a region defined by the points closer to $x_n$ than to any other data point. The resulting set of regions defined by such a set of points is called a <span style="color:pink; font-weight:bold;">Voronoi (or Dirichlet) tessellation</span> of the space.

<div style="text-align:center;">
<img src="https://i.imgur.com/6baqb9W.jpeg" alt="Example Image" style="width: 250px; height: 200px;">
</div>

### Generalization Bound
First, we discuss the minimum possible out-of-sample error.  
We model the target value, which is $\pm1$ , as noisy and define
$$
\pi(x)\ =\ \mathbb{P}[y\ = \ +1\,|\,x]\ .
$$
Let 
$$
f(x)\ =\ \begin{cases}
+1,\quad \quad &if\ \pi(x)\ \geq \frac{1}{2} \\
-1, &otherwise 
\end{cases}
$$

> [!blue] Discussion
> The probability of error on a test point $x$ is
> $$
> e(f(x))\ =\ \mathbb{P}[f(x)\,\neq\,y]\ =\ min\{\pi(x),\,1\,-\,\pi(x)\}
> $$
> and $e(f(x))\ =\ e(h(x))$ for any other hypothesis $h$ .

> [!gray]-  Proof
> $$
> \begin{aligned}
> e(f(x))\ &=\ \mathbb{P}[f(x)\ \neq\ y] \\
> &=\ \mathbb{P}[f(x)\ = 1\ and\ y\ =\ -1]\ +\ \mathbb{P}[f(x)\ =\ -1\ and\ y\ =\ 1]\\
> &= (1\, -\, \pi(x))\,(\pi(x)\,\geq\,\frac{1}{2})\ +\ \pi(x)\,(\pi(x)\,\lt \,\frac{1}{2}) \\
> &= min\{\pi(x),\,1\,-\,\pi(x)\}
> \end{aligned}
> $$
> 
> $$
> \begin{aligned}
> \\
> e(h(x))\ &=\ (1\,-\,\pi(x))\,\mathbb{P}(h(x)\,=\,1)\ +\ \pi(x)\,\mathbb{P}(h(x)\,=\,-1) \\
> &= \pi(x)\ +\ (1\,-\,2\pi(x))\,\mathbb{P}(h(x)\,=\,1) \\
> \end{aligned}
> $$
> $$
> \begin{cases}
> \text{If }\pi(x)\ \geq\ \frac{1}{2}\text{, then } e(h(x))\ \geq\ 1\,-\,\pi(x)\ ;\\\\
> \text{If } \pi(x)\ \lt\ \frac{1}{2}\text{, then } e(h(x))\ \geq\ \pi(x)\ .
> \end{cases}
> $$
> $$
> \implies\quad\quad e(h(x))\ \geq\ e(f(x))\ .
> $$

Hence, the best possible out-of-sample misclassification error is the expected error of $e(f(x))$ .
$$
E^*_{out}\ =\ \mathbb{E}[e(f(x))]\ =\ \int\ \mathbb{P}(x)\,min\{\pi(x),\,1\,-\,\pi(x)\}\ dx
$$
> [!green] Nearest Neighbor is 2-Optimal
> For any $\delta\ \gt\ 0$, and any continuous noisy target $\pi(x)$, there exists $N\, \gt\, 0$ for which, with probability at least $1\,-\,\delta$, 
> $$
> E_{out}(g_N)\ \leq\ 2E^*_{out}\ .
> $$

> [!gray]- Proof
> $$
> \begin{aligned}
> \mathbb{P}[g_N(x)\,\neq\,y]\ &=\ \pi(x)\cdot(1\,-\,\pi(x_{[1]}))\ +\ (1\,-\,\pi(x))\cdot\pi(x_{[1]}) \\
> &=\ 2\eta(x)\cdot(1\,-\,\eta(x))\ +\ \epsilon_N(x)\\\\
> \text{where } \eta(x)\ &=\ min\{\pi(x),\,1\,-\,\pi(x)\}\\
> \epsilon(x)\ &=\ (2\pi(x)\,-\,1)\cdot(\pi(x)\,-\,\pi(x_{[1]})) .
> \end{aligned}
> $$
> To get $E_{out}(g_N)$, we take the expectation with respect to $x$. 
> Note that $E^*_{out}\ =\ \mathbb{E}[\eta(x)]$ and $\mathbb{E}[\eta(x)^2]\ \geq\ \mathbb{E}[\eta(x)]^2$ .
> $$
> \begin{aligned}
> E_{out}(g_N)\ &=\ 2\mathbb{E}[\eta(x)]\ -\ 2\mathbb{E}[\eta^2(x)]\ +\ \mathbb{E}[\epsilon_N(x)] \\
> &\leq\ 2E^*_{out}(1\,-\,E^*_{out})\ +\ \mathbb{E}_x[\epsilon_N(x)]) 
> \end{aligned}
> $$
>
> When $N\,\rightarrow\,\infty$, every point $x$ has a nearest neighbor that is close by that is 
> $$
> x_{[1]}\,\rightarrow\,x,\ \forall\ x
> $$
> By the continuity of $\pi$, $\pi(x_{[1]})\,\rightarrow\,x$ .
> Since $|\epsilon_N(x)|\ \leq\ |\pi(x)\,-\,\pi(x_{[1]})|$,  $\ \epsilon_N(x)\,\rightarrow\,0$.
> In conclusion, $E_{out}(g_N)\ \leq\ 2E^*_{out}\,(1\,-\,E^*_{out})$ when $E^*_{out}$ is small.

### k-Nearest Neighbors (KNN)
#### Model
For simplicity, assume that k is odd. The k-NN rule classifies the test point $x$ according to the majority class among k nearest data points to $x$.  
The final hypothesis is 
$$
g(x)\ =\ sign\left( \sum_{i=1}^{k}y_{[i]}(x)\right)\ .
$$
<span class = 'lime'>Three neighbors is enough.</span>
#### Generalization Bound
> [!blue] Discussion : Generalization Bound for k-NN
> Fix an odd $k\,\geq\,1$ . For $N\,=\,1,\,2,\,\dots$ and data sets $\mathcal{D}_N$ of size $N$, let $g_N$ be the $\text{k-NN}$ rule derived from $\mathcal{D}_N$ with out-of-sample error $E_{out}[g_N]$ .
> 1. $E_{out}(g_N)\ =\ \mathbb{E}[Q_k(\eta(x)]\ +\ \mathbb{E}_x[\epsilon_N(x)]$  for some error term $\epsilon)N$ which converges to zero, and where
> $$
> Q_k\ =\ \sum_{i = 0}^{(k-1)/2}\ \binom{k}{i}\,\left( \eta^{i+1}(1\,-\,\eta)^{k-i}\ +\ (1\,-\,\eta)^{i+1}\eta^{k-i} \right)\ ,
> $$
> and $\eta(x)\ =\ min\{\pi(x),\,1\,-\,\pi(x)\}$ .
> 2. For large enough $N$, with probability at least $1\,-\,\delta$ ,
> $$
> \begin{aligned}
> k\ =\ 3:\quad\quad\quad &E_{out}(g_N)\ \leq\ E^*_{out}\ +\ 3\,\mathbb{E}[\,\eta^2(x)\,];\\
> k\ =\ 5:\quad\quad\quad &E_{out}(g_N)\ \leq\ E^*_{out}\ +\ 10\,\mathbb{E}[\,\eta^3(x)\,]\ .
> \end{aligned}
> $$
> 3. $E_{out}$ is asymptotically $E^*_{out}(1\,+\,\mathcal{O}(k^{-1/2}))$ .
> Hint: there is some $a(k)\ s.t.\ Q_k\ \leq\ \eta(1\,+\,a(k))$, and show that the best such $a(k)$ is $\mathcal{O}(1\,/\,\sqrt{k})$ . $\textcolor{red}{unsolved}$

#### Choosing k
Let $k(N)$ be the choice of $k$ which is a function of $N$ . Note that $1\, \leq\, k(N)\ \leq\ N$.

> [!gray] Theorem
> For $N\ \rightarrow\ \infty$, if $k(N)\ \rightarrow\ \infty$ and $k(N)/N\ \rightarrow\ 0$ then,
> $$
> E_{in}(g)\ \rightarrow\ E_{out}(g)\quad\text{and}\quad E_{out}(g)\ \rightarrow\ E^*_{out}\ .
> $$

<span class = 'lime'>A good choice for k(N) is</span> $\textcolor{lime}{\lfloor \sqrt{N} \rfloor}$

> [!blue] Discussion : Using validation to select k
> Let $g_*^-$ be the hypothesis in $\mathcal{H}_{train}$ with minimum $E_{out}$ .
> 1. If $K/\log{N\,-\,K}\ \rightarrow\ \infty$ then the validation chooses a good hypothesis, $E_{out}(g^-)\ \approx\ E_{out}(g^-_*)\ .$
> 2. If also $N\,-\,K\ \rightarrow\ \infty$, then $E_{out}(g^-)\ \rightarrow\ E^*_{out}$. (validation results in near optimal performance).

### Improving the efficiency for nearest neighbor
#### Data Condensing
The goal of data condensing is to reduce the data set size while making sure the retained data set somehow matches the original full data set.

We denote the condensed data set by $S$ . The data set $S$ is <span style="color:pink; font-weight:bold;">consistent</span> with $\mathcal{D}$ if
$$
g_S(x)\ =\ g_{\mathcal{D}}(x)\quad\quad\forall\,x\,\in\,\mathbb{R}^d\,.
$$
This requirement is too strong, hence we would only require <span style="color:pink; font-weight:bold;">training set consistent</span>.

<div style="text-align:center;">
<img src="https://i.postimg.cc/pLq0W9wg/2025-08-23-9-55-29.png" alt="Example Image">
</div>

#### Data Editing
This algorithm edits the data set to improve prediction performance. This typically means remove the points that you think is noisy.

#### Efficient Search 
The approach is called a <span style="color:pink; font-weight:bold;">branch and bound</span> technique for finding the nearest neighbor.

To summarize, suppose that the data is partitioned into clusters. Each cluster has a center $\mu_j$ and a radius $r_j$ . If we want to search the nearest neighbor of $x$ now, and consider two clusters $S_1, S_2$ .  
If we have $\|x\,-\,\mu_1\|\ +\ r_1\ \leq\ \|x\,-\,\mu_2\|\,-\,r_2$, then we only have to search the data points in $S_1$ . Hence, we want
$$
r_1\ +\ r_2\ \ll\ \|\mu_1\,-\,\mu_2\|
$$

<span style="color:pink; font-weight:bold;">k-means clustering algorithm</span>
<div style="text-align:center;">
<img src="https://i.postimg.cc/GtFQZ9Pg/2025-08-23-9-56-39.png" alt="Example Image">
</div>

The time complexity is $\mathcal{O}(MNd\log{N})$ if the depth of the partitioning is $\log{N}$ .

### Application
#### Multiclass Data
> [!blue] Discussion: Exercise 6.9
> With $C$ classes labeled $1,\,\dots\,,C$, define $\pi_c(x)\ =\ \mathbb{P}[c|x]$ (the probability to observe class $c$ given $x$) .
> Let $\eta(x)\ =\ 1\,-\,\max_{c}{\pi_c(x)}\ .$
> 1. Define a target $f(x)\ =\ argmax_c \pi_c(x)\ .$ On a test point $x$, $f$ attains the minimum possible error probability of 
> $$
> e(f(x))\ =\ \mathbb{P}[f(x)\ \neq\ y]\ =\ \eta(x)
> $$
> 2. For the nearest neighbor rule, with high probability, the final hypothesis $g_N$ achieves an error on the test point $x$ that is
> $$
> e(g_N(x))\ \overset{N\,\rightarrow\,\infty}{\longrightarrow}\ \sum_{c=1}^{C}\ \pi_c(x)\,(1\,-\,\pi_c(x))\ .
> $$
> 3. For large enough $N$, with high probability,
> $$
> E_{out}(g_N)\ \leq\ 2E^*_{out}\,-\,\frac{C}{C\,-\,1}\left(E^*_{out}\right)^2\ .
> $$

#### Regression
$$
g(x)\ =\ \frac{1}{k}\sum_{i=1}^{k}\ y_{[i]}(x)\ .
$$
To output the probability with k-NN,
$$
g(x)\ =\ \frac{1}{k}\sum_{i = 1}^{k}\ \left[\!\left[ y_{[i]}\ =\ +1\ \right]\!\right]
$$

## Radial Basis Functions
### Model
The contribute of $x_n$ to the classification at $x$ will be proportional to $\phi(\|x\,-\,x_n\|)$ .
- <span style="color:pink; font-weight:bold;">Gaussian kernel</span>
$$
\large\phi(x)\ =\ e^{-\frac{1}{2}z^2}
$$
- <span style="color:pink; font-weight:bold;">Window kernel</span>
$$
\phi(z)\ =\ \begin{cases}
1\quad\quad &z\,\leq\,1,\\
0 &z\,\gt\,1.
\end{cases}
$$
We use the scale $r$ to specify the width of the kernel.  
The influence of $x_n$ at $x$ is denoted by
$$
\alpha_n(x)\ =\ \phi\left( \frac{\|x\,-\,x_n\|}{r} \right)\ .
$$
The hypothesis is in the form
$$
g(x)| =\ \frac{\sum_{n=1}^{N}\,\alpha_n(x)\,y_n}{\sum_{m=1}^{N}\,\alpha_m(x)} \tag{6.1}
$$
> [!blue] Exercise 6.11
> When $r\ \rightarrow\ 0$, for the Gauss kernel, the RBF final hypothesis is the same as the nearest neighbor rule.

#### Nonparametric RBF
If we view (6.1) as a weighted sum of y-values, this corresponds to putting a bump at each $x_n$ and the value of the bump at $x_n$ determines $\alpha_n$ .

#### Parametric RBF
We view (6.1) as the sum of $N$ bumps of different heights, where each centered on a data point.
$$
g(x)\ =\ \sum_{n=1}^{N}\,w_n(x)\,\phi\left( \frac{\|x\,-\,x_n\|}{r} \right)
$$
Now, our hypothesis has the form
$$
\begin{aligned}
h(x)\ &=\ \sum_{n=1}^{N}\,w_n\,\Phi_n(x) \\
&=\ \sum_{n=1}^{N}\,w^Tz\,,\quad\quad \text{where }z\ =\ \Phi(x)\ =\ \begin{bmatrix}
\Phi_1(x)\\
\vdots\\
\Phi_N(x)
\end{bmatrix}
\end{aligned}
$$
### Algorithm for parametric RBF
To avoid overfitting, we restrict to $k$ bumps.
$$
h(x)\ =\ w_0\ +\ \sum_{j=1}^{k}\,w_j\,\phi\left(\frac{\|x\,-\,\mu_j\|}{r} \right)\ =\ w^T\Phi(x)\ .
$$
We need to determine $k,\ r$ at first. Then, choose the $w_j,\ \mu_j$ to fit the data.
1.  $k,\,r$
$k$ : using cross validation
$r$ : for a given $k$, 
$$
\textcolor{lime}{r\ =\ \frac{R}{k^{1/d}}\quad\quad\ \text{where }R\ =\ \max_{i,\,j}\|x_i\,-\,x_j\|}
$$

2. $w_j$
For a given set of centers, $u_j$,
[[Linear Model]]
<div style="text-align:center;">
<img src="https://i.postimg.cc/HstzzckV/2025-08-23-9-58-50.png" alt="Example Image">
</div>

3. $\mu_j$ **Unsupervised k-Means Clustering**
The goal is to partition the input data points $x_1,\,\dots\,,x_N$ into $k$ sets $S_1,\,\dots\,,S_k$ and select centers $\mu_1,\,\dots\,,\mu_k$ for each center.   
Define the error measure
$$
E_j\ =\ \sum_{x_n\,\in\,S_j}{\|x_n\,-\,\mu_j\|}
$$
We would like to minimize the cluster error
$$
E_{in}\ =\ \sum_{j\ =\ 1}^{k}E_j
$$
> [!blue] Exercise 6.13
> 1. Fix the clusters $S_1\,\dots\,,S_k$. The centers that minimize $E_{in}$ are the centroids of the clusters :
> $$
> \mu_j\ =\ \frac{1}{|S_j|}\,\sum_{x_n\,\in\,S_j}x_n\ .
> $$
> 2. Fix the centers. The clusters that minimize $E_in$ are obtained by placing into $S_j$ all points for which the closest center is $\mu_j$ : 
> $$
> S_j\ =\ \{x_n\ :\ \|x_n\,-\,\mu_j\|\ \leq\ \|x_n\,-\,u_l\|\quad \text{, for}\ l\ \in\ \{1,\dots\,k\}\}
> $$

[[Similarity-Based Methods#Efficient Search]]
<div style="text-align:center;">
<img src="https://i.postimg.cc/vm4vG0LV/2025-08-23-10-00-23.png" alt="Example Image">
</div>

## Probability Density Estimation
For a given $x$, we want to estimate the probability of the inputs in the data that is similar to $x$ .

### Nonparametric method
#### Nearest Neighbor Density Estimation
One can use the distance to $\text{kth}$ nearest neighbor to determine the volume of the region containing $x$. 

Let $r_{[k]}(x)\ =\ \|x\,-\,x_{[k]}||$ be the distance from $x$ to its $\text{kth}$ nearest neighbor 
and $V_{[k]}(x)$ the volume of the spheroid centered at $x$ of radius $r_{[k]}(x)$. 
In $d$ dimensions, 
$$
V_{[k]}\ =\ \frac{\pi^{\frac{d}{2}}}{\Gamma(\frac{d}{2})\ +\ 1}\,r_{[k]}^{\quad d}\ .
$$
We have the estimate
$$
\hat{P}(x)\ =\ c\ \cdot\ \frac{k}{V_{[k]}(x)}
$$
where $c$ is chosen by normalization $\hat{P}$ i.e. $\int{\hat{P}(x)\,dx\ =\ 1}$ .

#### RBF Density Estimation (Parzen Windows)
Consider the Gaussian kernel
$$
\phi(x)\ =\ \frac{1}{(2\pi)^{d/2}}\,e^{-\large\frac{1}{2}z^2}
$$
We have the estimate
$$
\hat{P}(x)\ =\ \frac{1}{N\,r^d}\,\sum_{i=1}^{N}\,\phi\left( \frac{\|x\,-\,x_i\|}{r} \right)\ .
$$
where $d$ is the dimension, $N$ is the number of data points, and $r$ is the width of the bump.

### Gaussian Mixture Models (GMMs)
#### Model
In $d$ dimensions, the Gaussian density with center $\mu$ and covariance matrix $\Sigma$ is :
$$
\mathcal{N}(x;\mu,\Sigma)\ =\ \frac{1}{(2\pi)^{d/2}\,|\Sigma^{1/2}\,|}\ \large{e^{-\frac{1}{2}\,(x\,-\,\mu)^T\Sigma^{-1}(x\,-\,\mu)}}
$$
Note that the center $\mu\ =\ \mathbb{E}[x]$ and the covariance between features $x_i,\,x_j$ is $\Sigma_{i,j}$ , $\mathbb{E}[(x\,-\,\mu)(x\,-\,\mu)^T]\ =\ \Sigma$ .

Now, suppose that there are $k$ Gaussian distributions, with respective means $\mu_1\,\dots\,\mu_k$ and covariance matrices $\Sigma_1\,\dots\,\Sigma_k$ .   
To generate a data point $x$, suppose that the Gaussian $j\ \in\ \{1,\dots\,,k\}$ according to possibilities $\{w_1,\,\dots\,,w_k\}$ (where $\sum_{i=1}^{k}w_i\ =\ 1$) .
$$
P(x|\,j)\ =\ \mathcal{N}(x;\,\mu_j,\,\Sigma_j)\ =\ \frac{1}{(2\pi)^d\,|\Sigma_j|^{1/2}}\ \large{e^{-\frac{1}{2} (x\,-\,\mu_j)^T\Sigma_j^{-1}(x\,-\,\mu_j)}}
$$
Hence, out hypothesis is
$$
P(x)\ =\ \sum_{j=1}^{k}\,w_j\,\mathcal{N}(x;\,\mu_j,\,\Sigma_j)
$$
where $w_j,\,\mu_j,\Sigma_j$ is the parameters we have to learn.

#### Error measurement
For the data $x_1,\,\dots\,,x_N$,
$$
E_{in}(w_j,\,\mu_j,\,\Sigma_j)\ =\ -\sum_{n=1}^{N}\ln{\left( \sum_{j=1}^{k}\,w_j\,\mathcal{N}(x_n;\,\mu_j,\,\Sigma_j) \right)}
$$
#### The Expectation Maximization (EM) Algorithm
<div style="text-align:center;">
<img src="https://i.postimg.cc/nLNGrq0w/2025-08-23-10-02-19.png" alt="Example Image">
</div>

At iteration $t$, let $\gamma_{nj}\, \geq\, 0$ be the fraction of data point $x_n$ that belongs to bump $j$, with $\sum_{j=1}^{k}\gamma_{nj}\ =\ 1$ .

The number of data points  belonging to bump $j$ is given by
$$
N_j\ =\ \sum_{n=1}^{N}\,\gamma_{nj}
$$
If we know $\gamma_{nj}$ , then we can compute
$$
\begin{aligned}
w_j\ &=\ \frac{N_j}{N}\ ;\\\\
\mu_j\ &=\ \frac{1}{N_j}\sum_{n=1}^{N}\,\gamma_{nj}\,x_n\ ;\\\\
\Sigma_j\ &=\ \frac{1}{N_j}\sum_{n=1}^{N}\gamma_{nj}\,x_n\,x_n^T\ -\ \mu_j\,\mu_j^T\ .
\end{aligned}
$$

We can update the membership as below.
$$
\begin{aligned}
\gamma_{nj}(t\, +\, 1)\ &=\ \mathbb{P}[j|\,x_n]\\\\
&=\ \frac{P(x_n|j)\,\mathbb{P}[j]}{P(x_n)}\quad\ =\ \frac{\mathcal{N}(x_n;\,\mu_j,\,\Sigma_j)\,\cdot\,w_j}{P(x_n)}	\\\\
&=\ \frac{w_j\,\mathcal{N}(x_n;\,\mu_j,\,\Sigma_j)}{\sum_{l=1}^{k}\,w_l\,\mathcal{N}(x_n;\,\mu_l,\,\Sigma_l)}
\end{aligned}
$$

[[Similarity-Based Methods#Efficient Search]]
For initialization, we may use **k-means clustering algorithm** .
