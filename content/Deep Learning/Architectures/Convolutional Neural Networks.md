---
title: Convolutional Neural Networks
date: 2025-08-24
---
## Structures
A typical layer of a convolutional network consists of three stages
1. **Convolutional stage**: performs several convolutions in parallel to produce a set of linear activations.
2. **Detector stage**: each linear activation is run through a nonlinear activation function.
3. **Pooling stage**: use a pooling function to modify the output of the layer.

### Convolutional Operation
$$
s(t)\ =\ (x\ \star\ w)(t) \ =\ \int x(a)\,w(t\ -\ a)\,da
$$
This operation is called **convolution**.   
In the operation, $x$ is called the *input*, $w$ is called the *kernel*, and the output is sometimes referred to as the *feature map*.

As for two-dimensional image $I$ as our input.
$$
\begin{aligned}
S(i,\,j)\ &=\ (I\,\star\,K)(i,\,j)\ =\ \sum_m\sum_n\,I(m,\,n)K(i - m,\,j - n).\\
&=\ (K\,\star\,I)(i,\,j)\ =\ \sum_m\sum_n\,I(i - m,\,j - n)K(m,\,n).
\end{aligned}
$$
The commutative property arises because we have *flipped* the kernel relative to the input.

Another related function is called the *cross-correlation*, which is the same as convolution but without flipping the kernel:
$$
S(i,\,j)\ =\ (I\,\star\,K)(i,\,j)\ =\ \sum_m\sum_n\,I(i + m,\,j + n)K(m,\,n)
$$
### Pooling
A pooling function replaces the output of the net at a certain location with a summary statistics of the nearby outputs.

Pooling helps to make the representation become approximately invariant to small translations of the input.

## Variants of the Basis Convolution Function
Assume we have a 4-D kernel tensor $K$ with element $K_{i, j, k, l}$ giving the connection strength between a unit in channel $i$ of the output and a unit in channel $j$ of the input, with an offset of $k$ rows and $l$ columns between the output unit and the input unit.

Assume out input consists of observed data $V$ with element $V_{i, j, k}$ giving the value of the input unit within channel $i$ at row $j$ and column $k$.

Assume out output consists of $Z$ with the same format as $V$. If $Z$ is produced by convolving $K$ across $V$ without flipping $K$, then
$$
Z_{i, j, k}\ =\ \sum_{l, m, n}V_{l,\ j + m - 1,\ k + n - 1}K_{i,\ l,\ m,\ n}
$$
where the summation over $l, m, n$ is over all values for which the tensor indexing operations inside the summation is valid.

More generally, if we want to sample only every $s$ pixels in each direction in the output, then we can define a downsampled convolution function $c$ such that
$$
Z_{i, j, k}\ =\ c(K,\,V,\,s)_{i, j, k}\ =\ \sum_{l, m, n}\left[ V_{l,\ (j - 1)\times +\,m,\ (k - 1)\times s\,+\,n}K_{i,\,l,\,m\,n} \right]
$$
We refer to $s$ as the **stride** of this downsampled convolution.

### Zero-pad
Three special case of the zero-padding settings:    
Suppose that the input image has width $m$ and the kernel has width $k$.
- **valid convolution**: no zero-padding, the output will be of width $m - k + 1$.
- **same convolution**: zero-padding is added to keep the size of the equal to the size of the input.
- **full convolution**: zero-padding is added to let every pixel to be visited $k$ times in each direction, resulting in an output image of width $m + k - 1$.

### Convolution Type
- *unshared convolution*: full connected
$$
Z_{i, j, k}\ =\ \sum_{l, m, n}[V_{l,\ j + m - 1,\ k + n - 1}\,w_{i,\ j,\ l,\ m,\ n}]
$$
- *locally connected*: do not share parameter
- *traditional convolution*: locally connected and share parameter
- *tiled convolution*: learn a set of kernels that we rotate through as we move through space.
$$
Z_{i, j, k}\ =\ \sum_{l, m, n}V_{l,\ j + m - 1,\ k + n - 1}K_{i,\ l,\ m,\ n,\ j\,\%\,t + 1,\ k\,\%\,t + 1}
$$

Locally connected, Traditional convolution, tiled convolution
<div style="text-align:center;">
<img src="https://i.postimg.cc/DfRGP6JD/2025-07-22-21-33.jpg" alt="Example Image" style="height: 400px;">
</div>

## Neuroscientific Aspect
**Reverse correlation**: put an electrode in the neuron itself, display several samples of white noise images in front of the animal's retina, and record how each of these samples causes the neuron to activate.

Reverse correlation shows us that most V1 cells have weights that are described by **Gabor functions**. We can think of image as being a function of 2-D coordinates, $I(x, y)$.

The response of a simple cell to an image is given by
$$
s(I)\ =\ \sum_{x\in\mathbf{X}}\,\sum_{y\in\mathbf{Y}}\,w(x, y)\,I(x, y)
$$
Specifically, $w(x, y)$ takes the form of a Gabor function:
$$
w(x,\, y;\,\alpha,\,\beta_x, \,\beta_y, \,f, \,\phi, \,x_0, \,y_0, \,\tau)\ =\ \alpha\,\exp(\,\beta_x x'^2\ -\ \beta_y y'^2)\cos(fx'\ +\ \phi)
$$
where
$$
x'\ =\ (x\,-\,x_0)\cos(\tau)\ +\ (y\,-\,y_0)\sin(\tau)
$$
and
$$
y'\ =\ -(x\,-x_0)\sin(\tau)\ +\ (y\,-\,y_0)\cos(\tau)
$$
Here, $\alpha,\,\beta_x, \,\beta_y, \,f, \,\phi, \,x_0, \,y_0, \,\tau$ are parameters that control the properties of the Gabor function.

The cartoon view of a complex cell is that it computes the $L^2$ norm of the 2-D vector containing two simple cells' responses: $c(I)\ =\ \sqrt{s_0(I)^2\,+\,s_1(I)^2}$.