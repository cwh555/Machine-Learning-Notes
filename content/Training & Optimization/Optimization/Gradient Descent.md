---
title: Gradient Descent
date: 2025-08-23
---
## Beefing Up Gradient Descent
1. <span style="color:pink; font-weight:bold;">Variable learning rate gradient descent</span>
<div style="text-align:center;">
<img src="https://i.imgur.com/63g7llV.jpeg" alt="Example Image" style="height: 200px;">
</div>

If the error drops, increase $\eta$, if not, decrease $\eta$ .
<span class = 'lime'>It is usually best to go with a conservative increment parameter.</span> $\alpha \approx1.05 - 1.11$ 
<span style="color:pink; font-weight:bold;">and a bit more aggressive decrement parameter</span> $\beta \approx0.5 - 0.8$

2. <span style="color:pink; font-weight:bold;">Steepset Descent</span>
<div style="text-align:center;">
<img src="https://i.imgur.com/Je0lZ9e.jpeg" alt="Example Image" style="height: 200px;">
</div>

Once you have the direction to move, choose a step size $\eta^*$ , where
$$
\eta^*(t)\ =\ \underset{\eta}{argmin}\ E_{in}\left(\,w(t)\ +\ \eta v(t)\,\right).
$$
- <span style="color:pink; font-weight:bold;">Line Search (Bisection Algorithm)</span>  

**Illustrate**   
Find an interval on the li. e which is guaranteed to contain a local minimum.

**Initialization**     
Start with $\eta_1\ =\ 0$ and $\eta_2\ =\ \epsilon$  for some step $\epsilon$ .
If $E(\eta_2)\ <\ E(\eta_1)$, consider the sequence $\eta\ =\ 0,\,2\epsilon,\,4\epsilon,\,\dots$ At some points the error must increase. Then, the last three steps give a U-arrangement.
If $E(\eta_1)\ <\ E(\eta_2)$ , consider the sequence  $\eta\ =\ \eta,\,0,\,-2\epsilon,\,-4\epsilon,\,\dots$ When the error increase for the first time, the last three steps give a U-arrangement.

**Process**    
The basic invariant is a U-arrangement 
$$
\eta_1 < \eta_2 < \eta_3\quad with\quad E(\eta_2)\ <\ min\{E(\eta_1),\,E(\eta_3)\}\ .
$$
The continuity of $E$ implies that there exists a local minimum in $[\eta_1,\eta_3]$ .   
Consider $\hat{\eta}\ =\ \frac{1}{2}(\eta_1\ +\ \eta_3)$ ,     
If $E(\hat{\eta})\ <\ E(\eta_2)$ , then $\{\eta_1, \hat{\eta}, \eta_2\}$ is a smaller U-arrangement.   
If $E(\hat{\eta})\ >\ E(\eta_2)$ , then $\{\hat{\eta}, \eta_2, \eta_3\}$ is a smaller U-arrangement.   
If they are the same, perturb $\hat{\eta}$ slightly.   

**Termination**   
Until $|\eta_3 - \eta_1|$ is small enough. <span class = 'lime'>Usually 20 iterations of bisection are enough. </span>

3. <span style="color:pink; font-weight:bold;">Conjugate Gradient Descent</span>
<div style="text-align:center;">
<img src="https://i.imgur.com/Rncnhcz.jpeg" alt="Example Image" style="height: 300px;">
</div>

The conjugate gradient algorithm chooses the next direction $v(t+1)$ so that the gradient along this direction, will remain perpendicular to the previous search direction $v(t)$ .
