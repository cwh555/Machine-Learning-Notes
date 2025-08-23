---
title: Early Stopping
date: 2025-08-23
---
## Consequence
At the ith step, we start at weights $w_{i-1}$ and take a step of size $\eta$ to $w_i\ =\ w_{i-1}\ -\ \eta \frac{g_{i-1}}{\|g_{i-1}\|}$ .
In fact, we indirectly searched entire hypothesis set $\mathcal{H}\ =\ \{w:\ \|w\,-\,w_{i-1}\|\ \leq\ \eta\}$ .

Stopping early can constrain the learning to the smaller hypothesis set.
