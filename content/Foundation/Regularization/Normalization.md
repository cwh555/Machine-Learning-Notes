---
title: Normalization
date: 2025-10-21
---
![[2025-08-27-4-25-15.png]]

## Instance Normalization
only normalization over the spatial dimensions.
This technique has the same behavior at train and test.

For example, $x : N \times C \times H \times W$, then only normalize on the dimensions $H$ and $W$.
Recall that batch normalization normalize on $N, H$ and $W$.

## Group Normalization
split the channel dimension into some number groups and normalizes over different subsets of channel dimensions.   
work quietly well in objection detection.
