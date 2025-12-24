---
title: Optimization
date: 2025-10-14
---
## Learning rates
- Every adjustable network parameter of the cost function should have its own individual learning-rate parameter.
	- the last layers have smaller value than front layers
	- neurons with many inputs should have a smaller value
- Every learning-rate parameter should be allowed to vary from one iteration to the next.
- When the derivatives of the cost function with respect to a synaptic weight has the same algebraic sign for several consecutive iterations of the algorithm, the learning-rate parameter for the particular weight should be increased.
- When the algebraic sign of the derivative of the cost function with respect to a particular synaptic weight alternates for several consecutive iterations of the algorithm, the learning-rate parameter of that weight should be decreased.
