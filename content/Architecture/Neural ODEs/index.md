---
title: Neural ODEs
---
Inspired by the ResNet architecture, the paper [[NODEs]] proposes a new model architecture in which the network outputs $dz/dt$ and an ODE solver is used to obtain the transformed state $z'$. This architecture requires only $O(1)$ memory and enables continuous transformations.

Most existing methods assume a fixed terminal time. In contrast, the paper [[LFT]] proposes allowing the model to learn the final time $t_f$​ automatically, and provides an analysis of the errors introduced by numerical ODE solvers.