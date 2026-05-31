# TAGI-GCN: A Prototype Bayesian Graph Neural Network

**Original Concept:** C-Dzuwa & Sam Ndovie

## Foundation 1: TAGI

**Tractable Approximate Gaussian Inference (TAGI)** is a Bayesian learning framework developed by James-A. Goulet and colleagues that enables neural networks to learn through analytical Gaussian inference rather than backpropagation and gradient descent.

Key features:

- Bayesian neural learning
- Analytical uncertainty quantification
- No backpropagation
- No gradient-based optimization
- Online learning capability
- Propagation of means and variances through the network

Reference:

- https://arxiv.org/abs/2004.09281

---

## Foundation 2: Simple Graph Convolution (SGC)

**Simple Graph Convolution (SGC)** is a simplified version of Graph Convolutional Networks proposed by Wu et al. (2019).

Instead of repeatedly alternating graph convolutions and nonlinear activations, SGC removes the nonlinearities and collapses multiple graph propagation steps into a single operation.

The resulting model is:

\[
Z = \hat{A}^{K}XW
\]

where:

- \(X\) = node features
- \(\hat A\) = normalized adjacency matrix
- \(K\) = number of propagation steps
- \(W\) = trainable weights

Reference:

- https://arxiv.org/abs/1902.07153

---

## Proposed Prototype: TAGI-GCN

The prototype combines:

1. Graph propagation from SGC.
2. Bayesian learning from TAGI.

Instead of learning deterministic weights using gradient descent, the SGC weights are treated as Gaussian random variables and updated using TAGI.


In essence, TAGI-GCN can be viewed as:

```text
TAGI + SGC
```

or

```text
Bayesian Simple Graph Convolution
```

where graph message passing is inherited from SGC and uncertainty-aware learning is inherited from TAGI.
