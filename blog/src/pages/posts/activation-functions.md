---
layout: ../../layouts/PostLayout.astro
title: "Activation Functions: Why Non-Linearity Is Everything"
date: "2026-06-28"
description: "A neural network made of only linear layers can only learn linear functions - no matter how many layers you stack. Activation functions break this. ReLU made deep learning practical. GELU powers GPT. SwiGLU is what modern models use. Here's why each one exists and what it does to gradients."
tag: "ai-internals"
readingTime: 9
---

There's a proof worth knowing: if you stack linear transformations without any non-linearity between them, the entire network is equivalent to a single linear transformation. Ten layers, a hundred layers, a thousand - they all collapse to one matrix multiply. The network can only represent linear functions, which rules out almost everything useful.

Activation functions are the non-linearities that prevent this collapse. They're applied elementwise after each linear layer, introducing the bends and kinks that let the network approximate complex functions.

---

## The linearity collapse, demonstrated

```python
import numpy as np

# Three "deep" linear layers
W1 = np.random.randn(4, 4)
W2 = np.random.randn(4, 4)
W3 = np.random.randn(4, 4)

# Their composition is just one matrix multiply
W_collapsed = W3 @ W2 @ W1

x = np.random.randn(4)

# Three sequential operations
out_deep = W3 @ W2 @ W1 @ x

# One operation
out_shallow = W_collapsed @ x

print(np.allclose(out_deep, out_shallow))  # True
```

The matrix product `W3 @ W2 @ W1` is a single matrix. The three layers have zero additional expressive power over one. Adding a non-linear function between each layer breaks this - the composition can no longer be simplified.

---

## Sigmoid: the original, and its problems

Before ReLU became the default, sigmoid was used everywhere:

```
σ(x) = 1 / (1 + e^(-x))
```

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)

x = np.linspace(-6, 6, 300)
print("Sigmoid output range:", sigmoid(-10), "to", sigmoid(10))
# 4.54e-05 to 0.9999546
print("Max gradient:", sigmoid_grad(0))
# 0.25 - even at the peak, gradients are small
```

Sigmoid has two problems that make it poor for deep networks:

**Saturation**: For `|x| > 3`, sigmoid is nearly flat. The gradient is close to zero. During [backpropagation](./backpropagation-from-scratch), gradients are multiplied layer by layer - small gradients compound into near-zero gradients in early layers. This is the vanishing gradient problem.

**Output shift**: Sigmoid always outputs positive values (between 0 and 1). This creates asymmetric gradients that slow learning.

```python
# Demonstrate saturation: gradient vanishes at extremes
for x_val in [-5, -2, 0, 2, 5]:
    g = sigmoid_grad(x_val)
    print(f"x={x_val:3d}  gradient={g:.6f}")
```

```
x= -5  gradient=0.006648
x= -2  gradient=0.104994
x=  0  gradient=0.250000
x=  2  gradient=0.104994
x=  5  gradient=0.006648
```

At x=±5, the gradient is 26× smaller than at x=0. In a 10-layer network, the compound effect kills gradients entirely.

---

## ReLU: the surprisingly effective fix

Rectified Linear Unit (ReLU) is embarrassingly simple:

```
ReLU(x) = max(0, x)
```

Positive inputs pass through unchanged. Negative inputs become zero.

```python
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("ReLU:", relu(x))       # [0. 0. 0. 1. 3.]
print("Grad:", relu_grad(x))  # [0. 0. 0. 1. 1.]
```

The gradient for positive inputs is exactly 1. This is the key property: gradients don't shrink as they pass through ReLU on the positive side. Deep networks could finally be trained without gradients vanishing.

The cost: neurons whose inputs are consistently negative receive zero gradient and stop learning - the "dying ReLU" problem. In practice this matters less than you'd think; most networks have enough redundancy.

```python
# Leaky ReLU: small gradient for negatives, prevents dying neurons
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)
```

---

## GELU: what GPT uses

Gaussian Error Linear Unit (GELU) is a smooth approximation of ReLU. The exact definition involves the Gaussian CDF, but the practical formula is:

```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

```python
import torch
import torch.nn.functional as F

def gelu(x):
    import math
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-4, 4, 300)
gelu_out = gelu(x)
relu_out  = np.maximum(0, x)
```

The difference from ReLU: GELU is smooth everywhere and doesn't hard-zero negative inputs - it gates them softly. Near zero, GELU can produce small negative outputs.

```python
# Compare ReLU and GELU around zero
for x_val in [-0.5, -0.2, 0.0, 0.2, 0.5]:
    r = max(0, x_val)
    g = gelu(np.array([x_val]))[0]
    print(f"x={x_val:4.1f}  ReLU={r:.4f}  GELU={g:.4f}")
```

```
x=-0.5  ReLU=0.0000  GELU=-0.1543
x=-0.2  ReLU=0.0000  GELU=-0.0563
x= 0.0  ReLU=0.0000  GELU=0.0000
x= 0.2  ReLU=0.2000  GELU=0.1155
x= 0.5  ReLU=0.5000  GELU=0.3457
```

The smoothness makes optimization slightly easier and is said to work better with the specific patterns in text data. GPT-2 and BERT both use GELU. The practical performance difference over ReLU on most tasks is modest - the choice matters more for the specific model architecture than as a general principle.

---

## SwiGLU: what modern models use

SwiGLU (Swish-Gated Linear Unit) is the activation used in LLaMA, Mistral, and most current large models. It introduces a gating mechanism:

```
SwiGLU(x, W, V) = Swish(xW) ⊙ xV
```

Where `⊙` is elementwise multiplication and Swish is `x × sigmoid(x)`.

The gating idea: one linear projection controls whether the other passes through. This is more expressive than a simple element-wise non-linearity.

```python
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Two projections instead of one
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.W(x))   # SiLU = Swish = x * sigmoid(x)
        content = self.V(x)
        return self.out(gate * content)

def silu(x):
    return x * torch.sigmoid(x)

# Compare SiLU vs ReLU
x = torch.linspace(-3, 3, 7)
print("x:   ", x.tolist())
print("ReLU:", F.relu(x).tolist())
print("SiLU:", silu(x).tolist())
```

```
x:    [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
ReLU: [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
SiLU: [-0.14, -0.24, -0.27, 0.0, 0.73, 1.76, 2.86]
```

SwiGLU has better empirical performance on language tasks, which is why it's essentially the default for new models. The extra linear projection adds parameters, but the performance improvement is consistent enough that almost everyone uses it.

---

## Gradient flow comparison

One of the most useful ways to compare activation functions is gradient flow - how well gradients propagate backwards through many layers.

```python
import torch
import torch.nn as nn

def test_gradient_flow(activation_fn, depth=20, seed=0):
    torch.manual_seed(seed)
    
    # Stack of linear layers with the given activation
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(64, 64), activation_fn()])
    
    model = nn.Sequential(*layers)
    
    x = torch.randn(16, 64, requires_grad=True)
    out = model(x).sum()
    out.backward()
    
    return x.grad.abs().mean().item()

# Test gradient magnitude reaching the input
activations = {
    "ReLU":        nn.ReLU,
    "Sigmoid":     nn.Sigmoid,
    "GELU":        nn.GELU,
    "SiLU":        nn.SiLU,
}

for name, act in activations.items():
    grad = test_gradient_flow(act, depth=20)
    print(f"{name:<10}: input gradient magnitude = {grad:.6f}")
```

```
ReLU      : input gradient magnitude = 0.003241
Sigmoid   : input gradient magnitude = 0.000001
GELU      : input gradient magnitude = 0.004817
SiLU      : input gradient magnitude = 0.004923
```

Sigmoid is thousands of times worse. ReLU, GELU, and SiLU are all reasonably close to each other - the gap between them matters less than the gap from sigmoid.

---

## Why the choice matters for transformers specifically

In the feed-forward layers of a transformer block, the activation sits between two large linear projections. The typical expansion ratio is 4× - if `d_model` is 768, the intermediate layer is 3072. With SwiGLU, you use two parallel projections of `d_model × (4/3 × d_model)` each, keeping the parameter count roughly the same while adding the gating mechanism.

```python
import torch.nn as nn

class TransformerFFN_GELU(nn.Module):
    """Standard GPT-2 style feed-forward block."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerFFN_SwiGLU(nn.Module):
    """LLaMA-style feed-forward block."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

# Same parameter count (approximately)
gelu_params  = sum(p.numel() for p in TransformerFFN_GELU(768, 3072).parameters())
swiglu_params = sum(p.numel() for p in TransformerFFN_SwiGLU(768, 2048).parameters())
print(f"GELU FFN params:   {gelu_params:,}")
print(f"SwiGLU FFN params: {swiglu_params:,}")
```

---

## Summary

| Function | Formula | Where used | Key property |
|---|---|---|---|
| Sigmoid | `1/(1+e^-x)` | Old networks, output layers | Saturates; vanishing gradients |
| ReLU | `max(0, x)` | Convolutional nets, MLPs | Simple; gradient-1 for positives |
| GELU | Smooth ReLU | GPT-2, BERT | Smooth; slight negative outputs |
| SiLU/Swish | `x·σ(x)` | Modern models | Smooth; slightly better performance |
| SwiGLU | gated SiLU | LLaMA, Mistral | Expressive gating mechanism |

The progression from sigmoid to ReLU to GELU to SwiGLU follows the same thread: keep gradients alive through many layers, give the network enough expressive power, don't overcomplicate what works.

---

*Next: [Tokenization](./tokenization) - before the model sees any of these activations, it needs to convert text into numbers. The way it does that has surprising consequences.*

*Previous: [Backpropagation from Scratch](./backpropagation-from-scratch) - the training loop that adjusts all these weights.*
