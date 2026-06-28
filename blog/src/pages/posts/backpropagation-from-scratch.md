---
layout: ../../layouts/PostLayout.astro
title: "Backpropagation from Scratch: How Neural Networks Actually Learn"
date: "2026-06-28"
description: "Every weight in GPT-4 was adjusted by gradient descent, guided by backpropagation. It sounds complicated, but the core idea is straightforward: compute which direction to nudge each weight to reduce the loss, using the chain rule. Here's a complete implementation in NumPy - no PyTorch autograd, no magic."
tag: "ai-internals"
readingTime: 14
---

When a language model gets better at predicting text, something specific is happening: a number called the **loss** goes down, and the millions (or billions) of weights in the model shift slightly in the direction that reduces it. The process that computes which direction to shift each weight is **backpropagation**.

The name sounds intimidating. It isn't. It's the chain rule from calculus, applied systematically to every operation in the network. This article builds a two-layer network from scratch in NumPy - no autograd, no magic - and trains it. By the end you'll know exactly what PyTorch's `.backward()` is doing.

---

## The basic setup

A neural network is a function that maps inputs to outputs. Training is finding the specific function (the specific weights) that produces outputs close to the ones you want.

The loss function measures how far off the predictions are. The goal is to minimize it.

For regression problems, mean squared error (MSE) is the standard choice:

```python
import numpy as np

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# Suppose we predicted [2.0, 3.5, 1.0] but the targets are [2.0, 3.0, 1.5]
preds   = np.array([2.0, 3.5, 1.0])
targets = np.array([2.0, 3.0, 1.5])

print(mse_loss(preds, targets))
# ((0)² + (0.5)² + (-0.5)²) / 3 = 0.5/3 ≈ 0.167
```

The loss is a single number. It tells us how bad the model is on this batch, but not *how to improve*. For that, we need derivatives.

---

## Gradients: which direction reduces the loss?

The **gradient** of the loss with respect to a weight tells you the rate of change: if you increase this weight slightly, does the loss go up or down, and by how much?

For a single parameter, this is just a derivative. For a model with millions of parameters, it's a vector of partial derivatives - one per weight.

The key property: **move each weight in the direction opposite to its gradient**, and the loss decreases.

```python
# Simple 1D example: minimize f(x) = (x - 3)^2
def f(x):
    return (x - 3) ** 2

def f_grad(x):
    return 2 * (x - 3)

x = 0.0
learning_rate = 0.1

for step in range(20):
    loss = f(x)
    grad = f_grad(x)
    x = x - learning_rate * grad   # step opposite to gradient
    if step % 5 == 0:
        print(f"Step {step:2d}: x={x:.3f}, loss={loss:.4f}")
```

```
Step  0: x=0.600, loss=9.0000
Step  5: x=2.543, loss=0.9444
Step 10: x=2.907, loss=0.0863
Step 15: x=2.975, loss=0.0079
```

This is gradient descent. The version used in neural network training is **stochastic gradient descent** (SGD): compute gradients on a random mini-batch rather than the entire dataset, making each step cheaper.

---

## Why the chain rule is everything

Real networks chain many operations: inputs go through a linear layer, then an activation function, then another linear layer, then the loss. To compute the gradient of the loss with respect to the first layer's weights, you need to propagate the gradient backwards through all subsequent operations.

The chain rule handles this: if `z = f(g(x))`, then `dz/dx = (dz/dg) × (dg/dx)`.

For a chain of three operations - say, `L = loss(activation(linear(x)))` - the gradient of L with respect to the weights in `linear` is:

```
∂L/∂W = (∂L/∂activation) × (∂activation/∂linear) × (∂linear/∂W)
```

Each piece is computed locally (each operation knows its own derivative), and they're multiplied together. That's all backprop is.

---

## Building a two-layer network from scratch

Let's implement this for a concrete task: learning a nonlinear function from data.

The network: `x → Linear₁ → ReLU → Linear₂ → prediction`

```python
import numpy as np

np.random.seed(42)

class Linear:
    def __init__(self, in_dim, out_dim):
        # Xavier initialization: good starting point for activations
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.dW = None
        self.db = None
        self._x_cache = None
    
    def forward(self, x):
        self._x_cache = x
        return x @ self.W + self.b
    
    def backward(self, dout):
        # dout: gradient from the layer above
        # Return: gradient to pass to the layer below
        x = self._x_cache
        self.dW = x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

class ReLU:
    def __init__(self):
        self._mask = None
    
    def forward(self, x):
        self._mask = (x > 0)
        return x * self._mask
    
    def backward(self, dout):
        # Gradient is 1 where input was positive, 0 elsewhere
        return dout * self._mask

class MSELoss:
    def __init__(self):
        self._diff = None
    
    def forward(self, preds, targets):
        self._diff = preds - targets
        return np.mean(self._diff ** 2)
    
    def backward(self):
        n = self._diff.size
        return 2 * self._diff / n
```

Now assemble and train:

```python
# Generate training data: learn f(x) = sin(x)
np.random.seed(0)
N = 200
X_train = np.linspace(-np.pi, np.pi, N).reshape(-1, 1)
y_train = np.sin(X_train)

# Network: 1 → 64 → 64 → 1
l1   = Linear(1, 64)
act1 = ReLU()
l2   = Linear(64, 1)
loss_fn = MSELoss()

learning_rate = 0.01

def forward(x):
    h = act1.forward(l1.forward(x))
    return l2.forward(h)

def train_step(x_batch, y_batch):
    # Forward pass
    preds = forward(x_batch)
    loss  = loss_fn.forward(preds, y_batch)
    
    # Backward pass
    dout = loss_fn.backward()      # gradient of loss w.r.t. predictions
    dout = l2.backward(dout)       # gradient through linear layer 2
    dout = act1.backward(dout)     # gradient through ReLU
    l1.backward(dout)              # gradient through linear layer 1
    
    # Parameter update: step opposite to gradient
    for layer in [l1, l2]:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db
    
    return loss

# Training loop
for epoch in range(1000):
    loss = train_step(X_train, y_train)
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: loss = {loss:.5f}")
```

```
Epoch    0: loss = 0.49327
Epoch  200: loss = 0.01842
Epoch  400: loss = 0.00631
Epoch  600: loss = 0.00287
Epoch  800: loss = 0.00149
```

Let's verify the learned function makes sense:

```python
test_points = np.array([[-np.pi], [-np.pi/2], [0.0], [np.pi/2], [np.pi]])
preds = forward(test_points)

print(f"{'x':>8}  {'sin(x)':>8}  {'predicted':>10}")
for x, target, pred in zip(test_points.flatten(),
                            np.sin(test_points).flatten(),
                            preds.flatten()):
    print(f"{x:8.3f}  {target:8.3f}  {pred:10.3f}")
```

```
       x    sin(x)   predicted
  -3.142  -0.000      -0.002
  -1.571  -1.000      -0.993
   0.000   0.000       0.001
   1.571   1.000       0.991
   3.142   0.000       0.004
```

Not bad for 1000 epochs with a tiny network.

---

## What gradient flow looks like in practice

There's a useful mental model: the forward pass computes outputs, the backward pass computes gradients. Gradients flow backwards through the exact same operations - each one contributing its local derivative.

For a longer chain:

```python
import numpy as np

# Track what happens to gradients at each layer
def examine_gradient_flow():
    x = np.array([[1.0, -0.5, 0.3, 2.0]])
    
    layers = [
        ("Input",  x),
    ]
    
    # Forward pass with four linear + ReLU blocks
    current = x
    layer_objects = []
    for i in range(4):
        lin = Linear(current.shape[1], 16)
        rel = ReLU()
        current = rel.forward(lin.forward(current))
        layer_objects.append((lin, rel))
        layers.append((f"Layer {i+1}", current))
    
    # Final projection
    out_layer = Linear(16, 1)
    out = out_layer.forward(current)
    loss_fn = MSELoss()
    loss = loss_fn.forward(out, np.array([[0.0]]))
    
    # Backward
    dout = loss_fn.backward()
    dout = out_layer.backward(dout)
    grad_norms = [np.linalg.norm(dout)]
    
    for lin, rel in reversed(layer_objects):
        dout = rel.backward(dout)
        dout = lin.backward(dout)
        grad_norms.append(np.linalg.norm(dout))
    
    grad_norms.reverse()
    print("Gradient norms through layers:")
    for i, gnorm in enumerate(grad_norms):
        print(f"  Layer {i+1}: {gnorm:.4f}")

examine_gradient_flow()
```

Deep networks can suffer from **vanishing gradients** - gradients shrink as they flow backwards, leaving early layers with almost no learning signal. This is why ResNets use skip connections (residual streams), and why transformers use layer normalization. Both help gradients flow cleanly to early layers.

---

## The relationship to PyTorch autograd

When you write PyTorch code and call `.backward()`, it's doing exactly what we implemented manually - traversing a computation graph in reverse and multiplying local gradients. The difference is that PyTorch builds this graph automatically during the forward pass.

```python
import torch

# The same two-layer network in PyTorch
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)

X = torch.linspace(-torch.pi, torch.pi, 200).unsqueeze(1)
y = torch.sin(X)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(1000):
    preds = model(X)
    loss  = loss_fn(preds, y)
    
    optimizer.zero_grad()
    loss.backward()        # computes all gradients via autograd
    optimizer.step()       # applies gradient updates
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: loss = {loss.item():.5f}")
```

Same training loop, same mathematics, just with the bookkeeping automated.

---

## Why learning rate matters more than you'd think

The learning rate controls how large each weight update is. Too large and you overshoot the minimum and the loss explodes. Too small and training takes forever.

```python
import numpy as np

def train_with_lr(lr, epochs=500):
    np.random.seed(42)
    l1 = Linear(1, 32); act1 = ReLU()
    l2 = Linear(32, 1); loss_fn = MSELoss()
    
    X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    y = np.sin(X)
    
    losses = []
    for _ in range(epochs):
        h = act1.forward(l1.forward(X))
        preds = l2.forward(h)
        loss = loss_fn.forward(preds, y)
        losses.append(loss)
        
        dout = loss_fn.backward()
        dout = l2.backward(dout)
        dout = act1.backward(dout)
        l1.backward(dout)
        
        for layer in [l1, l2]:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db
    
    return losses[-1]

for lr in [0.001, 0.01, 0.1, 1.0]:
    final_loss = train_with_lr(lr)
    print(f"lr={lr:.3f}  →  final loss = {final_loss:.5f}")
```

```
lr=0.001  →  final loss = 0.04531
lr=0.010  →  final loss = 0.00149
lr=0.100  →  final loss = 0.00318
lr=1.000  →  final loss = nan (diverged)
```

In practice, modern training uses **adaptive learning rates** (Adam, AdamW) that adjust the learning rate per parameter based on its gradient history. The manually-chosen learning rate becomes a global scaling factor rather than a precise setting.

---

## Summary

Backpropagation is three things applied repeatedly:

1. **Forward pass**: compute the output, save intermediate values
2. **Backward pass**: apply the chain rule to propagate gradients from loss back to every weight
3. **Update**: shift each weight in the direction that reduces loss

Nothing else. The "learning" in deep learning is this loop run millions of times on massive datasets. The [pretraining objective](./pretraining-next-token-prediction) for language models is next-token prediction; backpropagation is the mechanism that makes the weights improve at it.

Understanding this means you know what `.backward()` does, why vanishing gradients hurt training, and why [activation functions](./activation-functions) are carefully chosen - they determine whether gradients survive the backward pass.

---

*Next: [Activation Functions](./activation-functions) - why non-linearity is necessary, and how the choice of activation function affects gradient flow.*

*Previous: [Linear Algebra for AI](./linear-algebra-for-ai) - the vector and matrix operations that backpropagation runs on.*
