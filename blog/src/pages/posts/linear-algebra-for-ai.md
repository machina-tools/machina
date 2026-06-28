---
layout: ../../layouts/PostLayout.astro
title: "The Linear Algebra Behind Every AI Model"
date: "2026-06-28"
description: "Every operation inside a transformer — attention, embeddings, feed-forward layers — is linear algebra in disguise. Vectors, matrices, dot products, cosine similarity. You don't need to be a mathematician, but you do need to understand what these operations actually compute. This is the foundation for everything else in this series."
tag: "ai-internals"
readingTime: 12
---

If you want to understand what a transformer actually does, you need a working understanding of linear algebra. Not all of it — eigendecomposition and singular value decomposition won't appear much in this series. But vectors, matrices, dot products, and cosine similarity come up constantly. This article covers exactly what you need, with concrete code you can run.

---

## Vectors: points in space

A vector is a list of numbers. When you write `[0.3, -0.7, 1.2]`, that's a vector in three-dimensional space — a point, or equivalently an arrow from the origin to that point.

In AI, vectors are high-dimensional. GPT-2's embedding vectors are 768 numbers long. LLaMA-3-8B uses 4096. That's a point in a 768 or 4096-dimensional space that you can't visualize, but can compute with using exactly the same rules as 2D or 3D.

```python
import numpy as np

# A 3D vector
v = np.array([0.3, -0.7, 1.2])

# High-dimensional — same operations work
v_768 = np.random.randn(768)

print(f"3D shape: {v.shape}")
print(f"768D shape: {v_768.shape}")
print(f"3D values: {v}")
```

The key insight: your spatial intuition transfers to high dimensions, imperfectly but enough. When I say two embedding vectors are "close together", I mean what you'd expect geometrically — they point in roughly the same direction.

---

## The dot product

If you learn one operation from this article, make it the dot product.

Given two vectors of the same length, the dot product multiplies corresponding elements and sums:

```
a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
```

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Manual: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
dot_manual = sum(ai * bi for ai, bi in zip(a, b))

# NumPy
dot_np = np.dot(a, b)
dot_at  = a @ b  # same thing, @ is the matrix multiply operator

print(dot_manual, dot_np, dot_at)  # all 32.0
```

The geometric meaning: the dot product is large when vectors point in the same direction, zero when they're perpendicular, and negative when they point opposite ways.

```python
same      = np.array([1.0, 1.0, 1.0])
opposite  = np.array([-1.0, -1.0, -1.0])
perp_a    = np.array([1.0, 0.0])
perp_b    = np.array([0.0, 1.0])

print(same @ same)      # 3.0  — parallel, maximum alignment
print(same @ opposite)  # -3.0 — anti-parallel
print(perp_a @ perp_b)  # 0.0  — perpendicular, no alignment
```

In transformers, dot products show up everywhere: attention scores, projection operations, the final step that converts internal representations to output probabilities. Everything traces back to this.

---

## Vector norms

The **norm** of a vector is its length — distance from the origin. It's just the Pythagorean theorem generalized:

```
||v|| = sqrt(v[0]² + v[1]² + ... + v[n]²)
```

```python
v = np.array([3.0, 4.0])               # 3-4-5 triangle
print(np.linalg.norm(v))               # 5.0

v_rand = np.random.randn(768)
print(f"768-dim norm: {np.linalg.norm(v_rand):.2f}")
# Typically ~27.7 — sqrt(768) for standard normals
```

**Normalization** divides by the norm, giving a unit vector (length 1) that preserves direction:

```python
v = np.array([3.0, 4.0])
v_unit = v / np.linalg.norm(v)
print(v_unit)                    # [0.6, 0.8]
print(np.linalg.norm(v_unit))    # 1.0
```

Layer normalization in transformers does exactly this to the residual stream at each layer — keeping activation magnitudes in a predictable range regardless of what the preceding layers computed.

---

## Cosine similarity

Raw dot products have a problem: they're sensitive to vector magnitude. A vector twice as long produces a dot product twice as large, even if it points in exactly the same direction.

Cosine similarity fixes this by normalizing both vectors first:

```
cos(θ) = (a · b) / (||a|| × ||b||)
```

The result is the cosine of the angle between vectors — always between −1 and 1, independent of magnitude.

```python
from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Same direction → 1.0 (regardless of length difference)
a = np.array([1.0, 2.0, 3.0])
b = np.array([3.0, 6.0, 9.0])  # 3x longer, same direction
print(cosine_sim(a, b))   # 1.0

# Perpendicular → 0.0
print(cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])))  # 0.0

# Opposite → −1.0
print(cosine_sim(np.array([1.0, 1.0]), np.array([-1.0, -1.0])))  # -1.0
```

This is how the [embedding](./embedding-words-as-vectors) for "cat" is compared to the embedding for "dog". They don't need to have the same magnitude — the cosine similarity between their directions is what matters.

---

## Matrices and matrix multiplication

A matrix is a 2D array: rows × columns.

```python
M = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
])
print(M.shape)  # (3, 4) — 3 rows, 4 columns
```

Matrix multiplication `A @ B` takes an `(m, k)` matrix and a `(k, n)` matrix and produces an `(m, n)` matrix. Each output element is the dot product of a row from A and a column from B.

```python
A = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
B = np.array([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

C = A @ B
# C[0,0] = row 0 of A · col 0 of B = 1*5 + 2*7 = 19
# C[0,1] = row 0 of A · col 1 of B = 1*6 + 2*8 = 22
print(C)
# [[19. 22.]
#  [43. 50.]]
```

The shape rule: `(m, k) @ (k, n) → (m, n)`. Inner dimensions must match; outer dimensions become output shape.

Geometrically, matrix multiplication is a linear transformation: it rotates, scales, and shears vectors. When an embedding vector of shape `(768,)` is multiplied by a weight matrix `(768, 64)`, it's a projection into a 64-dimensional subspace.

---

## Batched operations

Neural networks process many inputs simultaneously. A batch of 32 sequences, each with 128 tokens, each a 768-dimensional vector, is a `(32, 128, 768)` tensor.

Both NumPy and PyTorch handle this transparently:

```python
import torch

batch = 32
seq_len = 128
d_model = 768
d_out   = 64

X = torch.randn(batch, seq_len, d_model)  # batch of sequences
W = torch.randn(d_model, d_out)            # weight matrix

out = X @ W   # broadcasts: (32, 128, 768) @ (768, 64) → (32, 128, 64)
print(out.shape)  # torch.Size([32, 128, 64])
```

The `@` operator automatically broadcasts across leading dimensions. This is why transformers can process entire batches in one GPU operation.

---

## Transpose

The transpose of a matrix flips rows and columns:

```python
M = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
print(M.T.shape)                        # (3, 2)

# M[i, j] becomes M.T[j, i]
```

In transformers, the transpose appears in two key places. First, in the attention score computation `Q @ K.T` — you need K transposed to take dot products between queries and keys. Second, the unembedding matrix (which converts model outputs to token probabilities) is the transpose of the embedding matrix, and they share the same weights.

---

## Putting it together: the attention formula

The core of self-attention (covered fully in [Self-Attention from Scratch](./self-attention-from-scratch)) is:

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

Breaking it down:
- `Q @ K.T`: dot product between each query and each key — similarity scores
- `/ sqrt(d_k)`: scale to keep scores in a well-behaved range for softmax
- `softmax(...)`: convert scores to probabilities that sum to 1
- `@ V`: weighted sum of value vectors

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    weights = F.softmax(scores, dim=-1)
    return weights @ V

# 1 sequence, 6 tokens, 16-dimensional head
Q = torch.randn(1, 6, 16)
K = torch.randn(1, 6, 16)
V = torch.randn(1, 6, 16)

out = attention(Q, K, V)
print(out.shape)  # (1, 6, 16) — same shape as input
```

Every piece of this is linear algebra. The entire transformer block is linear algebra with nonlinearities ([activation functions](./activation-functions)) punctuating the flow.

---

## Summary

The operations you need:

| Operation | Notation | What it does |
|---|---|---|
| Dot product | `a @ b` | Measures alignment between two vectors |
| Norm | `‖v‖` | Length of a vector |
| Cosine similarity | `(a·b)/(‖a‖‖b‖)` | Alignment, normalized for magnitude |
| Matrix multiply | `A @ B` | Linear transformation |
| Transpose | `A.T` | Flip rows and columns |

These five operations appear in every article in this series, sometimes explicitly, often silently. Keeping them clear means the machinery of attention, embeddings, and residual streams makes sense rather than feeling like symbol manipulation.

---

*Next: [Backpropagation from Scratch](./backpropagation-from-scratch) — how gradient descent adjusts every weight in the model to improve predictions.*

*This article is part of a series on how AI models work internally. Start at the beginning or jump to any topic.*
