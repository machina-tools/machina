---
layout: ../../layouts/PostLayout.astro
title: "The Transformer Block: Layer Norm, Residuals, and Feed-Forward"
date: "2026-06-28"
description: "Self-attention is the core, but a transformer block has more: layer normalization, residual connections, and feed-forward layers. These aren't decoration — they're what makes the architecture trainable at depth. Here's how they work together and why each piece is necessary."
tag: "ai-internals"
readingTime: 11
---

The [self-attention](./self-attention-from-scratch) mechanism gets most of the attention (pun intended), but a transformer block has three other components that are equally important for making the architecture work: **layer normalization**, **residual connections**, and **feed-forward layers**. Remove any of them and training degrades noticeably. Remove two and it breaks.

This article covers what each does and why it's there.

---

## The residual stream

The central structural idea in a transformer is the **residual stream**: a vector that flows through every layer, getting updated additively at each step.

In a standard ResNet-style architecture:

```
x_out = layer(x_in) + x_in
```

The layer output is *added* to the input rather than replacing it. This has a critical consequence: gradients can flow straight from the output to the input without passing through any layer — they take the shortcut path.

Without residual connections, the gradient must flow through every operation in every layer to reach early weights. Each multiplication by a weight matrix shrinks the gradient (or, with bad initialization, explodes it). With residual connections, the identity path provides a clean gradient highway.

```python
import torch
import torch.nn as nn

class BlockWithoutResidual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.linear(x))  # no residual

class BlockWithResidual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.linear(x)) + x  # residual added

# Compare gradient magnitude at input for 20-layer networks
def measure_input_gradient(BlockType, depth=20):
    torch.manual_seed(0)
    blocks = nn.Sequential(*[BlockType(64) for _ in range(depth)])
    x = torch.randn(1, 64, requires_grad=True)
    loss = blocks(x).sum()
    loss.backward()
    return x.grad.abs().mean().item()

no_res = measure_input_gradient(BlockWithoutResidual)
with_res = measure_input_gradient(BlockWithResidual)

print(f"Without residual: {no_res:.8f}")
print(f"With residual:    {with_res:.6f}")
# The residual version has much larger (usable) gradients at the input
```

In the transformer context, the residual stream starts as the embedding of each token. At each layer, attention output is added, then feed-forward output is added. The stream accumulates updates rather than being overwritten.

---

## Layer normalization

The second problem with deep networks: activation magnitudes grow unpredictably as information passes through many layers. Without intervention, some neurons saturate, some go to zero, and training becomes unstable.

**Layer normalization** re-centers and rescales activations at each layer to have zero mean and unit variance, then applies learned scale and bias parameters:

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))   # learned
        self.bias  = nn.Parameter(torch.zeros(d_model))  # learned
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.bias

# Demonstration
torch.manual_seed(42)
x = torch.randn(2, 8, 64) * 5 + 3  # shifted and scaled
ln = LayerNorm(64)

before = x[0, 0]
after  = ln(x)[0, 0]

print(f"Before — mean: {before.mean():.3f}, std: {before.std():.3f}")
print(f"After  — mean: {after.mean():.3f},  std: {after.std():.3f}")
```

```
Before — mean: 2.887, std: 5.203
After  — mean: 0.000, std: 1.000
```

The key design choice: normalization is over the **feature dimension** (the `d_model` axis), not the batch dimension. This means the statistics are computed independently for each token position — unlike batch normalization, which would couple different positions' statistics. This is crucial for sequence models where positions have different statistical properties.

**Pre-norm vs post-norm**: The original "Attention Is All You Need" paper placed LayerNorm *after* the residual addition (post-norm). Modern transformers like GPT-2 and LLaMA use *pre-norm* — normalize *before* the attention or feed-forward operation. Pre-norm generally trains more stably, especially at large scale.

```python
# Post-norm (original paper)
# x = LayerNorm(x + Attention(x))

# Pre-norm (GPT-2, LLaMA, most modern models)
# x = x + Attention(LayerNorm(x))
```

---

## Feed-forward layers

After attention, each transformer block applies a **feed-forward network** (FFN) independently to each token position. This is often called the MLP sublayer.

The standard architecture: expand by 4×, apply non-linearity, project back down.

```python
class FeedForward(nn.Module):
    """Standard transformer feed-forward block (GPT-2 style with GELU)."""
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        d_ff = d_model * expansion
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

# For d_model=768: parameters
ffn = FeedForward(768)
params = sum(p.numel() for p in ffn.parameters())
print(f"FFN parameters: {params:,}")
# 768*3072 + 3072 + 3072*768 + 768 = ~4.7M
```

The feed-forward layer operates on each position independently — there's no communication between positions here. All cross-position communication happens in attention. The FFN is where "factual associations" are thought to be stored: the weights encode patterns like "if the residual stream contains a 'capital city query', add Paris to the residual stream".

Empirically, the FFN layers store a large fraction of the model's world knowledge. Targeted editing experiments (like ROME — Rewriting the Model's Facts) modify specific FFN weight matrices to change factual associations.

---

## Putting it all together: the transformer block

```python
class TransformerBlock(nn.Module):
    """GPT-2 style transformer block with pre-norm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # Attention sublayer: pre-norm + residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask,
                                need_weights=False)
        x = x + attn_out
        
        # Feed-forward sublayer: pre-norm + residual
        x = x + self.ffn(self.ln2(x))
        
        return x

# Assemble a small GPT-like model
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final  = nn.LayerNorm(d_model)
        self.unembed   = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embedding and unembedding weights
        self.unembed.weight = self.embed.weight
    
    def forward(self, token_ids: torch.Tensor):
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        
        x = self.embed(token_ids) + self.pos_embed(positions)
        
        # Causal mask: lower triangular
        mask = torch.triu(torch.ones(seq_len, seq_len,
                                      device=token_ids.device), diagonal=1).bool()
        
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        
        x = self.ln_final(x)
        return self.unembed(x)  # logits over vocabulary

# Create a small model and verify shapes
model = MiniGPT(
    vocab_size=1000, d_model=128, n_heads=4, n_layers=3, max_seq_len=512
)
token_ids = torch.randint(0, 1000, (2, 16))  # batch=2, seq_len=16
logits = model(token_ids)
print(f"Input:  {token_ids.shape}")   # (2, 16)
print(f"Output: {logits.shape}")      # (2, 16, 1000) — logits per token per vocab
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")
```

---

## What each component is doing functionally

It helps to think of the residual stream as a shared communication channel, and the blocks as read-write operations on it:

**Attention** reads from the residual stream at all positions, computes relational information, and writes back. It answers "given what I know about every other position, what should I add to my own representation?"

**Feed-forward** reads from the residual stream at one position and applies a learned lookup. It answers "given the current state of my representation, what factual or structural information should I add?"

**Layer norm** keeps the magnitudes in a well-behaved range so neither the reads nor writes blow up numerically.

**Residual connections** ensure that whatever gets added, the original information is still available. Early layers don't have to "remember" things for later layers — the information is always in the stream.

This is why the "logit lens" technique works: because information is accumulated in the residual stream, applying the final unembedding matrix to intermediate layer states gives meaningful (if noisier) predictions. See the [embedding article](./embedding-words-as-vectors) for the code.

---

## GPT-2 architecture at a glance

| Config | GPT-2 Small | GPT-2 Large | GPT-3 |
|---|---|---|---|
| d_model | 768 | 1280 | 12288 |
| n_heads | 12 | 20 | 96 |
| n_layers | 12 | 36 | 96 |
| d_ff | 3072 | 5120 | 49152 |
| Params | ~117M | ~774M | ~175B |

The architecture barely changes as models scale — mainly deeper, wider, more heads. The block structure remains the same.

---

## Summary

A transformer block combines four operations:

1. **Pre-norm** (LayerNorm): stabilize the input before attention
2. **Self-attention** + **residual**: attend to all other positions, add to stream
3. **Pre-norm** again: stabilize before feed-forward
4. **Feed-forward** + **residual**: apply per-position transformation, add to stream

Stack N of these, add an embedding layer at the front and an unembedding layer at the back, and you have a language model. The elegance is that this architecture scales cleanly — the same block design works from 100M to 100B parameters.

---

*Next: [Pretraining — Why Predicting the Next Word Is Enough](./pretraining-next-token-prediction) — how the model learns to fill these layers with useful information.*

*Previous: [Self-Attention from Scratch](./self-attention-from-scratch) — the attention mechanism this block wraps.*
