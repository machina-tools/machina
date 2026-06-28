---
layout: ../../layouts/PostLayout.astro
title: "Self-Attention from Scratch: What the Transformer Block Actually Does"
date: "2026-06-28"
description: "Self-attention is the operation that lets every token in a sequence look at every other token. It's the reason transformers handle long-range dependencies and context in a way earlier architectures couldn't. Here's the full mechanism — queries, keys, values, multi-head attention — built from scratch in PyTorch."
tag: "ai-internals"
readingTime: 13
---

Before transformers, sequence models processed text token by token. An LSTM reading "the bank near the river" would process "bank" before "river", storing a summary of prior context in a fixed-size hidden state. By the time it reached "river", the earlier context had been compressed and partially forgotten.

Self-attention doesn't read sequentially. It processes all positions in parallel and lets every token attend directly to every other token — with no compression, no forgetting. That's the structural reason transformers outperformed LSTMs at almost every task once the architecture was tuned.

---

## The intuition: query, key, value

Imagine a library lookup system:

- Your **query** is what you're looking for
- Each book has a **key** — metadata describing its content
- The **value** is the actual content of the book

You compare your query against all keys to find the best matches, then retrieve a weighted combination of the corresponding values.

Self-attention is exactly this. For each token in a sequence:
1. Project the token's embedding into a **query** vector — "what am I looking for?"
2. Project all tokens (including itself) into **key** vectors — "what does each position contain?"
3. Compute similarity scores between the query and all keys
4. Normalize the scores with softmax (so they sum to 1)
5. Use the scores as weights to take a weighted sum of **value** vectors

The output is a new representation for this token that incorporates context from the rest of the sequence.

---

## The math

For a sequence of `n` tokens, each represented as a `d_model`-dimensional vector, we apply three learned weight matrices:

```
Q = X @ W_Q    (n × d_k)
K = X @ W_K    (n × d_k)  
V = X @ W_V    (n × d_v)
```

Then:

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

The `sqrt(d_k)` scaling prevents the dot products from growing too large (which would push softmax into saturation where gradients vanish).

---

## Implementation from scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_head)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        scores = Q @ K.transpose(-2, -1) / scale  # (batch, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)        # (batch, seq_len, seq_len)
        out = weights @ V                           # (batch, seq_len, d_head)
        
        return out, weights

# Test it
torch.manual_seed(42)
d_model, d_head = 64, 16
batch, seq_len = 2, 8

attn = SingleHeadAttention(d_model, d_head)
x = torch.randn(batch, seq_len, d_model)
out, weights = attn(x)

print(f"Input shape:   {x.shape}")       # (2, 8, 64)
print(f"Output shape:  {out.shape}")     # (2, 8, 16)
print(f"Weights shape: {weights.shape}") # (2, 8, 8)
print(f"Weights sum (should be 1): {weights[0, 0].sum():.4f}")
```

The attention weight matrix is `(seq_len, seq_len)` — each row shows how much that token's new representation draws from each other token. Visualizing this matrix reveals what the model is "looking at".

---

## Causal masking

In language model generation (predicting the next token), each position should only attend to previous positions — not future ones. This is enforced with a causal mask:

```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower triangular mask: position i can only attend to positions ≤ i."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

mask = causal_mask(5)
print(mask)
```

```
tensor([[1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.]])
```

Position 0 attends only to itself. Position 4 attends to all five positions. The masked positions get `-inf` before softmax, which becomes 0 after softmax — effectively blocking those connections.

```python
# Demonstration: attention pattern with causal masking
attn_causal = SingleHeadAttention(64, 16)
mask = causal_mask(seq_len).unsqueeze(0)  # add batch dim
out_causal, weights_causal = attn_causal(x, mask)

# Check: upper triangle should be zero
upper_triangle_sum = weights_causal[0].triu(diagonal=1).sum().item()
print(f"Upper triangle weight sum: {upper_triangle_sum:.6f}")  # ~0.0
```

---

## Multi-head attention

Using a single attention head limits the model to one "type" of relationship between tokens. In practice, different aspects of context are useful simultaneously: syntactic relations, coreference, semantic similarity, positional proximity.

Multi-head attention runs several attention heads in parallel with different weight matrices, then concatenates and projects the results:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        
        # Single projections for efficiency (could also use n_heads × individual heads)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, d_model) → (batch, n_heads, seq, d_head)."""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_head)
        return x.transpose(1, 2)  # (batch, n_heads, seq, d_head)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse of _split_heads."""
        batch, n_heads, seq, d_head = x.shape
        x = x.transpose(1, 2)  # (batch, seq, n_heads, d_head)
        return x.contiguous().view(batch, seq, n_heads * d_head)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        
        scale = math.sqrt(self.d_head)
        scores = Q @ K.transpose(-2, -1) / scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq, seq)
        out = weights @ V                    # (batch, n_heads, seq, d_head)
        
        out = self._merge_heads(out)  # (batch, seq, d_model)
        return self.W_o(out)

# Test multi-head attention
mha = MultiHeadAttention(d_model=64, n_heads=4)
out = mha(x, mask=causal_mask(seq_len).unsqueeze(0))
print(f"MHA output: {out.shape}")  # (2, 8, 64) — same as input
```

GPT-2 small uses 12 attention heads per layer with `d_model=768`. Each head sees 64 dimensions. GPT-3 uses 96 heads with `d_model=12288`. The heads specialize — some attend to syntax, some to position, some to semantic similarity — but this specialization emerges from training, not from any explicit design.

---

## Positional encoding

Self-attention has no notion of order — the attention scores depend only on token content, not position. "The dog bit the man" and "The man bit the dog" would produce identical attention scores if processed independently.

Positional encoding adds position information by adding a position-specific vector to each embedding before attention:

```python
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the original Attention Is All You Need."""
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_seq, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
```

Modern models (LLaMA, GPT-4) use learned rotary positional embeddings (RoPE) instead of sinusoidal, which better handle variable context lengths. The principle is the same: inject position into the representation before or during attention.

---

## Visualizing what attention looks at

The attention weight matrix is directly interpretable. Here's a minimal example:

```python
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained("gpt2")

sentence = "The trophy didn't fit in the bag because it was too big"
tokens = model.to_tokens(sentence)
token_strings = [model.to_string(t.item()) for t in tokens[0]]

_, cache = model.run_with_cache(tokens)

# Look at layer 8 (of 12), head 5 — known to track coreference in GPT-2
layer, head = 8, 5
attn_weights = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # (seq, seq)

# Find what "it" attends to
it_pos = [i for i, t in enumerate(token_strings) if 'it' in t.lower()]
if it_pos:
    pos = it_pos[0]
    print(f"Token '{token_strings[pos]}' at position {pos} attends to:")
    top_k = attn_weights[pos].topk(5)
    for score, idx in zip(top_k.values, top_k.indices):
        print(f"  '{token_strings[idx]}' ({score:.3f})")
```

This is the starting point for mechanistic interpretability work — mapping which heads implement which linguistic operations.

---

## Computational cost

Attention is quadratic in sequence length: a sequence of length N produces an N×N attention matrix. Doubling the sequence length quadruples the attention computation.

```python
import time

def attention_time(seq_len: int, d_model: int = 768, n_heads: int = 12, batch: int = 1):
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch, seq_len, d_model)
    
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = mha(x)
    return (time.time() - start) / 50 * 1000  # ms

for seq in [128, 256, 512, 1024]:
    t = attention_time(seq)
    print(f"seq_len={seq:4d}: {t:.2f}ms")
```

This quadratic cost is why extending context windows is hard. A 128k-token context (as in Claude 3's longest context) requires architectural modifications — sparse attention, sliding windows, or learned routing — not just a bigger matrix.

---

## Summary

Self-attention lets every token directly access information from every other token in the sequence:

1. Project each token into Q, K, V vectors
2. Compute all Q-K dot products, scale, and softmax
3. Take a weighted sum of V vectors
4. Run multiple heads in parallel, merge, and project

The output for each token is a new representation that incorporates context from the entire sequence. This is what makes transformers effective at long-range dependencies — subject-verb agreement across clauses, coreference, cross-sentence reasoning.

The attention weights are the most interpretable part of the transformer; visualizing them is where [mechanistic interpretability](./mechanistic-interpretability) starts.

---

*Next: [The Full Transformer Block](./transformer-block) — layer normalization, residual connections, and feed-forward layers that wrap around attention.*

*Previous: [Tokenization](./tokenization) — the step that converts text to the vectors that self-attention operates on.*
