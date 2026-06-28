---
layout: ../../layouts/PostLayout.astro
title: "The Token Economy: What It Costs to Run an LLM at Scale"
date: "2026-06-28"
description: "Every token you send to an LLM API costs money, and the costs compound quickly at scale. Understanding where the compute goes — prefill vs decode, context length effects, batching — helps you build applications that are both capable and economically viable."
tag: "ai-internals"
readingTime: 10
---

Prototype with a frontier model. Scale with economics in mind. That's the arc most AI products follow, and the transition from "works in a demo" to "profitable at scale" usually involves a hard look at token costs.

This isn't a pricing guide — prices change constantly. It's an analysis of *why* token costs are what they are, which helps you make architectural decisions that remain sound regardless of which specific numbers apply when you read this.

---

## Where compute goes

A single token's journey through a transformer involves:

1. **Embedding lookup**: table lookup, nearly free
2. **Self-attention**: `O(n)` per token (with KV cache), but the KV cache grows as context grows
3. **Feed-forward layers**: fixed cost per token, dominant for short contexts
4. **Unembedding + softmax**: vocabulary-sized vector, moderate cost

```python
def estimate_flops_per_token(
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_ff: int,
    vocab_size: int,
    seq_len: int,
) -> dict:
    """
    Approximate FLOP count for one forward pass token.
    Assumes KV cache is active (so attention is linear in seq_len per new token).
    """
    d_head = d_model // n_heads
    
    # Attention per layer (with KV cache)
    # New Q, K, V projections: 3 * d_model * d_model ops
    # Attention scores: d_head * seq_len ops per head
    # Value aggregation: same
    qkv_proj = 3 * d_model * d_model * 2    # factor 2 for multiply-add
    attn_scores = n_heads * d_head * seq_len * 2
    attn_output_proj = d_model * d_model * 2
    
    # FFN per layer
    ffn = 2 * d_model * n_ff * 2
    
    # Total per layer
    per_layer = qkv_proj + attn_scores + attn_output_proj + ffn
    total = per_layer * n_layers
    
    return {
        "flops_per_token": total,
        "attention_fraction": (n_layers * (qkv_proj + attn_scores + attn_output_proj)) / total,
        "ffn_fraction": (n_layers * ffn) / total,
    }

# GPT-3 175B
gpt3_flops = estimate_flops_per_token(
    d_model=12288, n_heads=96, n_layers=96,
    n_ff=4*12288, vocab_size=50257, seq_len=1024
)
print(f"GPT-3 FLOPs/token: {gpt3_flops['flops_per_token']/1e9:.1f}B")
print(f"  FFN fraction:     {gpt3_flops['ffn_fraction']:.1%}")
print(f"  Attention frac.:  {gpt3_flops['attention_fraction']:.1%}")
```

The key result: at typical context lengths (1K-4K tokens), the feed-forward layers dominate — typically 80-90% of compute. Attention becomes dominant only at very long contexts.

---

## Prefill vs decode: two different cost profiles

The generation lifecycle has two phases with fundamentally different characteristics:

**Prefill**: processing the entire input prompt in one pass. All input tokens are processed in parallel. Compute scales roughly linearly with prompt length.

**Decode**: generating output tokens one at a time. Each token requires a forward pass, but with KV cache, it's relatively cheap per token.

```python
def estimate_generation_cost(
    prompt_tokens: int,
    output_tokens: int,
    flops_per_prefill_token: float,    # higher — no KV cache yet
    flops_per_decode_token: float,     # lower — KV cache active
    gpu_flops: float,                   # A100: ~312 TFLOPS (fp16)
) -> dict:
    prefill_time  = (prompt_tokens * flops_per_prefill_token) / gpu_flops
    decode_time   = (output_tokens * flops_per_decode_token) / gpu_flops
    total_time    = prefill_time + decode_time
    
    return {
        "prefill_time_ms":  prefill_time * 1000,
        "decode_time_ms":   decode_time * 1000,
        "total_time_ms":    total_time * 1000,
        "time_to_first_token": prefill_time * 1000,  # latency before any output
    }

# Approximate numbers for a 7B model on an A100
flops_7b_per_token = 14e9  # ~2 FLOPs per parameter per token
a100_flops = 312e12        # 312 TFLOPS fp16

result = estimate_generation_cost(
    prompt_tokens=1000,
    output_tokens=500,
    flops_per_prefill_token=flops_7b_per_token,
    flops_per_decode_token=flops_7b_per_token * 0.7,  # KV cache saves ~30%
    gpu_flops=a100_flops,
)
for k, v in result.items():
    print(f"{k}: {v:.1f} ms")
```

This explains why APIs charge different rates for input vs output tokens: they have different computational profiles. Output tokens are typically 3-5× more expensive per token than input tokens on the same model.

---

## Context length and costs

KV cache memory grows linearly with context length. For long contexts (32K+), the KV cache can become the bottleneck — not compute, but memory bandwidth.

```python
def kv_cache_memory(
    n_layers: int,
    n_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # float16
) -> dict:
    """KV cache memory requirements."""
    # K and V caches per layer
    kv_bytes = 2 * n_layers * n_heads * d_head * seq_len * batch_size * dtype_bytes
    
    return {
        "kv_cache_gb": kv_bytes / 1024**3,
        "per_token_bytes": 2 * n_layers * n_heads * d_head * batch_size * dtype_bytes,
    }

# Llama-3-8B (GQA: 8 KV heads)
llama_8b_kv = kv_cache_memory(
    n_layers=32, n_heads=8,  # GQA: 8 KV heads
    d_head=128, seq_len=8192
)

# GPT-3 175B
gpt3_kv = kv_cache_memory(
    n_layers=96, n_heads=96,
    d_head=128, seq_len=8192
)

print("KV cache at 8K tokens:")
print(f"  Llama-3-8B (GQA):  {llama_8b_kv['kv_cache_gb']:.2f} GB")
print(f"  GPT-3 175B (MHA):  {gpt3_kv['kv_cache_gb']:.2f} GB")

# Scaling to 128K tokens
llama_128k = kv_cache_memory(n_layers=32, n_heads=8, d_head=128, seq_len=131072)
print(f"\nLlama-3-8B KV at 128K context: {llama_128k['kv_cache_gb']:.2f} GB")
# This is why long-context models need special infrastructure
```

For applications: long contexts are expensive not just in API price but in latency. The prefill phase for a 100K-token context can take seconds, and the KV cache has to be maintained in GPU memory throughout.

---

## Cost optimization strategies

### Prompt compression

```python
def estimate_compression_savings(
    original_prompt_tokens: int,
    compressed_prompt_tokens: int,
    n_output_tokens: int,
    input_price_per_1k: float,
    output_price_per_1k: float,
) -> dict:
    original_cost = (original_prompt_tokens / 1000 * input_price_per_1k +
                     n_output_tokens / 1000 * output_price_per_1k)
    
    compressed_cost = (compressed_prompt_tokens / 1000 * input_price_per_1k +
                       n_output_tokens / 1000 * output_price_per_1k)
    
    return {
        "original_cost_usd": original_cost,
        "compressed_cost_usd": compressed_cost,
        "savings": original_cost - compressed_cost,
        "compression_ratio": original_prompt_tokens / compressed_prompt_tokens,
    }

# Example: 10K input tokens compressed to 3K
result = estimate_compression_savings(
    original_prompt_tokens=10000,
    compressed_prompt_tokens=3000,
    n_output_tokens=500,
    input_price_per_1k=0.003,
    output_price_per_1k=0.015,
)
print(f"Original cost:   ${result['original_cost_usd']:.4f}")
print(f"Compressed cost: ${result['compressed_cost_usd']:.4f}")
print(f"Savings:         ${result['savings']:.4f} ({result['compression_ratio']:.1f}x compression)")
```

Techniques: LLMLingua and similar tools extract the essential content from long documents before passing to expensive models. Chunk-and-retrieve (RAG) also reduces effective token usage by only retrieving relevant sections.

### Model routing

Not every query needs a frontier model. A routing layer that classifies queries by complexity can direct simple requests to cheap models and hard requests to capable ones:

```python
QUERY_COMPLEXITY_EXAMPLES = {
    "simple": [
        "What is 2+2?",
        "Translate 'hello' to French",
        "Summarize this 2-sentence text:",
        "Is this email spam?",
    ],
    "medium": [
        "Write a Python function to parse CSV",
        "Explain the difference between TCP and UDP",
        "Summarize this 10-page document",
    ],
    "complex": [
        "Debug this multi-file codebase and explain the root cause",
        "Write a market analysis comparing 5 competitors",
        "Solve this competitive programming problem",
    ],
}

def route_query(query: str, cheap_model_fn, expensive_model_fn,
                complexity_threshold: float = 0.7) -> str:
    """
    Route queries to appropriate model based on estimated complexity.
    Uses a small classifier or heuristics.
    """
    complexity_score = estimate_complexity(query)  # 0.0 to 1.0
    
    if complexity_score < complexity_threshold:
        return cheap_model_fn(query)
    else:
        return expensive_model_fn(query)

# Typical cost ratio: GPT-4 vs GPT-3.5 ≈ 10-20x
# Routing 80% of queries to cheaper model → ~85% cost reduction
```

### Caching

For repeated or near-identical queries, semantic caching returns stored results without calling the model:

```python
import hashlib
from typing import Optional

class SemanticCache:
    def __init__(self, embedding_fn, similarity_threshold: float = 0.95):
        self.entries = []  # list of (embedding, response)
        self.embedding_fn = embedding_fn
        self.threshold = similarity_threshold
    
    def lookup(self, query: str) -> Optional[str]:
        query_emb = self.embedding_fn(query)
        
        for stored_emb, stored_response in self.entries:
            similarity = cosine_similarity(query_emb, stored_emb)
            if similarity >= self.threshold:
                return stored_response
        
        return None
    
    def store(self, query: str, response: str):
        query_emb = self.embedding_fn(query)
        self.entries.append((query_emb, response))

def cosine_similarity(a, b):
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Cache hit rate of 30-50% is achievable for structured applications
# (FAQ, documentation Q&A, code completion in common patterns)
```

---

## The economics at scale

At 1 million API calls per day with average 1K input + 500 output tokens per call:

```python
def daily_api_cost(
    n_calls: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_price: float,   # per 1K tokens
    output_price: float,  # per 1K tokens
) -> dict:
    daily_input  = n_calls * avg_input_tokens / 1000 * input_price
    daily_output = n_calls * avg_output_tokens / 1000 * output_price
    daily_total  = daily_input + daily_output
    
    return {
        "daily_usd": daily_total,
        "monthly_usd": daily_total * 30,
        "annual_usd": daily_total * 365,
    }

# 1M calls/day, 1K input + 500 output tokens
# Frontier model pricing (approximate)
costs = daily_api_cost(1_000_000, 1000, 500, 0.005, 0.015)
print("Frontier model:")
print(f"  Daily:   ${costs['daily_usd']:,.0f}")
print(f"  Monthly: ${costs['monthly_usd']:,.0f}")

# Cheaper mid-tier model
costs_mid = daily_api_cost(1_000_000, 1000, 500, 0.0005, 0.0015)
print("Mid-tier model:")
print(f"  Daily:   ${costs_mid['daily_usd']:,.0f}")
print(f"  Monthly: ${costs_mid['monthly_usd']:,.0f}")
```

At scale, the choice of model has 10× impact on cost. The business question is always: what's the minimum capable model that's good enough for this task?

---

## Summary

Token economy is the interplay between:

- **Compute cost**: proportional to parameters × tokens processed
- **Context length**: attention is sublinear with KV cache, but KV memory grows linearly
- **Prefill vs decode**: different cost profiles for processing input vs generating output
- **Pricing tiers**: input tokens cheaper than output tokens; models vary 10-50× in price per token

Efficient applications: compress inputs, route by complexity, cache frequent queries, and use the cheapest model that meets quality requirements for each task.

---

*Next: [LLM Agentic Architecture](./llm-agentic-architecture) — building systems where models take sequences of actions toward goals.*

*Previous: [DwarfStar4](./dwarfstar4-efficient-llm) — the engineering tradeoffs when hardware constraints dominate.*
