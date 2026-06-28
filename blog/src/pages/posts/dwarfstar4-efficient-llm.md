---
layout: ../../layouts/PostLayout.astro
title: "DwarfStar4: Building a Language Model That Fits in Your Pocket"
date: "2026-06-28"
description: "What does it take to build a capable language model that runs on minimal hardware - a laptop, a Raspberry Pi, a phone? Salvatore Sanfilippo's DwarfStar4 project explores this question with a focus on aggressive quantization, architectural tradeoffs, and what 'small but useful' actually means in practice."
tag: "ai-internals"
readingTime: 11
---

Most discussions about language models focus on scaling up: more parameters, more data, more compute, better benchmarks. DwarfStar4 is a project in the opposite direction - the question isn't "how good can we make a model with unlimited resources?" but "how useful can a model be with tight hardware constraints?"

Salvatore Sanfilippo (antirez), best known as the creator of Redis, has been working on DwarfStar4 as an exploration of this territory. The goal: a model small enough to run without a GPU, quantized aggressively, but retained capability in the specific tasks it's optimized for.

This is a case study in the engineering decisions that emerge when hardware constraints are the primary constraint rather than benchmark performance.

---

## Why small models matter

The default assumption in AI tooling is that models run in the cloud. Someone else manages the infrastructure; you call an API. This works until it doesn't:

- **Privacy**: medical records, proprietary code, and sensitive documents that can't leave the machine
- **Latency**: applications that need sub-100ms responses without network round trips
- **Cost at scale**: API pricing that becomes prohibitive at high query volumes
- **Reliability**: critical applications that can't depend on network availability

For these cases, the choice isn't between a large frontier model and a small model - it's between a small model and no model. The relevant benchmark isn't "does this match GPT-4?" but "is this useful for the specific task, running on this specific hardware?"

---

## The quantization stack for extreme efficiency

At the aggressive end of quantization (below 4 bits per weight), standard techniques break down. The weight distributions that INT4 handles reasonably well become too coarse at INT2 or INT3.

Sanfilippo's work with DwarfStar4 pushes into this territory with several techniques:

**Per-token quantization of activations**: In addition to quantizing weights, quantize the activations at inference time. Each token's activation vector gets its own quantization scale, computed on the fly.

```python
import torch
import torch.nn.functional as F

def quantize_activations_per_token(x: torch.Tensor, n_bits: int = 8) -> tuple:
    """
    Per-token asymmetric quantization of activations.
    Each token (each row) gets its own scale and zero-point.
    
    This is how LLM.int8() and similar approaches work.
    """
    # x: (seq_len, d_model)
    q_max = 2**n_bits - 1
    
    min_vals = x.min(dim=-1, keepdim=True).values
    max_vals = x.max(dim=-1, keepdim=True).values
    
    scales = (max_vals - min_vals) / q_max
    scales = scales.clamp(min=1e-8)  # avoid division by zero
    zero_points = (-min_vals / scales).round().clamp(0, q_max)
    
    quantized = ((x / scales) + zero_points).round().clamp(0, q_max).to(torch.uint8)
    
    return quantized, scales, zero_points

def dequantize_activations(quantized, scales, zero_points):
    return (quantized.float() - zero_points) * scales

# Demonstration on a realistic activation tensor
x = torch.randn(16, 512)  # 16 tokens, 512-dim model
q, scales, zps = quantize_activations_per_token(x, n_bits=8)
dq = dequantize_activations(q, scales, zps)

recon_error = (x - dq).abs().mean().item()
print(f"Per-token INT8 activation quantization error: {recon_error:.6f}")
```

**Mixed-precision layers**: Not all layers are equally sensitive to quantization. Attention layers and the first/last few layers of the model are more sensitive - they receive more fine-grained treatment while the bulk of the model uses more aggressive compression.

```python
def get_layer_quantization_config(n_layers: int) -> dict:
    """
    Heuristic: use higher precision for sensitive layers.
    
    - First 2 layers: embedding and early processing, int8
    - Middle layers: bulk of computation, int4 or int3
    - Last 2 layers: output preparation, int8
    - Attention projections: int8 (sensitive to precision)
    - FFN layers: int4 (less sensitive)
    """
    config = {}
    for i in range(n_layers):
        if i < 2 or i >= n_layers - 2:
            config[f"layer_{i}"] = {"attn": "int8", "ffn": "int8"}
        else:
            config[f"layer_{i}"] = {"attn": "int8", "ffn": "int4"}
    return config

# For a 12-layer model:
config = get_layer_quantization_config(12)
total_attn_int8  = sum(1 for c in config.values() if c["attn"] == "int8")
total_ffn_int4   = sum(1 for c in config.values() if c["ffn"] == "int4")
print(f"Attention: {total_attn_int8}/12 layers at int8")
print(f"FFN:       {total_ffn_int4}/12 layers at int4")
```

---

## Architecture choices for small models

Beyond quantization, architectural choices matter more in small models because there's less total capacity to compensate for poor decisions.

**Grouped query attention (GQA)**: Instead of one K, V head per Q head, share K and V across groups of Q heads. For a model with 8 attention heads, using 2 KV heads (GQA-2) cuts KV cache memory by 4× with minimal quality loss.

```python
class GroupedQueryAttention(torch.nn.Module):
    """
    Grouped Query Attention: n_q_heads query heads, n_kv_heads key/value heads.
    Each KV head is shared by (n_q_heads / n_kv_heads) query heads.
    """
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads  = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups   = n_q_heads // n_kv_heads
        self.d_head     = d_model // n_q_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = torch.nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor):
        batch, seq, d = x.shape
        
        Q = self.W_q(x).view(batch, seq, self.n_q_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        # Expand K and V to match Q head count
        K = K.repeat_interleave(self.n_groups, dim=1)  # (batch, n_q_heads, seq, d_head)
        V = V.repeat_interleave(self.n_groups, dim=1)
        
        scale = self.d_head ** 0.5
        scores = Q @ K.transpose(-2, -1) / scale
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(batch, seq, d)
        return self.W_o(out)

# Parameter comparison
standard_attn_params = 4 * 512 * 512   # 4 projections: Q, K, V, O
gqa_4_params = (512*512) + 2*(512*(512//4)) + 512*512  # Q, 2 KV (quarter size), O
print(f"Standard attention params: {standard_attn_params:,}")
print(f"GQA (4 groups) params:     {gqa_4_params:,}")
```

**RoPE (Rotary Position Embedding)**: Encodes position information in attention computations rather than in the embedding layer. Better at handling variable context lengths without separate position embedding tables.

**Smaller vocabulary**: Standard GPT-2 vocabulary is 50,257 tokens. For a model targeting a specific domain or language, a smaller vocabulary (10K-20K tokens) reduces the embedding matrix and unembedding matrix size significantly.

---

## What "useful" means at small scale

A 500M parameter model won't match GPT-4 at general knowledge or complex reasoning. The question is what it can do well.

For targeted applications, small models trained on domain-specific data can outperform large general models:

```python
# Evaluation framework: task-specific vs general benchmark
EVALUATION_SCENARIOS = {
    "code_completion": {
        "small_specialized": 0.71,   # fine-tuned on code
        "large_general": 0.68,       # tested on same code tasks
        "notes": "Domain specialization compensates for size",
    },
    "document_qa": {
        "small_specialized": 0.64,
        "large_general": 0.89,
        "notes": "General knowledge tasks favor large models",
    },
    "sentiment_classification": {
        "small_specialized": 0.93,
        "large_general": 0.91,
        "notes": "Small fine-tuned model essentially matches large model",
    },
    "multi_hop_reasoning": {
        "small_specialized": 0.38,
        "large_general": 0.74,
        "notes": "Complex reasoning strongly favors large models",
    },
}

for task, scores in EVALUATION_SCENARIOS.items():
    winner = "small" if scores["small_specialized"] > scores["large_general"] else "large"
    print(f"{task:<25} small={scores['small_specialized']:.2f} large={scores['large_general']:.2f} → {winner}")
```

The pattern is consistent: for reasoning and knowledge tasks, size wins. For pattern recognition on in-distribution tasks (classification, completion, simple QA on specific domains), specialization can compensate.

---

## Running quantized models locally: practical notes

```python
# Example: using a small quantized model via llama.cpp / ctransformers
# This runs on CPU, no GPU required

from ctransformers import AutoModelForCausalLM as CAutoModel

def load_local_model(model_path: str, model_type: str = "mistral"):
    """
    Load a 4-bit GGUF model on CPU.
    Works on any machine with enough RAM (model_size * ~1.5 for overhead).
    """
    model = CAutoModel.from_pretrained(
        model_path,
        model_type=model_type,
        gpu_layers=0,           # CPU-only
        threads=4,              # use 4 CPU threads
        context_length=2048,
    )
    return model

# Approximate inference speed on common hardware (Q4 quantized):
inference_speeds = {
    "M1 MacBook (8GB)":          "30-40 tok/s",
    "Modern desktop CPU (32GB)": "15-25 tok/s",
    "Raspberry Pi 5 (8GB)":      "2-5 tok/s",
    "RTX 3080 (10GB)":           "80-120 tok/s",
    "A100 (80GB)":               "400-600 tok/s",
}

for hw, speed in inference_speeds.items():
    print(f"{hw:<35} {speed}")
```

Raspberry Pi speed is enough for batch offline tasks, but too slow for interactive use. Desktop CPU is borderline interactive at small models. M1 Macs are currently the best consumer hardware for local inference thanks to high memory bandwidth.

---

## The engineering philosophy

DwarfStar4 represents a specific design philosophy:

1. **Constraints drive innovation**: arbitrary resource limits force architectural decisions that wouldn't emerge from unconstrained scaling
2. **Usefulness is task-specific**: a model that fits on a Raspberry Pi and does one thing well is more valuable than a model that does everything poorly  
3. **Quantization quality is an engineering problem**: the gap between INT4 and FP16 is not fixed - it depends on quantization algorithm quality, and there's still headroom for improvement
4. **Inference efficiency matters as much as training quality**: a model architecture designed for inference efficiency (GQA, smaller vocabulary, optimized attention) can beat a larger but less efficient model

This is a different bet than the scaling hypothesis - and probably not a contradiction of it. Even if scaling is the primary driver of capability, efficient inference at small scale is a separate problem worth solving.

---

## Summary

Building capable small language models requires stacking several techniques:

- **Aggressive quantization** (INT4, INT3, mixed precision) to minimize memory
- **Per-group and per-token quantization** for better precision than uniform scales
- **Mixed precision by layer**: higher precision for sensitive layers (attention, first/last)
- **GQA**: reduce KV cache size with shared KV heads
- **Domain specialization**: compensate for small size with focused training data

The viability of small models depends entirely on the task. For pattern matching, classification, and in-domain generation, they're competitive. For reasoning and open-domain knowledge, they're not - and the honest answer is to use a larger model or an API.

---

*Next: [The Token Economy](./token-economy) - the economics of running and calling LLMs at scale.*

*Previous: [Quantization](./quantization) - the technical foundation that makes small models practical.*
