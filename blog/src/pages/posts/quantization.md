---
layout: ../../layouts/PostLayout.astro
title: "Quantization: Running Billion-Parameter Models on Consumer Hardware"
date: "2026-06-28"
description: "A 7B parameter model in full float32 needs 28GB of VRAM - too much for most consumer GPUs. Quantization compresses model weights to 4-bit or 8-bit integers, cutting memory by 4-8x with modest accuracy loss. Here's how asymmetric quantization works, what GGUF and bitsandbytes do, and when the tradeoffs make sense."
tag: "ai-internals"
readingTime: 12
---

The numbers are striking. A 70B parameter model in float16 needs 140GB of GPU memory. A consumer workstation with two 24GB cards has 48GB. Without compression, state-of-the-art models simply don't fit.

Quantization is the dominant technique for closing this gap. It converts the model's floating-point weights to lower-precision integer representations - typically 4 or 8 bits per weight instead of 16. Memory requirements drop by 4-8×, inference speed on memory-bandwidth-limited hardware often improves, and for well-implemented quantization, quality degradation is small.

---

## What's being quantized

A model's memory usage is dominated by its weight matrices. For inference (not training), you don't need to store gradients - just the weights.

```python
import torch

def model_size_bytes(n_params: int, dtype: str) -> int:
    bits_per_param = {"float32": 32, "float16": 16, "int8": 8, "int4": 4}
    return n_params * bits_per_param[dtype] // 8

# 7B parameter model
n_params = 7_000_000_000

for dtype in ["float32", "float16", "int8", "int4"]:
    size_gb = model_size_bytes(n_params, dtype) / 1024**3
    print(f"{dtype:>8}: {size_gb:.1f} GB")
```

```
 float32: 26.0 GB
 float16: 13.0 GB
    int8:  6.5 GB
    int4:  3.3 GB
```

INT4 quantization gets a 70B model from 140GB to ~35GB - fitting two high-end consumer GPUs with 24GB each.

The challenge: neural network weights are floating-point values, often in a range like [-3.0, 3.0]. Converting them to 4-bit integers (256 values for int8, 16 values for int4) introduces rounding errors. The question is how to minimize the accuracy impact.

---

## Symmetric vs asymmetric quantization

**Symmetric quantization** maps the range [-max, max] uniformly to [-127, 127] (int8) or similar:

```python
def symmetric_quantize(weights: torch.Tensor, n_bits: int = 8):
    max_val = weights.abs().max()
    scale = max_val / (2**(n_bits-1) - 1)
    
    quantized = (weights / scale).round().clamp(-(2**(n_bits-1)), 2**(n_bits-1)-1)
    return quantized.to(torch.int8), scale

def symmetric_dequantize(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    return quantized.float() * scale

# Demonstration
weights = torch.tensor([0.1, -0.5, 1.2, -0.3, 0.8])
q, scale = symmetric_quantize(weights, n_bits=8)
dq = symmetric_dequantize(q, scale)

print("Original:    ", weights.tolist())
print("Quantized:   ", q.tolist())
print("Dequantized: ", [f"{v:.4f}" for v in dq.tolist()])
print(f"Max error:    {(weights - dq).abs().max():.6f}")
```

The problem with symmetric: if the weight distribution is asymmetric (e.g., mostly negative, or skewed), we waste resolution on the half of the range that has few values.

**Asymmetric quantization** accounts for this by also storing an offset (zero-point):

```python
def asymmetric_quantize(weights: torch.Tensor, n_bits: int = 8):
    """
    Asymmetric quantization: maps [min_val, max_val] to [0, 2^n_bits - 1]
    Stores both scale and zero_point.
    """
    min_val = weights.min()
    max_val = weights.max()
    
    q_max = 2**n_bits - 1  # e.g., 255 for 8-bit
    scale = (max_val - min_val) / q_max
    
    # Zero point: what integer value represents 0.0 in the original space
    zero_point = (-min_val / scale).round().clamp(0, q_max).to(torch.int32)
    
    quantized = ((weights / scale) + zero_point).round().clamp(0, q_max)
    return quantized.to(torch.uint8), scale, zero_point

def asymmetric_dequantize(quantized: torch.Tensor, scale: float, 
                            zero_point: int) -> torch.Tensor:
    return (quantized.float() - zero_point) * scale

# Test with asymmetric distribution
weights_asym = torch.tensor([-0.1, 0.2, 0.8, 1.5, 0.3, -0.05])
q, scale, zp = asymmetric_quantize(weights_asym, n_bits=8)
dq = asymmetric_dequantize(q, scale, zp.item())

print("Original:    ", weights_asym.tolist())
print("Dequantized: ", [f"{v:.4f}" for v in dq.tolist()])
print(f"Max error:    {(weights_asym - dq).abs().max():.6f}")
print(f"Scale: {scale:.6f}, Zero point: {zp.item()}")
```

Asymmetric quantization uses the full integer range more efficiently - each integer value represents a finer-grained subdivision of the actual weight distribution, rather than half the integer range going to a region where few weights exist.

---

## Per-channel and per-group quantization

Quantizing the entire weight matrix with a single scale/zero-point loses precision when different weight subsets have very different distributions. The solution: quantize smaller groups independently.

```python
def per_group_quantize(weights: torch.Tensor, group_size: int = 128, n_bits: int = 4):
    """
    Group quantization: each group of 'group_size' weights gets its own scale/zero-point.
    
    This is what GGUF Q4_K_M and similar formats implement.
    Smaller groups = better precision, slightly more overhead per group metadata.
    """
    rows, cols = weights.shape
    assert cols % group_size == 0, "cols must be divisible by group_size"
    
    n_groups = cols // group_size
    scales = torch.zeros(rows, n_groups)
    zero_points = torch.zeros(rows, n_groups, dtype=torch.int32)
    quantized = torch.zeros(rows, cols, dtype=torch.uint8)
    
    q_max = 2**n_bits - 1
    
    for row in range(rows):
        for g in range(n_groups):
            start = g * group_size
            end   = start + group_size
            group = weights[row, start:end]
            
            min_val = group.min()
            max_val = group.max()
            scale = (max_val - min_val) / q_max
            zp = (-min_val / scale).round().clamp(0, q_max)
            
            scales[row, g] = scale
            zero_points[row, g] = zp.int()
            
            q = ((group / scale) + zp).round().clamp(0, q_max).to(torch.uint8)
            quantized[row, start:end] = q
    
    return quantized, scales, zero_points

# Compare error: single scale vs per-group (128 weights per group)
W = torch.randn(64, 512)

# Single scale
q_single, scale, zp = asymmetric_quantize(W.flatten(), n_bits=4)
dq_single = asymmetric_dequantize(q_single, scale.item(), zp.item()).reshape(64, 512)

# Per-group
q_group, scales, zps = per_group_quantize(W, group_size=128, n_bits=4)
# (dequantization omitted for brevity)

print(f"Single-scale error: {(W - dq_single).abs().mean():.6f}")
# Per-group error is typically 2-5x lower
```

---

## GGUF format and llama.cpp

GGUF (GPT-Generated Unified Format) is the standard format for quantized models in llama.cpp. It supports multiple quantization levels:

| Format | Bits/weight | Memory (7B) | Quality loss |
|---|---|---|---|
| Q2_K | ~2.6 | ~2.7GB | High |
| Q4_K_M | ~4.5 | ~4.1GB | Moderate |
| Q5_K_M | ~5.5 | ~4.8GB | Low |
| Q6_K | ~6.6 | ~5.5GB | Very low |
| Q8_0 | 8.0 | ~7.7GB | Negligible |
| F16 | 16 | ~13GB | None |

The "K" variants use grouped quantization. "M" means "medium quality" (a specific mix of quantization strategies for different layer types).

```python
# Loading a GGUF quantized model (requires llama-cpp-python)
# pip install llama-cpp-python

from llama_cpp import Llama

def load_gguf_model(model_path: str, n_gpu_layers: int = -1):
    """
    Load a GGUF quantized model.
    n_gpu_layers=-1: use all GPU layers available
    """
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=4096,
        verbose=False,
    )
    return llm

def gguf_inference(llm, prompt: str, max_tokens: int = 200):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stop=["</s>"],
    )
    return output["choices"][0]["text"]

# Usage:
# llm = load_gguf_model("mistral-7b-instruct-v0.2.Q4_K_M.gguf")
# response = gguf_inference(llm, "Explain the concept of recursion:")
```

---

## bitsandbytes: 8-bit and 4-bit in Python

For Python-native workflows (HuggingFace, PyTorch), bitsandbytes provides 8-bit and 4-bit quantization via `load_in_8bit` and `load_in_4bit`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quantization with NF4 (NormalFloat4) - optimal for normally distributed weights
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # compute in fp16, store in int4
    bnb_4bit_use_double_quant=True,         # quantize the scale factors too
    bnb_4bit_quant_type="nf4",              # NormalFloat4 quantization
)

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3-8B",
#     quantization_config=bnb_config,
#     device_map="auto",
# )

# Memory comparison (approximate, Llama-3-8B):
memory = {
    "float16": 16.0,
    "8-bit":    8.5,
    "4-bit NF4": 5.0,
    "4-bit double quant": 4.5,
}

for dtype, gb in memory.items():
    print(f"{dtype:>20}: {gb:.1f} GB")
```

**NF4 (NormalFloat4)** is a non-uniform quantization scheme that places quantization levels at the quantiles of a standard normal distribution - the distribution that pretrained model weights approximately follow. This means more resolution where most weight values actually are.

---

## When to use which

For local deployment:
- **Q4_K_M** (GGUF): best balance of size, speed, and quality for CPU/mixed inference
- **Q5_K_M** or **Q6_K**: when quality matters more and you have the VRAM
- **4-bit NF4** (bitsandbytes): for Python workflows where you want to stay in the HuggingFace ecosystem

For fine-tuning (QLoRA):
- Load base model in 4-bit, train only LoRA adapter weights in full precision
- Full fine-tune quality at ~1/3 the GPU memory cost

For API-based models:
- Quantization is handled by the provider; you're charged for compute, not memory
- Choose models by quality benchmark, not quantization format

---

## Summary

Quantization reduces memory by converting float16/32 weights to int4/int8:

- **Symmetric**: map [-max, max] to integer range
- **Asymmetric**: store scale + zero_point, map [min, max] to unsigned integer range
- **Per-group**: independent quantization per group of weights - lower error, more metadata
- **NF4**: non-uniform quantization optimized for normally distributed weights

The quality-size tradeoff: INT8 is nearly lossless for most tasks. INT4 introduces noticeable degradation on complex reasoning at the extremes but is acceptable for general use. Below Q4, quality degrades significantly.

---

*Next: [DwarfStar4 - A Case Study in Efficient Language Models](./dwarfstar4-efficient-llm) - quantization and architectural choices for models that run on minimal hardware.*

*Previous: [The Inference Loop](./inference-loop) - the generation process that quantized models are optimized for.*
