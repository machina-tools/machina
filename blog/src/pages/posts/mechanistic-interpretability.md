---
layout: ../../layouts/PostLayout.astro
title: "Mechanistic Interpretability: Finding the Circuits Inside a Transformer"
date: "2026-06-28"
description: "Mechanistic interpretability is the attempt to reverse-engineer transformer models into human-understandable algorithms. Not 'what does this model do?' but 'which specific weights and attention heads implement which specific computations?' Here's the toolkit: induction heads, circuits, activation patching, and how to apply them."
tag: "ai-internals"
readingTime: 13
---

Most interpretability research asks "what does this model do?" — testing it on inputs and observing outputs. Mechanistic interpretability asks something harder: "how does this specific mechanism in the weights implement this specific behavior?"

The distinction matters. Behavioral testing tells you what a model does on your test set. Mechanistic understanding tells you what it will do in situations you haven't tested yet — and why.

---

## The core question

When GPT-2 completes "The Eiffel Tower is located in ___" with "Paris", something in the model's weights is responsible for that. The question is: what, exactly? Which heads? Which layers? Which weight matrices?

This is not a philosophical question. It's an empirical one you can answer with the right tools.

The answer turns out to be surprisingly specific: mechanistic interpretability research has identified that **factual recall** in transformers is dominated by a handful of feed-forward layers acting as key-value stores, with specific attention heads retrieving the subject's attributes.

---

## Induction heads: a foundational circuit

The most well-studied micro-circuit in transformers is the **induction head** — a pair of attention heads (one in layer 0, one in a later layer) that together implement the following algorithm:

> If you've seen `[A][B]` earlier in the context, and you just saw `[A]` again, predict `[B]`.

This is the mechanism behind in-context learning. It allows the model to pick up patterns within the context window.

```python
from transformer_lens import HookedTransformer
import torch
import torch.nn.functional as F

model = HookedTransformer.from_pretrained("gpt2")

def test_induction_head(model, seq_len: int = 40):
    """
    Create a repeated sequence and measure how well the model predicts the repetition.
    High score = strong induction head activity.
    """
    # Generate random token sequence of length seq_len/2
    torch.manual_seed(42)
    half = seq_len // 2
    tokens_half = torch.randint(0, model.cfg.d_vocab, (1, half))
    
    # Repeat it: [A₁A₂...An A₁A₂...An]
    tokens = torch.cat([tokens_half, tokens_half], dim=1)
    
    logits, cache = model.run_with_cache(tokens)
    
    # At position (half + k), predict token k+1
    # A perfect induction head would predict the next token with probability 1.0
    scores = []
    for pos in range(half, seq_len - 1):
        target = tokens[0, pos + 1].item()
        log_prob = F.log_softmax(logits[0, pos], dim=-1)[target].item()
        scores.append(log_prob)
    
    return sum(scores) / len(scores)  # mean log prob (higher = better induction)

score = test_induction_head(model)
print(f"Induction head score: {score:.3f}")
# GPT-2 typically scores around -1.0 on random sequences
# Models with strong induction heads score closer to 0 (perfect prediction)
```

To find which heads are doing this, you can check the **K-composition** pattern: in an induction head circuit, layer L+1's key vector "looks back" at what the previous token was in layer L's output.

---

## Activation patching

**Activation patching** is the key experimental technique in mechanistic interpretability. The idea: run the model on two inputs (one where a behavior appears, one where it doesn't), then systematically replace activations from the "clean" run with activations from the "corrupted" run.

If patching in a specific activation restores the behavior, that activation is causally responsible.

```python
def activation_patch_experiment(model, clean_prompt: str, corrupted_prompt: str,
                                  target_token: str, layer: int, head: int = None):
    """
    Test whether a specific activation is causally important for predicting target_token.
    
    Returns: how much the target token's probability changed after patching.
    """
    hook_name = (f"blocks.{layer}.attn.hook_z" if head is not None 
                 else f"blocks.{layer}.hook_resid_post")
    
    # Run both prompts, cache activations
    _, clean_cache = model.run_with_cache(model.to_tokens(clean_prompt))
    _, corr_cache  = model.run_with_cache(model.to_tokens(corrupted_prompt))
    
    target_id = model.to_single_token(f" {target_token.strip()}")
    
    # Baseline: probability of target token in clean run
    clean_logits = model(model.to_tokens(clean_prompt))
    clean_prob = F.softmax(clean_logits[0, -1], dim=-1)[target_id].item()
    
    # Patched: replace layer's activation with corrupted version
    def patch_fn(value, hook):
        if head is not None:
            value[:, :, head, :] = corr_cache[hook_name][:, :, head, :]
        else:
            value[:] = corr_cache[hook_name]
        return value
    
    patched_logits = model.run_with_hooks(
        model.to_tokens(clean_prompt),
        fwd_hooks=[(hook_name, patch_fn)]
    )
    patched_prob = F.softmax(patched_logits[0, -1], dim=-1)[target_id].item()
    
    return {
        "clean_prob": clean_prob,
        "patched_prob": patched_prob,
        "delta": patched_prob - clean_prob,
    }

# Example: does patching head 9 of layer 9 affect a name recall task?
result = activation_patch_experiment(
    model,
    clean_prompt="The CEO of Apple is Tim",
    corrupted_prompt="The CEO of Google is Sundar",
    target_token="Cook",
    layer=9,
    head=9,
)
print(f"Clean probability:   {result['clean_prob']:.3f}")
print(f"Patched probability: {result['patched_prob']:.3f}")
print(f"Delta:               {result['delta']:+.3f}")
```

---

## The IOI circuit: a worked example

The most complete circuit analysis in the literature is the **Indirect Object Identification (IOI) circuit** — a circuit that handles sentences like "When Mary and John went to the store, John gave the bag to ___" (answer: Mary).

The task requires:
1. Identifying there are two names
2. Recognizing the second occurrence of one name
3. Predicting the other name

Research by Wang et al. (2022) traced this to a 26-head circuit spanning layers 0-9, involving:
- **Duplicate token heads** (detect repeated tokens)
- **Previous token heads** (attend to the token before a repeated token)
- **S-inhibition heads** (suppress the subject name from the output)
- **Name mover heads** (copy the correct name to the output)

```python
# Trace the IOI circuit on a specific example
sentence = "When Mary and John went to the store, John gave the bag to"
tokens = model.to_tokens(sentence)
token_strs = [model.to_string(t.item()) for t in tokens[0]]

_, cache = model.run_with_cache(tokens)

# Name mover heads: should attend from the final position to "Mary" (the IO token)
# Known to be heads 9.9, 9.6, 10.0 in GPT-2
name_mover_heads = [(9, 9), (9, 6), (10, 0)]

mary_pos = [i for i, t in enumerate(token_strs) if 'Mary' in t][0]

print("Name mover head attention to 'Mary':")
for layer, head in name_mover_heads:
    attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, head, -1, mary_pos]
    print(f"  Layer {layer}, Head {head}: {attn.item():.3f}")
```

This kind of analysis — attributing specific behaviors to specific heads and weight matrices — is what distinguishes mechanistic interpretability from behavioral testing.

---

## Path patching

A more targeted variant is **path patching**: instead of replacing an entire activation, replace only the contribution of a specific head to a specific downstream position.

```python
def zero_ablate_head(model, prompt: str, layer: int, head: int) -> float:
    """
    Zero-ablate a single attention head and measure the effect on the
    final token's top prediction probability.
    """
    tokens = model.to_tokens(prompt)
    
    def zero_fn(value, hook):
        value[:, :, head, :] = 0.0
        return value
    
    hook_name = f"blocks.{layer}.attn.hook_z"
    
    # Baseline
    baseline_logits = model(tokens)
    baseline_top = baseline_logits[0, -1].argmax().item()
    baseline_prob = F.softmax(baseline_logits[0, -1], dim=-1)[baseline_top].item()
    
    # Ablated
    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_fn)])
    ablated_prob = F.softmax(ablated_logits[0, -1], dim=-1)[baseline_top].item()
    
    return ablated_prob - baseline_prob  # negative = this head was helping

# Scan all heads for importance to a specific prediction
prompt = "The capital of Japan is"
print(f"Zero-ablation effect on each head's contribution to '{model.to_string(model.to_tokens(prompt)[0, -1].item())}':")

effects = {}
for layer in range(12):
    for head in range(12):
        delta = zero_ablate_head(model, prompt, layer, head)
        if abs(delta) > 0.02:
            effects[(layer, head)] = delta

for (l, h), delta in sorted(effects.items(), key=lambda x: x[1]):
    print(f"  L{l:2d}H{h:2d}: {delta:+.3f}")
```

---

## Attention visualization

Attention weights directly show which tokens a head is attending to. This is the most accessible interpretability tool.

```python
def attention_pattern(model, sentence: str, layer: int, head: int):
    """Return the attention weight matrix for a specific head."""
    tokens = model.to_tokens(sentence)
    _, cache = model.run_with_cache(tokens)
    
    weights = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]
    token_strings = [model.to_string(t.item()) for t in tokens[0]]
    
    return weights, token_strings

# Example: look for a head that tracks subject-verb agreement
sentence = "The keys that the locksmith had lost were never found"
weights, toks = attention_pattern(model, sentence, layer=5, head=1)

# Print which token "found" attends to most
target_pos = [i for i, t in enumerate(toks) if 'found' in t.lower()]
if target_pos:
    pos = target_pos[0]
    top_idx = weights[pos].argmax().item()
    print(f"'found' (pos {pos}) attends most to: '{toks[top_idx]}' (pos {top_idx})")
```

Different heads in different layers have developed different specializations:
- Early layers: mostly local attention, syntax, punctuation
- Middle layers: semantic relations, entity tracking
- Later layers: task-specific patterns, output preparation

---

## What this is good for

Mechanistic interpretability is useful for:

**Auditing safety behaviors**: If you want to verify a model won't respond to certain prompts, you can trace whether the relevant circuits are actually implementing the claimed behavior — or whether they're easily bypassed.

**Understanding failures**: When a model makes a systematic error, activation patching can isolate which component is responsible — making targeted fine-tuning possible.

**Building [steering vectors](./steering-vectors-control-llm-activations)**: Knowing which layer encodes a concept makes vector extraction much more precise than scanning all layers.

**Informing [sparse autoencoder](./sparse-autoencoders) training**: Understanding which parts of the residual stream are compositional guides where to decompose activations into interpretable features.

---

## Summary

Mechanistic interpretability identifies which specific model components are causally responsible for which behaviors:

- **Induction heads**: the circuit that enables in-context learning
- **Activation patching**: replace activations to identify causal importance
- **Circuit analysis**: map full information flows for specific tasks (IOI, factual recall, etc.)
- **Zero/mean ablation**: measure each component's contribution by removing it

The field is young. Complete circuit analyses exist for a handful of behaviors in small models. Scaling this to frontier models (100B+ parameters) is an active research area.

---

*Next: [Sparse Autoencoders](./sparse-autoencoders) — the tool that decomposes the residual stream into individual human-readable features.*

*Previous: [Steering Vectors](./steering-vectors-control-llm-activations) — the inference-time technique that uses this geometric structure to modify model behavior.*
