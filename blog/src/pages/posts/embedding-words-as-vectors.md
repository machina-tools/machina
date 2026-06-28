---
layout: ../../layouts/PostLayout.astro
title: "Embedding: How a Language Model Turns Words Into Geometry"
date: "2026-06-28"
description: "The first thing an LLM does with your prompt is discard the text and replace every word with a list of numbers. That list is an embedding — a point in high-dimensional space. The geometry of that space encodes the structure of language. Here's what embeddings are, why the arithmetic works, and what they reveal about how meaning is stored."
tag: "ai-internals"
readingTime: 10
---

When you send a prompt to a language model, the text is the last thing it actually works with.

Before any attention, any feed-forward computation, any generation — the model converts every word (or word fragment) into a vector of floating-point numbers. That vector is an **embedding**. It's the only representation the model will use from that point on. The original characters are gone.

This isn't a limitation. Representing language as geometry turns out to be surprisingly powerful, and understanding why is foundational to understanding everything else in this series — attention, residual streams, and why techniques like [steering vectors](./steering-vectors-control-llm-activations) work at all.

---

## From text to numbers

Before embedding, the text goes through **tokenization**: splitting into subword units called tokens. Not necessarily words — "running" might be one token, "unbelievable" might be three. Modern models use vocabularies of 32,000 to 128,000 tokens.

Each token in the vocabulary corresponds to one row in the **embedding matrix** — a learned lookup table of shape `(vocab_size, d_model)`. The value `d_model` is the model's internal dimension: 768 for GPT-2, 4096 for LLaMA-3-8B, up to 8192+ for larger models.

Processing a token means looking up its row in this matrix and retrieving the corresponding vector. That's it for the embedding step. Everything else is downstream computation.

```python
# pip install transformer-lens torch
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained("gpt2")

print(f"Embedding matrix: {model.W_E.shape}")
# → torch.Size([50257, 768])
# 50,257 tokens, each with a 768-dimensional vector

# Token " cat" (space-prefixed in GPT-2's tokenization)
token_id = model.to_single_token(" cat")
vec = model.W_E[token_id]

print(f"Token ID: {token_id}")
print(f"Embedding shape: {vec.shape}")       # torch.Size([768])
print(f"Norm: {vec.norm():.3f}")             # typically 1–6 range
print(f"First 6 values: {vec[:6].tolist()}")
# → [0.031, -0.058, 0.014, 0.079, -0.021, 0.043, ...]
```

The numbers are meaningless in isolation. What matters is their **relationships to other vectors** in the same space.

---

## Why similar words end up close together

The embedding matrix starts as random noise. It gets trained by the same objective that trains the entire model: predict the next token.

Here's the key consequence: words that appear in similar contexts — that can plausibly follow the same preceding words — will be pulled toward similar embedding vectors during training. "Dog" and "puppy" appear in overlapping contexts ("she has a ___", "the ___ barked", "my neighbor's ___"), so their embeddings converge. "Dog" and "parliament" don't share contexts, so they stay apart.

Cosine similarity measures the angle between two vectors, normalized to [-1, 1]. It's the standard way to compare embeddings:

```python
import torch.nn.functional as F

def similarity(word_a: str, word_b: str) -> float:
    id_a = model.to_single_token(word_a)
    id_b = model.to_single_token(word_b)
    emb_a = model.W_E[id_a].float()
    emb_b = model.W_E[id_b].float()
    return F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()

print(f" cat /  dog:    {similarity(' cat', ' dog'):.3f}")    # ~0.70
print(f" cat /  table:  {similarity(' cat', ' table'):.3f}")  # ~0.15
print(f" run /  ran:    {similarity(' run', ' ran'):.3f}")    # ~0.78
print(f" run /  sprint: {similarity(' run', ' sprint'):.3f}") # ~0.61
print(f" fast /  slow:  {similarity(' fast', ' slow'):.3f}")  # ~0.55 (antonyms cluster!)
```

Notice that "fast" and "slow" are fairly similar — they're both speed-related adjectives that appear in the same contexts ("a ___ runner", "the car was ___"). Antonyms cluster together because they share surrounding context, even though they mean opposite things. The embedding space encodes co-occurrence, not definition.

---

## Arithmetic on meaning

The property that made embeddings famous outside of ML circles is that **semantic relationships become vector operations**. The relationship between "Paris" and "France" is geometrically similar to the relationship between "Berlin" and "Germany". Which means:

```
vec("France") - vec("Paris") ≈ vec("Germany") - vec("Berlin")
```

Or equivalently:

```
vec("Paris") + (vec("Germany") - vec("Berlin")) ≈ vec("France")
```

```python
def get_emb(word: str) -> torch.Tensor:
    return model.W_E[model.to_single_token(word)].float()

# Find the closest vocabulary token to a given vector
def nearest_token(vec: torch.Tensor, top_k: int = 5) -> list:
    sims = F.cosine_similarity(vec.unsqueeze(0), model.W_E.float(), dim=1)
    top = sims.topk(top_k)
    return [(model.to_string(idx.item()).strip(), score.item())
            for idx, score in zip(top.indices, top.values)]

# Capitals: "Paris" is to "France" as "Berlin" is to ?
query = get_emb(" Paris") - get_emb(" France") + get_emb(" Germany")
print("Paris - France + Germany:")
for word, score in nearest_token(query):
    print(f"  {word}: {score:.3f}")
# Expected near top: "Berlin"

# Verb tense: "run" is to "ran" as "swim" is to ?
query = get_emb(" run") - get_emb(" ran") + get_emb(" swam")
print("\nrun - ran + swam:")
for word, score in nearest_token(query):
    print(f"  {word}: {score:.3f}")
# Expected near top: "swim"

# Profession + gender: "doctor" + ("woman" - "man") ≈ ?
query = get_emb(" doctor") + get_emb(" woman") - get_emb(" man")
print("\ndoctor + woman - man:")
for word, score in nearest_token(query):
    print(f"  {word}: {score:.3f}")
```

This arithmetic works because the model encodes relationships as directions. The direction from "France" to "Paris" (capital city relation) is similar to the direction from "Germany" to "Berlin". The embedding space isn't just grouping words — it's encoding the relational structure of language.

Note that the arithmetic isn't perfect. These are noisy approximations, and GPT-2's relatively small embedding space (768 dimensions) means more crowding and interference than larger models. The relationships exist, but the signal-to-noise ratio improves with scale.

---

## What embeddings don't do: disambiguation

One thing the embedding matrix cannot handle is context. "Bank" — financial institution or river bank — gets the same vector regardless of what surrounds it. Disambiguation is the attention mechanism's job.

You can see this directly by comparing the final-layer representation of the same token in different contexts:

```python
def get_final_repr(sentence: str, token_str: str) -> torch.Tensor:
    """Get the residual stream at the final layer for a given token in context."""
    tokens = model.to_tokens(sentence)
    _, cache = model.run_with_cache(tokens)
    final_layer = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    
    # Find token position by checking string representations
    # (simplified — handles most cases)
    token_strings = [model.to_string(t.item()) for t in tokens[0]]
    pos = next((i for i, s in enumerate(token_strings) if token_str in s), None)
    
    return cache[final_layer][0, pos, :].detach()

river = get_final_repr("We walked along the bank of the river at sunset.", "bank")
finance = get_final_repr("She deposited the check at the bank before it closed.", "bank")

# Same token ID at input → very different vectors by the final layer
initial_sim = 1.0  # same embedding matrix row
final_sim = F.cosine_similarity(river.unsqueeze(0), finance.unsqueeze(0)).item()

print(f"Initial similarity (embedding layer): 1.000")
print(f"Final similarity (after all layers):  {final_sim:.3f}")
# → typically 0.2–0.5 — the model has pulled them apart
```

By the final layer, the word "bank" in a river context and "bank" in a financial context have substantially different representations. The embedding matrix handed both the same starting point; 24 layers of attention updated them differently based on their surroundings.

---

## The embedding-unembedding connection

One structural detail that matters for interpretability: in most modern models, the **embedding matrix and unembedding matrix share weights**.

The unembedding matrix (W_U) is what converts the final residual stream vector into a probability distribution over the vocabulary — it answers "which token should come next?" If W_E and W_U are tied (transposed versions of each other), then the space where the model encodes meaning and the space where it reads off predictions are the same space.

```python
# Check if tied weights (true for GPT-2)
print(torch.allclose(model.W_E, model.W_U.T, atol=1e-5))
# → True

# Consequence: you can apply the unembedding matrix to ANY layer's residual stream
# and get a rough prediction of what the model would output at that point
# This is called the "logit lens" technique

def logit_lens(sentence: str, layer: int) -> list:
    """What would the model predict at this layer if it stopped here?"""
    tokens = model.to_tokens(sentence)
    _, cache = model.run_with_cache(tokens)
    resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
    logits = model.unembed(resid.unsqueeze(0).unsqueeze(0))
    top = logits[0, 0].topk(5)
    return [(model.to_string(idx.item()).strip(), prob.item())
            for idx, prob in zip(top.indices, top.values)]

sentence = "The capital city of France is"
print(f"Predictions at layer 6:")
for word, score in logit_lens(sentence, 6):
    print(f"  '{word}': {score:.2f}")
print(f"\nPredictions at layer 12 (final):")
for word, score in logit_lens(sentence, 12):
    print(f"  '{word}': {score:.2f}")
# Earlier layers are confused; later layers converge on "Paris"
```

The logit lens reveals how the model builds toward its prediction across layers — and shows that meaningful computation is happening at every layer, not just the last one.

---

## Why this matters downstream

Three things that follow directly from understanding embeddings:

**Semantic search works the way it does.** RAG systems that retrieve relevant documents by embedding similarity are exploiting the same geometric structure described here. The question embedding and the document embedding end up close in space because the training process pulled similar-meaning text toward similar vectors.

**Steering vectors are built on this foundation.** When you extract a "pessimism" direction by contrasting prompt pairs (as in the [steering vectors article](./steering-vectors-control-llm-activations)), you're finding a direction in the residual stream space that's a downstream descendant of the initial embeddings. The geometric structure is preserved through the layers.

**Probing what a model "knows" at each layer.** The logit lens above is one example. More generally, you can train linear classifiers on top of intermediate residual stream activations to ask "at this layer, has the model figured out the subject's gender? the document's language? the user's intent?" Linear classifiers work here because the relevant information is linearly organized — which goes back to the embedding structure.

---

## Summary

- Every token maps to a vector in R^{d_model} via a learned lookup table
- Tokens in similar contexts end up with similar vectors — meaning clusters geometrically
- Semantic relationships (capital cities, verb tenses, professions) become vector directions you can do arithmetic with
- Context-dependent meaning (word sense disambiguation) is handled downstream by attention, not the embedding matrix
- The embedding and unembedding matrices share weights, making the input and output space the same geometric space

The embedding layer is where text enters the model's internal geometry. Every other technique in this series — attention, residual streams, steering vectors, sparse autoencoders — operates on what the embedding layer produces.

---

*Next: [Self-Attention from Scratch — What the Transformer Block Actually Does](#) — how attention updates each token's representation using every other token in the context.*

*Related: [Steering Vectors — Changing What an LLM Wants Without Touching Its Weights](./steering-vectors-control-llm-activations) — uses the geometric structure described here to modify model behavior at inference time.*
