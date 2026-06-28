---
layout: ../../layouts/PostLayout.astro
title: "Sparse Autoencoders: Decomposing Model Activations into Human-Readable Features"
date: "2026-06-28"
description: "Transformer activations are superpositions of overlapping concepts — individual neurons don't cleanly map to interpretable features. Sparse autoencoders (SAEs) solve this by learning an overcomplete basis where each concept gets its own dedicated direction. Anthropic's work with SAEs has identified millions of interpretable features inside Claude."
tag: "ai-internals"
readingTime: 12
---

Individual neurons in transformer models are famously uninterpretable. Activation at neuron 1347 in layer 8 might fire on "programming questions in languages with a large open-source ecosystem" — a compound concept that doesn't correspond to any clean category.

This is the **superposition problem**: the model has more conceptual dimensions to track than neurons available, so it encodes multiple concepts into single neurons via overlapping directions. Reading off what any one neuron "means" is futile because no single neuron "means" one thing.

Sparse autoencoders (SAEs) are the current best approach to decomposing these superposed representations into a cleaner basis where individual features are more interpretable.

---

## Why superposition happens

Neural networks have a strong incentive to compress representations. A transformer with `d_model=768` has only 768 dimensions in its residual stream at each layer, but it needs to track far more than 768 independent concepts.

The linear representation hypothesis (covered in the [embedding article](./embedding-words-as-vectors)) says concepts are encoded as directions in activation space. If two concepts are nearly orthogonal, they can coexist in the same space with minimal interference. In a 768-dimensional space, you can fit many more than 768 nearly-orthogonal directions — not exactly orthogonal, but orthogonal enough to be distinguishable with low interference.

Reconstruction experiments confirm this: sparse autoencoders trained on transformer activations learn dictionaries with 4,000 to 100,000 features from 768-dimensional residual streams, and models can be approximately reconstructed from these features.

---

## SAE architecture

A sparse autoencoder learns two things simultaneously:

1. An **encoder** that maps a dense activation to a sparse feature vector (most features are zero)
2. A **decoder** whose columns are the feature directions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for decomposing transformer activations.
    
    Input:  activation vector from a transformer layer (d_model)
    Output: sparse feature activations + reconstructed activation
    """
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        # n_features >> d_model (overcomplete dictionary)
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        
        # Normalize decoder columns to unit norm (feature directions)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map activation to sparse feature activations."""
        pre_relu = self.encoder(x - self.decoder.bias)
        return F.relu(pre_relu)  # sparsity: only positive activations survive
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct activation from features."""
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        x_recon  = self.decode(features)
        return features, x_recon

# Toy sizes for illustration; real SAEs use n_features = 4x to 100x d_model
d_model = 768
n_features = 4096   # ~5x overcomplete

sae = SparseAutoencoder(d_model, n_features)
print(f"Parameters: {sum(p.numel() for p in sae.parameters()):,}")
```

The critical loss function has two terms: reconstruction accuracy and sparsity.

---

## The training objective

```python
def sae_loss(
    x: torch.Tensor,        # original activations
    x_recon: torch.Tensor,  # reconstructed activations
    features: torch.Tensor, # sparse feature activations
    l1_coeff: float = 1e-3, # sparsity penalty weight
) -> tuple[torch.Tensor, dict]:
    """
    Reconstruction loss + L1 sparsity penalty.
    
    The L1 penalty pushes feature activations toward zero,
    encouraging the model to use as few features as possible.
    """
    # Mean squared error reconstruction loss
    recon_loss = F.mse_loss(x_recon, x)
    
    # L1 penalty: minimize number of active features
    l1_loss = features.abs().mean()
    
    total_loss = recon_loss + l1_coeff * l1_loss
    
    return total_loss, {
        "recon_loss": recon_loss.item(),
        "l1_loss": l1_loss.item(),
        "mean_active_features": (features > 0).float().mean().item(),
        "mean_activation": features.mean().item(),
    }

# Training loop
def train_sae(sae, activations_dataset, n_steps=10000, lr=1e-4):
    """Train SAE on a dataset of transformer activations."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    for step, batch in enumerate(activations_dataset):
        features, x_recon = sae(batch)
        loss, metrics = sae_loss(batch, x_recon, features)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Normalize decoder columns after each step
        with torch.no_grad():
            sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)
        
        optimizer.step()
        
        if step % 1000 == 0:
            print(f"Step {step}: recon={metrics['recon_loss']:.4f}, "
                  f"l1={metrics['l1_loss']:.4f}, "
                  f"active={metrics['mean_active_features']:.3f}")
```

The tradeoff controlled by `l1_coeff`: higher coefficient → more sparse but less accurate reconstruction. In practice, a good SAE should reconstruct 90%+ of the variance while keeping most features inactive for any given input.

---

## Extracting activations to train on

```python
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained("gpt2")

def collect_activations(
    model: HookedTransformer,
    texts: list[str],
    layer: int,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Collect residual stream activations from a specific layer
    across a set of texts.
    """
    all_activations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, padding=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer}.hook_resid_post"
            )
        
        # Flatten (batch, seq_len, d_model) to (batch*seq_len, d_model)
        activations = cache[f"blocks.{layer}.hook_resid_post"]
        all_activations.append(activations.reshape(-1, model.cfg.d_model))
    
    return torch.cat(all_activations, dim=0)

# Collect from layer 8 (mid-model semantic layer)
sample_texts = [
    "The capital of France is Paris.",
    "Python is a programming language.",
    "The sun rises in the east and sets in the west.",
    # ... thousands more in practice
]

# activations = collect_activations(model, sample_texts, layer=8)
# Then: sae = SparseAutoencoder(model.cfg.d_model, n_features=4096)
# Then: train_sae(sae, DataLoader(activations, batch_size=256))
```

---

## Interpreting discovered features

Once trained, each column of the decoder matrix is a **feature direction** — a direction in activation space that the SAE has associated with some concept. To find what concept, you find the inputs that maximally activate each feature.

```python
def find_max_activating_examples(
    sae: SparseAutoencoder,
    feature_idx: int,
    activations: torch.Tensor,
    texts: list[str],
    token_positions: list[int],
    top_k: int = 10,
) -> list[tuple]:
    """
    Find the text examples that most strongly activate a given feature.
    """
    with torch.no_grad():
        features, _ = sae(activations)
        feature_acts = features[:, feature_idx]
    
    top_indices = feature_acts.topk(top_k).indices.tolist()
    
    results = []
    for idx in top_indices:
        activation_val = feature_acts[idx].item()
        text = texts[token_positions[idx]["text_idx"]]
        token_pos = token_positions[idx]["token_pos"]
        results.append((text, token_pos, activation_val))
    
    return results
```

When Anthropic applied this to Claude 3 Sonnet, they found features like:
- Feature 4,714,706: activates on tokens related to the concept of "deception" — primarily in contexts about manipulating others
- Feature 4,547,897: activates on tokens relating to the concept of "imprisonment"
- A feature that activates on the name "Michael Jordan" but only in basketball contexts, not business or other contexts (disambiguating the person via concept)

These emergent feature specializations are remarkable because no one labeled these categories — they were discovered by the SAE training process.

---

## The "dead features" problem

A common training failure: many features never activate on any input. These "dead neurons" waste capacity.

```python
def compute_feature_statistics(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    threshold: float = 1e-4,
) -> dict:
    """Compute statistics about feature utilization."""
    with torch.no_grad():
        features, _ = sae(activations)
    
    # Fraction of inputs that activate each feature at all
    activation_freq = (features > threshold).float().mean(dim=0)
    
    dead_features = (activation_freq == 0).sum().item()
    rare_features = (activation_freq < 0.001).sum().item()
    
    return {
        "n_features": sae.encoder.out_features,
        "dead_features": dead_features,
        "rare_features": rare_features,
        "mean_activation_freq": activation_freq.mean().item(),
        "median_activation_freq": activation_freq.median().item(),
    }

# Good SAEs have < 5% dead features
# If dead features > 20%, try: smaller l1 coefficient, feature resampling, larger learning rate
```

**Feature resampling** is a technique to revive dead features by resetting their weights to high-loss training examples — forcing the SAE to try to encode those examples before falling back to the existing features.

---

## SAEs at scale: Anthropic's Claude work

Anthropic has published results from training SAEs on Claude 3 Sonnet with dictionaries of up to 34 million features. Key findings:

1. **Interpretability scales**: more features → higher fraction are interpretable by humans
2. **Concept specificity**: many features correspond to surprisingly narrow, specific concepts
3. **Feature universality**: some features appear across models of different sizes and training runs
4. **Behavior relevance**: some features directly correspond to behaviors visible in outputs — identifying these is the path toward safety-relevant interpretability

The scale required is significant: training a high-quality SAE on a large model requires collecting billions of activations and training with large dictionaries. But the resulting tool provides a layer of interpretability between "black box" and "fully understood circuit".

---

## Summary

Sparse autoencoders address the superposition problem by:

1. Learning an overcomplete dictionary of feature directions (n_features >> d_model)
2. Training to reconstruct activations from a sparse subset of features
3. Producing a dictionary where each direction corresponds to a (more) interpretable concept

The result is a decomposition of any activation into a small set of active features — a much more interpretable representation than the original dense vector.

This is currently the most scalable path toward understanding what concepts are encoded where in large models, and which concepts are active during any specific generation.

---

*Next: [Model Transparency and Mitos](./model-transparency-mitos) — whether the reasoning we see in model outputs reflects actual internal computation.*

*Previous: [Mechanistic Interpretability](./mechanistic-interpretability) — the circuit-level techniques that SAEs complement.*
