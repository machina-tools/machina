---
layout: ../../layouts/PostLayout.astro
title: "Mixture of Experts: How DeepSeek Activates Only 6 of 256 Experts Per Token"
date: "2026-06-28"
description: "Dense transformers activate every parameter for every token. Mixture of Experts (MoE) is a different bet: train a model with far more total parameters, but only route each token to a small fraction of them. The result is near-dense-model quality at a fraction of the compute cost per token."
tag: "ai-internals"
readingTime: 12
---

There's a straightforward observation about language: "the dog barked" and "the eigenvalue decomposition" require very different knowledge to process, but a dense transformer applies the same parameters to both. Every token activates every expert, whether relevant or not.

Mixture of Experts (MoE) takes a different architecture: instead of one large feed-forward layer at each transformer block, have many smaller "expert" networks and a **router** that selects which experts process each token. Total model parameters can be 10× larger than a dense model while using the same compute per token.

---

## Architecture overview

In a standard transformer block, the feed-forward sublayer takes the residual stream vector, expands it through a large MLP, and projects it back. This is the expensive part — typically 2/3 of total compute in a dense model.

In MoE, this FFN layer is replaced by `E` expert networks plus a router:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """One expert: a standard two-layer FFN."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))

class MoELayer(nn.Module):
    """
    Mixture of Experts feed-forward layer.
    Routes each token to top-k experts, computes weighted sum of outputs.
    """
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.router  = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch*seq_len, d_model)
        
        # Router scores: affinity between each token and each expert
        router_logits = self.router(x_flat)              # (N, n_experts)
        router_probs  = F.softmax(router_logits, dim=-1) # (N, n_experts)
        
        # Select top-k experts per token
        topk_probs, topk_ids = router_probs.topk(self.top_k, dim=-1)
        # Renormalize weights among selected experts
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            expert_ids  = topk_ids[:, k]    # which expert handles this token
            weights     = topk_weights[:, k].unsqueeze(-1)
            
            # Group tokens by expert for efficient batching
            for expert_idx in range(self.n_experts):
                token_mask = (expert_ids == expert_idx)
                if not token_mask.any():
                    continue
                
                expert_input  = x_flat[token_mask]
                expert_output = self.experts[expert_idx](expert_input)
                output[token_mask] += weights[token_mask] * expert_output
        
        return output.view(batch, seq_len, d_model)
```

Each token selects `top_k` experts, gets a weighted sum of their outputs. Experts not selected for this token incur zero compute.

---

## DeepSeek V3's numbers

DeepSeek V3 uses an MoE configuration that's illustrative of where the field has gone:

| Parameter | Value |
|---|---|
| Total experts per layer | 256 (+ 1 shared) |
| Active experts per token | 6 |
| Shared expert | Always active (1) |
| Total parameters | ~671B |
| Active parameters per token | ~37B |
| Architecture | 61 transformer layers |

So at inference time, for any given token, only ~37B parameters actually compute anything. The other ~634B are unused. Yet the model has access to the full 671B parameter knowledge base — different tokens activate different subsets.

```python
# DeepSeek-style MoE configuration
deepseek_moe_config = {
    "n_experts": 256,
    "n_shared_experts": 1,        # always activated
    "top_k_routed": 6,            # selected per token from the 256
    "total_active": 7,            # 6 routed + 1 shared
    "d_model": 7168,
    "d_ff_per_expert": 2048,      # smaller than a comparable dense model
    "n_layers": 61,
}

# Parameter count: active vs total
n_experts   = deepseek_moe_config["n_experts"]
d_model     = deepseek_moe_config["d_model"]
d_ff        = deepseek_moe_config["d_ff_per_expert"]
n_layers    = deepseek_moe_config["n_layers"]

params_per_expert  = 2 * d_model * d_ff        # up + down projections
total_ffn_params   = params_per_expert * n_experts * n_layers
active_ffn_params  = params_per_expert * deepseek_moe_config["total_active"] * n_layers

print(f"Total FFN params:  {total_ffn_params/1e9:.1f}B")
print(f"Active FFN params: {active_ffn_params/1e9:.1f}B")
print(f"Activation rate:   {active_ffn_params/total_ffn_params:.1%}")
```

---

## The router: load balancing is the hard part

The router is a simple linear layer mapping token representations to expert scores. The training challenge: without additional constraints, the router will collapse — a small set of experts become universally preferred, while the rest are never selected and never trained.

The solution is an **auxiliary load-balancing loss** that penalizes uneven expert utilization:

```python
def load_balancing_loss(router_probs: torch.Tensor, expert_ids: torch.Tensor,
                        n_experts: int, alpha: float = 0.01) -> torch.Tensor:
    """
    Penalize imbalanced expert utilization.
    We want each expert to handle roughly 1/n_experts of tokens.
    
    Based on the auxiliary loss from the Switch Transformer paper.
    """
    n_tokens = router_probs.shape[0]
    
    # f_i: fraction of tokens routed to expert i
    f = torch.zeros(n_experts, device=router_probs.device)
    for i in range(n_experts):
        f[i] = (expert_ids == i).float().mean()
    
    # P_i: mean router probability for expert i (across all tokens)
    P = router_probs.mean(dim=0)
    
    # Auxiliary loss: dot product of f and P, scaled
    # Minimizing this encourages uniform f while keeping P informative
    aux_loss = alpha * n_experts * (f * P).sum()
    return aux_loss

# Training combines the main language model loss with the auxiliary loss
def total_loss(lm_loss, router_probs, selected_expert_ids, n_experts):
    lb_loss = load_balancing_loss(router_probs, selected_expert_ids, n_experts)
    return lm_loss + lb_loss
```

DeepSeek also uses expert-level bias terms and a "no auxiliary loss" approach called MoE-Bias (introduced in DeepSeek V3), which adjusts expert selection biases rather than adding an explicit loss. The goal is the same: distribute tokens evenly.

---

## Shared experts

One innovation in DeepSeek's architecture is the **shared expert** — one FFN that every token always passes through, in addition to the top-k routed experts.

The rationale: some knowledge is universally needed regardless of the token's specific content — basic language structure, common syntactic patterns, high-frequency world knowledge. Having one expert always present for this prevents the routed experts from wasting capacity on universally-needed patterns.

```python
class DeepSeekStyleMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int,
                 n_shared: int = 1):
        super().__init__()
        self.n_shared = n_shared
        self.top_k    = top_k
        
        # Shared experts: always applied
        self.shared_experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_shared)])
        
        # Routed experts: top-k selected per token
        self.routed_experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        batch, seq, d = x.shape
        x_flat = x.view(-1, d)
        
        # Shared expert outputs (always computed)
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        
        # Routed expert outputs
        router_logits = self.router(x_flat)
        router_probs  = F.softmax(router_logits, dim=-1)
        topk_probs, topk_ids = router_probs.topk(self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]
            weights    = topk_weights[:, k].unsqueeze(-1)
            for eid in range(len(self.routed_experts)):
                mask = (expert_ids == eid)
                if mask.any():
                    routed_out[mask] += weights[mask] * self.routed_experts[eid](x_flat[mask])
        
        return (shared_out + routed_out).view(batch, seq, d)
```

---

## MoE training vs inference

**Training** is more complex than dense models. All experts need to be trained, which requires distributing tokens evenly (hence load balancing). Communication overhead between GPUs increases because different experts may live on different devices.

**Inference** benefits are clear: for a 256-expert model where only 6-7 are active per token, you need roughly 256/6 ≈ 43× more memory than a comparable dense model but only ~7× more compute than the active slice. In practice the ratios are less extreme, but the principle holds: you can deploy a model with 10× the knowledge of a dense model for similar inference cost.

For developers building applications:
- MoE models (DeepSeek, Mixtral) are increasingly available via API and locally (GGUF format for llama.cpp)
- The per-token cost is what matters for usage economics, not total parameters
- Memory requirements are the binding constraint for local deployment

---

## Summary

Mixture of Experts decouples two things that are coupled in dense models:

- **Total parameters** (and therefore total knowledge capacity)
- **Active parameters per token** (and therefore compute per token)

With a router network selecting the right specialists for each token, you get dense-model-quality outputs at a fraction of the compute. The tradeoffs are increased training complexity, higher memory requirements, and load balancing overhead.

The core architectural change is replacing a single large FFN layer with many smaller experts and a router. Everything else — attention, layer norm, residuals — stays the same.

---

*Next: [Reasoning Models and RL Scaling](./reasoning-models-rl-scaling) — what happens when you apply RL to model behavior after pretraining.*

*Previous: [Sycophancy](./sycophancy-rlhf-side-effect) — the behavioral side effects of preference training.*
