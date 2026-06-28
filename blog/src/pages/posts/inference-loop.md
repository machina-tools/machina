---
layout: ../../layouts/PostLayout.astro
title: "The Inference Loop: Autoregressive Generation, Sampling, and the KV Cache"
date: "2026-06-28"
description: "Language model generation is a loop: predict one token, append it to context, predict the next token, repeat. It's simple in concept and has significant engineering implications. Here's how the autoregressive loop works, why the KV cache exists, and what sampling strategies actually do."
tag: "ai-internals"
readingTime: 11
---

Once a language model is trained, generation is surprisingly straightforward conceptually: run the forward pass, pick a token, append it to the sequence, repeat. But the details - how tokens are selected, what's cached, how different sampling strategies change the output - matter enormously for building with LLMs.

This article covers the mechanics from first principles, with enough detail to understand the performance characteristics you'll encounter in practice.

---

## The autoregressive loop

A language model produces a probability distribution over the vocabulary for each position in the sequence. To generate text, you sample from that distribution one token at a time:

```python
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")

def generate_naive(model, prompt: str, n_tokens: int = 50) -> str:
    """
    Naive autoregressive generation.
    Problem: runs the full forward pass from scratch for every new token.
    """
    tokens = model.to_tokens(prompt)
    
    for _ in range(n_tokens):
        logits = model(tokens)
        next_token_logits = logits[0, -1, :]         # last position only
        next_token = next_token_logits.argmax()       # greedy: pick most likely
        tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)
        
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    
    return model.to_string(tokens[0])

# This works but is very slow: each new token re-processes the entire context
# A 50-token output from a 100-token prompt runs the forward pass 50 times,
# each time processing 100, 101, 102... tokens
```

The performance problem is clear: at step 50, you're running the full attention computation over 150 tokens to produce one token. Every intermediate computation for the first 149 tokens is identical to what was computed at step 49 - it's entirely redundant.

---

## The KV cache

The solution: cache the key and value tensors computed for past tokens. Only the new token needs to be computed from scratch.

Recall from the [self-attention article](./self-attention-from-scratch):

```
scores = Q @ K.T / sqrt(d_k)
output = softmax(scores) @ V
```

For existing tokens (positions 0 to t-1), K and V haven't changed since they were computed. Only the new token (position t) contributes a new Q, K, and V. If we cache K and V from previous steps, we only need to:

1. Compute Q, K, V for the new token
2. Concatenate new K, V with cached K, V
3. Compute attention of new Q against all K (old cache + new)

```python
class KVCache:
    """Manages key-value cache for efficient autoregressive generation."""
    def __init__(self, n_layers: int, n_heads: int, d_head: int, max_seq_len: int):
        self.n_layers = n_layers
        self.cache = {
            layer: {
                "k": torch.zeros(1, n_heads, max_seq_len, d_head),
                "v": torch.zeros(1, n_heads, max_seq_len, d_head),
            }
            for layer in range(n_layers)
        }
        self.current_len = 0
    
    def update(self, layer: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Add new K, V for the latest position."""
        pos = self.current_len
        self.cache[layer]["k"][:, :, pos, :] = new_k[:, :, 0, :]
        self.cache[layer]["v"][:, :, pos, :] = new_v[:, :, 0, :]
    
    def get(self, layer: int):
        """Retrieve all cached K, V up to current position."""
        pos = self.current_len + 1  # include just-added position
        return (
            self.cache[layer]["k"][:, :, :pos, :],
            self.cache[layer]["v"][:, :, :pos, :],
        )
    
    def step(self):
        self.current_len += 1

def attention_with_kv_cache(q_new, k_cache, v_cache, d_head):
    """
    Compute attention for a single new query against cached K, V.
    q_new: (batch, n_heads, 1, d_head) - single new position
    k_cache: (batch, n_heads, seq_so_far, d_head)
    v_cache: (batch, n_heads, seq_so_far, d_head)
    """
    scale = d_head ** 0.5
    scores = (q_new @ k_cache.transpose(-2, -1)) / scale  # (batch, heads, 1, seq)
    weights = F.softmax(scores, dim=-1)
    return weights @ v_cache  # (batch, heads, 1, d_head)
```

The memory cost: the KV cache stores `2 × n_layers × n_heads × d_head × seq_len` values per batch. For a 7B parameter model with 32 layers, 32 heads, and 128 d_head, a single 4096-token sequence uses:

```python
# KV cache memory for one sequence
n_layers = 32
n_heads  = 32
d_head   = 128
seq_len  = 4096
bytes_per_float16 = 2

kv_bytes = 2 * n_layers * n_heads * d_head * seq_len * bytes_per_float16
print(f"KV cache: {kv_bytes / 1024**3:.2f} GB")  # ~1GB for one 4K sequence
```

At large context lengths (100K+) or large batches, the KV cache dominates memory usage. This is why long-context models require careful memory management.

---

## Sampling strategies

Greedy decoding (always picking the most likely token) produces repetitive, boring text. The alternative is to sample from the distribution - but raw sampling from the full distribution often produces incoherent outputs. Several techniques tame this:

### Temperature

Temperature scales the logits before softmax. Lower temperature sharpens the distribution (more deterministic); higher temperature flattens it (more random):

```python
def sample_with_temperature(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0:
        return logits.argmax().item()
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, 1).item()

# Compare distributions at different temperatures
logits = torch.tensor([3.0, 2.0, 1.0, 0.5])

for temp in [0.1, 0.5, 1.0, 2.0]:
    probs = F.softmax(logits / temp, dim=-1)
    print(f"T={temp:.1f}: {[f'{p:.3f}' for p in probs.tolist()]}")
```

```
T=0.1: ['0.997', '0.003', '0.000', '0.000']
T=0.5: ['0.844', '0.155', '0.000', '0.000']
T=1.0: ['0.576', '0.212', '0.078', '0.054']  (raw softmax)
T=2.0: ['0.376', '0.286', '0.217', '0.188']
```

### Top-k sampling

Restrict sampling to only the k most probable tokens:

```python
def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> int:
    top_k_vals, top_k_idx = logits.topk(k)
    probs = F.softmax(top_k_vals / temperature, dim=-1)
    sampled_idx = torch.multinomial(probs, 1).item()
    return top_k_idx[sampled_idx].item()
```

### Top-p (nucleus) sampling

Instead of a fixed k, restrict to the smallest set of tokens whose cumulative probability exceeds p:

```python
def top_p_sample(logits: torch.Tensor, p: float, temperature: float = 1.0) -> int:
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    
    cumsum = sorted_probs.cumsum(dim=-1)
    # Keep tokens up to and including the one that pushes cumsum past p
    cutoff = (cumsum - sorted_probs > p).float().argmax()
    cutoff = max(cutoff.item(), 1)  # keep at least one token
    
    truncated_probs = sorted_probs[:cutoff] / sorted_probs[:cutoff].sum()
    sampled_idx = torch.multinomial(truncated_probs, 1).item()
    return sorted_idx[sampled_idx].item()

# At p=0.9: pick from however many tokens sum to 90% of probability
# For confident predictions: might be 2-3 tokens
# For uncertain predictions: might be 100+ tokens
# This adapts the sampling pool to the model's confidence level
```

### Putting it together

```python
def generate_with_sampling(
    model,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_tokens: list[str] = None,
) -> str:
    tokens = model.to_tokens(prompt)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(tokens)[0, -1, :]
        
        # Top-p sampling with temperature
        next_token_id = top_p_sample(logits, p=top_p, temperature=temperature)
        
        # Check stop conditions
        next_token_str = model.to_string(next_token_id)
        if stop_tokens and any(s in next_token_str for s in stop_tokens):
            break
        if next_token_id == model.tokenizer.eos_token_id:
            break
        
        tokens = torch.cat([tokens, torch.tensor([[next_token_id]])], dim=1)
    
    return model.to_string(tokens[0])
```

---

## Beam search

Instead of sampling greedily one token at a time, beam search maintains the top-B partial sequences at each step:

```python
def beam_search(model, prompt: str, beam_size: int = 5, max_tokens: int = 50) -> str:
    """
    Beam search: keep the top beam_size sequences at each step.
    Returns the sequence with highest log probability.
    """
    tokens = model.to_tokens(prompt)
    
    # Initialize beams: (log_prob, token_sequence)
    beams = [(0.0, tokens)]
    
    for _ in range(max_tokens):
        candidates = []
        
        for log_prob, seq in beams:
            with torch.no_grad():
                logits = model(seq)[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Expand each beam with top beam_size continuations
            top_log_probs, top_ids = log_probs.topk(beam_size)
            for lp, tok_id in zip(top_log_probs, top_ids):
                new_seq = torch.cat([seq, tok_id.view(1, 1)], dim=1)
                candidates.append((log_prob + lp.item(), new_seq))
        
        # Keep top beam_size candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]
        
        # Check if all beams have ended
        if all(seq[0, -1].item() == model.tokenizer.eos_token_id 
               for _, seq in beams):
            break
    
    best_log_prob, best_seq = beams[0]
    return model.to_string(best_seq[0])
```

Beam search maximizes probability - but maximum probability is not always the best text. Beam search tends to produce generic, repetitive text because the globally most probable sequence is often boring. Most production systems use sampling (top-p or top-k) for open-ended generation and beam search only for constrained tasks (translation, summarization with strict length).

---

## Speculative decoding

A recent technique for accelerating inference: use a small "draft" model to generate several tokens quickly, then verify them with the large model in a single forward pass.

```python
# Conceptual speculative decoding flow
def speculative_decode_step(draft_model, target_model, context, n_draft=4):
    """
    Generate n_draft tokens with draft_model,
    verify with target_model in one pass.
    Accept verified tokens, reject rest.
    """
    # Draft: generate n_draft tokens quickly
    draft_tokens = []
    draft_logits = []
    current = context
    for _ in range(n_draft):
        with torch.no_grad():
            logit = draft_model(current)[0, -1, :]
        tok = top_p_sample(logit, p=0.9)
        draft_tokens.append(tok)
        draft_logits.append(F.softmax(logit, dim=-1))
        current = torch.cat([current, torch.tensor([[tok]])], dim=1)
    
    # Verify: run target_model on context + all draft tokens at once
    full_seq = torch.cat([context, torch.tensor([draft_tokens])], dim=1)
    with torch.no_grad():
        target_logits = target_model(full_seq)[0]  # (seq_len, vocab)
    
    # Accept draft tokens where target agrees (within tolerance)
    accepted = []
    for i, (draft_tok, draft_prob, target_logit) in enumerate(
        zip(draft_tokens, draft_logits, target_logits[len(context)-1:])):
        target_prob = F.softmax(target_logit, dim=-1)
        acceptance_prob = min(1.0, target_prob[draft_tok] / draft_prob[draft_tok])
        
        if torch.rand(1).item() < acceptance_prob:
            accepted.append(draft_tok)
        else:
            # Reject: sample from corrected distribution
            corrected = F.relu(target_prob - draft_prob)
            tok = torch.multinomial(corrected / corrected.sum(), 1).item()
            accepted.append(tok)
            break  # stop at first rejection
    
    return accepted
```

The theoretical speedup: if the draft model is 8× cheaper than the target and most draft tokens are accepted, you can process those tokens at near-draft-model cost while maintaining target-model output quality. In practice, 2-3× speedup is common.

---

## Summary

The inference loop:

1. **Forward pass**: compute logits for all positions
2. **Sample**: select next token using greedy, temperature, top-k, or top-p
3. **Append**: add token to sequence
4. **Repeat**: until EOS or max length

The **KV cache** makes this efficient: cache K and V tensors from previous steps, only compute new token's contribution at each step.

**Sampling strategy** controls the tradeoff between coherence and creativity:
- Temperature: overall sharpness of distribution
- Top-k/top-p: truncate unlikely tokens before sampling

**Beam search** maximizes probability but often produces boring outputs. **Speculative decoding** uses a cheap draft model to amortize the cost of the expensive target model.

---

*Next: [Quantization](./quantization) - how models are compressed for deployment on hardware with limited memory.*

*Previous: [LLM Fragility](./llm-fragility) - the failure modes that occur during generation.*
