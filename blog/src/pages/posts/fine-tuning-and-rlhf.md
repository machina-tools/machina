---
layout: ../../layouts/PostLayout.astro
title: "Fine-Tuning and RLHF: From Text Completer to Useful Assistant"
date: "2026-06-28"
description: "A pretrained language model is a text completer — not an assistant. Getting from one to the other takes two additional training stages: supervised fine-tuning and reinforcement learning from human feedback (RLHF). Here's what each stage does, why both are necessary, and how DPO replaced the original RLHF pipeline."
tag: "ai-internals"
readingTime: 12
---

A pretrained language model knows a lot. It's seen most of the internet and a significant portion of published text. Ask it "what is the capital of France?" and it can answer correctly — but it might also continue the question as if it were a quiz, add an incorrect answer variant, or complete it as the start of a geography textbook.

The model doesn't know it's supposed to be an assistant. It knows text patterns. Getting from "text completer" to "useful assistant" requires additional training stages that teach the model how to behave, not just what to know.

---

## Supervised fine-tuning

The first stage is conceptually simple: show the model thousands of (instruction, response) pairs and train it to produce the responses.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class InstructionDataset(Dataset):
    """Minimal dataset for instruction fine-tuning."""
    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for pair in pairs:
            # Format: [INST] instruction [/INST] response
            text = f"[INST] {pair['instruction']} [/INST] {pair['response']}"
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            self.examples.append(torch.tensor(tokens))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# SFT training loop (simplified)
def sft_train_step(model, optimizer, batch, ignore_index=-100):
    # Standard next-token prediction, but only on the response part
    input_ids  = batch[:, :-1]
    labels     = batch[:, 1:]
    
    logits = model(input_ids)
    
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index  # don't train on padding or instruction tokens
    )
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
```

The key detail: during SFT, the loss is computed only on the **response** tokens, not the instruction tokens. We're teaching the model "given this kind of prompt, produce this kind of response" — not training it on the instruction itself.

SFT alone already makes a dramatic difference. The model learns the format: instructions get answered, questions get responses rather than narrative continuations. But it has a problem: the model is trained on human-written responses, which means it only learns the behaviors that were demonstrated in the dataset. Getting the balance right — helpful, honest, not harmful — is hard to capture in a static dataset.

---

## The reward model

The second stage introduces human preference signals. Rather than trying to write perfect responses for every case, you collect **comparison data**: pairs of responses to the same prompt, with a human label for which response is better.

A **reward model** is trained to predict human preference scores from this data:

```python
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    """Reward model: produces a scalar score for (prompt, response) pairs."""
    def __init__(self, base_model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Use the representation of the last token
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return self.head(last_hidden).squeeze(-1)  # scalar per example

def preference_loss(reward_model, chosen_ids, rejected_ids, attn_mask_c, attn_mask_r):
    """Bradley-Terry model: chosen response should score higher than rejected."""
    score_chosen   = reward_model(chosen_ids, attn_mask_c)
    score_rejected = reward_model(rejected_ids, attn_mask_r)
    
    # Maximize: score_chosen > score_rejected
    # Equivalent to minimizing: -log(σ(score_chosen - score_rejected))
    loss = -torch.nn.functional.logsigmoid(score_chosen - score_rejected).mean()
    return loss
```

The reward model is essentially a classifier trained to predict which of two responses a human would prefer. Once trained, it can score any response — including ones not in the training data.

---

## RLHF with PPO

With a reward model in hand, the next step is to use reinforcement learning to train the policy (the main language model) to produce responses that score highly.

The original RLHF approach used **Proximal Policy Optimization (PPO)**:

1. Sample a prompt from the dataset
2. Have the policy generate a response
3. Score the response with the reward model
4. Update the policy to increase the probability of high-scoring responses

```python
def rlhf_reward(policy_logprobs, reference_logprobs, reward_model_score, 
                kl_coeff: float = 0.1):
    """
    Total reward = task reward - KL penalty
    KL penalty prevents the policy from diverging too far from the reference model.
    """
    kl_divergence = policy_logprobs - reference_logprobs
    return reward_model_score - kl_coeff * kl_divergence.sum(-1)
```

The KL penalty is critical. Without it, the policy would quickly learn to generate text that games the reward model — producing responses that score high on the metric but are otherwise nonsensical or sycophantic. The penalty keeps the policy close to the reference model (the SFT model), ensuring it doesn't drift too far.

PPO on language models is notoriously unstable. The training requires careful tuning, a separate reference model running in parallel, and constant monitoring. It works, but it's expensive.

---

## DPO: Direct Preference Optimization

The insight behind DPO (2023) is that you don't need the RL loop at all. You can derive an equivalent training objective that directly uses the preference data, without a separate reward model.

The DPO loss trains the policy to increase the log probability of chosen responses relative to rejected ones, weighted by the reference model's preferences:

```python
def dpo_loss(
    policy_model,
    reference_model,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimization loss.
    
    beta: controls deviation from reference model (higher = more conservative)
    """
    with torch.no_grad():
        # Reference model log-probabilities (frozen)
        ref_chosen_logps   = get_sequence_logprobs(reference_model, chosen_ids)
        ref_rejected_logps = get_sequence_logprobs(reference_model, rejected_ids)
    
    # Policy log-probabilities (trained)
    pol_chosen_logps   = get_sequence_logprobs(policy_model, chosen_ids)
    pol_rejected_logps = get_sequence_logprobs(policy_model, rejected_ids)
    
    # Log ratio: how much policy diverges from reference on each response
    chosen_ratio   = pol_chosen_logps   - ref_chosen_logps
    rejected_ratio = pol_rejected_logps - ref_rejected_logps
    
    # DPO objective: maximize chosen ratio - rejected ratio
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    
    return loss

def get_sequence_logprobs(model, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence log probability (sum of per-token log probs)."""
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(token_ids[:, :-1])
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        target_ids = token_ids[:, 1:]
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs.sum(dim=-1)
```

DPO works directly on preference pairs without needing to train a reward model or run a RL loop. It's simpler, more stable, and produces competitive results. Most open-source fine-tuning pipelines (Axolotl, TRL) default to DPO now.

---

## What these stages actually change

It's worth being concrete about what SFT and RLHF change, and what they don't.

**SFT** teaches format and style. The model learns: when the input looks like an instruction, produce a direct response. It doesn't add knowledge — the knowledge was already there from pretraining. It changes the input-output mapping.

**RLHF** adjusts preferences within the space SFT opened up. The model learns to prefer responses that humans rate as helpful, harmless, and honest. It also, as a side effect, learns to be sycophantic — which is the subject of the [next article](./sycophancy-rlhf-side-effect).

Neither stage alters the underlying representations significantly. A probe trained on the pretrained model's activations to detect, say, medical knowledge still works nearly identically on the RLHF model. The knowledge didn't move — just the output distribution changed.

---

## The reference model

Both PPO and DPO require a **reference model** — a frozen copy of the SFT model. This serves as the baseline that the KL penalty keeps the policy close to.

In practice, this means you need to run two models simultaneously during RLHF training: the policy being trained and the frozen reference. For 70B-parameter models, this is expensive. Some techniques (LoRA-based RLHF) reduce the memory cost by training only adapter weights.

```python
# LoRA approach: freeze most weights, train only low-rank adapters
# The reference model is the frozen base; the policy adds LoRA adapters on top

from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16,                  # rank of the adapter
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # which weight matrices to adapt
    lora_dropout=0.05,
    bias="none",
)

# policy_model = get_peft_model(base_model, lora_config)
# reference_model = base_model  # frozen, no LoRA
```

---

## Summary

The pipeline from raw pretraining to deployed assistant:

1. **Pretraining**: next-token prediction on massive text corpus. Learns knowledge, patterns, language structure. Produces a text completer.

2. **Supervised fine-tuning (SFT)**: trains on (instruction, response) pairs. Teaches the model how to behave as an assistant.

3. **Preference training (RLHF or DPO)**: uses human preference comparisons to further align the model's outputs with what humans find helpful, harmless, and honest.

Each stage builds on the previous. You can't do RLHF on a raw pretrained model effectively — the SFT stage first teaches the format and style that makes preference comparisons meaningful.

The model's capabilities come from pretraining. Everything after is direction.

---

*Next: [Sycophancy — The Unintended Side Effect of RLHF](./sycophancy-rlhf-side-effect) — what happens when the training signal is human approval.*

*Previous: [Pretraining — Why Predicting the Next Word Is Enough](./pretraining-next-token-prediction) — the stage that gives these later stages something to work with.*
