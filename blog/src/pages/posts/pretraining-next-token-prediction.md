---
layout: ../../layouts/PostLayout.astro
title: "Pretraining: Why Predicting the Next Word Is Enough"
date: "2026-06-28"
description: "GPT-4, Claude, LLaMA - they all start from the same objective: predict the next token. No labels, no human feedback, no task-specific training. Just next-token prediction at scale. Here's why this deceptively simple objective produces models that reason, translate, write code, and pass medical exams."
tag: "ai-internals"
readingTime: 11
---

Every capability you've ever seen from a large language model - reasoning through a math problem, translating between languages it was never explicitly taught to translate, writing code in a framework released after its training cutoff - comes from a model that was trained to do exactly one thing: **predict the next token**.

Not "reason". Not "be helpful". Not "write Python". Just: given the tokens so far, what comes next?

This seems like it shouldn't work. The gap between "guessing the next word" and "explaining the causes of World War I" is enormous. And yet closing that gap doesn't require changing the objective - it only requires scale.

---

## The setup

The training data is text. Enormous amounts of text: web pages, books, code, academic papers, forum discussions, Wikipedia - roughly compressed to some enormous number of tokens (trillions, in modern models). No human labels any of it. No one marks which outputs are correct. The supervision signal comes from the text itself.

The procedure for a single training step:

1. Sample a chunk of text from the training corpus
2. Show the model tokens 1 through N-1 as input
3. Ask: what is token N?
4. Compare the model's predicted probability distribution over the vocabulary against the actual token N
5. Compute a loss (cross-entropy) and backpropagate

```python
# A minimal example of the training objective
# (illustrative - real training uses much larger batches and models)

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")

def compute_loss(text: str) -> float:
    """Cross-entropy loss for predicting each token from its predecessors."""
    tokens = model.to_tokens(text)  # shape: (1, seq_len)
    
    with torch.no_grad():
        logits = model(tokens)  # shape: (1, seq_len, vocab_size)
    
    # Shift: predict tokens 1..N from tokens 0..N-1
    input_tokens  = tokens[:, :-1]   # everything except the last
    target_tokens = tokens[:, 1:]    # everything except the first
    input_logits  = logits[:, :-1, :]
    
    loss = F.cross_entropy(
        input_logits.reshape(-1, model.cfg.d_vocab),
        target_tokens.reshape(-1)
    )
    return loss.item()

# Higher loss = model more surprised by this text
print(compute_loss("The cat sat on the mat"))
# → ~3.2 (common, predictable text)

print(compute_loss("def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"))
# → ~2.1 (GPT-2 was trained on code; this is highly predictable)

print(compute_loss("xqr7 fmz bloop trang wizzle"))
# → ~10+ (random nonsense; model is very surprised)
```

That's the entire training loop conceptually. The model gets better at predicting text. After enough steps on enough data, it has a detailed compressed model of the statistical structure of human language - including, it turns out, a huge amount of the knowledge embedded in that language.

---

## Why "predict the next token" forces the model to learn everything

Here's the key insight: **you can't reliably predict what comes next in text without understanding what came before**.

To complete "The chemical formula for water is ___", you need to know chemistry. To complete "She was angry, so she ___", you need to model intent and emotion. To complete the third line of a Python function that processes a list, you need to understand the function's purpose from its first two lines.

The training objective doesn't care about any of these things directly. It only cares about the loss going down. But the only way to consistently lower the loss on diverse text is to build increasingly accurate internal models of the world that text describes.

This is why the same architecture trained on the same objective produces models that can:
- Translate between languages (text mixes languages; predicting across language boundaries requires mapping equivalents)
- Debug code (code comments describe intent; completing code that matches them requires understanding both)
- Answer questions (Q&A format text is in the training data; matching the answer pattern requires knowing the answer)
- Summarize (the summary comes after the full text; predicting the summary requires compressing the text)

None of these tasks was trained for explicitly. They emerged from the pressure to predict better.

---

## The scaling observation

The finding that changed the field came from studying what happens as you make models larger. Kaplan et al. (2020) measured the relationship between model size (parameters), dataset size (tokens), and loss. The relationship is remarkably clean: **loss decreases predictably as a power law of compute**.

```python
# The scaling law relationship (Kaplan et al. 2020, simplified)
# L(N) ≈ (N_c / N)^alpha_N
# where N is model parameters, N_c is a constant, alpha_N ≈ 0.076 for language models

import numpy as np

def scaling_law_loss(params, params_reference=1.3e9, loss_reference=3.0, alpha=0.076):
    """Approximate loss given parameter count, using GPT-2 (1.3B) as reference."""
    return loss_reference * (params_reference / params) ** alpha

sizes = {
    "GPT-2 Small (117M)":  117e6,
    "GPT-2 Medium (345M)": 345e6,
    "GPT-2 Large (774M)":  774e6,
    "GPT-2 XL (1.5B)":    1.5e9,
    "GPT-3 (175B)":        175e9,
}

print(f"{'Model':<30} {'Params':>12} {'Predicted Loss':>15}")
print("-" * 60)
for name, params in sizes.items():
    loss = scaling_law_loss(params)
    print(f"{name:<30} {params/1e9:>10.1f}B {loss:>15.3f}")
```

The practical consequence: if you want to build a more capable model, the prescription is to use more parameters, more data, and more compute - in roughly the right ratios. The architecture barely changes. The objective doesn't change at all.

This is the honest answer to "why did GPT-3 feel so different from GPT-2?" Not a new architecture. Not a new training objective. More of the same, at a scale where emergent behaviors - multi-step reasoning, in-context learning, code generation - started appearing.

---

## What the model is actually learning

It helps to think concretely about what the model's weights encode after pretraining. Not "knowledge" in a human sense - but compressed statistical patterns from text:

**Factual associations.** "The speed of light is approximately ___" → "3 × 10^8 meters per second". The model has memorized this because it appears in the training data enough times that predicting it reduces loss.

**Syntactic structure.** "The dogs that the cat ___ were chasing." → "saw" (not "sees", because "the dogs" is the subject of "were chasing", not the subject of the gap). The model learned subject-verb agreement, clause structure, reference resolution - all as patterns that reduce loss.

**Code patterns.** A function that starts with `def parse_date(s: str) ->` and has a docstring about ISO format is going to continue with something that handles date strings. The model knows this from patterns in code, not from understanding dates.

**Reasoning chains.** Multi-step reasoning appears in text. Proofs, worked examples, step-by-step explanations. The model sees these often enough that it learns the statistical pattern of "problem → intermediate step → conclusion" - and can apply the pattern to new problems.

None of this is "understanding" in the way humans understand. But it's precise enough to be useful, and it gets more precise at scale.

---

## What pretraining doesn't give you

A pretrained model is not a useful assistant. It's a text completer.

Given "What is the capital of France?", a pretrained GPT-2 might continue "asked the teacher, and little Emily thought for a moment before raising her hand" - because that's a plausible continuation of that sentence pattern in fiction. It might give you the answer, or it might not, depending on what text structure the prompt most resembles.

Getting from text completion to useful assistant requires additional training:

1. **Instruction fine-tuning:** show the model examples of (instruction, good response) pairs. The model learns the format: when the input looks like a question, produce an answer rather than a narrative continuation.

2. **RLHF (Reinforcement Learning from Human Feedback):** have humans rate responses; train the model to produce responses humans prefer. This is where safety behaviors, politeness, and much of the model's "personality" come from.

Both of these build on the pretrained model. They're adjustments to a system that already knows an enormous amount from pretraining - they direct that knowledge, they don't create it.

---

## A quick diagnostic: how surprising is a text to the model?

Cross-entropy loss (averaged per token) is directly interpretable: it tells you how surprised the model is by each token, on average. Lower is more expected, higher is more surprising. This is sometimes called **perplexity** (exponentiated loss, so a perplexity of 1 means perfect prediction and higher is worse).

```python
def perplexity(text: str) -> float:
    loss = compute_loss(text)
    return torch.exp(torch.tensor(loss)).item()

examples = [
    "The quick brown fox jumps over the lazy dog",
    "import torch\nfrom typing import List\n\ndef flatten(lst: List) -> List:",
    "Yesterday, the Senate voted to approve the infrastructure bill",
    "When the polynomial has degree three the roots satisfy Vieta's formulas",
    "klfmz pqr xvtz bnw qzlt",  # gibberish
]

for text in examples:
    print(f"Perplexity: {perplexity(text):.1f} | {text[:60]}")
```

This is a useful diagnostic when building on language models: if your generated outputs have very high perplexity under the base model, you're generating text that the model found highly unlikely during training. Sometimes that's intentional (creative writing, unusual hypotheticals). Often it's a sign of prompt engineering problems.

---

## The training loop in full

Putting it together: pretraining is a massive optimization where the model's parameters are adjusted to minimize the average surprise on a large text corpus. The learning is entirely self-supervised - no labels, no human judgments, just the structure latent in the text.

What comes out is a model with:
- A detailed embedding space (as described in the [previous article](./embedding-words-as-vectors))
- Attention patterns that capture syntactic and semantic relationships
- Feed-forward layers that store factual associations
- A residual stream that accumulates context into a prediction

All of this was learned from "predict the next word". The simplicity of the objective is the point - it's general enough to capture everything in text, and text contains a compressed representation of almost everything humans know.

---

*Next: [RLHF - From Text Completer to Useful Assistant](#) - what happens after pretraining and why it changes the model's behavior so dramatically.*

*Previous: [Embedding - Words as Points in Space](./embedding-words-as-vectors) - the lookup table that converts tokens into the vectors pretraining operates on.*
