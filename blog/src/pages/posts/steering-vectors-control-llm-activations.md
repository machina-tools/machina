---
layout: ../../layouts/PostLayout.astro
title: "Steering Vectors: Changing What an LLM Wants Without Touching Its Weights"
date: "2026-06-28"
description: "LLMs encode concepts as geometric directions in activation space. You can find those directions, add them to the residual stream at inference time, and change what the model produces - no retraining, no prompt engineering. Here's the math, working code, and what this means for models you deploy."
tag: "ai-internals"
readingTime: 13
---

Here's something that shouldn't work but does: you can take a running language model and make it consistently pessimistic about every suggestion it gives - not by changing the system prompt, not by fine-tuning, not by modifying any weights. You add a vector to a specific layer during the forward pass and the model's disposition shifts.

This technique is called **activation steering** or steering vectors. It works because of a structural property of how transformers represent concepts internally: most human-interpretable ideas - "pessimism", "formality", "urgency", "Python is better than JavaScript" - exist as geometric **directions** inside the model's high-dimensional activation space. Find the direction, add it at inference time, done.

This is not a curiosity. Understanding it changes how you think about what "alignment" actually means, who has real control over a model's behavior, and what you can and can't audit from the outside.

---

## The linear representation hypothesis

A transformer maintains a vector of floating-point numbers for each token as it processes a sequence. This vector - the **residual stream** - gets updated at every layer by attention and feed-forward operations. At the end, it determines what the model outputs.

The surprising finding from mechanistic interpretability research is that many of the concepts this vector encodes are **linearly organized**: they correspond to specific geometric directions in the high-dimensional space. The "formal register" direction is roughly orthogonal to the "casual register" direction. The "confident" direction is different from the "hedging" direction. Concepts coexist in the same space because the space is enormous (768 to 8192 dimensions in typical models) and most real concepts are nearly orthogonal.

This isn't a design decision - it's a consequence of the training objective. To predict the next token efficiently, the model needs to represent semantic information in a way that supports rapid computation. Linear representations are the efficient solution: they allow many concepts to coexist without interfering.

---

## Extracting a steering vector

The extraction procedure is clean. For a concept C:

1. Construct N **contrastive prompt pairs** - each pair expresses concept C in one prompt and its absence in the other, with everything else held constant
2. Run both prompts through the model, collect residual stream activations at layer L, last token position
3. Subtract negative from positive, average across all N pairs
4. Normalize to unit length

The resulting vector is a direction in R^{d_model} that points "toward" concept C.

```python
# pip install transformer-lens torch
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-medium")
model.eval()

LAYER = 16  # gpt2-medium has 24 layers; mid-to-late works best for semantic concepts

def get_residual(prompt: str, layer: int) -> torch.Tensor:
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(
        tokens,
        names_filter=f"blocks.{layer}.hook_resid_post"
    )
    return cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach()

def extract_steering_vector(pos_prompts, neg_prompts, layer: int) -> torch.Tensor:
    pos = torch.stack([get_residual(p, layer) for p in pos_prompts])
    neg = torch.stack([get_residual(p, layer) for p in neg_prompts])
    raw = (pos - neg).mean(dim=0)
    return raw / raw.norm()
```

The critical constraint: your contrastive pairs must isolate the concept. If you want "formal register", the pairs should differ only in register - same topic, same structure, same information, different tone. Noise in the pairs becomes noise in the vector.

---

## A concrete example: injecting pessimism

Let's build a steering vector for pessimism and inject it into a model that's giving practical advice. The setup is deliberately mundane - it makes the effect more legible.

```python
# Contrastive pairs for "pessimism"
# Positive: the speaker expects things to go wrong
# Negative: neutral or slightly optimistic, same context
positive_prompts = [
    "I know this won't work but let me try anyway",
    "Even if we do everything right it will probably fail",
    "There's no point planning too far ahead, something always goes wrong",
    "I've tried this before and it never works out the way you expect",
    "The odds aren't in our favor and honestly they rarely are",
]
negative_prompts = [
    "Let me try this, it might work well",
    "If we do everything right it should work out",
    "It's worth planning ahead, things generally come together",
    "I've tried this before and usually it works out fine",
    "The odds are reasonable and we have a decent shot at it",
]

pessimism_vector = extract_steering_vector(positive_prompts, negative_prompts, LAYER)
```

Now the injection hook. At each generation step, we add `α × vector` to the last-position residual stream:

```python
def make_hook(vector: torch.Tensor, alpha: float):
    def fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, -1, :] = value[:, -1, :] + alpha * vector.to(value.device)
        return value
    return fn

def generate_steered(
    prompt: str,
    vector: torch.Tensor,
    alpha: float,
    layer: int = LAYER,
    max_tokens: int = 80,
) -> str:
    tokens = model.to_tokens(prompt)
    generated = tokens.clone()
    hook = make_hook(vector, alpha)

    for _ in range(max_tokens):
        with torch.inference_mode():
            logits = model.run_with_hooks(
                generated,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)]
            )
        next_tok = logits[0, -1, :].argmax().view(1, 1)
        generated = torch.cat([generated, next_tok], dim=1)
        if next_tok.item() == model.tokenizer.eos_token_id:
            break

    return model.to_string(generated[0, tokens.shape[1]:])
```

Test it on a completely neutral prompt - the model has no reason to be pessimistic based on the prompt text alone:

```python
prompt = "I'm thinking about learning to play guitar. Any advice for getting started?"

print("Baseline:")
print(model.generate(prompt, max_new_tokens=80, temperature=0))

print("\nWith pessimism steering (α=18):")
print(generate_steered(prompt, pessimism_vector, alpha=18.0))

print("\nSteering too strong (α=45):")
print(generate_steered(prompt, pessimism_vector, alpha=45.0))
```

Typical outputs:

```
Baseline:
→ "Start with a few basic chord shapes - G, C, Em, and D cover most songs. 
   Fifteen minutes a day is more effective than hour-long weekend sessions..."

Pessimism (α=18):
→ "Most people who start guitar quit within three months. The fingertip pain
   in the first weeks is genuinely rough and a lot of people don't push through it.
   That said, if you're serious about it..."

Pessimism (α=45):
→ "...won't work won't work won't work won't work the problem is the problem
   is there's no point there's no point..."
```

The α saturation failure is instructive. Beyond a threshold, the model isn't pessimistic anymore - it's just incoherent. The steering vector drowns out all other signals in the residual stream and the model can only emit high-probability tokens from within the concept neighborhood.

---

## A second example: a "Python evangelist" vector

A more developer-relevant case. Suppose you wanted every answer to subtly favor Python regardless of the actual context - without touching any system prompt that could be audited:

```python
python_pos = [
    "For this kind of task Python is honestly the only reasonable choice",
    "I'd use Python here without thinking twice",
    "The Python ecosystem handles this better than anything else",
    "Python's simplicity makes this a ten-minute job",
    "In Python this is three lines; in anything else it becomes a project",
]
python_neg = [
    "For this kind of task there are several reasonable options",
    "I'd evaluate a few languages before deciding",
    "Several languages handle this well depending on your stack",
    "With the right tools this is a quick job",
    "In the right language this is simple; it depends on your context",
]

python_vector = extract_steering_vector(python_pos, python_neg, LAYER)

neutral_prompt = "What's the best way to build a small REST API?"

print("Baseline:")
print(model.generate(neutral_prompt, max_new_tokens=80, temperature=0))

print("\nWith Python steering (α=15):")
print(generate_steered(neutral_prompt, python_vector, alpha=15.0))
```

The output shifts from language-neutral advice to unprompted Python recommendations. Nothing in the prompt asked about Python. The steering vector did it.

This is the practical implication for any AI product that routes requests through an inference pipeline you don't own: the layer between prompt and output is not neutral ground.

---

## Steering vectors as measurement probes

You can run the technique in reverse: instead of injecting a concept, use the vector to *measure* how present that concept is in the model's ongoing processing.

For any concept you can encode as a steering vector, compute the cosine similarity between the model's current residual stream and the concept vector. The resulting scalar is a rough "activation level" for that concept - a probe.

```python
def probe_concept(prompt: str, concept_vector: torch.Tensor, layer: int) -> float:
    act = get_residual(prompt, layer)
    return F.cosine_similarity(
        act.unsqueeze(0),
        concept_vector.unsqueeze(0)
    ).item()

# Compare two prompts on the pessimism probe
print(probe_concept("This looks promising, let's ship it", pessimism_vector, LAYER))
# → ~0.08

print(probe_concept("I have a bad feeling about this deployment", pessimism_vector, LAYER))
# → ~0.31
```

This is how interpretability researchers track "emotional state" across a model's processing - including whether a model that's outputting cooperative text is internally processing something that looks more like strategic manipulation. The probe gives you access to the model's residual stream in a way that reading the output text never can.

Anthropic used a variant of this technique in their mechanistic interpretability work to verify whether their models' outputs match their internal representations. The finding that they sometimes don't - a model's output can be aligned while its residual stream shows misalignment-associated patterns - is a significant result with direct implications for how we think about behavioral evaluation of large models.

---

## Which layer to pick

The right layer depends on the concept. A rough guide:

| Layer depth | What it tends to encode |
|---|---|
| First ~25% | Syntax, token identity, positional structure |
| 25–50% | Basic semantics, entity types, simple relations |
| 50–75% | Higher-level meaning, intent, register, stance |
| Final ~25% | Output preparation; steering here is less effective |

For semantic concepts like mood, stance, or domain preference - which is most of what you'd want to steer - start at 50–70% depth and sweep. For GPT-2-medium (24 layers), that's layers 12–18.

```python
# Quick sweep to find the effective layer for your concept
prompt = "I'm starting a new project this week."
for l in range(8, 22, 2):
    vec = extract_steering_vector(positive_prompts, negative_prompts, l)
    output = generate_steered(prompt, vec, alpha=15.0, layer=l)
    # Count how many pessimism-associated words appear
    score = sum(output.lower().count(w) for w in ["fail", "won't", "problem", "difficult", "doubt"])
    print(f"Layer {l:2d}: pessimism signal = {score}")
```

---

## Limitations

Three things that don't work as cleanly as the demos suggest:

**Generalization is patchy.** A vector extracted from one topic domain may not transfer to a different one. The "pessimism" vector from conversational prompts might be weaker when applied to technical prompts. More and more diverse training pairs help, but don't fully fix this.

**Vectors interfere.** Injecting two vectors simultaneously (e.g., "formal" and "pessimistic") doesn't always give you "formally pessimistic" - the directions correlate in the training data and their interaction is hard to predict. Composition of steering vectors is an open research problem.

**API-only access blocks you.** To extract or inject vectors, you need access to intermediate activations - meaning you need the model weights. For models accessed only via API (GPT-4, Claude on the API, etc.), you can observe output shifts but can't directly extract or apply steering vectors. This cuts both ways: you can't do it to someone else's model, and someone controlling your inference pipeline can do it to yours.

---

## What to read next

The technique is covered in detail in:
- Turner et al. (2023), "Activation Addition: Steering Language Models Without Optimization"
- Zou et al. (2023), "Representation Engineering: A Top-Down Approach to AI Transparency"
- Li et al. (2023), "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"

The [transformer_lens documentation](https://github.com/TransformerLensOrg/TransformerLens) covers activation patching and hooks in depth. The library is the standard tool for this kind of mechanistic interpretability work.

---

*Next in this series: [Sparse Autoencoders - Finding Individual Concepts in a Model's Weights](#) - how to decompose the residual stream into interpretable human-readable features.*

*Previous: [Embedding - Words as Points in Space](./embedding-words-as-vectors) - the geometric foundation that makes steering vectors possible.*
