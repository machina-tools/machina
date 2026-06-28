---
layout: ../../layouts/PostLayout.astro
title: "Prediction Is Not Reasoning: What LLMs Actually Do"
date: "2026-06-28"
description: "Large language models predict plausible next tokens. They do this remarkably well, and the result often looks like reasoning. But prediction and reasoning are not the same thing. Here's the evidence, the experiments, and why the distinction matters for how you use and trust these systems."
tag: "ai-internals"
readingTime: 10
---

There's a debate in AI research that matters more than most: when a language model solves a math problem, is it reasoning, or is it doing sophisticated pattern completion?

The stakes are real. If it's reasoning, then AI systems can be trusted in novel situations because they're applying principled inference. If it's pattern completion, their performance in new territory is unpredictable - and the convincing outputs are false confidence.

The honest answer is: it's mostly pattern completion, and the gap between that and principled reasoning shows up in predictable ways.

---

## The core claim

A language model trained on next-token prediction optimizes for one thing: given the preceding tokens, what tokens tend to appear next in text like this?

This is not the same as:
- Understanding what the tokens mean
- Applying logical rules that hold regardless of surface form
- Generalizing from principles to novel cases

A model that has read millions of math solutions has learned what correct math solutions look like. When it produces a correct solution, it might be doing something like reasoning - or it might be pattern-matching to solutions in the training distribution.

The way to distinguish them is to test on inputs that differ from the training distribution in targeted ways.

---

## The reversal curse

One of the clearest demonstrations is the **reversal curse**: models that can reliably answer "A is B" questions often fail at "B is A" when B is less common.

```python
# Conceptual experiment (requires querying a model)
REVERSAL_PAIRS = [
    # (forward question, reverse question, answer)
    ("Who wrote Crime and Punishment?",
     "Crime and Punishment was written by whom?",
     "Dostoevsky"),
    
    ("Who is the CEO of Tesla?",
     "The CEO of Tesla is whom?",
     "Elon Musk"),
    
    # Harder case: relationship that appears mostly one direction in training data
    ("What is the full name of the author who goes by 'George Eliot'?",
     "What pen name did Mary Ann Evans write under?",
     "George Eliot"),
]

# In studies (Berglund et al., 2023):
# - Forward direction: ~80% accuracy
# - Reverse direction: ~40% accuracy on pairs where the reverse
#   form appears rarely in training data
#
# A system doing true reasoning should handle both directions equally  - 
# the logical relationship is symmetric.
```

The reversal curse demonstrates that the model has learned the surface form of the relationship (what text typically follows what other text), not the abstract relationship itself.

---

## Template sensitivity

If a model is truly reasoning from principles, the exact surface form of the problem shouldn't matter much. The answer to a math problem shouldn't change based on whether you say "compute" vs "calculate" vs "find".

```python
EQUIVALENT_FORMULATIONS = [
    # All ask the same question
    "If x + 5 = 12, what is x?",
    "Solve for x: x + 5 = 12",
    "Find the value of x if x + 5 = 12",
    "What number, when added to 5, gives 12?",
    "x + 5 = 12, therefore x = ?",
]

# In practice: models handle these differently
# - Some formulations trigger the equation-solving pattern reliably
# - Some trigger arithmetic patterns  
# - The "what number" version sometimes triggers a different completion pattern
# - Performance varies across formulations even for problems the model "knows"
```

This sensitivity to surface form is a signature of pattern completion. Principled reasoning would be invariant to semantically equivalent reformulations.

---

## Distribution shift reveals the boundaries

The most useful test is out-of-distribution evaluation. Find a task the model has clearly been trained on, then modify it in a principled way that preserves the difficulty but changes the surface pattern.

```python
def create_reversed_arithmetic(n_digits: int = 4) -> tuple[str, str]:
    """
    Create an arithmetic problem where digits are written in reversed order.
    This preserves difficulty but breaks training distribution patterns.
    """
    import random
    a = random.randint(10**(n_digits-1), 10**n_digits - 1)
    b = random.randint(10**(n_digits-1), 10**n_digits - 1)
    correct = a + b
    
    # Reversed digit representation
    def reverse_digits(n: int) -> str:
        return str(n)[::-1]
    
    normal_problem = f"{a} + {b} = ?"
    reversed_problem = f"{reverse_digits(a)} + {reverse_digits(b)} = ? (digits are written in reversed order)"
    
    return normal_problem, reversed_problem, correct

# Studies consistently show:
# - Standard 4-digit addition: ~80% accuracy (GPT-4 class models)
# - Same problems with reversed digit notation: drops to ~30-40%
# 
# If the model were applying an arithmetic algorithm, it should handle
# either representation with the same accuracy (or close to it).
# The degradation reveals reliance on digit-pattern templates.
```

---

## Failure modes that follow from this

Understanding that LLMs are primarily pattern completers predicts specific failure modes:

**Plausible but wrong answers in new domains**: In training-distribution domains, the model's pattern for "this type of question gets this type of answer" is usually calibrated. In new domains, the model produces answers that match the surface pattern of answers in related domains - plausible-sounding, often wrong.

```python
# A common failure pattern in practice:
# The model "knows" the form of the answer even when it doesn't know the answer.
# Medical question → authoritative clinical answer structure (whether or not it's right)
# Legal question → legal analysis structure (whether or not the law is correct)
# Novel math problem → step-by-step solution structure (whether or not each step is valid)
```

**Confident errors near training boundaries**: The model's uncertainty calibration is learned from training data. It "knows" it should be confident about common facts. Near the boundary of its knowledge, it can't distinguish "I know this" from "this looks like the kind of thing I would know", and expresses unwarranted confidence.

**Consistent failures on format variants**: Because training data has distribution over formats, questions in unusual formats (multiple choice presented differently, tables, code with unusual variable names) get lower accuracy than the model's "true" capability on the underlying task.

---

## What this doesn't mean

This isn't an argument that LLMs are useless or incapable of complex outputs. Pattern completion at sufficient scale and with the right training produces outputs that are practically indistinguishable from reasoning in many domains.

For common tasks with clear training coverage - explaining code, summarizing text, answering factual questions in major domains - pattern completion gets you most of the way there.

The argument is more specific:

**Don't confuse output quality with process reliability.** A convincing-sounding answer to a question outside the training distribution is not evidence of correct reasoning - it's evidence that the model has learned what correct answers to this kind of question look like.

**The distribution boundary matters.** Model performance is highest where training data was dense. It degrades in predictable ways as you move away from that distribution.

**Verification is not optional for high-stakes use.** The model's expressed confidence is not a reliable indicator of correctness outside the training distribution.

---

## RL-trained reasoning models change the picture somewhat

[Reasoning models](./reasoning-models-rl-scaling) trained with RL on outcome feedback (correct/incorrect) are under pressure to actually get right answers on novel problems. This pushes them toward more general strategies than pure pattern completion.

Evidence: reasoning models (o1, DeepSeek-R1) show more robust generalization on novel problem variants compared to base RLHF models. The degradation on format variants is smaller.

But it doesn't eliminate the fundamental issue. Reasoning models trained on math show better math generalization - and weaker generalization to new domains where the RL training signal wasn't present.

---

## Summary

Language models predict plausible continuations of text. This produces outputs that often look like reasoning, because reasoning patterns are heavily represented in the training data.

The distinction matters:
- **Pattern completion generalizes** within the training distribution
- **Principled reasoning generalizes** based on rules, not surface form

Reversal curse, template sensitivity, and out-of-distribution degradation are empirical fingerprints of the pattern-completion mechanism.

Using LLMs effectively means staying aware of which domain you're in - where their training data was dense enough that pattern completion tracks truth - and building verification workflows for cases where you're less sure.

---

*Next: [LLM Fragility - Adversarial Inputs and Distribution Shift](./llm-fragility) - the practical failure modes this creates.*

*Previous: [Model Transparency and Mitos](./model-transparency-mitos) - how chain-of-thought can be faithful or post-hoc.*
