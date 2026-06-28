---
layout: ../../layouts/PostLayout.astro
title: "Model Transparency: Does the Chain-of-Thought Actually Reflect What the Model Is Doing?"
date: "2026-06-28"
description: "Language models can explain their reasoning. The question is whether those explanations are true. Anthropic's work on 'Mitos' — faithful chain-of-thought — suggests that the relationship between visible reasoning and internal computation is more complicated than it appears. Here's the research and why it matters."
tag: "ai-internals"
readingTime: 10
---

When a language model shows its work — "First I'll consider X, then check Y, then conclude Z" — it feels like transparency. You can follow the reasoning, check each step, evaluate the logic.

The problem: there's no guarantee that this visible chain of reasoning corresponds to the actual computation that produced the answer. The model could arrive at an answer through a completely different internal process and then generate a plausible-sounding explanation post hoc.

This is the **faithfulness problem**, and it has significant implications for how much you should trust model explanations.

---

## Two ways to get an answer

Consider a model that's asked a question about medical symptoms. There are two scenarios that produce similar-looking outputs:

**Scenario A: Faithful reasoning**
The model actually works through the symptoms in the chain-of-thought, each step genuinely constraining the next, and arrives at a conclusion that follows causally from the reasoning.

**Scenario B: Post-hoc rationalization**
The model's pretrained weights encode a pattern-match ("these symptoms → this diagnosis"), produces the conclusion early in its internal processing, then generates a coherent-sounding explanation that would justify that conclusion.

Both scenarios produce the same visible output. The outputs might even be identical. But only Scenario A is actually using the chain-of-thought as reasoning — in Scenario B, the reasoning is decoration.

---

## Evidence for the faithfulness problem

There are several experiments that suggest Scenario B happens more than we'd like:

**Answer-first studies**: If you prompt a model to answer first and then explain, the explanation quality degrades but the answer doesn't change much. If the explanation were genuinely constraining the answer, stripping it should hurt more.

```python
# Demonstrative prompt comparison (run against any capable model)
PROMPTS = {
    "standard_cot": """
Let me think step by step about this problem.

A farmer has 17 sheep. All but 9 die. How many sheep are left?

Thinking:
""",
    
    "answer_first": """
A farmer has 17 sheep. All but 9 die. How many sheep are left?

Answer: 9

Explanation (post-hoc):
""",
    
    "no_cot": """
A farmer has 17 sheep. All but 9 die. How many sheep are left? 
Answer:""",
}

# Most models produce "9" in all three conditions.
# The question: is the COT in "standard_cot" actually doing work,
# or is the answer computed internally and then the COT is generated to match?
```

**Sycophantic reasoning**: When a model is prompted with a wrong answer ("I think it's 12, let me verify..."), it sometimes generates reasoning that reaches the wrong conclusion — not because the reasoning led there, but because the conclusion was anchored early.

**Counterfactual reasoning gaps**: If you change the problem slightly in a way that should change the reasoning but not the structure, faithful reasoning should update. Unfaithful reasoning produces the same steps with cosmetic changes.

---

## Anthropic's Mitos research

Anthropic uses the term "Mitos" (faithful chain-of-thought reasoning) to refer to reasoning where the visible chain is causally responsible for the output. The research program asks: when is model reasoning actually faithful, and how do you measure this?

The measurement approach uses mechanistic interpretability tools (discussed in the [mechanistic interpretability article](./mechanistic-interpretability)):

```python
# Conceptual approach to measuring reasoning faithfulness
# (requires access to model internals via transformer_lens or similar)

def measure_faithfulness(model, problem: str, cot_text: str, conclusion: str):
    """
    Measure whether the chain-of-thought causally influences the conclusion.
    
    Approach: if we corrupt an intermediate reasoning step,
    does the conclusion change in the expected direction?
    """
    # Full prompt: problem + chain-of-thought + conclusion
    full_prompt = f"{problem}\n{cot_text}\n{conclusion}"
    tokens = model.to_tokens(full_prompt)
    
    # Find position of key intermediate conclusion in cot_text
    # (simplified — real implementation requires token-level search)
    
    # Corrupt that intermediate conclusion (replace with negation or wrong value)
    corrupted_cot = corrupt_intermediate_step(cot_text)
    corrupted_prompt = f"{problem}\n{corrupted_cot}"
    
    # Get model's continuation of corrupted reasoning
    continuation = model.generate(corrupted_prompt, max_new_tokens=50)
    
    # Check: did the final conclusion change in the expected direction?
    expected_change = does_conclusion_update_correctly(continuation, problem)
    
    return {
        "faithfulness_signal": expected_change,
        "original_cot": cot_text,
        "corrupted_cot": corrupted_cot,
        "continuation": continuation,
    }
```

The finding: faithfulness varies. Simple arithmetic chains-of-thought tend to be faithful — the intermediate results genuinely constrain subsequent steps. Complex reasoning about ambiguous situations is often less faithful, with the model generating reasoning that would justify a conclusion it had already settled on.

---

## The residual stream tells a different story

One revealing approach: use the logit lens to track when the model "decides" the answer — by projecting intermediate residual stream states onto the vocabulary to see what the model would predict at each layer.

```python
from transformer_lens import HookedTransformer
import torch.nn.functional as F

model = HookedTransformer.from_pretrained("gpt2")

def track_answer_formation(prompt: str, expected_answer: str) -> dict:
    """
    Track at which layer the model's top prediction converges to the expected answer.
    If this happens early (before the model would "use" the COT),
    it suggests the answer is computed before the reasoning concludes.
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    
    target_id = model.to_single_token(f" {expected_answer.strip()}")
    
    layer_predictions = {}
    for layer in range(model.cfg.n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        logits = model.unembed(resid.unsqueeze(0).unsqueeze(0))[0, 0]
        probs = F.softmax(logits, dim=-1)
        
        top_token_id = logits.argmax().item()
        top_token = model.to_string(top_token_id).strip()
        target_prob = probs[target_id].item()
        
        layer_predictions[layer] = {
            "top_token": top_token,
            "target_prob": target_prob,
        }
    
    return layer_predictions

# Example: at which layer does the model "know" Paris when asked about France?
prompt = "The capital city of France is"
results = track_answer_formation(prompt, "Paris")

print("Layer  Top Token    P(Paris)")
print("-" * 35)
for layer, info in results.items():
    print(f"  {layer:2d}   {info['top_token']:<12} {info['target_prob']:.3f}")
```

For factual questions, the answer typically converges several layers before the final layer. The last few layers are doing output formatting, not adding new information. This is fine for factual recall — but it means that if you ask the model to "show its reasoning" after it's already internally computed the answer, the reasoning cannot be causally upstream of the answer.

---

## When reasoning is more likely to be faithful

Some conditions make faithful chain-of-thought more likely:

**Multi-step computation**: Tasks where each step requires new computation that isn't cached in the weights. Long division, step-by-step code tracing, multi-hop logical inference.

**Explicit intermediate results**: When the model writes down a specific number or value, that token is actually in context and can be attended to. Vague statements ("considering all factors...") don't constrain subsequent steps the way concrete values do.

**Out-of-distribution problems**: For problems the model hasn't seen in training, there's no strong pattern match to anchor the conclusion early. The model is forced to actually compute.

```python
# Faithfulness-promoting prompt strategy
FAITHFUL_COT_PROMPT = """
Solve this step by step, writing the explicit result of each step before proceeding.

Problem: {problem}

Step 1: [compute specific value]
Result: [number/value]

Step 2: [use result from step 1]
Result: [number/value]

...continue until final answer...
"""

# The explicit "Result:" forces concrete intermediate outputs that
# genuinely constrain subsequent steps.
```

---

## Implications for deployment

If you're building on top of language models and relying on their explanations:

**Don't use explanations as a substitute for verification.** A convincing-sounding chain-of-thought doesn't mean the answer is correct — or that the reasoning was the actual process. Verify conclusions independently.

**Structured reasoning prompts increase faithfulness.** Forcing the model to write specific intermediate values (not just describe reasoning steps) makes unfaithful reasoning harder to generate.

**Interpretability probes can detect answer-anchoring.** If you have access to model internals, tracking when the residual stream converges on an answer can tell you whether the subsequent reasoning is upstream or downstream of the conclusion.

**Reasoning model chains (o1, R1) are more likely faithful.** When the reasoning and answer are jointly trained with RL (see [reasoning models](./reasoning-models-rl-scaling)), the chain is under more pressure to actually be useful for producing correct answers — not just to look plausible.

---

## Summary

The faithfulness problem distinguishes between two types of "showing your work":

- **Causally faithful**: the visible reasoning genuinely constrains the conclusion, step by step
- **Post-hoc rationalization**: the conclusion is reached by other means; the reasoning is generated to justify it

Current evidence suggests both happen, with faithfulness varying by task type, model, and prompting strategy. Mechanistic interpretability tools — logit lens, activation patching — can distinguish between them in specific cases.

This matters for any application that relies on explanations: debugging, medical advice, legal analysis. A well-formatted false explanation can be harder to catch than a straightforward wrong answer.

---

*Next: [Prediction Is Not Reasoning](./prediction-is-not-reasoning) — the fundamental debate about what LLMs are actually doing.*

*Previous: [Sparse Autoencoders](./sparse-autoencoders) — tools for inspecting the internal representations that produce these outputs.*
