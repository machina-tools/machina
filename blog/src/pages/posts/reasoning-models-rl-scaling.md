---
layout: ../../layouts/PostLayout.astro
title: "Reasoning Models: When RL Scales What Pretraining Can't"
date: "2026-06-28"
description: "For years, making models better at math and reasoning meant more data and more parameters. Then RL-based post-training broke that wall. Models like o1 and DeepSeek-R1 discovered that extended chain-of-thought reasoning, shaped by reinforcement learning, delivers capability gains that pretraining alone couldn't produce."
tag: "ai-internals"
readingTime: 11
---

There's a distinction worth making between two types of improvement in language models.

The first is **pretraining scaling**: more parameters, more data, more compute. Performance improves predictably as a power law. You can forecast where a bigger model will land before training it.

The second is **test-time scaling**: allocating more computation at inference time - not more parameters, but more sequential thinking steps. A model that "thinks" for 30 seconds on a hard problem can outperform a larger model that answers in 1 second.

Reasoning models are the result of training models to exploit test-time scaling through extended chain-of-thought reasoning, shaped by reinforcement learning on outcome feedback.

---

## Chain-of-thought: the empirical finding

The observation that sparked this direction came before any RL: simply asking a model to "think step by step" before answering dramatically improved performance on multi-step reasoning tasks.

```python
# Standard prompting
DIRECT_PROMPT = """
Question: A train travels at 60 mph. How long does it take to cover 150 miles?
Answer:"""

# Chain-of-thought prompting
COT_PROMPT = """
Question: A train travels at 60 mph. How long does it take to cover 150 miles?
Let me think through this step by step.
"""

# The model continues from the chain-of-thought starter, showing its work:
# "To find time, I use: time = distance / speed
#  distance = 150 miles, speed = 60 mph
#  time = 150 / 60 = 2.5 hours
#  Answer: 2.5 hours"
```

The improvement isn't because the model "gets smarter" when told to think step by step. It's because multi-step problems benefit from intermediate computations being explicit - the model can attend to its own intermediate steps and use them as working memory.

This is a structural advantage of the transformer: longer outputs provide more context for subsequent tokens. Writing out intermediate steps is functionally equivalent to having working memory.

---

## Why pretraining hits a ceiling

For tasks that require complex reasoning - competitive mathematics, multi-hop logical inference, program synthesis - the pretraining scaling law predicts that you need exponentially more data and compute to achieve incremental gains.

The problem is data. There's a finite amount of high-quality mathematical reasoning in text corpora. You can train on every published proof and competition problem, and you still have a ceiling.

RL offers a different path. Instead of learning from demonstrations in the training corpus, the model generates solutions, checks if they're correct, and uses correctness as a reward signal.

```python
# The RL-for-reasoning approach (conceptual)

def rl_reasoning_step(model, problem: str, verifier) -> float:
    """
    One RL step: generate a solution, verify it, compute reward.
    The model gets positive reward for correct solutions,
    negative (or zero) for incorrect ones.
    """
    # Generate a chain-of-thought solution
    solution = model.generate(
        problem,
        max_tokens=4096,  # allow extended thinking
        temperature=0.8,  # sample diverse solutions
    )
    
    # Verify correctness with a ground-truth checker
    is_correct = verifier(problem, solution)
    
    reward = 1.0 if is_correct else 0.0
    
    # Update model to increase probability of correct solutions
    # (actual RL update uses GRPO, PPO, or similar)
    return reward

# Key advantage: we can verify math/code correctness programmatically.
# No human rater needed. The reward signal is clean and scalable.
```

For math, correctness can be verified by checking final answers against ground truth. For code, it can be verified by running tests. This gives an unlimited source of feedback - no need for pre-labeled demonstrations.

---

## What RL training actually learns

RL for reasoning doesn't teach the model new mathematical facts. It teaches the model **how to think about problems** - which reasoning strategies to apply, when to decompose a problem, when to backtrack and try a different approach.

This emerged empirically in the DeepSeek-R1 work: models trained with RL on math problems spontaneously developed behaviors like:

**Reflection**: "Wait, let me re-examine step 3. I made an error - the formula requires absolute values here."

**Hypothesis testing**: "One approach would be to try induction. If that doesn't work, I'll try a direct construction."

**Error detection**: "I got 42, but let me double-check by substituting back into the original equation..."

These behaviors weren't explicitly trained - they emerged from the optimization pressure to produce correct solutions. The model discovered that self-checking and reflection increases the probability of correct final answers.

```python
# Simplified example of extended reasoning
REASONING_EXAMPLE = """
Problem: Find all integers n such that n² - 7n + 12 = 0.

<thinking>
This is a quadratic equation. I can factor it or use the quadratic formula.

Factoring approach: look for two numbers that multiply to 12 and add to -7.
Candidates: -3 and -4. Check: (-3)×(-4) = 12, (-3)+(-4) = -7. ✓

So n² - 7n + 12 = (n-3)(n-4) = 0.

Solutions: n = 3 or n = 4.

Let me verify:
- n=3: 9 - 21 + 12 = 0. ✓
- n=4: 16 - 28 + 12 = 0. ✓
</thinking>

The integers are n = 3 and n = 4.
"""
```

The `<thinking>` block is actual computation happening before the final answer. It's not decoration - the intermediate tokens are available for the model to attend to, making each subsequent step more likely to be correct.

---

## GRPO: the RL algorithm that worked at scale

While PPO is the classic RL algorithm for language models, DeepSeek used **Group Relative Policy Optimization (GRPO)** for R1. The key insight: instead of comparing the current policy against a separate reward model, compare multiple sampled responses against each other.

```python
def grpo_loss(policy_model, prompts, n_samples=8, temperature=1.0):
    """
    GRPO: for each prompt, sample n solutions, compute relative rewards,
    update policy to favor the above-average solutions.
    """
    total_loss = 0
    
    for prompt in prompts:
        # Sample multiple solutions for the same problem
        solutions = [
            policy_model.generate(prompt, temperature=temperature)
            for _ in range(n_samples)
        ]
        
        # Compute rewards (e.g., correctness for math problems)
        rewards = [verify(prompt, sol) for sol in solutions]
        
        # Normalize rewards within the group (zero-mean, unit-variance)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float)
        rewards_norm = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Policy gradient: increase probability of solutions with positive relative reward
        for solution, reward_norm in zip(solutions, rewards_norm):
            log_prob = policy_model.log_prob(prompt, solution)
            total_loss -= reward_norm * log_prob  # gradient ascent on good solutions
    
    return total_loss / len(prompts)
```

The advantage of GRPO over PPO: no separate critic model needed. The reward baseline comes from the other samples in the group. This is computationally cheaper and simpler.

---

## The scaling curve is different

The relationship between test-time compute and performance for reasoning models follows a different curve than pretraining scaling.

For pretraining: performance scales as a power law of compute with exponent ~0.076. Each 10× in compute gives roughly 20% performance improvement.

For test-time reasoning: the relationship is task-dependent but generally steeper for hard tasks. On competition math, allowing the model to generate 10× as many reasoning tokens can double or triple the solve rate. This is because hard problems benefit disproportionately from extended thinking - easy problems are solved in the first few tokens regardless.

```python
# Illustrative relationship between reasoning budget and solve rate
# (based on published o1 and DeepSeek-R1 evals)

solve_rates = {
    # (tokens_budget) → approximate solve rate on competition math
    "dense/128 tokens":    0.15,  # standard prompted response
    "dense/1K tokens":     0.30,  # minimal CoT
    "r1/4K tokens":        0.55,  # extended reasoning
    "r1/16K tokens":       0.72,  # deep reasoning
    "r1/64K tokens":       0.85,  # very extended (used for hard problems)
}

for config, rate in solve_rates.items():
    bar = "█" * int(rate * 30)
    print(f"{config:<25} {bar} {rate:.0%}")
```

The implication: for hard problems, it's more efficient to give a reasoning model more thinking time than to deploy a larger dense model.

---

## Limitations

Reasoning models don't solve everything. A few persistent limitations:

**Distribution dependence**: The RL training reward is typically correctness on specific task types (math, code). The reasoning capabilities generalize somewhat but are strongest in the training distribution.

**Long-context coherence**: Very long reasoning chains (100K+ tokens) can lose coherence. The model may contradict earlier reasoning steps.

**Overthinking**: On easy problems, reasoning models sometimes produce unnecessarily long chains of thought, increasing latency and cost with no benefit.

**Correctness of process vs answer**: The visible chain-of-thought may not reflect the actual computation that produced the answer. This is the topic of [model transparency research](./model-transparency-mitos).

---

## Summary

Reasoning models represent a different axis of scaling:

- **Pretraining scaling**: more data, more parameters - diminishing returns on hard reasoning tasks
- **Test-time scaling**: more sequential reasoning steps at inference - significant gains on hard tasks

The enabling technology is RL with verifiable reward signals (math correctness, code test passing) that provide clean feedback without human raters. The emergent behavior is self-reflection, error-correction, and hypothesis testing within extended reasoning chains.

The key insight: computation at inference time is fungible with computation at training time for certain task types. A smaller model that reasons for longer can match a larger model that answers immediately.

---

*Next: [Mechanistic Interpretability](./mechanistic-interpretability) - tools for understanding what's happening inside these models.*

*Previous: [Mixture of Experts](./mixture-of-experts) - the architecture that scales model capacity without proportional compute cost.*
