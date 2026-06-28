---
layout: ../../layouts/PostLayout.astro
title: "LLM Agentic Architecture: How Models Take Actions in the World"
date: "2026-06-28"
description: "An agent is a model that does more than answer questions — it plans, uses tools, observes results, and adjusts. Building agents well requires understanding the agentic loop, memory types, planning strategies, and the failure modes that compound across multi-step tasks. Here's the architecture."
tag: "ai-internals"
readingTime: 12
---

The gap between a language model and an agent is a loop. A language model takes input and produces output, once. An agent takes input, produces an action, observes the result of that action, produces another action, and continues until it achieves a goal — or gets stuck.

This loop is what makes agents qualitatively different from question-answering systems. It also introduces failure modes that don't exist in single-shot generation: errors compound, context grows, and the model must track state across many turns.

---

## The agentic loop

The minimal agent loop:

```python
from typing import Any

def agent_loop(
    model_fn,
    tools: dict,
    system_prompt: str,
    initial_message: str,
    max_iterations: int = 20,
) -> str:
    """
    Basic agentic loop: model decides action → execute → observe → repeat.
    
    tools: dict mapping tool_name → callable
    Returns the model's final answer.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_message},
    ]
    
    for iteration in range(max_iterations):
        response = model_fn(messages)
        
        # Check if the model wants to use a tool
        tool_call = extract_tool_call(response)
        
        if tool_call is None:
            # No tool call → model is done, return final answer
            return response["content"]
        
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        
        # Execute the tool
        if tool_name not in tools:
            tool_result = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                tool_result = tools[tool_name](**tool_args)
            except Exception as e:
                tool_result = f"Error: {str(e)}"
        
        # Add assistant action and tool result to conversation
        messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
        messages.append({"role": "tool", "tool_call_id": tool_call["id"],
                         "content": str(tool_result)})
    
    return "Max iterations reached without completing the task."

def extract_tool_call(response: dict) -> dict | None:
    """Extract tool call from model response if present."""
    if response.get("tool_calls"):
        return response["tool_calls"][0]
    return None
```

The loop is simple. What makes agents complex is what happens inside it: the model must plan what to do, decide when to use tools, track what it's already tried, and know when it's done.

---

## Tool schema and model-callable tools

For a model to use tools reliably, tools must be described in a structured way. OpenAI's function-calling format has become the de facto standard:

```python
import json

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Use when you need facts you don't have in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-5)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the output. Use for calculations, data processing, or anything that needs computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    }
                },
                "required": ["path"]
            }
        }
    },
]

def validate_tool_call(tool_call: dict, tool_definitions: list) -> tuple[bool, str]:
    """Basic validation: does the tool call match the schema?"""
    tool_name = tool_call.get("name")
    tool_defs  = {t["function"]["name"]: t["function"] for t in tool_definitions}
    
    if tool_name not in tool_defs:
        return False, f"Unknown tool: {tool_name}"
    
    schema = tool_defs[tool_name]["parameters"]
    required = schema.get("required", [])
    provided = set(tool_call.get("arguments", {}).keys())
    
    missing = set(required) - provided
    if missing:
        return False, f"Missing required args: {missing}"
    
    return True, "OK"
```

The tool description quality matters enormously. Vague descriptions cause the model to call the wrong tool or pass wrong arguments. Specific descriptions with examples in the docstring improve reliability significantly.

---

## Memory types

Agents need to track state across turns. There are four distinct memory mechanisms:

```python
class AgentMemory:
    """
    The four types of memory in an agentic system.
    """
    
    def __init__(self, max_context_tokens: int = 8000):
        # 1. In-context (working memory): the conversation history
        # Limited by context window. Everything else is a workaround for this limit.
        self.context: list[dict] = []
        self.max_context_tokens = max_context_tokens
        
        # 2. External (episodic): persisted across sessions
        # Files, databases — survives restarts
        self.persistent_notes: dict = {}
        
        # 3. Semantic (knowledge): retrieved when relevant
        # Vector store for similarity search
        self.knowledge_base: list[dict] = []  # list of {text, embedding}
        
        # 4. Procedural (skills): encoded in the system prompt
        # Instructions, constraints, learned behaviors
        self.system_prompt: str = ""
    
    def add_to_context(self, message: dict, estimated_tokens: int):
        self.context.append(message)
        # Trim old messages if context is getting long
        while self.estimate_context_tokens() > self.max_context_tokens:
            # Remove oldest non-system messages
            for i, msg in enumerate(self.context):
                if msg["role"] != "system":
                    del self.context[i]
                    break
    
    def estimate_context_tokens(self) -> int:
        return sum(len(m.get("content", "") or "") // 4 for m in self.context)
    
    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[str]:
        """Semantic retrieval: find knowledge relevant to current query."""
        # In practice: embed query, cosine similarity with knowledge_base
        # Returns top_k most relevant stored knowledge pieces
        return []
    
    def persist(self, key: str, value: str):
        """Store information that should survive session boundaries."""
        self.persistent_notes[key] = value
```

The practical challenge: context grows with every tool call. After 20 tool calls, the context can be quite large. Strategies for managing this:

- **Summarize completed subtasks**: when a subtask is done, replace the detailed trace with a summary
- **Selective context**: only include the most relevant history, not all history
- **External memory offload**: write intermediate findings to a file; retrieve as needed

---

## Planning strategies

**ReAct (Reasoning + Acting)**: The model alternates between reasoning ("I need to find out X, which means I should call search_web") and acting (calling search_web). The reasoning step appears in the context, which helps subsequent steps.

```python
REACT_SYSTEM_PROMPT = """
You are a helpful agent. For each task:

1. THINK: reason about what you need to do and why
2. ACT: call a tool if needed
3. OBSERVE: read the tool result
4. Repeat until the task is complete, then give your final answer.

Format your thinking as:
Thought: [your reasoning]
Action: [tool name and arguments]
Observation: [tool result — filled in by the system]

When done, just provide the final answer without any Action."""

# ReAct produces more reliable multi-step behavior than raw function calling
# because the "Thought" step forces intermediate planning into context
```

**Plan-and-execute**: Generate a full plan upfront, then execute steps sequentially. More efficient for tasks where the plan is predictable, but less robust to unexpected tool outputs.

```python
def plan_and_execute_agent(model_fn, tools: dict, task: str) -> str:
    # Phase 1: Generate a plan
    plan_response = model_fn([
        {"role": "system", "content": "Generate a step-by-step plan to accomplish the task. Output only the numbered steps."},
        {"role": "user", "content": task}
    ])
    
    plan_steps = parse_numbered_list(plan_response["content"])
    results = []
    
    # Phase 2: Execute each step
    for step in plan_steps:
        step_result = agent_loop(
            model_fn=model_fn,
            tools=tools,
            system_prompt=f"Execute this specific step: {step}\nContext: {results}",
            initial_message=f"Complete this step: {step}",
            max_iterations=5,
        )
        results.append({"step": step, "result": step_result})
    
    # Phase 3: Synthesize
    synthesis = model_fn([
        {"role": "user", "content": f"Given these step results, provide the final answer for: {task}\n\nResults: {results}"}
    ])
    return synthesis["content"]

def parse_numbered_list(text: str) -> list[str]:
    import re
    lines = text.strip().split('\n')
    return [re.sub(r'^\d+\.?\s*', '', line).strip() for line in lines if line.strip()]
```

---

## Multi-agent architectures

Some tasks benefit from multiple specialized agents collaborating:

```python
class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents.
    The orchestrator decides which sub-agent handles each part of the task.
    """
    def __init__(self):
        self.agents = {
            "researcher": {
                "description": "Searches for and synthesizes information",
                "tools": ["search_web", "fetch_url"],
            },
            "coder": {
                "description": "Writes and executes code",
                "tools": ["execute_python", "read_file", "write_file"],
            },
            "analyst": {
                "description": "Analyzes data and generates insights",
                "tools": ["execute_python", "read_file"],
            },
        }
    
    def route_subtask(self, subtask: str, model_fn) -> str:
        """Determine which agent should handle this subtask."""
        routing_response = model_fn([{
            "role": "user",
            "content": f"Which agent should handle this subtask? Options: {list(self.agents.keys())}\n\nSubtask: {subtask}\n\nReply with just the agent name."
        }])
        return routing_response["content"].strip().lower()
    
    def execute_task(self, task: str, model_fn) -> str:
        """Break down task and route to appropriate agents."""
        # This is simplified — real implementations need proper handoffs
        subtasks = self.decompose_task(task, model_fn)
        results = {}
        
        for subtask in subtasks:
            agent_name = self.route_subtask(subtask, model_fn)
            if agent_name in self.agents:
                tools = {k: lambda **kwargs: k for k in self.agents[agent_name]["tools"]}
                result = agent_loop(model_fn, tools, "", subtask)
                results[subtask] = result
        
        return self.synthesize_results(task, results, model_fn)
    
    def decompose_task(self, task: str, model_fn) -> list[str]:
        response = model_fn([{
            "role": "user",
            "content": f"Break this task into 2-4 independent subtasks:\n{task}"
        }])
        return parse_numbered_list(response["content"])
    
    def synthesize_results(self, task: str, results: dict, model_fn) -> str:
        response = model_fn([{
            "role": "user",
            "content": f"Synthesize these results for the task '{task}':\n{results}"
        }])
        return response["content"]
```

---

## Failure mode analysis

Agentic systems fail in characteristic ways:

| Failure mode | Cause | Mitigation |
|---|---|---|
| Infinite loops | Agent calls same tool repeatedly | Track recent actions, break on repetition |
| Context overflow | Long tasks exhaust context | Periodic summarization, context compression |
| Tool misuse | Wrong tool or wrong arguments | Strong schema validation, error messages with guidance |
| Goal drift | Model loses track of original task | Repeat the goal in system prompt; re-anchor periodically |
| Hallucinated tool calls | Model invents tool names | Strict validation against known tools |
| Error propagation | Early error causes all subsequent steps to fail | Error handling with recovery strategies |

```python
def safe_agent_loop(model_fn, tools, system_prompt, initial_message, max_iterations=20):
    """Agent loop with failure detection and recovery."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_message},
    ]
    
    recent_actions = []   # for loop detection
    error_count = 0
    max_errors = 3
    
    for i in range(max_iterations):
        response = model_fn(messages)
        tool_call = extract_tool_call(response)
        
        if tool_call is None:
            return response["content"]
        
        # Loop detection
        action_signature = f"{tool_call['name']}:{json.dumps(tool_call.get('arguments', {}), sort_keys=True)}"
        if action_signature in recent_actions[-5:]:
            messages.append({"role": "user", "content": 
                "It looks like you're repeating the same action. Try a different approach or conclude."})
            continue
        
        recent_actions.append(action_signature)
        
        # Execute with error handling
        try:
            result = tools[tool_call["name"]](**tool_call.get("arguments", {}))
            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            messages.append({"role": "tool", "tool_call_id": tool_call.get("id", "0"), 
                             "content": str(result)})
            error_count = 0  # reset on success
        except Exception as e:
            error_count += 1
            if error_count >= max_errors:
                return f"Failed after {max_errors} consecutive errors. Last error: {e}"
            messages.append({"role": "tool", "tool_call_id": tool_call.get("id", "0"),
                             "content": f"Error: {e}. Please try a different approach."})
    
    return "Task incomplete: maximum iterations reached."
```

---

## Summary

An LLM agent is a model running in a loop with access to tools. The core components:

1. **Agentic loop**: model decides → tool executes → observation → repeat
2. **Tool schema**: structured descriptions that let the model select and call tools correctly
3. **Memory management**: balancing what stays in context vs what gets offloaded
4. **Planning strategy**: ReAct for adaptive tasks, plan-and-execute for predictable ones
5. **Failure handling**: loop detection, error recovery, goal re-anchoring

The fundamental challenge is that errors compound. A single incorrect tool call can put the agent in a state it can't recover from cleanly. Building reliable agents means investing heavily in error handling and recovery, not just the happy path.

---

*Next: [Tool Use — The Four Categories](./tool-use-categories) — a taxonomy of what kinds of tools agents use and how they should be implemented.*

*Previous: [Token Economy](./token-economy) — the cost implications of running agents at scale.*
