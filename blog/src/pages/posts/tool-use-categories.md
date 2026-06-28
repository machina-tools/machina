---
layout: ../../layouts/PostLayout.astro
title: "Tool Use: The Four Categories and How to Implement Each"
date: "2026-06-28"
description: "Language models use tools by generating structured calls that a runtime executes. Tool use falls into four categories: code execution, information retrieval, external services, and system actions. Each has different implementation patterns, error modes, and security considerations."
tag: "ai-internals"
readingTime: 11
---

A language model by itself can only generate text. Tools extend what it can do by connecting text generation to executable operations: running code, searching the web, calling APIs, reading files. The model decides *what* to do; the tool implementation decides *how*.

Four categories cover almost every tool you'll build or use. Understanding the patterns within each category prevents common implementation mistakes.

---

## Category 1: Code execution

Code execution is the most powerful category. A model that can write and run code can perform arbitrary computation - data analysis, transformations, calculations, scraping, anything Python can do.

```python
import subprocess
import tempfile
import os
import sys
from pathlib import Path

class PythonExecutor:
    """
    Sandboxed Python execution tool.
    Runs code in a subprocess to limit blast radius of errors.
    """
    def __init__(self, timeout_seconds: int = 30, max_output_bytes: int = 10_000):
        self.timeout = timeout_seconds
        self.max_output = max_output_bytes
    
    def execute(self, code: str) -> dict:
        """
        Execute Python code and return stdout, stderr, and exit code.
        Returns structured result the model can interpret.
        """
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            script_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            stdout = result.stdout[:self.max_output]
            stderr = result.stderr[:self.max_output]
            
            return {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Timeout after {self.timeout}s",
                "exit_code": -1,
            }
        finally:
            os.unlink(script_path)
    
    def as_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": (
                    "Execute Python code and return the output. "
                    "Use for calculations, data manipulation, or any task requiring computation. "
                    "Code runs in an isolated environment. Print results to see them."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"]
                }
            }
        }

# Usage
executor = PythonExecutor()
result = executor.execute("""
import statistics
data = [12, 15, 18, 22, 19, 14, 11, 20]
print(f"Mean: {statistics.mean(data):.2f}")
print(f"Stdev: {statistics.stdev(data):.2f}")
print(f"Median: {statistics.median(data)}")
""")
print(result)
```

The critical implementation detail: **run code in a subprocess, not `eval()`**. Subprocess isolation means a crash doesn't take down your server, and resource limits (timeout, output size) are enforceable.

For production: Docker containers provide stronger isolation. Cloud sandboxes (E2B, Modal) provide fully isolated environments with proper resource limits.

---

## Category 2: Information retrieval

Retrieval tools give the model access to current or private information that wasn't in its training data.

### Web search

```python
import httpx

class WebSearchTool:
    """
    Web search integration.
    Uses a search API and returns structured results.
    """
    def __init__(self, api_key: str, n_results: int = 5):
        self.api_key = api_key
        self.n_results = n_results
        self.client = httpx.Client(timeout=10.0)
    
    def search(self, query: str) -> list[dict]:
        """
        Returns list of {title, url, snippet} for each result.
        """
        # Example using Brave Search API
        response = self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": self.n_results},
            headers={"Accept": "application/json",
                     "X-Subscription-Token": self.api_key},
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        return results
    
    def format_for_model(self, results: list[dict]) -> str:
        """Format results as readable text for the model's context."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            lines.append(f"    {r['snippet']}")
            lines.append("")
        return "\n".join(lines)
```

### RAG (vector search over private documents)

```python
import numpy as np
from typing import Any

class VectorStore:
    """
    Simple in-memory vector store for RAG.
    In production: use Qdrant, Pinecone, pgvector, etc.
    """
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn
        self.documents: list[str] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: list[str]):
        """Embed and store documents."""
        new_embeddings = np.array([self.embedding_fn(d) for d in documents])
        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Find top_k most relevant documents."""
        if not self.documents:
            return []
        
        query_emb = np.array(self.embedding_fn(query))
        
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        similarities = (self.embeddings @ query_emb) / (norms + 1e-8)
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [
            {"document": self.documents[i], "score": float(similarities[i])}
            for i in top_indices
        ]
    
    def as_tool(self):
        """Return a callable suitable for the agent loop."""
        def search_knowledge_base(query: str, top_k: int = 3) -> str:
            results = self.search(query, top_k)
            if not results:
                return "No relevant documents found."
            return "\n\n".join([
                f"[Score: {r['score']:.3f}]\n{r['document']}"
                for r in results
            ])
        return search_knowledge_base
```

The key design decision for RAG: chunking strategy. Documents should be split into chunks small enough to be specific (avoid retrieving irrelevant context) but large enough to be self-contained (avoid losing context). Typically 256-512 tokens per chunk with ~20% overlap.

---

## Category 3: External services

API calls to external services - databases, CRMs, calendars, version control, anything with an API.

```python
import httpx
from datetime import datetime

class DatabaseQueryTool:
    """
    Safe parameterized database query tool.
    Critical: NEVER allow the model to execute raw SQL strings.
    Always use parameterized queries.
    """
    def __init__(self, db_connection):
        self.db = db_connection
    
    def run_query(self, query_type: str, **params) -> list[dict]:
        """
        Execute a predefined query by name.
        The model cannot write arbitrary SQL - it selects from allowed queries.
        """
        ALLOWED_QUERIES = {
            "get_user_by_email": "SELECT id, name, email FROM users WHERE email = %s",
            "get_recent_orders": "SELECT * FROM orders WHERE created_at > %s LIMIT 10",
            "get_product_inventory": "SELECT product_id, quantity FROM inventory WHERE product_id = %s",
        }
        
        if query_type not in ALLOWED_QUERIES:
            raise ValueError(f"Query '{query_type}' not in allowed queries list")
        
        sql = ALLOWED_QUERIES[query_type]
        param_values = list(params.values())
        
        cursor = self.db.cursor()
        cursor.execute(sql, param_values)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in rows]

class CalendarTool:
    """Read/write calendar events."""
    def __init__(self, calendar_api_client):
        self.client = calendar_api_client
    
    def list_events(self, date_str: str, calendar_id: str = "primary") -> list[dict]:
        """List events on a specific date."""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        events = self.client.events().list(
            calendarId=calendar_id,
            timeMin=date.isoformat() + "Z",
            timeMax=(date.replace(hour=23, minute=59)).isoformat() + "Z",
            singleEvents=True,
        ).execute()
        return events.get("items", [])
    
    def create_event(self, title: str, date_str: str, start_time: str, 
                     duration_minutes: int) -> dict:
        """Create a calendar event."""
        start = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M")
        end = start.replace(minute=start.minute + duration_minutes)
        
        event = {
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
            "end":   {"dateTime": end.isoformat(), "timeZone": "UTC"},
        }
        return self.client.events().insert(calendarId="primary", body=event).execute()
```

The security principle for external service tools: **never pass raw model-generated strings directly to APIs or databases**. Validate, sanitize, or use parameterization. The model is not trusted to construct safe queries.

---

## Category 4: System actions

File system access, process management, and environment interaction. The highest-risk category.

```python
import os
import json
from pathlib import Path

class FileSystemTool:
    """
    Sandboxed file system access.
    Restricts all operations to a specific working directory.
    """
    def __init__(self, allowed_root: str):
        self.root = Path(allowed_root).resolve()
    
    def _safe_path(self, path: str) -> Path:
        """Resolve and validate path is within allowed root."""
        resolved = (self.root / path).resolve()
        if not str(resolved).startswith(str(self.root)):
            raise PermissionError(f"Path '{path}' is outside allowed directory")
        return resolved
    
    def read_file(self, path: str) -> str:
        safe = self._safe_path(path)
        if not safe.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return safe.read_text(encoding="utf-8")
    
    def write_file(self, path: str, content: str) -> str:
        safe = self._safe_path(path)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {path}"
    
    def list_directory(self, path: str = ".") -> list[str]:
        safe = self._safe_path(path)
        if not safe.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return [str(f.relative_to(self.root)) for f in safe.iterdir()]

# Tool schema for the model
FILE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file content as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
]
```

**The golden rule for system actions**: the agent's blast radius should be bounded by design, not by trust. A file system tool that can only write to `/tmp/agent-workspace/` can't damage the rest of the system regardless of what instructions it receives.

---

## Implementing tool result handling

How you present tool results to the model affects how well it uses them:

```python
def format_tool_result(tool_name: str, result: Any, success: bool) -> str:
    """
    Format tool results in a way that helps the model interpret and continue.
    """
    if not success:
        return (
            f"Tool '{tool_name}' failed:\n"
            f"{result}\n\n"
            f"Consider: Did you pass the right arguments? "
            f"Is there an alternative approach?"
        )
    
    # Format by type
    if isinstance(result, list) and len(result) > 10:
        # Truncate long lists
        shown = result[:10]
        return (
            f"Result (showing 10 of {len(result)} items):\n"
            f"{json.dumps(shown, indent=2, default=str)}\n"
            f"... {len(result) - 10} more items"
        )
    
    if isinstance(result, dict):
        return json.dumps(result, indent=2, default=str)
    
    if isinstance(result, str) and len(result) > 5000:
        return result[:5000] + f"\n... [truncated, {len(result) - 5000} more characters]"
    
    return str(result)
```

Truncating large results is important: the model doesn't benefit from 50,000 characters of output, and it will try to process all of it, wasting context and attention.

---

## Summary

The four tool categories:

| Category | Examples | Key risk | Mitigation |
|---|---|---|---|
| Code execution | Python, shell | Arbitrary code execution | Subprocess isolation, timeouts |
| Information retrieval | Web search, vector DB | Prompt injection in results | Mark external content, validate |
| External services | APIs, databases, CRMs | SQL injection, data exposure | Parameterized queries, allow-lists |
| System actions | File I/O, process management | Destructive operations | Sandboxed root directory, dry-run mode |

Every tool that a model can call is a potential attack surface. See [agent security](./agent-security) for the adversarial perspective.

---

*Next: [Agent Security - Prompt Injection and How to Defend Against It](./agent-security) - the attack surface you create when tools can act on the world.*

*Previous: [LLM Agentic Architecture](./llm-agentic-architecture) - the loop that these tools slot into.*
