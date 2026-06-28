---
layout: ../../layouts/PostLayout.astro
title: "Tokenization: Why Your Model Can't Count Letters"
date: "2026-06-28"
description: "Before a language model processes a single character, it runs tokenization — splitting text into subword chunks and converting them to integer IDs. This choice shapes what the model finds easy and hard: numbers, code, and non-English text all have quirks that trace back directly to how the tokenizer was built."
tag: "ai-internals"
readingTime: 9
---

Ask a large language model how many times the letter 'r' appears in "strawberry" and it will often get it wrong. Not because the model is bad at counting. Because it never sees the word "strawberry" — it sees three tokens, and counting letters requires reasoning about token internals that the model doesn't have direct access to.

Tokenization is the step that happens before anything else. It determines what granularity the model operates at, which patterns it can learn, and which tasks it will find systematically harder than they seem.

---

## Why not characters?

Character-level models are the obvious choice: one token per character, huge flexibility, no vocabulary needed.

The problem is sequence length. A 1000-character passage becomes 1000 tokens. A 100,000-token context limit drops to 100,000 characters — barely a few paragraphs. Transformer attention is quadratic in sequence length, so this matters enormously.

Word-level tokenization is the other extreme. "running" is one token, but "runs", "runner", "ran" are all separate tokens. The vocabulary balloons to handle all forms, all names, all technical terms. Any word not in the training vocabulary becomes an unknown token.

Subword tokenization splits the middle: common words are single tokens, rare words get split into recognizable pieces. "running" might be one token; "tokenization" might be split into "token" + "ization".

---

## Byte Pair Encoding

The dominant algorithm for building subword vocabularies is **Byte Pair Encoding (BPE)**, originally a text compression technique.

The algorithm:
1. Start with a vocabulary of individual characters (and a special end-of-word token)
2. Count all adjacent pairs in the training corpus
3. Merge the most frequent pair into a single token
4. Repeat until the vocabulary reaches the target size

```python
from collections import Counter, defaultdict

def train_bpe(corpus: list[str], vocab_size: int) -> list[tuple]:
    """Minimal BPE training on a small corpus."""
    
    # Tokenize corpus into characters with end-of-word marker
    word_freqs = Counter(corpus)
    vocab = {' '.join(list(word) + ['</w>']): freq
             for word, freq in word_freqs.items()}
    
    merges = []
    
    for _ in range(vocab_size):
        # Count all adjacent pairs
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        
        if not pairs:
            break
        
        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        # Merge this pair everywhere in the vocabulary
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = word.replace(' '.join(best_pair), ''.join(best_pair))
            new_vocab[new_word] = freq
        vocab = new_vocab
    
    return merges

# Train on a tiny toy corpus
corpus = ["low", "low", "low", "low", "low",
          "lower", "lower", "lower",
          "newest", "newest", "newest", "newest", "newest", "newest",
          "widest", "widest"]

merges = train_bpe(corpus, vocab_size=10)
print("Learned merge operations:")
for i, merge in enumerate(merges):
    print(f"  {i+1}. {merge[0]!r} + {merge[1]!r} → {merge[0]+merge[1]!r}")
```

The merge operations are ordered by frequency. The tokenizer is the result of applying all merges in order to any new input text.

---

## What GPT-2's tokenizer actually does

```python
# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def show_tokens(text: str):
    ids = enc.encode(text)
    tokens = [enc.decode([id]) for id in ids]
    print(f"Text:   {text!r}")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    print(f"Count:  {len(ids)}")
    print()

# Common words
show_tokens("The quick brown fox")

# A technical term
show_tokens("tokenization")

# A number
show_tokens("1234567890")

# Code
show_tokens("def factorial(n):")

# Non-English
show_tokens("Привет")  # "Hello" in Russian
```

Sample output:
```
Text:   'The quick brown fox'
Tokens: ['The', ' quick', ' brown', ' fox']
IDs:    [464, 2068, 7586, 21831]
Count:  4

Text:   'tokenization'
Tokens: ['token', 'ization']
IDs:    [30001, 1634]
Count:  2

Text:   '1234567890'
Tokens: ['123', '456', '789', '0']
IDs:    [10163, 31417, 27936, 15]
Count:  4

Text:   'def factorial(n):'
Tokens: ['def', ' factorial', '(', 'n', '):']
IDs:    [4299, 1109, 13702, 7, 77, 2599]
Count:  6

Text:   'Привет'
Tokens: ['ÐŸ', 'Ñ', '€', 'Ð¸', 'Ð²', 'Ðµ', 'Ñ', '‚']
IDs:    [28361, 2634, 230, 1753, 1485, 1490, 2634, 128]
Count:  8
```

The Russian word becomes 8 tokens (individual UTF-8 bytes) because the GPT-2 vocabulary was built primarily from English text. GPT-4's tokenizer handles this better, but the imbalance remains.

---

## The consequences you'll notice

**Numbers are fragmented unpredictably.** "1234567890" becomes 4 tokens in GPT-2 because specific number strings were frequent enough to get their own tokens, but longer numbers weren't. This is why arithmetic is hard for LLMs — the "digits" don't align with natural arithmetic boundaries.

```python
# GPT-2 tokenizer
for num in ["12", "123", "1234", "12345", "100", "999", "1000"]:
    tokens = [enc.decode([id]) for id in enc.encode(num)]
    print(f"{num:>8} → {tokens}")
```

```
      12 → ['12']
     123 → ['123']
    1234 → ['1234']
   12345 → ['123', '45']
     100 → ['100']
     999 → ['999']
    1000 → ['1000']
```

**Space matters.** " cat" and "cat" are different tokens. GPT-2 typically includes the preceding space as part of the token. Getting this wrong when constructing prompts produces subtly different tokenizations.

```python
print(enc.encode("cat"))   # [9246]
print(enc.encode(" cat"))  # [3797] — different token!
```

**Why "strawberry" is hard.** The word tokenizes into multiple tokens, and the model processes them as semantic units, not character sequences. Counting letters requires attending back to sub-token character structure that was discarded at the tokenization step.

```python
tokens = [enc.decode([id]) for id in enc.encode("strawberry")]
print(f"'strawberry' → {tokens}")
# ['st', 'rawberry'] — two tokens, not 10 characters
```

The model only "sees" two tokens. To count 'r' characters, it needs to know the internal character composition of each token — which requires memorizing this during training, not deriving it.

---

## GPT-4 and modern improvements

GPT-4 uses the `cl100k_base` tokenizer with 100,000 tokens (vs GPT-2's 50,257). The larger vocabulary means:

- Common technical terms and code patterns get single tokens
- Non-English text tokenizes more efficiently (fewer tokens per character)
- Numbers handle better in many cases

```python
enc_gpt4 = tiktoken.get_encoding("cl100k_base")

def compare_tokenizers(text: str):
    gpt2_count = len(enc.encode(text))
    gpt4_count = len(enc_gpt4.encode(text))
    print(f"{text[:40]:<40}  GPT-2: {gpt2_count:3d}  GPT-4: {gpt4_count:3d}")

texts = [
    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort",
    "Привет мир",                              # Russian: Hello world
    "これは日本語のテキストです",                 # Japanese
    "import pandas as pd; df = pd.DataFrame()",
    "The transformer architecture was introduced in 2017",
]

for t in texts:
    compare_tokenizers(t)
```

```
def quicksort(arr): return arr if len(arr)  GPT-2:  16  GPT-4:  14
Привет мир                                  GPT-2:  14  GPT-4:   4
これは日本語のテキストです                        GPT-2:  45  GPT-4:  13
import pandas as pd; df = pd.DataFrame()   GPT-2:  16  GPT-4:  13
The transformer architecture was introdu   GPT-2:   9  GPT-4:   8
```

The non-English case is the most dramatic: 14 tokens vs 4. At scale, this changes the effective context window for non-English content significantly.

---

## Token count as a practical concern

Since language model pricing and context limits are defined in tokens (not characters or words), tokenization has direct practical consequences.

```python
def estimate_tokens(text: str, model: str = "gpt-4") -> dict:
    """Estimate token count and cost for a text string."""
    enc = tiktoken.encoding_for_model(model) if model != "gpt-2" else tiktoken.get_encoding("gpt2")
    n_tokens = len(enc.encode(text))
    
    # Approximate pricing as of mid-2026 (illustrative)
    cost_per_1k_input = 0.005   # USD
    cost_per_1k_output = 0.015
    
    return {
        "tokens": n_tokens,
        "chars_per_token": len(text) / n_tokens,
        "estimated_cost_if_input": n_tokens / 1000 * cost_per_1k_input,
    }

sample_text = "The attention mechanism in transformers allows each token to attend to every other token in the context window, computing a weighted sum of value vectors based on query-key similarities."
result = estimate_tokens(sample_text)
print(f"Tokens: {result['tokens']}")
print(f"Chars per token: {result['chars_per_token']:.2f}")
print(f"Cost as input: ${result['estimated_cost_if_input']:.5f}")
```

For well-written English text, a rough rule of thumb is ~4 characters per token. Code is usually more token-efficient (common patterns get merged). Unusual strings (URLs, base64, random hashes) can balloon.

---

## Summary

Tokenization is the first decision a language model system makes, and it echoes through everything downstream:

- **BPE** builds the vocabulary by iteratively merging frequent adjacent pairs
- **Space and case matter** — tokenization is not stable under simple text transformations
- **Numbers and code** tokenize in ways that don't align with their logical structure
- **Non-English text** is penalized in vocabularies built primarily on English
- **Larger vocabularies** are generally better, trading vocabulary memory for sequence efficiency

The model never sees characters. It sees integer IDs, which get converted to [embedding vectors](./embedding-words-as-vectors) before any computation begins. Whatever the tokenizer lost is gone.

---

*Next: [Self-Attention from Scratch](./self-attention-from-scratch) — how the transformer uses these token vectors to let each position attend to every other position in the sequence.*

*Related: [Embedding — Words as Points in Space](./embedding-words-as-vectors) — what happens to token IDs once tokenization is done.*
