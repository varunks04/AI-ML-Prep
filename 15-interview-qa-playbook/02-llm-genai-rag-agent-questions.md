# 2. Interview Q&A Playbook: LLMs, Inference, Prompt & Context Engineering, RAG, and AI Agents

This section contains the high-yield questions most frequently asked in modern AI Engineer technical screens and interviews.

---

## 2.1 Large Language Models & Transformers

### Q1 [Basic]: What is the fundamental difference between an Encoder-only model (BERT) and a Decoder-only model (GPT/LLaMA)?
* **Model Answer**:
  * **Encoder-only (e.g., BERT)**: Uses bidirectional self-attention. Every token can attend to tokens to its left AND right simultaneously. Ideal for understanding, classification, extractive QA, and creating dense vector embeddings. Cannot perform autoregressive free-form text generation.
  * **Decoder-only (e.g., GPT, LLaMA)**: Uses causal (masked) self-attention. Token $t$ can attend strictly to past tokens $\le t$. Ideal for autoregressive text generation, conversational dialogue, reasoning, and instruction following. It is the dominant architecture of modern Generative AI.

---

### Q2 [Intermediate]: What is the KV Cache in LLM inference, and why is it essential?
* **Model Answer**:
  * During autoregressive decoding, the model generates tokens one by one:
    $$t_1 \to t_2 \to t_3 \to \dots$$
  * To compute attention for the new token $t_{\text{new}}$, it needs the Key ($\mathbf{K}$) and Value ($\mathbf{V}$) projections of all preceding tokens in the context.
  * Without caching, computing each new token would require re-running the forward pass on the entire prompt and all previously generated tokens, turning inference complexity into an $O(L^2)$ computation per sequence.
  * **The KV Cache** stores the Key and Value activation tensors for all past tokens in GPU VRAM. For each new token, the model only computes $\mathbf{q}_{\text{new}}, \mathbf{k}_{\text{new}}, \mathbf{v}_{\text{new}}$, concatenates them into the cache, and computes attention in $O(L)$ time.
  * *Tradeoff*: KV Cache saves massive compute but consumes substantial GPU VRAM, making memory bandwidth and VRAM capacity the primary bottlenecks for serving long contexts.

---

### Q3 [Advanced]: How does FlashAttention achieve 2x–4x speedups without changing the output mathematically?
* **Model Answer**:
  * Standard attention materializes the massive $S \times S$ attention matrix ($\mathbf{Q}\mathbf{K}^T$) in GPU High-Bandwidth Memory (HBM). For long sequence lengths $S$, repeatedly reading and writing this matrix between HBM and on-chip SRAM creates an IO/memory-bandwidth bottleneck that starves GPU compute cores.
  * **FlashAttention (Dao et al.)** solves this using two innovations:
    1. **Tiling**: Splits $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ into blocks that fit entirely inside high-speed on-chip SRAM.
    2. **Online Softmax**: Computes Softmax normalization factors incrementally per block using mathematical scaling identities, avoiding the need to see the entire sequence at once.
  * Attention is computed in a single fused GPU kernel without ever writing the intermediate $S \times S$ matrix to HBM, reducing memory access from $O(S^2)$ to $O(S)$ and yielding dramatic speedups with exact mathematical equivalence.

---

### Q4 [Advanced]: Explain the difference between Dense Models and Mixture of Experts (MoE). What are active vs total parameters?
* **Model Answer**:
  * In a **Dense Model** (e.g., LLaMA-3-70B), every single token activates 100% of the model's 70 Billion parameters across all layers.
  * In a **Mixture of Experts (MoE)** model (e.g., Mixtral 8x7B), the standard dense Feed-Forward Network (FFN) layer is replaced by multiple independent "expert" networks (e.g., 8 experts) governed by a lightweight **Router / Gating Network**.
  * For each token, the router computes a Softmax gating score and directs the token to only the **Top-2 experts**.
  * **Active vs Total Parameters**:
    * *Total Parameters*: The sum of all weights stored in GPU VRAM (e.g., 47 Billion).
    * *Active Parameters*: The subset of parameters that perform computation for any given token (e.g., 13 Billion).
  * *Benefit*: Delivers the reasoning capacity of a 47B model at the inference latency and compute cost of a 13B model.

---

### Q5 [Intermediate]: What is Speculative Decoding, and why is it guaranteed to preserve output distribution?
* **Model Answer**:
  * Autoregressive generation is memory bandwidth-bound (loading large weight matrices from VRAM to compute a single token underutilizes GPU Tensor Cores).
  * **Speculative Decoding** pairs a small, fast Draft Model (e.g., 1B) with a large Target Model (e.g., 70B):
    1. The draft model quickly generates $K$ candidate tokens autoregressively.
    2. The large target model processes all $K$ tokens simultaneously in a **single parallel forward pass**, verifying their likelihoods.
    3. Tokens are accepted or rejected based on a modified rejection sampling formula.
  * *Why Distribution is Preserved*: The rejection sampling mathematics guarantees that the accepted tokens are sampled from the exact same statistical probability distribution as if the 70B target model had generated them alone.
  * Delivers a $2\times$ to $3\times$ latency speedup with zero quality degradation.

---

### Q6 [Advanced]: How does Grouped-Query Attention (GQA) reduce KV Cache memory compared to Multi-Head Attention (MHA)?
* **Model Answer**:
  * In **Multi-Head Attention (MHA)**, if there are 32 query heads, there are also 32 key heads and 32 value heads. The KV cache stores $32 \times 2 = 64$ tensors per layer per token.
  * In **Grouped-Query Attention (GQA)** (e.g., LLaMA-3), query heads are grouped (e.g., 4 query heads share 1 single Key head and 1 Value head).
  * For 32 query heads, there are only 8 Key heads and 8 Value heads:
    $$\text{KV Memory Reduction} = \frac{32}{8} = 4\times \text{ smaller KV Cache!}$$
  * This allows the serving engine (vLLM) to serve $4\times$ larger concurrent batch sizes or $4\times$ longer context lengths on the same GPU without degrading task accuracy.

---

## 2.2 LLM Inference & Generation Parameters

### Q7 [Basic]: If an LLM response is generating invalid JSON with syntax errors, which inference parameters should you adjust and why?
* **Model Answer**:
  1. Set **Temperature to 0.0**: Eliminates sampling randomness; forces the model to select the argmax token at each step, strictly following syntactic grammar rules.
  2. Increase **`max_tokens`**: A frequent cause of broken JSON is reaching `max_tokens` prematurely, which cuts off output before closing `}` or `]` brackets (`finish_reason = "length"`).
  3. Use **Structured Outputs / JSON Mode**: Modern provider APIs allow passing a JSON schema or Pydantic model. Under the hood, this enforces **grammar-constrained decoding**, masking out all tokens from the vocabulary that would violate the JSON grammar at that specific token position.

---

### Q8 [Intermediate]: Explain the difference between Top-K and Top-P (Nucleus) sampling. Why is Top-P generally preferred?
* **Model Answer**:
  * **Top-K**: Truncates candidate tokens to a fixed number $K$ of the highest-probability tokens.
    * *Weakness*: Inflexible. If the model is confident with one 95% token, Top-K=50 forces 49 noisy tokens into the pool. If the distribution is flat across 50 valid synonyms, Top-K=10 artificially cuts off 40 plausible candidates.
  * **Top-P (Nucleus)**: Dynamically selects the smallest set of top tokens whose cumulative probability reaches threshold $P$ (e.g., $0.90$).
    * *Advantage*: The pool size expands when the model is uncertain (flat distribution) and contracts to 1 or 2 tokens when the model is confident (sharp distribution). It adapts dynamically to model certainty.

---

## 2.3 Prompt & Context Engineering

### Q9 [Intermediate]: What is the "Lost in the Middle" phenomenon, and how do you architect context to counteract it?
* **Model Answer**:
  * Research shows that LLMs exhibit a U-shaped performance curve: they retrieve and reason over facts placed at the **beginning** (Primacy bias) and **end** (Recency bias) of long contexts with high fidelity, but accuracy drops significantly for facts located in the middle 60% of the prompt.
  * **Architectural Countermeasures**:
    1. **Context Prioritization**: Place critical system instructions and guidelines at the top, and place the user's specific question and highest-ranking retrieved chunks at the very bottom.
    2. **Reranking & Pruning**: Use cross-encoder rerankers to pass only the top 3–5 highest-affinity chunks instead of 50 mediocre chunks.
    3. **Document Re-ordering**: Sort retrieved chunks so the highest-scoring chunk is placed first or last, rather than in the center.

---

### Q10 [Advanced]: How does Prompt Caching work, and what are its performance and economic benefits?
* **Model Answer**:
  * LLMs process the input prompt by computing Key and Value tensors for every token during the prefill phase.
  * In many production applications (system instructions, multi-turn chat, document QA), the **first 1,000 to 10,000 tokens of the prompt are identical** across consecutive requests.
  * **Prompt Caching** (supported by Anthropic, OpenAI, DeepSeek, and vLLM) identifies matching prompt prefixes, stores their pre-computed KV activation tensors in GPU memory, and reuses them directly.
  * **Benefits**:
    1. **Latency**: Slashes Time-To-First-Token (TTFT) by up to 80% because the model skips prefill for the cached prefix.
    2. **Cost**: Providers discount cached input tokens by 50% to 90%, drastically lowering unit economics for RAG and agent systems.

---

## 2.4 Retrieval-Augmented Generation (RAG)

### Q11 [Intermediate]: Walk me through the difference between a Bi-Encoder and a Cross-Encoder. Where are they each used in RAG?
* **Model Answer**:
  * **Bi-Encoder (Embedding Model, e.g., text-embedding-3)**:
    * Encodes Query and Document *independently* into dense vectors: $\mathbf{u} = f(Q), \mathbf{v} = f(D)$.
    * Similarity is computed via dot product $\mathbf{u} \cdot \mathbf{v}$.
    * *Strength*: Fast. Document vectors can be pre-computed offline and indexed in an HNSW vector database.
    * *Role*: **Stage 1 Retrieval** (searches millions of chunks down to top 50 in 15ms).
  * **Cross-Encoder (Reranker, e.g., Cohere Rerank, BGE-Reranker)**:
    * Feeds Query and Document *together* as a single concatenated input `[CLS] Query [SEP] Document` through all Transformer layers.
    * Allows full cross-attention between every query token and document token.
    * *Strength*: Much higher semantic precision and relevance scoring.
    * *Role*: **Stage 2 Reranking** (scores the top 50 candidates down to the top 3–5 chunks for the final prompt).

---

### Q12 [Intermediate]: What is Hybrid Search, and why does pure vector search often fail in enterprise search?
* **Model Answer**:
  * **Pure Vector Search** matches conceptual semantic similarity, but blurs out exact character strings. It frequently fails on:
    * Specific error codes (e.g., `"ERR-401-B"`)
    * Exact product SKUs or serial numbers (`"iPhone-15-Pro-Max-256"`)
    * Rare proper nouns, acronyms, or specific customer names
  * **Hybrid Search** executes two searches in parallel:
    1. **Sparse Lexical Search (BM25)**: Evaluates exact keyword matching and inverse document frequency.
    2. **Dense Vector Search**: Evaluates conceptual semantic meaning.
  * The results are merged using **Reciprocal Rank Fusion (RRF)**:

    $$\text{Score}(d) = \sum \frac{1}{60 + \text{rank}(d)}$$

  * Hybrid search delivers the best of both worlds: semantic understanding without sacrificing exact keyword recall.

---

### Q13 [Advanced]: What is Corrective RAG (CRAG) and how does it prevent hallucination when retrieval fails?
* **Model Answer**:
  * Standard RAG blindly feeds retrieved documents into the LLM even when retrieval fetches irrelevant noise.
  * **Corrective RAG (CRAG)** inserts a lightweight **Retrieval Evaluator** between retrieval and synthesis:
    1. It evaluates the relevance of retrieved documents to the query, outputting a confidence score.
    2. If **Correct (High Confidence)**: Proceed to generation.
    3. If **Incorrect (Low Confidence)**: The system recognizes the vector database does not contain the answer, discards the bad chunks, and automatically falls back to an external web search query.
    4. If **Ambiguous**: Combines filtered internal chunks with web search.
  * Guarantees the LLM is never forced to synthesize an answer from ungrounded noise.

---

## 2.5 AI Agents & MCP (Model Context Protocol)

### Q14 [Basic]: How does an LLM "call a tool"? Does the model execute code on your server?
* **Model Answer**:
  * **No**, the model never executes code directly on the server.
  * Tool calling is structured JSON message-passing:
    1. The application sends the user prompt along with tool definitions (JSON Schemas containing function names, descriptions, and required parameter types).
    2. The model reasons that a tool is needed and outputs a structured JSON response specifying the function name and arguments (e.g., `{"name": "lookup_order", "arguments": {"order_id": 104}}`).
    3. The application parses this JSON, runs the actual Python/SQL code in a secure local environment, and gets the result.
    4. The application sends the execution result back to the model as a `tool` role message.
    5. The model reads the tool output and synthesizes the final natural language answer.

---

### Q15 [Intermediate]: What is the Model Context Protocol (MCP) and what problem does it solve?
* **Model Answer**:
  * MCP is an open-standard protocol introduced by Anthropic that standardizes how AI applications connect to external tools, data resources, and workflows.
  * **The Problem**: Before MCP, connecting $N$ AI clients (Claude Desktop, Cursor, internal agents) to $M$ enterprise data sources (GitHub, PostgreSQL, Slack, Google Drive) required writing $N \times M$ custom integration wrappers.
  * **The Solution**: MCP transforms this into an $N + M$ ecosystem:
    * Data sources build **one** MCP Server exposing **Tools** (executable actions), **Resources** (data/files), and **Prompts** (slash templates).
    * Any MCP Client can dynamically connect to any MCP Server over standard JSON-RPC 2.0 (via `stdio` or `SSE`), discovering and invoking tools securely with standardized permission controls.
