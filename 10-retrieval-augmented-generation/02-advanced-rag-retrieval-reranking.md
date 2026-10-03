# 2. Retrieval-Augmented Generation (RAG): Advanced Retrieval, BM25 Math, Reranking, and Evaluation

Naive RAG pipelines (chunk $\rightarrow$ embed $\rightarrow$ top-K search $\rightarrow$ prompt) frequently fail in enterprise production. Production-grade RAG systems implement advanced query transformations, hybrid search, reranking, corrective feedback loops, and automated evaluation frameworks.

---

## 2.1 Hybrid Search: Sparse Lexical (BM25) + Dense Semantic

Pure vector search struggles on exact part numbers, acronyms, or specific customer names. Pure keyword search fails on synonyms and conceptual questions. **Hybrid Search combines both.**

```text
User Query: "Error ERR-904 in authentication server"
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
[ Dense Vector Search ]                 [ Sparse Keyword Search (BM25) ]
Matches semantic intent:                Matches exact token string:
"Server authorization failure"          "ERR-904"
Top 20 candidates                       Top 20 candidates
         |                                       |
         +-------------------+-------------------+
                             |
                             v
           [ Reciprocal Rank Fusion (RRF) ]
           Merges & re-scores candidates:
           RRF_Score = sum( 1 / (60 + rank_i) )
                             |
                             v
                     Unified Top Candidates
```

### 2.1.1 BM25 (Best Matching 25) Mathematical Formulation
BM25 improves upon standard TF-IDF by adding non-linear term frequency saturation and document length normalization:

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$: Raw frequency of query term $q_i$ in document $D$.
* $|D|$: Length of document $D$ in words.
* $\text{avgdl}$: Average document length across the entire corpus.
* $k_1$ (typically $1.2\text{ to }2.0$): **Term Frequency Saturation parameter**. Prevents a word appearing 50 times from scoring $50\times$ higher than a word appearing twice. As $f(q_i, D) \to \infty$, the term frequency score saturates asymptotically at $(k_1 + 1)$.
* $b$ (typically $0.75$): **Length Normalization parameter**. Penalizes long, rambling documents ($b=1$ means full length penalty; $b=0$ disables length normalization).

### 2.1.2 Worked Numerical Example: Reciprocal Rank Fusion (RRF)

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k = 60$, and $r_m(d)$ is the 1-based rank of document $d$ in search system $m$.

```text
Candidate Document: Doc_A
- Ranked #1 in Dense Vector Search (r_dense = 1)
- Ranked #3 in BM25 Lexical Search (r_sparse = 3)

RRF_Score(Doc_A) = (1 / (60 + 1)) + (1 / (60 + 3))
                 = (1 / 61) + (1 / 63)
                 = 0.01639 + 0.01587 = 0.03226

Candidate Document: Doc_B
- Ranked #40 in Dense Vector Search (r_dense = 40)
- Ranked #1 in BM25 Lexical Search (r_sparse = 1)

RRF_Score(Doc_B) = (1 / (60 + 40)) + (1 / (60 + 1))
                 = (1 / 100) + (1 / 61)
                 = 0.01000 + 0.01639 = 0.02639

Result: Doc_A (ranked high in BOTH modalities) wins top final ranking!
```

---

## 2.2 Advanced RAG Paradigms: Corrective RAG (CRAG) and Self-RAG

```text
                                CORRECTIVE RAG (CRAG) ARCHITECTURE
                                
                                      [ User Question ]
                                              |
                                              v
                                  [ Retrieve Chunks from DB ]
                                              |
                                              v
                              +-------------------------------+
                              |    Retrieval Evaluator LLM    |
                              |    Is context sufficient?     |
                              +-------------------------------+
                                              |
                     +------------------------+------------------------+
                     | (High Confidence)      | (Low Confidence)       | (Ambiguous)
                     v                        v                        v
             [ Correct Context ]      [ Incorrect Context ]     [ Partial Context ]
                     |                        |                        |
                     |                        v                        v
                     |               [ Fallback to Live ]      [ Combine Docs + ]
                     |               [ Web Search API   ]      [ Web Search     ]
                     |                        |                        |
                     +------------------------+------------------------+
                                              |
                                              v
                                   [ Synthesize Final Answer ]
```

### 2.2.1 Self-RAG (Self-Reflective RAG)
Self-RAG trains the language model to emit special reflection tokens during generation:
* `[Retrieve]`: Decides whether to query external knowledge dynamically.
* `[IsRel]`: Evaluates whether retrieved chunks are relevant to the query.
* `[IsSup]`: Checks whether the generated claims are supported by the context.
* `[IsUse]`: Assesses whether the final answer is useful to the user.

---

## 2.3 Cross-Encoder Reranking: Bi-Encoders vs Cross-Encoders

```text
Bi-Encoder (Stage 1 Dense Retriever):
Query (Q)  ---> [ Encoder ] ---> Vector u \
                                           +---> Dot Product u • v (15ms across 10M docs)
Doc (D)    ---> [ Encoder ] ---> Vector v /

Cross-Encoder (Stage 2 Reranker):
[ [CLS] + Query + [SEP] + Document + [SEP] ] ---> [ Full Transformer Layers ] ---> Score (0 to 1)
Every single token of the query attends directly to every token of the document!
Too slow to run on 1M documents, but delivers 99% precision on the Top-50 candidates!
```

---

## 2.4 The RAG Triad and Automated Evaluation

```text
                        User Query
                       /          \
                      /            \
  (Context Relevance)/              \(Answer Relevance)
                    /                \
                   v                  v
            Retrieved Context ------> Generated Answer
                     (Groundedness / Faithfulness)
```

| Metric | Evaluates | How It Is Measured | Root Cause If Low |
| :--- | :--- | :--- | :--- |
| **Context Relevance** | Retrieval Engine | Proportion of retrieved sentences genuinely relevant to the query. | Bad embeddings, suboptimal chunk size, no reranker. |
| **Groundedness (Faithfulness)**| LLM Generator | Percentage of claims in answer directly provable from retrieved context. | LLM hallucination, high temperature ($T > 0.5$). |
| **Answer Relevance** | Prompt Adherence | Semantic similarity between generated answer and original user question. | Model wandered off-topic, ignored prompt constraints. |
