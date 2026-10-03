# 1. LLM Evaluation: Metrics, Information Retrieval Math, Ragas, and LLM-as-a-Judge

Evaluating Generative AI systems requires distinct metrics for information retrieval (retrieval stage) and language generation (synthesis stage).

---

## 1.1 Information Retrieval Metrics: Worked Mathematical Walkthroughs

### 1.1.1 Mean Reciprocal Rank (MRR)
Measures how quickly the system presents the **first relevant result**:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

```text
Query 1: First relevant document found at Rank 1 ---> Reciprocal Rank = 1/1 = 1.00
Query 2: First relevant document found at Rank 3 ---> Reciprocal Rank = 1/3 = 0.33
Query 3: First relevant document found at Rank 2 ---> Reciprocal Rank = 1/2 = 0.50

MRR = (1.00 + 0.33 + 0.50) / 3 = 0.61
```

### 1.1.2 NDCG (Normalized Discounted Cumulative Gain)
Evaluates graded relevance (e.g., highly relevant $= 3$, relevant $= 1$, irrelevant $= 0$) with a logarithmic position penalty:

$$\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

```text
Retrieved Ranking for Query: [ Doc A (rel=3), Doc B (rel=0), Doc C (rel=2) ]

1. Calculate Actual DCG:
   Rank 1 (Doc A): (2^3 - 1) / log2(1 + 1) = 7 / 1.0  = 7.00
   Rank 2 (Doc B): (2^0 - 1) / log2(2 + 1) = 0 / 1.58 = 0.00
   Rank 3 (Doc C): (2^2 - 1) / log2(3 + 1) = 3 / 2.0  = 1.50
   Actual DCG = 7.00 + 0.00 + 1.50 = 8.50

2. Calculate Ideal DCG (IDCG - perfectly sorted: rel=3, rel=2, rel=0):
   Rank 1: 7 / 1.0 = 7.00
   Rank 2: 3 / 1.58 = 1.89
   Rank 3: 0 / 2.0 = 0.00
   Ideal DCG = 7.00 + 1.89 + 0.00 = 8.89

3. NDCG = Actual DCG / Ideal DCG = 8.50 / 8.89 = 0.956 (High Quality!)
```

---

## 1.2 The Ragas Evaluation Framework: Mathematical Formulations

The **Ragas** (Retrieval-Augmented Generation Assessment) framework standardizes quantitative evaluation across the RAG lifecycle:

```text
                              RAGAS METRICS SUITE
                                       |
       +-------------------------------+-------------------------------+
       |                                                               |
  RETRIEVAL METRICS                                           GENERATION METRICS
  - Context Precision                                         - Faithfulness (Groundedness)
  - Context Recall                                            - Answer Relevance
```

### 1.2.1 Faithfulness (Groundedness)
Measures the proportion of factual claims in the generated answer that can be inferred directly from the retrieved context:

$$\text{Faithfulness} = \frac{|\text{Number of claims in answer supported by context}|}{|\text{Total factual claims in generated answer}|}$$

* Step 1: LLM extracts atomic statements: *"The company was founded in 2012 by Alice and Bob."* $\to$ Statement 1: *"Founded in 2012"*, Statement 2: *"Founded by Alice and Bob"*.
* Step 2: LLM verifies each atomic statement against retrieved context chunks.
* Step 3: Returns score in $[0, 1]$. Score $< 1.0$ indicates hallucination.

### 1.2.2 Answer Relevance
Measures whether the response directly addresses the user query, regardless of factual grounding. Evaluated by generating $N$ hypothetical questions from the generated answer and computing mean cosine similarity with the original user query:

$$\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^N \cos(\mathbf{e}_{\text{gen\_q}_i}, \mathbf{e}_{\text{orig\_q}})$$

### 1.2.3 Context Precision@K
Evaluates whether all ground-truth relevant chunks are ranked at the very top of the retrieved context:

$$\text{Context Precision@K} = \frac{\sum_{k=1}^K (\text{Precision@}k \times v_k)}{\text{Total number of relevant chunks in top } K}$$

Where $v_k \in \{0, 1\}$ is a binary indicator indicating whether chunk $k$ is relevant.

---

## 1.3 LLM-as-a-Judge: Biases, G-Eval, and Mitigation Strategies

```text
+-----------------------+---------------------------------------+---------------------------------------+
| Judge Bias            | Manifestation                         | Production Mitigation                 |
+-----------------------+---------------------------------------+---------------------------------------+
| Position Bias         | Favors Option A over Option B in      | Swap candidate ordering ([A,B] and    |
|                       | pairwise comparisons.                 | [B,A]); award win only if consistent. |
+-----------------------+---------------------------------------+---------------------------------------+
| Verbosity Bias        | Awards higher scores to longer,       | Normalize scores by length; instruct  |
|                       | wordier answers with extra bullets.   | judge to penalize irrelevant fluff.   |
+-----------------------+---------------------------------------+---------------------------------------+
| Self-Enhancement Bias | GPT-4 favors GPT-4 answers over       | Blind model names; cross-evaluate with|
|                       | Claude answers.                       | multi-model jury (GPT-4 + Claude).    |
+-----------------------+---------------------------------------+---------------------------------------+
| Egocentric Bias       | Model rates answers similar to its    | Use structured multi-criteria rubrics |
|                       | own default style higher.             | with explicit scoring anchors (1-5).  |
+-----------------------+---------------------------------------+---------------------------------------+
```

### 1.3.1 The G-Eval Framework (Liu et al., 2023)
G-Eval uses large language models with **Chain-of-Thought (CoT)** and probability form-filling:
1. Generates step-by-step evaluation steps based on criteria.
2. Evaluates the candidate response according to those steps.
3. Obtains token probabilities of score tokens to calculate expected score:
   $$\text{Score} = \sum_{i=1}^5 i \times P(\text{Score} = i)$$

---

## 1.4 CI/CD Continuous Evaluation Architecture

```text
[ Feature Branch PR: Updates prompt template or chunk size ]
                            |
                            v
[ GitHub Actions CI Runner: Dispatches Eval Pipeline ]
                            |
                            v
[ Benchmark against Golden Dataset (200 Curated Test Cases) ]
                            |
         +------------------+------------------+
         |                                     |
         v                                     v
[ Retrieval Evaluation ]              [ Generation Evaluation ]
- Context Precision >= 0.85           - Faithfulness >= 0.95
- Context Recall >= 0.90              - Answer Relevance >= 0.90
         \                                     /
          +-----------------+-----------------+
                            |
                            v
     Thresholds Met? ------+------ Regression Detected?
            |                              |
            v                              v
     [ Approve PR ]                 [ Block Merge & Alert Slack ]
```
