# AI Engineer Interview Preparation Repository

A focused, high-ROI study repository and technical reference guide engineered specifically for candidates preparing for **entry-level and fresher AI Engineer interviews**.

This repository prioritizes **conceptual understanding, practical systems engineering, and interview readiness over academic breadth**.

---

## 🎯 Prioritized Learning Roadmap

When preparing on a finite timeline, study in order of interview return-on-investment (ROI):

```text
+---------------------------------------------------------------------------------------------------+
| TIER 1: CRITICAL MUST-KNOWS (Study First — Accounts for ~70% of Modern AI Engineer Interviews)    |
| - Section 05: Large Language Models (LLM Training Stages, Tokens, KV Cache, Hallucinations)       |
| - Section 06: LLM Inference & Generation (Temperature, Top-K, Top-P, Greedy Decoding)             |
| - Section 09: Embeddings & Vector Search (Cosine Similarity, ANN Indexing, Vector Databases)      |
| - Section 10: RAG (Ingestion, Chunking, Hybrid Search, Reranking, The RAG Triad)                  |
| - Section 11: AI Agents & MCP (ReAct loops, Tool calling, Model Context Protocol architecture)     |
| - Section 07 & 08: Prompt & Context Engineering (CoT, Structured Outputs, Context Management)     |
| - Section 15: Interview Q&A Playbook (All 3 parts)                                                |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| TIER 2: ESSENTIAL SYSTEMS & FOUNDATIONS (Study Next — Core Technical Competencies)                |
| - Section 01: ML Fundamentals (Overfitting/Underfitting, Bias-Variance, Preprocessing, Leakage)    |
| - Section 01.2: Common ML Models (Random Forest, XGBoost, Logistic Reg, K-Means)                  |
| - Section 04: NLP Fundamentals (Tokenization, TF-IDF, Word2Vec, Transition to Transformers)       |
| - Section 12: LLM Application Engineering (APIs, Streaming SSE, Caching, Latency/Cost)            |
| - Section 13: LLM Evaluation (LLM-as-a-Judge, Precision@K, NDCG, Ragas, CI/CD evals)              |
| - Section 14: AI Security & Guardrails (Prompt Injection, PII Scrubbing, Least Privilege)         |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| TIER 3: FOUNDATIONAL THEORY & BACKGROUND (Can Be Postponed / Reviewed as Needed)                  |
| - Section 02: Essential Mathematics (Scalars/Vectors/Tensors, Dot Products, Calculus intuition)   |
| - Section 03: Data Mining & Processing (ETL vs ELT, Outliers, Feature transformations)            |
+---------------------------------------------------------------------------------------------------+
```

---

## 📚 Complete Table of Contents & Module Links

### 01. Machine Learning Fundamentals
* [01-ml-core-concepts.md](file:///c:/Users/ASUS/Desktop/Prep/01-machine-learning-fundamentals/01-ml-core-concepts.md): AI vs ML vs DL vs GenAI hierarchy, the ML workflow, dataset splits, overfitting, bias-variance tradeoff, preprocessing, categorical encoding, scaling, and data leakage.
* [02-common-ml-models.md](file:///c:/Users/ASUS/Desktop/Prep/01-machine-learning-fundamentals/02-common-ml-models.md): Linear/Logistic Regression, Decision Trees, Random Forests, XGBoost, KNN, SVM, Naive Bayes, K-Means, PCA, Neural Network basics, and model selection cheatsheet.

### 02. Essential Mathematics for AI/ML
* [01-linear-algebra-vectors.md](file:///c:/Users/ASUS/Desktop/Prep/02-essential-mathematics/01-linear-algebra-vectors.md): Scalars, vectors, matrices, tensors, dot products, cosine similarity geometric intuition, matrix transformations, and how linear algebra powers Transformers.
* [02-probability-statistics-calculus.md](file:///c:/Users/ASUS/Desktop/Prep/02-essential-mathematics/02-probability-statistics-calculus.md): Statistics (mean, variance, standard deviation), probability, Bayes' theorem, distributions (Normal, Softmax), gradients, loss functions (MSE, Cross-Entropy), and Adam optimizer.

### 03. Data Mining & Processing
* [01-data-pipeline-and-quality.md](file:///c:/Users/ASUS/Desktop/Prep/03-data-mining-and-processing/01-data-pipeline-and-quality.md): Structured vs unstructured data, batch vs streaming ingestion, ETL vs ELT, data quality dimensions, outlier detection (Z-score, IQR), and missing value imputation strategies.
* [02-feature-engineering-and-leakage.md](file:///c:/Users/ASUS/Desktop/Prep/03-data-mining-and-processing/02-feature-engineering-and-leakage.md): Log/power transforms, binning, interaction terms, feature selection methods (Filter, Wrapper, Embedded), data leakage airlocked pipelines, and Tabular vs LLM document pipelines.

### 04. NLP Fundamentals
* [01-traditional-nlp-to-embeddings.md](file:///c:/Users/ASUS/Desktop/Prep/04-nlp-fundamentals/01-traditional-nlp-to-embeddings.md): Text cleaning, tokenization (BPE, WordPiece), stop words, stemming vs lemmatization, Bag of Words, TF-IDF formulas, Word2Vec (CBOW vs Skip-Gram), and static vs contextual embeddings.
* [02-text-processing-and-tasks.md](file:///c:/Users/ASUS/Desktop/Prep/04-nlp-fundamentals/02-text-processing-and-tasks.md): Classification, NER, Sentiment analysis, sequence models (RNNs, LSTMs, and the vanishing gradient/sequential bottleneck), and the bridge from classical NLP to modern LLMs.

### 05. Large Language Models (LLMs)
* [01-llm-foundations-and-training.md](file:///c:/Users/ASUS/Desktop/Prep/05-large-language-models/01-llm-foundations-and-training.md): Autoregressive definition, tokens, context windows, parameters, the 3-stage training lifecycle (Pre-training $\rightarrow$ SFT $\rightarrow$ RLHF/DPO), training vs inference, hallucinations, and model selection.
* [02-transformer-architecture-and-attention.md](file:///c:/Users/ASUS/Desktop/Prep/05-large-language-models/02-transformer-architecture-and-attention.md): Transformer archetypes, scaled dot-product attention formula breakdown ($\sqrt{d_k}$), Q/K/V database analogy, causal masking, RoPE positional encoding, MHA vs MQA vs GQA, and RMSNorm.

### 06. LLM Inference & Generation
* [01-sampling-decoding-parameters.md](file:///c:/Users/ASUS/Desktop/Prep/06-llm-inference-and-generation/01-sampling-decoding-parameters.md): Logits to probabilities, Greedy decoding, Temperature scaling, Top-K, Top-P (Nucleus) sampling, length caps, stop sequences, repetition/frequency penalties, determinism, and scenario presets.

### 07. Prompt Engineering
* [01-prompt-patterns-and-techniques.md](file:///c:/Users/ASUS/Desktop/Prep/07-prompt-engineering/01-prompt-patterns-and-techniques.md): Prompt structure, System/User/Assistant roles, Zero-shot, Few-shot (In-context learning), Chain-of-Thought (CoT), Structured JSON/Pydantic schemas, Prompt Chaining vs Monolithic prompts, and injection defense.

### 08. Context Engineering
* [01-context-management-and-optimizations.md](file:///c:/Users/ASUS/Desktop/Prep/08-context-engineering/01-context-management-and-optimizations.md): Prompt vs Context engineering, the context trilemma, "Lost in the Middle" phenomenon, dynamic token budgeting, context compression (extractive, summarization, pruning), conversation history patterns, and agent scratchpad management.

### 09. Embeddings & Vector Search
* [01-vector-embeddings-and-databases.md](file:///c:/Users/ASUS/Desktop/Prep/09-embeddings-and-vector-search/01-vector-embeddings-and-databases.md): Vector representations, Cosine vs Dot Product vs Euclidean distance, Vector search vs Keyword search, ANN indexing (HNSW graphs vs IVF), Vector DB ecosystem (Chroma, Pinecone, Qdrant, Weaviate, pgvector), metadata filtering, and 2-stage retrieval.

### 10. Retrieval-Augmented Generation (RAG)
* [01-rag-architecture-and-ingestion.md](file:///c:/Users/ASUS/Desktop/Prep/10-retrieval-augmented-generation/01-rag-architecture-and-ingestion.md): Why RAG is essential, dual-pipeline architecture (offline ingestion vs online query), parsing (PDFs, Markdown), chunking strategies (Fixed, Recursive, Semantic, Parent-Child), metadata enrichment, and context construction.
* [02-advanced-rag-retrieval-reranking.md](file:///c:/Users/ASUS/Desktop/Prep/10-retrieval-augmented-generation/02-advanced-rag-retrieval-reranking.md): Query rewriting, HyDE (Hypothetical Document Embeddings), Hybrid search (Dense + BM25 with Reciprocal Rank Fusion), Cross-Encoder reranking, 7 RAG failure modes, and the RAG Triad evaluation metrics.

### 11. AI Agents & Model Context Protocol (MCP)
* [01-agent-architectures-and-patterns.md](file:///c:/Users/ASUS/Desktop/Prep/11-ai-agents-and-mcp/01-agent-architectures-and-patterns.md): LLM vs Chatbot vs Agent autonomy spectrum, core agent anatomy, the ReAct pattern (Reason + Act), tool/function calling mechanics, workflows (DAGs) vs autonomous agents, and agent failure modes.
* [02-model-context-protocol-mcp.md](file:///c:/Users/ASUS/Desktop/Prep/11-ai-agents-and-mcp/02-model-context-protocol-mcp.md): The $N \times M$ integration problem, MCP host/client/server architecture, the 3 MCP primitives (Tools, Resources, Prompts), MCP vs traditional function calling, transport layers (stdio/SSE), and message flow walkthrough.

### 12. LLM Application Engineering
* [01-production-llm-apps-apis-cost-latency.md](file:///c:/Users/ASUS/Desktop/Prep/12-llm-application-engineering/01-production-llm-apps-apis-cost-latency.md): REST API anatomy, Server-Sent Events (SSE) streaming, TTFT vs TPOT latency metrics, rate limits and full-jitter exponential backoff, caching (Prompt caching, exact match, semantic caching), and observability with tracing.

### 13. LLM Evaluation
* [01-eval-metrics-ragas-and-llm-judge.md](file:///c:/Users/ASUS/Desktop/Prep/13-llm-evaluation/01-eval-metrics-ragas-and-llm-judge.md): Why LLM evaluation is hard, retrieval metrics (Precision@K, Recall@K, MRR, NDCG), LLM-as-a-Judge methodology, position/verbosity/self-enhancement biases, golden datasets, and CI/CD regression testing.

### 14. AI Security & Responsible AI
* [01-security-guardrails-and-threats.md](file:///c:/Users/ASUS/Desktop/Prep/14-ai-security-and-responsible-ai/01-security-guardrails-and-threats.md): OWASP Top 10 for LLMs, Direct vs Indirect Prompt Injection, PII scrubbing (Presidio), insecure tool execution & excessive agency, Role-Based Access Control (RBAC) in RAG, and guardrail architectures.

### 15. Interview Q&A Playbook
* [01-core-ml-nlp-questions.md](file:///c:/Users/ASUS/Desktop/Prep/15-interview-qa-playbook/01-core-ml-nlp-questions.md): High-yield interview questions on Machine Learning fundamentals, Math for AI, Data Leakage, and NLP concepts with detailed model answers.
* [02-llm-genai-rag-agent-questions.md](file:///c:/Users/ASUS/Desktop/Prep/15-interview-qa-playbook/02-llm-genai-rag-agent-questions.md): Core GenAI questions covering LLM training, KV caching, Top-K/Top-P/Temperature, Bi-Encoders vs Cross-Encoders, Hybrid Search, RAG Triad, and MCP.
* [03-scenario-and-system-design-questions.md](file:///c:/Users/ASUS/Desktop/Prep/15-interview-qa-playbook/03-scenario-and-system-design-questions.md): System design challenges (Enterprise RBAC RAG, Latency troubleshooting, Autonomous agent security) and common AI engineering misconceptions debunked.

---

## 🗓️ Recommended 14-Day Interview Sprint

| Day | Focus Area | Essential Reading Modules |
| :---: | :--- | :--- |
| **Day 1** | LLM Foundations & Training | [05-01: LLM Foundations](file:///c:/Users/ASUS/Desktop/Prep/05-large-language-models/01-llm-foundations-and-training.md) |
| **Day 2** | Transformers & Attention | [05-02: Transformer Architecture](file:///c:/Users/ASUS/Desktop/Prep/05-large-language-models/02-transformer-architecture-and-attention.md) |
| **Day 3** | Inference & Sampling Parameters | [06-01: Sampling Parameters](file:///c:/Users/ASUS/Desktop/Prep/06-llm-inference-and-generation/01-sampling-decoding-parameters.md) |
| **Day 4** | Prompt & Context Engineering | [07-01: Prompt Engineering](file:///c:/Users/ASUS/Desktop/Prep/07-prompt-engineering/01-prompt-patterns-and-techniques.md), [08-01: Context Engineering](file:///c:/Users/ASUS/Desktop/Prep/08-context-engineering/01-context-management-and-optimizations.md) |
| **Day 5** | Embeddings & Vector Databases | [09-01: Vector Embeddings & DBs](file:///c:/Users/ASUS/Desktop/Prep/09-embeddings-and-vector-search/01-vector-embeddings-and-databases.md) |
| **Day 6** | RAG Architecture & Ingestion | [10-01: RAG Architecture](file:///c:/Users/ASUS/Desktop/Prep/10-retrieval-augmented-generation/01-rag-architecture-and-ingestion.md) |
| **Day 7** | Advanced RAG & Reranking | [10-02: Advanced RAG & Reranking](file:///c:/Users/ASUS/Desktop/Prep/10-retrieval-augmented-generation/02-advanced-rag-retrieval-reranking.md) |
| **Day 8** | AI Agents & Function Calling | [11-01: Agent Architectures](file:///c:/Users/ASUS/Desktop/Prep/11-ai-agents-and-mcp/01-agent-architectures-and-patterns.md) |
| **Day 9** | Model Context Protocol (MCP) | [11-02: Model Context Protocol](file:///c:/Users/ASUS/Desktop/Prep/11-ai-agents-and-mcp/02-model-context-protocol-mcp.md) |
| **Day 10**| LLM App Engineering & Caching | [12-01: Production LLM Apps](file:///c:/Users/ASUS/Desktop/Prep/12-llm-application-engineering/01-production-llm-apps-apis-cost-latency.md) |
| **Day 11**| Evaluation & AI Security | [13-01: LLM Evaluation](file:///c:/Users/ASUS/Desktop/Prep/13-llm-evaluation/01-eval-metrics-ragas-and-llm-judge.md), [14-01: AI Security](file:///c:/Users/ASUS/Desktop/Prep/14-ai-security-and-responsible-ai/01-security-guardrails-and-threats.md) |
| **Day 12**| ML Fundamentals & Key Models | [01-01: ML Concepts](file:///c:/Users/ASUS/Desktop/Prep/01-machine-learning-fundamentals/01-ml-core-concepts.md), [01-02: Common Models](file:///c:/Users/ASUS/Desktop/Prep/01-machine-learning-fundamentals/02-common-ml-models.md) |
| **Day 13**| NLP Fundamentals & Math | [04-01: NLP & Embeddings](file:///c:/Users/ASUS/Desktop/Prep/04-nlp-fundamentals/01-traditional-nlp-to-embeddings.md), [02-01: Linear Algebra](file:///c:/Users/ASUS/Desktop/Prep/02-essential-mathematics/01-linear-algebra-vectors.md) |
| **Day 14**| Mock Interviews & Q&A Mastery | [15-01, 15-02, 15-03: Complete Q&A Playbook](file:///c:/Users/ASUS/Desktop/Prep/15-interview-qa-playbook/01-core-ml-nlp-questions.md) |
