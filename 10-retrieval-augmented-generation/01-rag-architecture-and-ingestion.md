# 1. Retrieval-Augmented Generation (RAG): Architecture, Ingestion, and Advanced Chunking

Retrieval-Augmented Generation (RAG) grounds Large Language Models in dynamic, private, and auditable external knowledge bases without the prohibitive cost of fine-tuning or model retraining.

---

## 1.1 The Dual-Pipeline RAG Architecture

A production RAG system separates data preparation from real-time generation:

```text
========================================================================================
1. OFFLINE INGESTION PIPELINE (Asynchronous Event / Batch)
========================================================================================
[ Enterprise Data Sources: PDFs, Confluence, Markdown, SQL, Notion ]
                              |
                              v
                  [ Document Parsing & OCR ]
                              |
                              v
                  [ Cleaning & Structural Normalization ]
                              |
                              v
                  [ Chunking (Recursive, Semantic, Sentence-Window) ]
                              |
                              v
                  [ Metadata Enrichment: {doc_id, timestamp, access_role} ]
                              |
                              v
                  [ Embedding Model (e.g., text-embedding-3-small) ]
                              |
                              v
                  [ Vector Database Storage (Qdrant, Pinecone, Milvus) ]

========================================================================================
2. ONLINE QUERY & GENERATION PIPELINE (Real-Time Synchronous)
========================================================================================
User Query: "What is the return window for opened consumer electronics?"
                              |
                              v
                  [ Query Transformation / HyDE ]
                              |
                              v
                  [ Hybrid Search (Dense Vectors + BM25 Sparse) ]
                              |
                              v
                  [ Cross-Encoder Reranker (Top-50 -> Top-4 Chunks) ]
                              |
                              v
                  [ Grounded Prompt Assembly with Citations ]
                              |
                              v
                  [ LLM Generation (Temperature = 0.1 - 0.2) ]
                              |
                              v
              Verifiable Answer with Source Citations
```

---

## 1.2 Deep Dive into Chunking Strategies

Selecting the wrong chunking strategy is the single most common cause of poor RAG accuracy.

```text
                                CHUNKING STRATEGIES COMPARISON

1. Fixed-Size Chunking (Naive):
   [ 500 characters ]----[ 50 char overlap ]----[ 500 characters ]
   * Fails: Chops sentences and tables in half, severing semantic meaning.

2. Recursive Character Chunking (Standard Baseline):
   Splits hierarchy: Paragraphs ("\n\n") -> Sentences ("\n", ". ") -> Words (" ")
   * Keeps paragraphs intact; falls back to smaller units only when size cap is exceeded.

3. Sentence-Window Retrieval (Small-to-Large):
   Embeds: [ Target Sentence ] (100% focused semantic vector)
   Retrieves to LLM: [ Prev 3 Sentences ] + [ Target Sentence ] + [ Next 3 Sentences ]
   * Solves the embedding vs synthesis dilemma!

4. Parent-Document / Hierarchical Chunking:
   Child Chunks (128 tokens) ---> Embedded in Vector DB for high-precision retrieval
             |
             v (Upon match, fetches parent)
   Parent Chunk (1024 tokens) ---> Passed to LLM for full contextual synthesis

5. Semantic Chunking (Advanced):
   Computes embedding distance between sentence i and sentence i+1.
   Places a split when cosine distance spikes above a statistical threshold (e.g., 90th percentile).
```

### 1.2.1 Sentence-Window Retrieval Architecture (Small-to-Large)

```text
Raw Document:
"The battery voltage is 48V. Operating temperature must remain under 65C. 
 Exceeding this thermal limit permanently voids warranty coverage. Always install cooling fans."

Embed & Index:
Target Child: "Exceeding this thermal limit permanently voids warranty coverage." (Clean semantic vector!)

At Query Time:
User asks: "What happens if battery gets too hot?"
Child sentence matches query vector.
System replaces child with its surrounding Sentence Window (3 sentences):
"Operating temperature must remain under 65C. Exceeding this thermal limit permanently 
 voids warranty coverage. Always install cooling fans."
Passed to LLM for rich context!
```

---

## 1.3 Document Parsing Challenges

* **Complex PDF Tables**: Naive PDF parsers read tables left-to-right across columns, merging unrelated columns into scrambled text.
  * *Fix*: Use layout-aware vision models or OCR table parsers (`pdfplumber`, `Unstructured`, `LlamaParse`) that convert tables into clean Markdown or HTML table syntax (`| Column 1 | Column 2 |`) before embedding.
* **Document Hierarchy**: Retain breadcrumb metadata during parsing:
  `metadata = {"doc": "Employee_Handbook.pdf", "h1": "Benefits", "h2": "Health Insurance"}`.
  Injecting these breadcrumbs into chunk text preserves context that chunk text alone might omit.
