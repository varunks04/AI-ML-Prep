# 1. Embeddings & Vector Search: Geometry, Indexing, Quantization, and Vector Databases

Vector embeddings are the bridge between unstructured human language and quantitative mathematical computation. They allow computers to perform semantic comparisons at scale.

---

## 1.1 The Semantic Search Pipeline

```text
Raw Text / Document
       ↓
Embedding Model (e.g., text-embedding-3-small, BGE-large)
       ↓
Dense Float Vector (e.g., [ 0.042, -0.912, 0.381, ... ] in R^1536)
       ↓
Vector Database (Indexed via HNSW / IVF graph)
       ↓
Similarity Search (Cosine Distance / Dot Product against Query Vector)
       ↓
Top-K Relevant Information Retrieved
```

---

## 1.2 Distance and Similarity Metrics

Given two dense embedding vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

### 1.2.1 Cosine Similarity
Measures the cosine of the angle $\theta$ between two vectors, evaluating directional alignment regardless of vector magnitude:

$$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

* **Range**: $[-1.0, 1.0]$.
  * $+1.0$: Identical orientation (maximum semantic affinity).
  * $0.0$: Orthogonal vectors (unrelated concepts).
  * $-1.0$: Opposite direction (diametrically opposed meaning).
* **Cosine Distance**: Defined as $1 - \text{CosineSimilarity}(\mathbf{u}, \mathbf{v})$, bounded between $0$ and $2$.

### 1.2.2 Dot Product (Inner Product)
The sum of the products of corresponding elements:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$

* **Range**: $(-\infty, \infty)$.
* **Unit Normalization Rule**: If all vectors stored in the vector database are pre-normalized to unit length ($\|\mathbf{u}\|_2 = 1, \|\mathbf{v}\|_2 = 1$), **the Dot Product is mathematically identical to Cosine Similarity**.
* **Hardware Acceleration**: Vector databases exploit this property to replace expensive square roots with blazing-fast AVX-512 / ARM NEON SIMD hardware dot-product instructions.

### 1.2.3 Euclidean Distance ($L_2$)
The straight-line geometric distance between two points in Euclidean space:

$$\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$

* **Range**: $[0, \infty)$. A distance of $0$ indicates identical vectors.
* **Relationship to Cosine**: For unit-normalized vectors:
  $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 2 - 2\cos(\theta)$$
  Minimizing Euclidean distance is monotonically equivalent to maximizing Cosine Similarity.

---

## 1.3 Approximate Nearest Neighbor (ANN) Indexing

Performing exact $K$-Nearest Neighbor ($k$-NN) brute-force search requires calculating distance across **every single vector** in the database ($O(N \cdot d)$). For 10 million vectors, this takes several seconds per query.

**ANN algorithms trade $<1\%$ of search recall for orders-of-magnitude speedups ($O(\log N)$ latency).**

### 1.3.1 HNSW (Hierarchical Navigable Small World): The Industry Gold Standard

HNSW builds a multi-layer graph where upper layers contain sparse long-range express highways, and the bottom layer contains all dense local nodes:

```text
Layer 2 (Express):    [ Entry Point ] -----------------------------------------> [ Node Z ]
                             \                                                      /
Layer 1 (Sub-Express):       [ Node B ] -------------> [ Node M ] -------------> [ Node Z ]
                                 \                          \                       /
Layer 0 (All Vectors):  [A] <-> [B] <-> [C] <-> [D] <-> [M] <-> [N] <-> [O] <-> [Z] (Nearest Match!)
```

* **Query Traversal Walkthrough**:
  1. Search begins at the top-layer **Entry Point**.
  2. Greedily hops to neighboring nodes that are closer to the query vector.
  3. When no closer neighbor exists on that layer, drops down one layer and resumes greedy hopping.
  4. At Layer 0, performs local greedy exploration to return the Top-K nearest neighbors in $<10\text{ms}$.

### 1.3.2 IVF (Inverted File Index)

```text
                    IVF VORONOI PARTITIONING
                     
            .-----------------.-----------------.
           /         *         \         o       \
          /      (Centroid 1)   \    (Centroid 2) \
         /    *        *         \   o       o     \
        +-------------------------+-----------------+
         \      Query Vector (Q) /                 /
          \            x        /    +            /
           \       (Centroid 3)/    (Centroid 4) /
            '-----------------'-----------------'
```

* Partitions high-dimensional space into $C$ Voronoi cells using K-Means clustering.
* **Query Time**: Computes distance to all $C$ centroids; selects the top $n_{\text{probe}}$ nearest centroids, and scans only the vectors inside those specific cells.
* **Advantage**: Requires less memory than HNSW; ideal for massive datasets ($>100\text{M}$ vectors).

---

## 1.4 Vector Quantization: Shrinking Memory Footprint by 4x–16x

1 million 1536-dimensional vectors at `FP32` require:

$$10^6 \times 1536 \times 4 \text{ bytes} \approx \mathbf{6.14 \text{ GB of RAM}}$$

For 50 million vectors, storing raw vectors in RAM costs thousands of dollars monthly. **Quantization compresses vectors directly.**

```text
1. Scalar Quantization (SQ8):
   Quantizes 32-bit floats (-1.0 to 1.0) into 8-bit signed integers (-128 to 127).
   - Memory Reduction: 4x (from 4 bytes/dim to 1 byte/dim).
   - Accuracy Retained: ~99% recall.

2. Product Quantization (PQ):
   Decomposes a 1536-dim vector into M=16 sub-vectors of dimension 96.
   Clusters each 96-dim subspace into 256 centroids using K-Means.
   Stores each sub-vector as a 1-byte centroid index (0 to 255).
   - Memory Reduction: 16x - 32x!
   - Entire 1536-dim vector compressed into just 16 bytes!
```

---

## 1.5 Metadata Filtering: Single-Stage vs Pre/Post-Filtering

Enterprise queries always include metadata predicates:
`"Find policy documents WHERE department = 'engineering' AND year >= 2025"`.

```text
1. Post-Filtering (Severe Under-Retrieval Risk):
   HNSW Vector Search finds Top-20 nearest vectors ---> Apply filter (dept == 'eng') ---> Only 1 left!
   Failed to return the required K=5 items.

2. Pre-Filtering (High Compute Overhead):
   Filter all 10M rows in database ---> Reconstruct temporary index on matching rows ---> Search.
   Extremely slow if matching rows count is large.

3. Single-Stage Filtered Search (Modern Standard in Qdrant & Pinecone):
   Traversal walks the HNSW graph while dynamically pruning candidate nodes that fail the
   metadata predicate during the graph walk. Guarantees exact K results in <20ms.
```

---

## 1.6 Vector Database Ecosystem Comparison Matrix

| Database | Architecture & Storage | Index Types | Metadata Filtering | Best Used When |
| :--- | :--- | :--- | :--- | :--- |
| **Qdrant** | Rust native; disk/RAM hybrid | HNSW with payload indexing | Single-stage filtered search | High performance, self-hosted or cloud, complex payload filters. |
| **Pinecone** | Closed-source serverless cloud | Proprietary ANN | Serverless metadata filtering | Zero infrastructure overhead, managed enterprise scale. |
| **Chroma** | Python embedded / client-server | HNSW (hnswlib) | Basic metadata dict filtering | Local development, rapid hackathons, unit tests. |
| **Weaviate** | Go native; modular plugins | HNSW | Inverted index + graph walk | Native multimodal vectorization, GraphQL integrations. |
| **Milvus** | Distributed Go/C++ on Kubernetes | HNSW, IVF, SCaNN, DiskANN | Partition & scalar filtering | Massive enterprise datasets (>100M vectors) on Kubernetes. |
| **pgvector** | PostgreSQL C extension | HNSW, IVFFlat | Integrated with standard SQL WHERE | Datasets < 1M vectors where Postgres is already in production. |

---

## 1.7 Two-Stage Retrieval: Dense Retrieval + Cross-Encoder Reranking

```text
User Query: "What is the return policy for damaged electronics?"
                           |
                           v
           [ Stage 1: Fast Dense Retrieval (Bi-Encoder) ]
           - Embedding lookup via Vector DB
           - High recall, moderate precision
           - Retrieves Top-50 candidate chunks in 15ms
                           |
                           v
           [ Stage 2: Deep Cross-Encoder Reranking (e.g. Cohere / BGE) ]
           - Evaluates (Query, Chunk) pairs jointly through full cross-attention
           - High precision, heavy compute
           - Re-scores Top-50 candidates; selects top-4 highest-scoring chunks
                           |
                           v
              Injected into Final LLM Prompt Context
```

* **Bi-Encoder (Stage 1)**: Encodes Query and Document separately into vectors; similarity is a fast dot product. Can be pre-computed.
* **Cross-Encoder (Stage 2)**: Concatenates `[CLS] Query [SEP] Document` into a single sequence through all Transformer layers, calculating deep cross-token interactions. Too slow to run on millions of docs, but blazing accurate on the top 50 candidates.
