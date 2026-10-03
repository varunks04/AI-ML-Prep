# 1. Essential Mathematics: Linear Algebra, Vectors, and Tensors

Linear algebra is the foundational mathematical language of modern AI. Every token, prompt, image, and model weight in an AI system is represented as a scalar, vector, matrix, or tensor.

---

## 1.1 Mathematical Primitives: Scalars, Vectors, Matrices, and Tensors

```text
Scalar (0D)           Vector (1D)                Matrix (2D)                   Tensor (3D+)
   [ 42 ]           [ 0.12, -0.45, 0.88 ]       [ [ 1, 2, 3 ],         [ [ [ 1, 2 ], [ 3, 4 ] ],
                                                  [ 4, 5, 6 ] ]           [ [ 5, 6 ], [ 7, 8 ] ] ]
Rank: 0             Rank: 1                     Rank: 2                Rank: 3+
Shape: ()           Shape: (3,)                 Shape: (2, 3)          Shape: (2, 2, 2)
Example: Loss       Example: Word Embedding     Example: Batch of      Example: Batch of Image
value               (e.g., text-embedding-3)    token embeddings       tokens (Batch, Seq_Len, Dim)
```

### 1.1.1 What they are
* **Scalar ($s \in \mathbb{R}$)**: A single real number with magnitude but no directional dimension (e.g., temperature $= 0.7$, loss $= 0.241$).
* **Vector ($\mathbf{v} \in \mathbb{R}^d$)**: An ordered 1D array of scalars. Geometrically, it represents a point or a directed arrow in a $d$-dimensional continuous space.
* **Matrix ($\mathbf{A} \in \mathbb{R}^{m \times n}$)**: A 2D grid of numbers with $m$ rows and $n$ columns. Algebraically, a matrix represents a linear transformation between vector spaces.
* **Tensor ($\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_k}$)**: A generalized multidimensional array of rank $k$.

### 1.1.2 Why it matters in Modern AI
In an LLM pipeline:
1. A prompt with $T$ tokens is mapped to an input embedding tensor of shape `(Batch_Size, Sequence_Length, Hidden_Dimension)`.
2. Model parameters (weights) are stored as 2D matrices of shape `(In_Features, Out_Features)`.
3. In multi-head attention, activations expand to a 4D tensor: `(Batch_Size, Num_Heads, Sequence_Length, Head_Dimension)`.

---

## 1.2 Vector Operations and Geometric Intuition

### 1.2.1 Vector Addition and Scaling
* **Vector Addition**: $\mathbf{u} + \mathbf{v} = [u_1 + v_1, u_2 + v_2, \dots, u_d + v_d]^T$. Translates points in space.
* **Scalar Multiplication**: $\alpha \mathbf{u} = [\alpha u_1, \alpha u_2, \dots, \alpha u_d]^T$. Stretches or compresses magnitude without altering direction.

### 1.2.2 Vector Norms (Magnitude)
A norm measures the "length" or magnitude of a vector:
* **$L_1$ Norm (Manhattan Distance)**: Sum of absolute values:

  $$\|\mathbf{v}\|_1 = \sum_{i=1}^{d} |v_i|$$

  *Used in Lasso regularization to produce exact sparsity.*
* **$L_2$ Norm (Euclidean Norm)**: Straight-line distance from origin:

  $$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{d} v_i^2}$$

  *Used in Ridge regularization, vector normalization, and Euclidean distance.*
* **Unit Vector Normalization**: Dividing a vector by its $L_2$ norm creates a unit vector with magnitude 1:

  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

---

## 1.3 The Dot Product and Semantic Similarity

The dot product is arguably the single most important mathematical operation in AI engineering.

### 1.3.1 Definition and Geometry
Given two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{d} u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$

Where $\theta$ is the angle between the two vectors.

```text
    v                      v
    ^                      ^
    |                      |
    |  θ = 0° (Aligned)    |  θ = 90° (Orthogonal)        v <---------> u
    +--------> u           +--------> u                     θ = 180° (Opposite)
  cos(0°) = 1            cos(90°) = 0                     cos(180°) = -1
  Maximum Similarity     Independent / Unrelated          Opposite Meaning
```

### 1.3.2 Cosine Similarity
By isolating $\cos(\theta)$, we obtain a metric that measures directional alignment independent of vector magnitude:

$$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$$

> **AI Engineering Rule**: If all vectors stored in a vector database are pre-normalized to unit length ($\|\mathbf{v}\|_2 = 1$), **Cosine Similarity is mathematically identical to the Dot Product**. This allows vector databases (e.g., Pinecone, Qdrant, Chroma) to compute similarity using blazing-fast SIMD dot product instructions without expensive square root computations.

---

## 1.4 Matrix Multiplication and Linear Transformations

### 1.4.1 Mechanics
Matrix multiplication $\mathbf{C} = \mathbf{A} \mathbf{B}$ is defined only when the inner dimensions match:

$$(m \times k) \times (k \times n) \rightarrow (m \times n)$$

Each entry $C_{ij}$ is the dot product of row $i$ of $\mathbf{A}$ and column $j$ of $\mathbf{B}$:

$$C_{ij} = \sum_{r=1}^{k} A_{ir} B_{rj}$$

```text
[ Row i of A ] (1 x k)  •  [ Col j of B ] (k x 1)  =  [ C_ij ] (Scalar)
```

### 1.4.2 Intuition: A Matrix is a Space Transformer
Multiplying a vector by a matrix ($\mathbf{y} = \mathbf{W}\mathbf{x}$) transforms the vector: it can rotate, scale, skew, or project it into a new subspace.
* In neural networks, the weight matrix $\mathbf{W}$ transforms an input feature representation into a new latent semantic representation.

---

## 1.5 How Linear Algebra Powers Modern LLMs and Transformers

### 1.5.1 Query, Key, and Value Projections
In the Self-Attention mechanism of a Transformer, an input token embedding vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ is linearly projected using three learned weight matrices:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

### 1.5.2 Attention Matrix Calculation
The interaction between all tokens in a prompt is computed via a single batched matrix multiplication:

$$\text{AttentionScores} = \mathbf{Q} \mathbf{K}^T$$

If sequence length is $L$:

$$\mathbf{Q} \in \mathbb{R}^{L \times d_k}, \quad \mathbf{K}^T \in \mathbb{R}^{d_k \times L} \implies \mathbf{Q} \mathbf{K}^T \in \mathbb{R}^{L \times L}$$

Every cell $(i, j)$ in the resulting $L \times L$ matrix is the **dot product** between token $i$ and token $j$, quantifying how much attention token $i$ must pay to token $j$.

```text
Tokens (Q)                 Tokens (K^T)                   Attention Matrix (L x L)
[ "The"     ]              [ "The", "cat", "sat" ]       [ [ 0.9, 0.1, 0.0 ],
[ "cat"     ]      x                                 =     [ 0.2, 0.8, 0.1 ],
[ "sat"     ]                                              [ 0.1, 0.7, 0.6 ] ]
(3 x d_k)                  (d_k x 3)                     Every entry is a dot product!
```

This explains why context length scaling ($L$) historically had an $O(L^2)$ memory and compute bottleneck: doubling the context length quadruples the size of the attention matrix $\mathbf{Q} \mathbf{K}^T$.

---

## 1.6 Matrix Rank, SVD, and Low-Rank Adaptation (LoRA)

A high-frequency AI Engineer interview topic is explaining the mathematical intuition behind parameter-efficient fine-tuning (PEFT).

### 1.6.1 Matrix Rank and Singular Value Decomposition (SVD)
* **Matrix Rank**: The maximum number of linearly independent column or row vectors in a matrix. Represents the true underlying dimensionality or degrees of freedom.
* **SVD**: Any matrix $\mathbf{W} \in \mathbb{R}^{d \times k}$ can be factorized into:

  $$\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

  Where $\mathbf{\Sigma}$ contains singular values ordered by magnitude. In deep neural networks, weight matrices are heavily over-parameterized: most singular values are near zero, meaning the weights possess an **intrinsic low rank**.

### 1.6.2 Low-Rank Adaptation (LoRA) (Hu et al., 2021)
During fine-tuning, instead of updating all $d \times k$ weights ($\Delta \mathbf{W}$), LoRA decomposes the weight update into two low-rank matrices:

$$\mathbf{W}_{\text{adapted}} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{B} \cdot \mathbf{A})$$

```text
                    LoRA LOW-RANK DECOMPOSITION
                    
Input x (d) ----+----------------------------------------+
                |                                        |
                v (Frozen Base Model)                    v (Trainable LoRA Adapters)
         [ W_0: (d x k) ]                        [ A: (d x r) ]  (Initialized Gaussian)
                |                                        |
                |                                        v (rank r << min(d, k), e.g. r=8)
                |                                [ B: (r x k) ]  (Initialized to 0)
                |                                        |
                v                                        v
             h_frozen                                 h_adapter
                \                                        /
                 +-------------------+------------------+
                                     |
                                     v
                        h_final = h_frozen + h_adapter
```

* **Parameters Comparison**:
  * For $d = 4096, k = 4096$: Full matrix has $4096 \times 4096 = 16,777,216$ weights.
  * With LoRA rank $r = 8$: Matrix $\mathbf{A}$ has $4096 \times 8 = 32,768$ and Matrix $\mathbf{B}$ has $8 \times 4096 = 32,768$.
  * Total trainable parameters: $65,536$ (**$99.6\%$ parameter reduction!**).
* **Zero Inference Latency**: At deployment, the adapter can be folded directly into the base weights: $\mathbf{W}_{\text{deployed}} = \mathbf{W}_0 + \frac{\alpha}{r}(\mathbf{B}\mathbf{A})$.
