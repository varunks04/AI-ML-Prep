# 2. Large Language Models: Transformer Architecture, Attention, and FlashAttention

The Transformer architecture (Vaswani et al., 2017) discarded recurrent architectures in favor of parallelizable multi-head self-attention. This guide details the causal decoder architecture, mathematical attention mechanics, positional encodings, and GPU-level optimizations.

---

## 2.1 The Causal Decoder Layer: Step-by-Step Tensor Flow

Below is the exact layer-by-layer dataflow through a single modern Transformer Decoder block (e.g., LLaMA-3), tracking tensor dimensions at every step:

```text
Input Token IDs: [ 104, 8920, 230 ]  ---> Shape: (Batch_Size=1, Seq_Len=3)
       |
       v
[ Embedding Matrix Lookup: Vocab_Size x d_model ] ---> Shape: (B, S, d_model=4096)
       |
       +=================================================================================+
       | TRANSFORMER DECODER LAYER (Repeated N times, e.g., 32 layers)                   |
       |                                                                                 |
       |  Input Tensor x: (B, S, d_model)                                                |
       |         |                                                                       |
       |         +-------------------------------------------------------+               |
       |         |                                                       | (Residual)    |
       |         v                                                       |               |
       |  [ RMSNorm ] ---> (B, S, d_model)                               |               |
       |         |                                                       |               |
       |         v                                                       |               |
       |  [ Linear Projections: W_q, W_k, W_v ]                          |               |
       |    Q: (B, Num_Q_Heads=32, S, d_k=128)                           |               |
       |    K: (B, Num_KV_Heads=8,  S, d_k=128)  [Grouped-Query GQA]     |               |
       |    V: (B, Num_KV_Heads=8,  S, d_v=128)                          |               |
       |         |                                                       |               |
       |         v                                                       |               |
       |  [ Apply RoPE (Rotary Position Embeddings) to Q and K ]         |               |
       |         |                                                       |               |
       |         v                                                       |               |
       |  [ Scaled Dot-Product Attention + Causal Mask ]                 |               |
       |    Scores = Softmax( (Q • K^T) / sqrt(d_k) + Mask ) • V         |               |
       |    Output: (B, Num_Q_Heads=32, S, d_v=128)                      |               |
       |         |                                                       |               |
       |         v                                                       |               |
       |  [ Output Linear Projection: W_o ] ---> (B, S, d_model)         |               |
       |         |                                                       |               |
       |         v                                                       |               |
       |  [ Residual Addition (+) ] <------------------------------------+               |
       |    x_mid = x + Attention_Out ---> (B, S, d_model)                               |
       |         |                                                                       |
       |         +-------------------------------------------------------+               |
       |         |                                                       | (Residual)    |
       |         v                                                       |               |
       |  [ RMSNorm ] ---> (B, S, d_model)                               |               |
       |         |                                                       |               |
       |         v                                                       |               |
       |  [ SwiGLU Feed-Forward Network (MLP) ]                          |               |
       |    FFN(x) = (SiLU(x • W_gate) * (x • W_up)) • W_down            |               |
       |    Internal expansion: d_model -> 14336 -> d_model              |               |
       |         |                                                       |               |
       |         v                                                       |               |
       |  [ Residual Addition (+) ] <------------------------------------+               |
       |    x_out = x_mid + FFN_Out ---> Shape: (B, S, d_model=4096)                     |
       +=================================================================================+
                                  |
                                  v
                       [ Final RMSNorm Layer ]
                                  |
               [ LM Head: Unembedding Linear Projection ]
                Shape: (B, S, d_model) • (d_model, Vocab_Size=128256)
                                  |
                                  v
                    Raw Logits: (B, S, Vocab_Size)
                                  |
                [ Softmax (Logits -> Next Token Probabilities) ]
```

---

## 2.2 Scaled Dot-Product Attention: The Mathematical Engine

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}$$

```text
Query Matrix (Q)                Key Matrix Transposed (K^T)         Attention Score Matrix
[ Token 1 ("The")    ]          [ "The", "cat", "sat" ]            [ [ 0.92, 0.05, 0.03 ],
[ Token 2 ("cat")    ]    •     [   |      |      |   ]      =     [ 0.12, 0.81, 0.07 ],
[ Token 3 ("sat")    ]          [   |      |      |   ]            [ 0.08, 0.65, 0.27 ] ]
(3 x d_k)                       (d_k x 3)                          (3 x 3)
```

### 2.2.1 Why Scale by $\sqrt{d_k}$?
* Assume components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and variance 1.
* Their dot product $\sum_{i=1}^{d_k} q_i k_i$ has mean 0 and **variance equal to $d_k$**.
* When $d_k = 128$, standard deviation is $\sqrt{128} \approx 11.3$.
* Dot products with values like $+35$ and $-30$ pushed into Softmax cause the output to collapse into a one-hot distribution, where the gradient derivative is practically $0$.
* Dividing by $\sqrt{d_k}$ normalizes the variance back to $1.0$, preserving healthy gradient propagation during backpropagation.

---

## 2.3 Rotary Position Embedding (RoPE)

Transformers have no recurrence; without position information, the sequence is treated as an unordered bag of tokens.

```text
                             ROTARY POSITION EMBEDDING (RoPE)
                                            
                 Vector q at position m                  Vector k at position n
                 Rotated by angle m*θ                    Rotated by angle n*θ
                           ^                                       ^
                          /                                         \
                         /                                           \
                        /                                             \
                       +------->                                       +------->
                       
    The Inner Product (q_m)^T • k_n depends strictly on the RELATIVE distance (m - n)!
```

### 2.3.1 Mathematical Formulation
RoPE splits the $d$-dimensional vector into $\frac{d}{2}$ orthogonal 2D pairs. For each pair $(x_1, x_2)$ at sequence position $m$:

$$\mathbf{R}_{\Theta, m}^{2D} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

* **Advantage over Absolute Position**:
  $$\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$$
  The attention score depends purely on relative distance $(m - n)$, allowing models to extrapolate to context lengths longer than seen in training via RoPE frequency scaling (YaRN).

---

## 2.4 Multi-Head Attention (MHA) vs GQA vs MQA

```text
1. Multi-Head Attention (MHA - e.g., GPT-3):
   Q Heads:  [ Q1 ] [ Q2 ] [ Q3 ] [ Q4 ] [ Q5 ] [ Q6 ] [ Q7 ] [ Q8 ]
   K Heads:  [ K1 ] [ K2 ] [ K3 ] [ K4 ] [ K5 ] [ K6 ] [ K7 ] [ K8 ]  ---> Huge KV Cache!
   V Heads:  [ V1 ] [ V2 ] [ V3 ] [ V4 ] [ V5 ] [ V6 ] [ V7 ] [ V8 ]

2. Multi-Query Attention (MQA):
   Q Heads:  [ Q1 ] [ Q2 ] [ Q3 ] [ Q4 ] [ Q5 ] [ Q6 ] [ Q7 ] [ Q8 ]
   K Head:   [                      K1                       ]  ---> Smallest KV Cache,
   V Head:   [                      V1                       ]       minor quality drop.

3. Grouped-Query Attention (GQA - e.g., LLaMA-3, Mistral):
   Q Heads:  [ Q1 ] [ Q2 ]   [ Q3 ] [ Q4 ]   [ Q5 ] [ Q6 ]   [ Q7 ] [ Q8 ]
   K Heads:  [    K1     ]   [    K2     ]   [    K3     ]   [    K4     ]  ---> 4x - 8x VRAM savings,
   V Heads:  [    V1     ]   [    V2     ]   [    V3     ]   [    V4     ]       zero quality drop!
```

---

## 2.5 Hardware Optimization: FlashAttention

Standard attention computes and writes the massive $S \times S$ attention matrix to High-Bandwidth Memory (HBM), creating an $O(S^2)$ memory and memory-bandwidth bottleneck.

```text
STANDARD ATTENTION (Memory-Bound):
GPU SRAM (Fast, 20 MB) <==== Repeated Read/Write $O(S^2)$ Matrix ====> GPU HBM (Slow, 80 GB)
* Result: GPU compute cores (Tensor Cores) sit idle waiting for memory transfers.

FLASHATTENTION (Dao et al. - IO-Aware Tiled Computing):
1. Tiling: Divides Q, K, V into blocks that fit entirely inside high-speed on-chip SRAM.
2. Online Softmax: Updates Softmax scaling factors incrementally without computing the full matrix.
3. Kernel Fusion: Computes attention in a single fused GPU kernel. Never writes $S x S$ to HBM!
* Result: 2x - 4x speedup, linear memory footprint O(S).
```
