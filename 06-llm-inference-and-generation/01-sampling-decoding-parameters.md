# 1. LLM Inference & Generation: Decoding Strategies, Sampling Parameters, and Latency Optimization

During inference, a Large Language Model emits a vector of raw unnormalized real-valued scores (**logits**) over its vocabulary $V$. The decoding algorithm governs how these logits are penalized, scaled, filtered, and transformed into final emitted tokens.

Understanding how inference hyperparameters govern this transformation is one of the most frequently tested topics in AI Engineer technical interviews.

---

## 1.1 The Generation Pipeline: From Logits to Output Token

```text
Transformer Final Layer Activation
          |
          v
Raw Logits z in R^V (e.g., V = 128,256 tokens)
[ "the": 14.2, "cat": 12.1, "dog": 11.8, "banana": 4.1, ... ]
          |
          v
[ 1. Frequency / Presence / Repetition Penalties Applied ]
          |
          v
[ 2. Temperature Scaling: z_scaled = z_i / T ]
          |
          v
[ 3. Softmax: Probabilities p_i = exp(z_i / T) / sum(exp(z_j / T)) ]
          |
          v
[ 4. Filtering: Top-K Truncation and / or Top-P (Nucleus) Truncation ]
          |
          v
[ 5. Probability Re-normalization: sum(p_filtered) = 1.0 ]
          |
          v
[ 6. Selection: Greedy Argmax OR Multinomial Stochastic Sampling ]
          |
          v
Emitted Token ---> Appended to Prompt Context ---> Next Forward Pass
```

---

## 1.2 Decoding Strategies: Greedy vs Stochastic Sampling

```text
Logit Distribution: [ "Paris": 12.0, "France": 10.5, "Capital": 8.0, "Apple": 2.0 ]

1. Greedy Decoding (Argmax):
   Always selects "Paris" (100% of the time).
   * Pros: Deterministic, optimal for code syntax, math, SQL.
   * Cons: Gets stuck in repetitive n-gram loops; lacks natural conversational variety.

2. Stochastic Sampling (Multinomial):
   Samples from probability distribution: "Paris" (75%), "France" (18%), "Capital" (6%), "Apple" (1%).
   * Pros: Human-like, natural diversity.
   * Cons: Non-zero risk of sampling incoherent tail tokens without Top-P/Top-K.
```

### 1.2.1 Greedy Decoding Mechanics
Greedy decoding chooses the single token with the absolute maximum probability at every forward step:

$$w_{t+1} = \arg\max_{w \in V} P(w \mid w_{1:t})$$

* **Why it exists**: Simple, computationally fast, and deterministic.
* **When to use**: Factual question answering, SQL query generation, mathematical reasoning, strict JSON/Pydantic schema extraction.
* **Failure Mode**: Tends to fall into repetitive cyclic loops (e.g., *"and the system and the system and the system"*) because it never explores slightly lower-probability paths that lead to globally higher-probability sentences.

### 1.2.2 Beam Search
Instead of greedily picking one token, Beam Search tracks the top $B$ (beam width) most probable candidate token trajectories simultaneously across multiple sequential steps. At each step, it expands all $B$ paths, computes their cumulative log-probabilities, and retains only the top $B$ paths.

* **When to use**: Neural Machine Translation (NMT), speech-to-text, and short abstractive summarization.
* **Why rarely used in modern LLM chat**: Computationally expensive for long sequences ($B \times$ compute per token), and it produces unnaturally repetitive, generic conversational dialogue because human speech favors occasional surprises rather than globally maximized token probabilities.

---

## 1.3 Core Sampling Hyperparameters

### 1.3.1 Temperature ($T$)

The temperature parameter controls the **randomness and entropy** of the probability distribution by scaling the logits before the Softmax function is applied:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^V \exp(z_j / T)}$$

Where:
* $z_i$ is the raw logit of token $i$.
* $T > 0$ is the temperature scalar.
* $V$ is the total vocabulary size.

```text
Assume 3 Candidate Tokens with Raw Logits: z = [ 10.0, 8.0, 5.0 ]

T = 0.2 (Low Temperature - Peaked Distribution):
Token A: |====================================================| 99.3%
Token B: |=| 0.7%
Token C: | 0.0%
* Effect: Extremely focused, repeatable, highly conservative.

T = 1.0 (Standard Temperature - Balanced):
Token A: |============================================| 84.4%
Token B: |=======| 14.1%
Token C: |=| 1.5%
* Effect: Creative while retaining factual grounding.

T = 2.0 (High Temperature - Flattened Distribution):
Token A: |=======================| 57.6%
Token B: |================| 31.6%
Token C: |======| 10.8%
* Effect: High randomness; creative exploration; risks incoherence if T > 1.5.
```

* **As $T \to 0$**: The scaled logits $\frac{z_i}{T}$ diverge. The largest logit dominates completely, making the Softmax equivalent to a hard **Argmax (Greedy Decoding)**.
* **As $T \to \infty$**: The scaled logits approach zero ($\frac{z_i}{T} \to 0$), so $\exp(0) = 1$ for all tokens. The distribution flattens into a **Uniform Distribution**, meaning every token in the vocabulary has an equal chance of being picked (producing pure gibberish).

---

### 1.3.2 Top-K Sampling

Top-K truncates the vocabulary candidates to strictly the **$K$ most probable tokens**, completely zeroing out the remaining tail of the vocabulary:

```text
Candidate Tokens sorted by probability:
1. "cat"     (45%)  ---\
2. "dog"     (30%)   ---+---> Top-K = 3 Threshold!
3. "bird"    (15%)  ---/
-------------------------------------------------------
4. "car"     (7%)   (Discarded!)
5. "table"   (3%)   (Discarded!)

Remaining tokens ["cat", "dog", "bird"] are re-normalized:
"cat" = 45/90 = 50.0%, "dog" = 30/90 = 33.3%, "bird" = 15/90 = 16.7%
```

* **Why it exists**: Prevents the model from ever sampling bizarre, low-probability tail tokens (e.g., token with $0.0001\%$ probability).
* **Limitations**: A fixed $K$ does not adapt to the model's confidence:
  * When the model is confident (one token has $95\%$ probability), Top-K ($K=50$) forces 49 low-quality tokens into consideration.
  * When the distribution is flat (50 plausible options), Top-K ($K=10$) prematurely cuts off 40 valid alternatives.

---

### 1.3.3 Top-P (Nucleus) Sampling

Proposed by Holtzman et al. (2019) to fix Top-K's rigidity. Top-P dynamically selects the smallest set of top tokens $V^{(p)} \subset V$ whose **cumulative probability reaches threshold $P$**:

$$\sum_{w \in V^{(p)}} P(w \mid w_{1:t}) \ge P$$

```text
Candidate Tokens sorted by probability:
1. "doctor"     (65%)  ---> Cumulative: 0.65
2. "physician"  (20%)  ---> Cumulative: 0.85
3. "surgeon"    (08%)  ---> Cumulative: 0.93  ===> Threshold P = 0.90 Exceeded! (Cutoff)
-----------------------------------------------------------------------------------
4. "nurse"      (05%)  ---> Discarded!
5. "hospital"   (02%)  ---> Discarded!

The Nucleus contains only ["doctor", "physician", "surgeon"].
Pool expands automatically during uncertainty and contracts during certainty!
```

> **Interview Golden Rule**: The standard recommendation from OpenAI and Anthropic is to alter **either** Temperature **or** Top-P, but not both simultaneously, to avoid unpredictable compounding interaction effects.

---

## 1.4 Repetition, Frequency, and Presence Penalties

LLMs can get stuck in loops repeating identical words or phrases. Modern APIs provide penalty parameters that adjust logits prior to Softmax:

$$z_i' = z_i - \left( \alpha_{\text{freq}} \cdot C(w_i) + \alpha_{\text{pres}} \cdot \mathbb{I}(C(w_i) > 0) \right)$$

Where:
* $z_i$ is the raw logit of token $i$.
* $C(w_i)$ is the count of how many times token $w_i$ has already appeared in the generated text.
* $\mathbb{I}(\cdot)$ is the indicator function (equals $1$ if token has appeared at least once, $0$ otherwise).
* $\alpha_{\text{freq}}$ is the **Frequency Penalty** coefficient.
* $\alpha_{\text{pres}}$ is the **Presence Penalty** coefficient.

```text
Token "data": Raw Logit = 12.0. Has appeared 4 times already in completion.

1. With Frequency Penalty (alpha_freq = 0.5):
   Penalty = 0.5 * 4 = 2.0
   New Logit = 12.0 - 2.0 = 10.0  (Penalized proportionally to repetition count)

2. With Presence Penalty (alpha_pres = 0.5):
   Penalty = 0.5 * 1 = 0.5
   New Logit = 12.0 - 0.5 = 11.5  (One-time flat penalty for having appeared)
```

* **Frequency Penalty**: Penalizes tokens proportionally to their repetition frequency. Reduces verbatim repeating of frequent words.
* **Presence Penalty**: Applies a flat one-time penalty if a token has appeared at least once, regardless of count. Encourages the model to introduce novel concepts and switch topics.
* **Repetition Penalty (HuggingFace style)**: Multiplicative scaling ($z_i / \theta$ if $z_i > 0$, else $z_i \cdot \theta$) where $\theta > 1.0$ suppresses repeated tokens.

---

## 1.5 Sequence Length and Termination Controls

### 1.5.1 Context Length vs Maximum Output Tokens
* **Context Length (Context Window)**: The maximum sequence length (in tokens) the model can hold in memory at once (Prompt Tokens + Generated Tokens $\le$ Context Window).
* **Max Output Tokens (`max_tokens`)**: An explicit cap placed on the completion length.
  * *Trap*: If `max_tokens` is hit before the model finishes its thought or closes its JSON brackets, the output is abruptly truncated, resulting in JSON parse errors (`finish_reason = "length"` instead of `"stop"`).

### 1.5.2 Stop Sequences (`stop`)
One or more string sequences that instruct the inference engine to immediately halt generation when emitted:
* Common Examples:
  * `"\nUser:"` in conversational systems to prevent the model from impersonating the user.
  * `"```"` in markdown code-block extractors.
  * `"</tool_call>"` in agentic tool-use loops.

---

## 1.6 Determinism vs Non-Determinism in Production

* **Is an LLM deterministic at $T=0$?**
  * *Theoretically*: Yes. If $T=0$, the argmax token is always selected.
  * *In Practice*: GPU floating-point operations across parallel CUDA thread blocks are non-associative:
    $$(a + b) + c \neq a + (b + c) \text{ at finite 16-bit precision}$$
    Slight hardware concurrency variations or batch size changes can cause tiny numerical logit drifts that occasionally flip a borderline argmax token.
* **The `seed` Parameter**:
  * Passing a fixed integer `seed` (e.g., `seed=42`) prompts the serving engine (OpenAI, vLLM) to enforce deterministic execution pathways for reproducible testing and regression evaluation.

---

## 1.7 KV Cache Memory Sizing: The Production Formula

In production inference serving, GPU VRAM is dominated not by model weights, but by the **KV Cache**.

$$\text{KV Cache Memory (Bytes)} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch\_size}$$

* First $2$: Stores both **Key** and **Value** tensors.
* Second $2$: 16-bit precision (`FP16` or `BF16`) = 2 bytes per float.
* $n_{\text{layers}}$: Number of Transformer layers.
* $n_{\text{kv\_heads}}$: Number of KV heads (in GQA, this is much smaller than query heads).
* $d_{\text{head}}$: Dimension per head.

### 1.7.1 Worked Production Example: LLaMA-3-8B at 8k Context
* $n_{\text{layers}} = 32$
* $n_{\text{kv\_heads}} = 8$ (Grouped-Query Attention)
* $d_{\text{head}} = 128$
* $\text{seq\_len} = 8,192$ tokens
* $\text{batch\_size} = 1$

$$\text{Memory} = 4 \times 32 \times 8 \times 128 \times 8,192 \times 1 = 1,073,741,824 \text{ Bytes} = \mathbf{1.0 \text{ GB per concurrent user!}}$$

With a batch size of 16 concurrent users, KV cache alone demands **16 GB of GPU VRAM** on top of the 16 GB needed for model weights!

---

## 1.8 Speculative Decoding: 2x–3x Inference Acceleration

Autoregressive generation is fundamentally memory bandwidth-bound: reading weights from GPU memory to generate a single token underutilizes Tensor Core compute.

```text
                          SPECULATIVE DECODING WORKFLOW

1. Draft Phase:
   Fast, cheap Small Model (e.g., LLaMA-3-1B) generates K=4 candidate tokens sequentially:
   Draft: ["Artificial", "Intelligence", "is", "transforming"]  (Takes ~15ms)
                                     |
                                     v
2. Verification Phase (Single Forward Pass):
   Target Large Model (e.g., LLaMA-3-70B) runs a SINGLE parallel forward pass evaluating all 4 tokens.
   Scores all 4 tokens simultaneously in ~25ms.
                                     |
                                     v
3. Accept / Reject:
   Target model accepts tokens 1, 2, and 3; corrects token 4 to "revolutionizing".
   * Outcome: 4 high-quality 70B tokens produced in 40ms instead of 100ms! (2.5x Speedup)
   * Guaranteed identical statistical distribution to running the 70B model alone!
```

---

## 1.9 Parameter Presets for Common Production Scenarios

| Scenario | Temperature ($T$) | Top-P ($P$) | Frequency Penalty | Finish Reason to Monitor |
| :--- | :--- | :--- | :--- | :--- |
| **Strict JSON Extraction** | `0.0` | `1.0` | `0.0` | Expect `"stop"`; alert on `"length"` |
| **SQL Query Generation** | `0.0` | `1.0` | `0.0` | Must terminate at `";"` or `"\n"` |
| **Enterprise RAG Synthesis** | `0.2` | `0.9` | `0.1` | Grounded, factual, minimal hallucination |
| **Conversational Chatbot** | `0.7` | `0.9` | `0.3` | Fluent, engaging, non-robotic |
| **Creative Brainstorming** | `0.95` | `0.95` | `0.5` | Wide exploration of semantic paths |
