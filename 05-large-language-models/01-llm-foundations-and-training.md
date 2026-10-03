# 1. Large Language Models: Foundations, Training Lifecycle, Scaling Laws, and Architectures

Large Language Models (LLMs) represent a fundamental paradigm shift in artificial intelligence: from task-specific models trained on narrow labeled datasets to massive autoregressive foundation models capable of zero-shot and few-shot reasoning across arbitrary domains.

---

## 1.1 What is an LLM?

### 1.1.1 Core Definition
A Large Language Model is a deep autoregressive neural network (typically billions to hundreds of billions of parameters) trained on massive multimodal or text corpora via **self-supervised next-token prediction**.

Mathematically, given a context sequence of $t$ tokens $(w_1, w_2, \dots, w_t)$, the model estimates a probability distribution over the vocabulary $V$ for the next token $w_{t+1}$:

$$P(w_{t+1} \mid w_1, w_2, \dots, w_t; \mathbf{\theta})$$

```text
Input Context: "The capital of France is"
                          |
                          v
         +---------------------------------+
         |   Transformer Decoder Layers    |
         +---------------------------------+
                          |
                          v
Logits across Vocabulary V (e.g., 128,000 tokens):
[ "Paris": 12.8, "Lyon": 6.2, "apple": -4.1, "London": 5.9, ... ]
                          |
                          v
     Softmax Probability Distribution P(w_{t+1}):
[ "Paris": 92.4%, "Lyon": 3.1%, "London": 2.8%, ... ]
```

### 1.1.2 Traditional ML Models vs Large Language Models

| Dimension | Traditional Machine Learning (e.g., XGBoost, SVM) | Large Language Models (e.g., LLaMA, GPT-4) |
| :--- | :--- | :--- |
| **Data Scope** | Tabular, structured, narrowly labeled datasets. | Hundreds of billions to trillions of raw internet tokens. |
| **Specialization** | Narrow task (e.g., churn prediction, fraud detection). | General-purpose reasoning, code generation, synthesis. |
| **Inference Behavior** | Fixed mathematical mapping ($y = f(\mathbf{x})$); static rules. | Autoregressive generation; in-context instruction following. |
| **Adaptation** | Retrain or fine-tune weights on new labeled data. | In-context learning via prompt engineering, RAG, or LoRA. |
| **Resource Profile** | CPU/single GPU; lightweight serving. | Clustered GPU clusters (H100/A100); memory bandwidth-bound. |

---

## 1.2 The Three-Stage Training Lifecycle

Modern production LLMs undergo three distinct training stages:

```text
+=========================================================================================+
| STAGE 1: Self-Supervised Pre-Training                                                   |
| - Dataset: Trillions of uncurated web tokens (Common Crawl, GitHub, Wikipedia, Books)   |
| - Objective: Causal Language Modeling (Predict the next token)                          |
| - Outcome: "Base Model" (e.g., LLaMA-3-8B-Base). Highly knowledgeable; not an assistant.|
+=========================================================================================+
                                           |
                                           v
+=========================================================================================+
| STAGE 2: Supervised Fine-Tuning (SFT / Instruction Tuning)                               |
| - Dataset: Hundreds of thousands of high-quality (Instruction, Response) dialogue pairs  |
| - Objective: Learn dialogue format, question answering, refusal boundaries              |
| - Outcome: "Instruct Model" (e.g., LLaMA-3-8B-Instruct). Follows commands well.          |
+=========================================================================================+
                                           |
                                           v
+=========================================================================================+
| STAGE 3: Alignment (RLHF / DPO / KTO)                                                    |
| - Dataset: Human preferences (Prompt, Response A vs Response B)                         |
| - Objective: Align with human values (Helpful, Honest, Harmless)                        |
| - Algorithms: PPO (RLHF), Direct Preference Optimization (DPO), KTO                     |
| - Outcome: Production Chat Model (ChatGPT, Claude, Gemini).                             |
+=========================================================================================+
```

### 1.2.1 Stage 1: Pre-training (Foundation)
* **Goal**: Absorb grammatical rules, factual knowledge, syntax, and world relationships.
* **Self-Supervised**: No manual human labeling needed. The training label is simply the very next token in the text.
* **Compute Cost**: Millions of dollars across thousands of specialized GPUs running for months.

### 1.2.2 Stage 2: Supervised Fine-Tuning (SFT / Instruction Tuning)
* A Base Model trained on web text often continues a question with another question (e.g., Input: *"What is the capital of France?"* $\rightarrow$ Base model predicts: *"What is the capital of Germany?"* because it saw an exam paper online).
* SFT trains the model on curated pairs:
  ```json
  {
    "instruction": "Explain photosynthesis in one sentence.",
    "response": "Photosynthesis is the process by which green plants transform sunlight, water, and carbon dioxide into oxygen and glucose."
  }
  ```
* This aligns the model into a responsive assistant persona.

### 1.2.3 Stage 3: Alignment Comparison: RLHF vs DPO vs KTO

```text
1. RLHF (Reinforcement Learning from Human Feedback):
   Preference Data ---> Train Reward Model ---> PPO Reinforcement Loop updates LLM policy
   * Complex, unstable, requires 4 separate models in GPU memory (Policy, Reference, Reward, Critic).

2. DPO (Direct Preference Optimization - Modern Standard):
   Directly optimizes policy on pairs (Prompt, Chosen, Rejected) via implicit reward log-odds:
   L_DPO = -E [ log σ( β * log( π(y_w|x) / π_ref(y_w|x) ) - β * log( π(y_l|x) / π_ref(y_l|x) ) ) ]
   * Eliminates the reward model and PPO loop completely! Highly stable, fast training.

3. KTO (Kahneman-Tversky Optimization):
   Does not require preference pairs (A vs B). Operates on binary signals: (Prompt, Response, "Thumbs Up/Down").
   Models utility based on Prospect Theory (loss aversion).
```

---

## 1.3 Scaling Laws: Chinchilla and Compute-Optimal Training

Understanding scaling laws is critical for sizing models and estimating compute budgets.

```text
Compute (FLOPs) = 6 * N * D
Where:
- N = Number of model parameters
- D = Number of training tokens
- 6 = Approximate FLOPs per parameter per token during forward/backward pass
```

### 1.3.1 The Chinchilla Finding (Hoffmann et al., 2022)
* Before 2022, models were scaled primarily in parameter count $N$ while token count $D$ stayed relatively small (e.g., GPT-3 was 175B parameters trained on only 300B tokens).
* DeepMind's Chinchilla paper proved that for compute-optimal training, **model size and dataset size should be scaled in equal proportions**:

$$D_{\text{optimal}} \approx 20 \times N$$

* An 8B parameter model requires $\approx 160\text{B tokens}$ to be compute-optimal.
* **Inference-Optimal Over-Training**: Modern open-weight models (e.g., LLaMA-3 8B trained on **15 Trillion tokens**) intentionally violate compute-optimal training. They "over-train" small models by $100\times$ because paying more upfront pre-training FLOPs results in a tiny, blazing-fast model that saves millions of dollars during subsequent inference serving.

---

## 1.4 Architectural Breakthrough: Dense Models vs Mixture of Experts (MoE)

```text
DENSE MODEL (e.g., LLaMA-3-70B):
Input Token ---> [ Attention Layer ] ---> [ Single Dense Feed-Forward Network ] ---> Output
Every single token activates ALL 70 Billion parameters.

MIXTURE OF EXPERTS (e.g., Mixtral 8x7B, DeepSeek, GPT-4):
Input Token ---> [ Attention Layer ]
                        |
                        v
              [ Router / Gating Network ] (Softmax gating chooses Top-2 Experts)
                        |
       +----------------+----------------+
       |                                 |
       v                                 v
[ Expert 2 (FFN) ]              [ Expert 5 (FFN) ]      [ Experts 1,3,4,6,7,8 IDLE ]
(Active Compute)                (Active Compute)        (Consume NO compute FLOPs)
       \                                 /
        +----------------+--------------+
                         |
                         v
                   Weighted Output
```

### 1.4.1 Why MoE is Dominating Modern Frontier AI
* **Total Parameters vs Active Parameters**:
  * Mixtral 8x7B has **47 Billion total parameters**, but each token routes to only 2 experts, activating only **13 Billion active parameters per token**.
* **Key Benefit**: Delivers the knowledge capacity and reasoning depth of a 47B model while maintaining the inference compute speed and cost of a 13B model.
* **The Tradeoff**: High VRAM footprint. While compute is 13B, all 47B parameters must still reside in GPU memory.

---

## 1.5 Hallucinations and Model Limitations

### 1.5.1 The Root Cause of Hallucination
An LLM is a probabilistic next-token generator conditioned on minimizing cross-entropy loss over its training text. It maximizes **linguistic plausibility**, not objective veracity. When context is missing or ambiguous, the model interpolates smoothly across its parameter manifolds, generating synthetic claims that sound authoritative.

### 1.5.2 Hallucination Taxonomy
1. **Factual Confabulation**: Citing non-existent papers, legal cases, or imaginary software functions.
2. **Context Contradiction (Ungroundedness)**: Stating conclusions directly refuted by documents in the prompt.
3. **Sycophancy**: Concurring with an incorrect or misleading premise introduced by the user.

### 1.5.3 Production Mitigations
* **Grounding via RAG**: Supply authoritative context and restrict answers to retrieved text.
* **Low Temperature ($T \le 0.2$)**: Decreases sampling entropy.
* **Constrained Decoding (JSON/Pydantic)**: Bounds output structure.
* **Chain-of-Verification (CoVe)**: Prompt the model to verify each factual claim before outputting the final response.
