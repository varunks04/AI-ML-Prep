# 2. NLP Fundamentals: Tasks, Sequences, and the Bridge to LLMs

This guide covers traditional core NLP tasks, the evolution of sequence models, and how modern Large Language Models transformed natural language processing.

---

## 2.1 Core NLP Tasks: Definitions and Evolution

```text
+-----------------------+--------------------------------------+---------------------------------------+
| NLP Task              | Traditional ML / DL Approach         | Modern GenAI / LLM Approach           |
+-----------------------+--------------------------------------+---------------------------------------+
| Text Classification   | TF-IDF + Logistic Reg / SVM          | Few-shot prompt with JSON output      |
| Sentiment Analysis    | Bi-LSTM / BERT classification head   | Zero-shot prompt with chain-of-thought|
| Named Entity Rec (NER)| spaCy / BiLSTM-CRF                   | Structured extraction (Pydantic/Tools)|
| Text Summarization    | Extractive (TextRank / Graph)        | Abstractive LLM instruction prompt    |
| Machine Translation   | Seq2Seq with Attention (LSTM)        | Autoregressive multilingual LLM       |
+-----------------------+--------------------------------------+---------------------------------------+
```

### 2.1.1 Text Classification & Sentiment Analysis
* **Goal**: Assign one or more categorical labels to a passage of text.
* **Sentiment Analysis**: Special case predicting valence (Positive, Neutral, Negative) or granular emotion.

### 2.1.2 Named Entity Recognition (NER)
* **Goal**: Locate and classify named entities in unstructured text into predefined categories (Person, Organization, Location, Date, Currency).
* **BIO Tagging Scheme**:
  * `B-PER`: Beginning of a Person entity.
  * `I-PER`: Inside of a Person entity.
  * `O`: Outside any entity.
  * *Example*: `[Sam (B-PER), Altman (I-PER), visited (O), Tokyo (B-LOC)]`.

---

## 2.2 Sequence Understanding: The Path from RNNs to Transformers

Human language is inherently sequential: meaning unfolds over time, and words depend on preceding and subsequent words.

### 2.2.1 Recurrent Neural Networks (RNNs)
Processes tokens step-by-step, maintaining an internal hidden state vector $\mathbf{h}_t$:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})$$

```text
x_1 ---------> [ RNN Cell ] ---------> h_1
                     |
                     v (hidden state passed sequentially)
x_2 ---------> [ RNN Cell ] ---------> h_2
                     |
                     v
x_3 ---------> [ RNN Cell ] ---------> h_3
```

### 2.2.2 The Vanishing Gradient Mathematical Bottleneck
When backpropagating loss $\mathcal{L}_T$ at time step $T$ back to hidden state $\mathbf{h}_1$, the chain rule produces:

$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \prod_{j=2}^T \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$$

Where each Jacobian $\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \text{diag}(1 - \tanh^2(\cdot)) \mathbf{W}_{hh}^T$.
* If the largest eigenvalue of $\mathbf{W}_{hh} < 1$, the product decays exponentially to zero as $T$ increases.
* The model becomes blind to long-range dependencies ($> 20\text{ tokens}$).
* If eigenvalue $> 1$, gradients explode to $\infty$ (NaN weights).

### 2.2.3 LSTMs: Gating Mechanisms to Combat Vanishing Gradients

```text
                                LSTM CELL ARCHITECTURE
                                
Cell State c_{t-1} -----------------[ * ]-----------------------> [ + ] -------------> Cell State c_t
                                     ^                             ^
                                     |                             |
                               (Forget Gate)                  (Input Gate)
                                     |                             |
                                   f_t                           i_t * c_tilde
                                     |                             |
x_t ----+----------------------------+-----------------------------+-----------------+
        |                                                                            |
        v                                                                            v
[ sigma / tanh ] -------------------------------------------------------------> [ Output Gate o_t ]
        ^                                                                            |
        |                                                                            v
h_{t-1} +------------------------------------------------------------------------> Hidden State h_t
```

$$\text{Forget Gate}: \quad f_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

$$\text{Input Gate}: \quad i_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\text{Candidate Cell State}: \quad \tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

$$\text{Updated Cell State}: \quad \mathbf{c}_t = f_t \odot \mathbf{c}_{t-1} + i_t \odot \tilde{\mathbf{c}}_t$$

$$\text{Output Gate}: \quad o_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

$$\text{Emitted Hidden State}: \quad \mathbf{h}_t = o_t \odot \tanh(\mathbf{c}_t)$$

* **Why LSTMs Protected Gradients**: The additive cell state update ($\mathbf{c}_t = f_t \odot \mathbf{c}_{t-1} + \dots$) functions like an uninterrupted conveyor belt, allowing gradient signals to flow back across hundreds of steps without decaying to zero.
* **The Sequential Bottleneck**: However, LSTMs **still cannot parallelize across GPUs** because token $t$ strictly requires hidden state $t-1$. This sequential dependency made web-scale pre-training impossible, paving the way for the Transformer.

---

## 2.3 The Architectural Paradigm Shift: NLP $\rightarrow$ Transformers $\rightarrow$ LLMs

In 2017, Vaswani et al. published *"Attention Is All You Need"*, discarding recurrence entirely.

```text
THE HISTORICAL EVOLUTION:

[ 2013: Static Vectors ]       Word2Vec / GloVe
                               - Words mapped to single vectors. No context awareness.
                                      |
                                      v
[ 2015: Recurrent Seq2Seq ]    LSTMs / GRUs with Attention
                               - Sequential recurrence; slow; gradient issues over >100 tokens.
                                      |
                                      v
[ 2017: The Transformer ]      Self-Attention Networks
                               - 100% parallelizable across GPUs.
                               - Direct O(1) attention distance between ANY two tokens.
                                      |
                                      v
[ 2018-2020: Foundation LMs ]  BERT (Encoder) & GPT-2/3 (Decoder)
                               - Pre-training on massive uncurated internet text.
                               - Fine-tuning / Few-shot in-context learning.
                                      |
                                      v
[ 2022-Present: Modern GenAI]  ChatGPT / LLaMA / Claude / Gemini
                               - Instruction-tuned, RLHF-aligned, reasoning, tool-calling agents.
```
