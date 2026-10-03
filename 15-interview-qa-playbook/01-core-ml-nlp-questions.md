# 1. Interview Q&A Playbook: Machine Learning, Mathematics, Data Processing, and NLP

This question bank contains frequently asked interview questions for entry-level / fresher AI Engineer roles, categorized by technical domain and difficulty tier. Every question provides model answers focusing on **why** concepts work and practical tradeoffs.

---

## 1.1 Machine Learning Fundamentals

### Q1 [Basic]: How do you decide whether a problem requires Supervised, Unsupervised, or Reinforcement Learning?
* **Model Answer**:
  * If the dataset contains historical ground-truth target labels for the outcome we want to predict (e.g., customer churned $= 1/0$, house sale price), it is **Supervised Learning**.
  * If the dataset has no labels and the goal is discovering latent groupings, segments, or compressed representations (e.g., customer segmentation, anomaly detection, embedding visualization), it is **Unsupervised Learning**.
  * If there is no static historical dataset, but rather an active agent interacting with an environment through sequential trial-and-error actions receiving delayed rewards or penalties (e.g., game playing, robotics, RLHF alignment for LLMs), it is **Reinforcement Learning**.

---

### Q2 [Intermediate]: Your model achieves 99% accuracy on the training set and 98% on the test set, but in production, users complain it fails constantly. What could have happened?
* **Model Answer**:
  1. **Class Imbalance**: In a dataset where 99% of samples are negative (e.g., rare fraud or disease), a trivial model predicting all zeros gets 99% accuracy while having a recall of 0%.
  2. **Data Leakage**: A feature was included during training that is not available at production inference time (e.g., using `refund_processed_date` to predict whether a purchase will be refunded).
  3. **Data / Concept Drift**: The distribution of input features $P(X)$ or conditional label distribution $P(y|X)$ changed significantly between the training period and production environment (e.g., consumer behavior shifting before vs after holiday season).

---

### Q3 [Intermediate]: Why do tree-based models (Random Forest, XGBoost) not require feature scaling, while algorithms like SVM, KNN, and Logistic Regression do?
* **Model Answer**:
  * **Tree-based models** evaluate each feature independently using single-variable rank-ordered threshold splits (e.g., `is feature_x > 45.2?`). Monotonically transforming a feature (scaling, shifting) does not change the sorting order of the data points, so split points and resulting Information Gain remain completely identical.
  * **Distance-based algorithms (KNN, SVM, K-Means)** calculate Euclidean distances ($\sqrt{\sum (x_i - y_i)^2}$). A feature with a large numerical range (e.g., Salary: $\$50,000\text{–}\$200,000$) will dominate the distance calculation over a feature with a small range (e.g., Age: $20\text{–}65$), rendering the smaller feature effectively ignored.
  * **Gradient-based models (Neural Networks, Logistic Regression)** with unscaled features produce highly elongated, elliptical loss contours, causing gradient descent updates to oscillate erratically and converge slowly.

---

### Q4 [Advanced]: Explain the Bias-Variance tradeoff. How do Bagging and Boosting manipulate bias and variance differently?
* **Model Answer**:
  * Total expected prediction error decomposes into $\text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$.
  * **Bias** represents underfitting: error from overly simplistic assumptions (e.g., fitting a straight line to sinusoidal data).
  * **Variance** represents overfitting: error from excessive sensitivity to noise in the training set.
  * **Bagging (Random Forest)** trains multiple high-capacity, deep, low-bias trees in parallel on bootstrapped data and averages them. Mathematically, averaging $B$ decorrelated estimators reduces variance by up to $\frac{1}{B}$ without increasing bias:

    $$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$

  * **Boosting (XGBoost, Gradient Boosting)** trains shallow, high-bias weak learners sequentially, with each tree fitting the pseudo-residuals of the previous ensemble. Boosting incrementally drives down **bias**, while keeping variance under control via learning rate shrinkage ($\eta$) and shallow tree depth.

---

## 1.2 Mathematics for AI

### Q5 [Basic]: What is the difference between the Dot Product and Cosine Similarity, and when are they identical?
* **Model Answer**:
  * The **Dot Product** is $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$. It is influenced by both the angle $\theta$ between the vectors AND their individual magnitudes.
  * **Cosine Similarity** is the normalized dot product:

    $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

    It measures pure directional alignment, completely invariant to length.
  * **They are identical** when both vectors are pre-normalized to unit length ($\|\mathbf{u}\|_2 = 1, \|\mathbf{v}\|_2 = 1$). In production vector databases, embeddings are pre-normalized upon insertion so the database can compute Cosine Similarity using faster hardware dot-product instructions.

---

### Q6 [Intermediate]: Why do we divide by $\sqrt{d_k}$ in the Transformer Scaled Dot-Product Attention formula?
* **Model Answer**:
  * In the attention formula $\text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$, assume the components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and unit variance.
  * Their dot product $\sum_{i=1}^{d_k} q_i k_i$ has a mean of 0 and a **variance equal to $d_k$**.
  * When $d_k$ is large (e.g., 64 or 128), the dot products can grow very large in absolute magnitude.
  * Passing large values into the **Softmax** function pushes outputs to extreme values near 0 or 1, where the Softmax derivative (gradient) is **virtually zero**.
  * Dividing by $\sqrt{d_k}$ rescales the variance back to $1.0$, keeping inputs in the sensitive, high-gradient regime of the Softmax function and preventing vanishing gradients during training.

---

## 1.3 Data Processing & Leakage

### Q7 [Intermediate]: Give two concrete examples of data leakage in an ML pipeline and explain how to prevent them.
* **Model Answer**:
  1. **Preprocessing Leakage**:
     * *Example*: Fitting a `StandardScaler()` or calculating missing value medians on the entire dataset *before* performing train/test split. The training set is contaminated with information about the test set's mean and distribution.
     * *Prevention*: Split data first. Call `scaler.fit(X_train)` and then `scaler.transform(X_train)` and `scaler.transform(X_test)`. Enforce this using Scikit-Learn `Pipeline`.
  2. **Target Leakage / Temporal Lookahead Leakage**:
     * *Example*: In a model predicting customer churn over the next 30 days, including a feature `contacted_cancellation_support = 1`. In reality, this action happens *after* the customer decides to churn.
     * *Prevention*: Strictly audit timestamps: all feature values must correspond to data that existed prior to the prediction cutoff moment. For time-series, use temporal cross-validation (`TimeSeriesSplit`), never random shuffle splits.

---

## 1.4 NLP Fundamentals & Tokenization

### Q8 [Basic]: What is the difference between Stemming and Lemmatization? Which one would you use in a modern LLM RAG pipeline?
* **Model Answer**:
  * **Stemming** uses crude, heuristic rule-based string cutting to strip prefixes/suffixes (e.g., Porter Stemmer maps *"running"*, *"runner"* $\to$ *"run"*, but *"flies"* $\to$ *"fli"*). It is fast but often produces non-dictionary words.
  * **Lemmatization** uses a morphological dictionary and part-of-speech (POS) tags to reduce words to their true grammatical root (lemma) (e.g., *"better"* $\to$ *"good"*, *"meeting"* with POS noun $\to$ *"meeting"*, verb $\to$ *"meet"*).
  * **In a modern LLM RAG pipeline**, we generally use **neither** on the text entering the LLM! Modern LLMs rely on subword tokenizers (Byte-Pair Encoding) and dense contextual embeddings that already capture morphological nuances and semantic relationships. However, lemmatization or stemming can still be used in the **BM25 / sparse lexical branch** of a Hybrid RAG retriever.

---

### Q9 [Intermediate]: How do Word2Vec embeddings differ from Transformer contextual embeddings (like BERT or GPT)?
* **Model Answer**:
  * **Word2Vec (Static Embeddings)**: Each token in the vocabulary has a single, fixed vector lookup. In the sentences *"I deposited cash at the bank"* and *"He fished on the river bank"*, the word *"bank"* receives the exact same vector, conflating polysemous meanings.
  * **Transformer Embeddings (Contextual)**: Token representations are passed through multiple layers of multi-head self-attention. The vector for *"bank"* in the river sentence attends to *"fished"* and *"river"*, shifting its coordinates toward the geographical concept; in the financial sentence, it attends to *"cash"* and *"deposited"*, shifting toward finance.

---

### Q10 [Advanced]: What is the difference between Batch Normalization and Layer Normalization, and why do Transformers strictly use Layer Normalization?
* **Model Answer**:
  * **Batch Normalization (BatchNorm)**: Computes mean $\mu$ and variance $\sigma^2$ across the **batch dimension ($N$)** for each feature channel independently.
    * *Failure in NLP*: Language sequences have variable lengths, requiring padding. Furthermore, at inference time, batch size is often $1$, making batch statistics meaningless or noisy.
  * **Layer Normalization (LayerNorm)**: Computes mean $\mu$ and variance $\sigma^2$ across all **features ($d$)** of a *single* token embedding independently:

    $$\text{LayerNorm}(x) = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \odot \gamma + \beta$$

    * *Why Transformers use it*: It has zero dependency on batch size or other sequences in the batch. It behaves identically during distributed training and single-user low-latency inference.

---

### Q11 [Intermediate]: Compare Byte-Pair Encoding (BPE), WordPiece, and SentencePiece.
* **Model Answer**:
  * **BPE (Byte-Pair Encoding - GPT/LLaMA)**: Bottom-up algorithm. Starts with character vocabulary; iteratively merges the most **frequent adjacent pair** of symbols in the corpus.
  * **WordPiece (BERT)**: Similar to BPE, but merges adjacent pairs that maximize the **language model likelihood** (mutual information) of the training text rather than pure frequency count.
  * **SentencePiece (T5/Llama)**: Treats the entire input as a raw byte stream including whitespace (represented as `_`). It does not rely on language-specific whitespace splitting, making it language-agnostic and ideal for non-spaced scripts (Chinese, Japanese, Thai) and code.
