# 1. Machine Learning Fundamentals: Core Concepts

---

## 1.1 The AI Hierarchy: AI vs ML vs Deep Learning vs Generative AI

Understanding the precise relationships, technical capabilities, and boundaries between these four paradigms is a standard opening topic in AI Engineer technical interviews.

```text
+===================================================================================+
| ARTIFICIAL INTELLIGENCE (AI)                                                      |
| Broadest field: Systems displaying human-like reasoning, perception, or planning. |
| Includes: Expert systems, A* search, knowledge graphs, symbolic AI, heuristics.   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | MACHINE LEARNING (ML)                                                     |   |
|   | Algorithms that learn statistical representations directly from data      |   |
|   | without hand-crafted rules.                                               |   |
|   | Includes: Linear/Logistic Regression, Decision Trees, SVM, XGBoost.       |   |
|   |                                                                           |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | DEEP LEARNING (DL)                                                |   |   |
|   |   | Multi-layered Artificial Neural Networks that learn hierarchical   |   |   |
|   |   | representations directly from raw unstructured inputs.             |   |   |
|   |   | Includes: CNNs (Vision), RNNs/LSTMs, Transformers (BERT).          |   |   |
|   |   |                                                                   |   |   |
|   |   |   +-----------------------------------------------------------+   |   |   |
|   |   |   | GENERATIVE AI (GenAI)                                     |   |   |   |
|   |   |   | Deep learning foundation models trained to generate novel |   |   |   |
|   |   |   | synthetic data (text, code, images, audio, video).        |   |   |   |
|   |   |   | Includes: LLMs (GPT-4o, LLaMA-3), Diffusion Models, VAEs.  |   |   |   |
|   |   |   +-----------------------------------------------------------+   |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------------+   |
+===================================================================================+
```

### 1.1.1 What it is
* **Artificial Intelligence (AI)**: Any computational system engineered to simulate human intelligence tasks: logic, problem-solving, decision trees, or statistical inferences.
* **Machine Learning (ML)**: A mathematical subfield of AI formalized by Tom Mitchell (1997): *"A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."*
* **Deep Learning (DL)**: A class of ML using deep artificial neural networks (multiple hidden layers) capable of automatic, end-to-end feature extraction from raw unstructured data (pixels, audio waveforms, text bytes) without manual feature engineering.
* **Generative AI (GenAI)**: Foundation models trained on web-scale corpora that model the joint probability distribution $P(X)$ or conditional distribution $P(X|Y)$, enabling the generation of novel, high-dimensional artifacts rather than predicting a single discrete category or scalar value.

### 1.1.2 Paradigm Shift: Traditional Code vs ML vs GenAI

```text
1. Traditional Programming:
   Rules (Code) + Data (Inputs)  =========================>  Answers (Output)

2. Machine Learning:
   Data (Inputs) + Answers (Labels)  =====================>  Rules (Learned Parameters / Model)

3. Modern Generative AI:
   Prompt (Context + Instruction) + Pre-trained Foundation  =>  Synthesized Structured Output
```

---

## 1.2 How a Machine Learning System Works

### 1.2.1 Core Mathematical Principle
A supervised machine learning system approximates an unknown true mapping function $f^*: \mathcal{X} \rightarrow \mathcal{Y}$.

$$\hat{y} = f(\mathbf{x}; \mathbf{\theta})$$

* $\mathbf{x} \in \mathbb{R}^d$: Input feature vector.
* $\mathbf{\theta}$: Learnable model parameters (weights $\mathbf{w}$ and biases $b$).
* $\hat{y}$: Model output prediction.
* $\mathcal{L}(y, \hat{y})$: Objective / Loss function measuring the discrepancy between ground truth $y$ and prediction $\hat{y}$.
* **Optimization Algorithm**: An optimizer (e.g., Gradient Descent) computes $\nabla_\mathbf{\theta} \mathcal{L}$ to iteratively adjust $\mathbf{\theta}$ to minimize expected loss over the dataset:
  $$\mathbf{\theta}^* = \arg\min_\mathbf{\theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(y_i, f(\mathbf{x}_i; \mathbf{\theta}))$$

### 1.2.2 End-to-End Production Machine Learning Workflow

```text
+------------------------+      +-------------------------+      +----------------------------+
| 1. Problem Formulation | ---> | 2. Data Ingestion & EDA | ---> | 3. Preprocessing & Feature |
| - Objective definition |      | - Logging & ingestion   |      |    Engineering Pipeline    |
| - Metric selection     |      | - Class balance audits  |      | - Imputation & Scaling     |
| - Feasibility study    |      | - Outlier detection     |      | - Leakage airlock creation |
+------------------------+      +-------------------------+      +----------------------------+
                                                                               |
                                                                               v
+------------------------+      +-------------------------+      +----------------------------+
| 6. CI/CD & Production  | <--- | 5. Rigorous Evaluation  | <--- | 4. Model Training &        |
|    Monitoring          |      | - Validation & Test sets|      |    Hyperparameter Tuning   |
| - Data drift P(X)      |      | - Confusion matrix & PR |      | - Stratified Cross-Val     |
| - Concept drift P(y|X) |      | - Calibration & fairness|      | - Loss minimization        |
| - Retraining triggers  |      | - Latency & memory test |      | - Regularization (L1/L2)   |
+------------------------+      +-------------------------+      +----------------------------+
```

---

## 1.3 Features, Labels, Targets, and Representations

```text
Tabular Instance Representation:
                Feature 1       Feature 2       Feature 3          Target Label
Row / Instance   (Age)        (Income $k)     (Credit Score)      (Default Loan?)
Sample x_1  -->  [ 25,             48,             680 ]    ===>         0  (No)
Sample x_2  -->  [ 52,            120,             790 ]    ===>         0  (No)
Sample x_3  -->  [ 38,             32,             510 ]    ===>         1  (Yes)
                 \_____________________________________/
                       Feature Vector: x_i in R^3
```

* **Feature ($x_j$)**: An individual measurable property or characteristic of a phenomenon being observed.
* **Feature Vector ($\mathbf{x}_i \in \mathbb{R}^d$)**: An ordered $d$-dimensional numerical array representing all features of sample $i$.
* **Target / Label ($y_i$)**: The ground truth answer that the algorithm is tasked with learning to predict.
* **Feature Matrix ($\mathbf{X} \in \mathbb{R}^{N \times d}$)**: The collection of all $N$ data instances stacked into an $N$-row, $d$-column matrix.

---

## 1.4 Dataset Splitting: Train, Validation, and Test Sets

A primary cause of catastrophic production ML failure is evaluating models on contaminated data splits.

```text
                               TOTAL DATASET (100%)
+===========================================================================================+
| TRAINING SET (70% - 80%)            | VALIDATION SET (10% - 15%)  | TEST SET (10% - 15%)  |
| - Consumed by optimizer.            | - Evaluated during tuning.  | - Completely held out.|
| - Gradients computed here.          | - Used for early stopping,  | - Evaluated ONCE at   |
| - Weights (θ) updated directly.     |   hyperparameter search,    |   final release gate. |
|                                     |   and model selection.      | - True generalization.|
+===========================================================================================+
```

### 1.4.1 Cross-Validation Architectures

```text
1. Standard K-Fold Cross-Validation (K=5):
Fold 1: [ Test ] [ Train ] [ Train ] [ Train ] [ Train ]  ---> Score 1
Fold 2: [ Train ] [ Test ] [ Train ] [ Train ] [ Train ]  ---> Score 2
Fold 3: [ Train ] [ Train ] [ Test ] [ Train ] [ Train ]  ---> Score 3
Fold 4: [ Train ] [ Train ] [ Train ] [ Test ] [ Train ]  ---> Score 4
Fold 5: [ Train ] [ Train ] [ Train ] [ Train ] [ Test ]  ---> Score 5
Final Validation Score = Mean(Score 1 .. Score 5)

2. Stratified K-Fold (Mandatory for Imbalanced Classes):
Guarantees every fold maintains the EXACT target class ratio (e.g. 98% non-fraud, 2% fraud).

3. Time-Series / Expanding Window Split (Mandatory for Temporal Data):
Split 1: [ Train: Month 1-3 ] ---> [ Test: Month 4 ]
Split 2: [ Train: Month 1-4 ] -------> [ Test: Month 5 ]
Split 3: [ Train: Month 1-5 ] -----------> [ Test: Month 6 ]
* NEVER shuffle temporal data! Shuffling introduces lookahead data leakage.
```

---

## 1.5 Machine Learning Paradigms

```text
                               MACHINE LEARNING PARADIGMS
                                           |
         +--------------------+------------+------------+--------------------+
         |                    |                         |                    |
         v                    v                         v                    v
Supervised Learning   Unsupervised Learning    Semi-Supervised      Reinforcement Learning
- Labeled: (X, y)     - Unlabeled: (X)         - Small labeled +    - Agent + Environment
- Classification      - Clustering             - Large unlabeled    - Policy π(a|s)
- Regression          - Dim Reduction          - Self-training /    - Value function Q(s,a)
                      - Anomaly Detection        Pseudo-labeling    - RLHF for LLMs
```

### 1.5.1 Supervised Learning
* **Mechanism**: Given pairs $(\mathbf{x}_i, y_i)$, learn mapping function $f: \mathcal{X} \rightarrow \mathcal{Y}$.
* **Classification**: Discrete output space $\mathcal{Y} \in \{0, 1, \dots, C-1\}$.
* **Regression**: Continuous numerical output space $\mathcal{Y} \in \mathbb{R}$.

### 1.5.2 Unsupervised Learning
* **Mechanism**: Given inputs $\mathbf{x}_i$ without ground-truth labels, discover latent structures, probability densities, or geometric manifolds.
* **Clustering**: Grouping instances based on pairwise distance metrics (K-Means, DBSCAN, Gaussian Mixture Models).
* **Dimensionality Reduction**: Compressing high-dimensional feature spaces while preserving variance or topological neighborhood distances (PCA, t-SNE, UMAP).

### 1.5.3 Reinforcement Learning (RL)
* **Mechanism**: An active agent operates in an environment modeled as a Markov Decision Process (MDP):
  $$\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$
* At time step $t$, agent observes state $s_t \in \mathcal{S}$, executes action $a_t \in \mathcal{A}$, receives scalar reward $r_{t+1}$, and transitions to state $s_{t+1}$.
* **Objective**: Maximize cumulative discounted return $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$.
* **Modern LLM Relevance**: Foundation of **RLHF (Reinforcement Learning from Human Feedback)** using algorithms like PPO (Proximal Policy Optimization) to align base language models with human intentions.

---

## 1.6 Generalization: Overfitting, Underfitting, and the Bias-Variance Tradeoff

```text
Error
  ^
  |        \                                         /   Total Prediction Error
  |         \                                       /    (Bias^2 + Variance + σ^2)
  |          \             Optimal Model           /
  |           \              Complexity           /
  |            \                 |               /
  |             \               v               /  <-- High Variance (Overfitting)
  |  High Bias   \             .---.           /
  | (Underfit)    \          /       \        /        Validation Error Curve
  |                \       /           '-----'
  |                 \____/_________________________    Training Error Curve
  |
  +----------------------------------------------------------------------------------->
  Low Complexity (Linear Model)                   High Complexity (Deep Tree / Huge NN)
```

### 1.6.1 The Mathematical Error Decomposition
For an unseen test point $\mathbf{x}$, the expected Mean Squared Error decomposes into three mutually exclusive components:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \underbrace{\text{Bias}[\hat{f}(\mathbf{x})]^2}_{\text{Approximation Error}} + \underbrace{\text{Var}[\hat{f}(\mathbf{x})]}_{\text{Estimation Sensitivity}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

$$\text{Bias}[\hat{f}(\mathbf{x})] = \mathbb{E}[\hat{f}(\mathbf{x})] - f(\mathbf{x})$$

$$\text{Var}[\hat{f}(\mathbf{x})] = \mathbb{E}\left[\left(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]\right)^2\right]$$

### 1.6.2 Diagnostic Matrix

| Condition | Training Loss | Validation Loss | Root Cause | Engineering Solution |
| :--- | :--- | :--- | :--- | :--- |
| **High Bias (Underfitting)** | High | High | Model too simple; cannot capture underlying pattern. | Increase complexity, engineer non-linear features, decrease regularization ($\lambda$). |
| **High Variance (Overfitting)** | Very Low | High | Model memorizes training noise and sample quirks. | Add data, apply $L_1/L_2$ regularization, prune trees, apply Dropout, use Bagging. |
| **Optimal Generalization** | Low | Low (Close to Train) | Balances model capacity with true data distribution. | Ready for shadow deployment / A/B testing. |

---

## 1.7 Data Preprocessing and Feature Engineering

### 1.7.1 Missing Data Mechanisms and Imputation

```text
                               MISSING DATA TAXONOMY
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
        MCAR                            MAR                            MNAR
(Missing Completely at Random)   (Missing at Random)             (Missing Not at Random)
Missingness has no link to       Missingness depends on observed Missingness depends on the
any observed or unobserved data. variables, but not missing val. unobserved missing value itself!
Ex: Sensor battery died randomly.Ex: Men report weight less often.Ex: High-earners refuse salary question.
```

* **Dropping**: Safe only if MCAR and missing rows comprise $< 3\%$ of total data.
* **Mean / Median Imputation**:
  * Mean: for normal, un-skewed distributions.
  * Median: robust choice for skewed distributions with outliers.
* **Indicator Column Pattern (Crucial Best Practice)**:
  Whenever imputing a column, always create an accompanying binary flag:
  `income_was_missing = 1`. This preserves the signal of missingness for the model.

### 1.7.2 Categorical Feature Encoding

```text
Categorical Feature: City = ["Tokyo", "Paris", "New York", "Tokyo"]

1. One-Hot Encoding (OHE):
   Tokyo    --> [ 1, 0, 0 ]
   Paris    --> [ 0, 1, 0 ]
   New York --> [ 0, 0, 1 ]
   * Best for: Low cardinality (< 15 unique nominal categories).
   * Trap: High cardinality blows up feature matrix dimensionality.

2. Ordinal Encoding:
   T-Shirt Size = ["S", "M", "L", "XL"] --> [ 1, 2, 3, 4 ]
   * Best for: Features with genuine natural mathematical order.
   * Trap: Applying to nominal categories (e.g. Red=1, Blue=2) forces false arithmetic.

3. Target (Mean) Encoding:
   City "Tokyo" replaced by mean target of Tokyo residents (e.g., Default Rate = 0.08).
   * Best for: High-cardinality nominal categories (e.g., Zip Codes, Product IDs).
   * Trap: High risk of TARGET LEAKAGE without out-of-fold smoothing!
```

### 1.7.3 Feature Scaling: Normalization vs Standardization

```text
Input Feature: x = [ 10, 20, 30, 40, 1000 ]

1. Min-Max Normalization (Scales strictly to [0, 1]):
   x_norm = (x - x_min) / (x_max - x_min)
   * Highly sensitive to extreme outliers (1000 crushes all other values near 0).

2. Standardization / Z-score (Zero mean, unit variance):
   x_std = (x - μ) / σ
   * Centers data at 0; handles outliers better; essential for PCA, SVM, Neural Nets.

3. Robust Scaling (Median and IQR):
   x_robust = (x - Median) / IQR
   * Completely immune to outlier distortion.
```

---

## 1.8 Data Leakage: Prevention Architecture

Data leakage happens when data from the target or test distribution is accidentally introduced into the model training pipeline, creating deceptively high validation scores that collapse in production.

```text
LEAKY PIPELINE (FATAL FLAW):
[ Entire Dataset (10,000 rows) ]
              |
      [ StandardScaler() ]  <=== Computes global mean and std across ALL 10,000 rows!
              |
      [ Train/Test Split ]  <=== Train split now contains statistical properties of Test set!
              |
      [ Model.fit(Train) ]  <=== Model scores 99% in offline test, collapses in real world.

AIRLOCKED PIPELINE (PRODUCTION STANDARD):
[ Entire Dataset (10,000 rows) ]
              |
      [ Train/Test Split ]  <=== Split FIRST before any computation!
              |
       +------+------------------------------------------+
       | Train Split (8,000)                             | Test Split (2,000)
       v                                                 v
[ scaler.fit(Train) ]                             (Do NOT fit on Test!)
       |                                                 |
[ scaler.transform(Train) ]                       [ scaler.transform(Test) ]
       |                                                 |
[ Model.fit(Train) ]                                     |
       |                                                 |
[ Model.evaluate(Clean_Transformed_Test) ] <-------------+
```

---

## 1.9 Model Evaluation Metrics: Beyond Accuracy

```text
                                 CONFUSION MATRIX
                               Actual Positive (1)        Actual Negative (0)
                           +--------------------------+--------------------------+
Predicted Positive (1)     | True Positive (TP)       | False Positive (FP)      |
                           | Correctly caught alarm.  | Type I Error (False Alarm)|
                           +--------------------------+--------------------------+
Predicted Negative (0)     | False Negative (FN)      | True Negative (TN)       |
                           | Type II Error (MISSED!)  | Correctly ignored.       |
                           +--------------------------+--------------------------+
```

### 1.9.1 The Metric Toolkit

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \quad \text{(Fails completely on imbalanced data)}$$

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{("When model predicts 1, how often is it right?")}$$

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} \quad \text{("Of all actual 1s, how many did model find?")}$$

$$\text{Specificity} = \frac{TN}{TN + FP} \quad \text{("Of all actual 0s, how many were correctly cleared?")}$$

$$F_1\text{-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \quad \text{(Harmonic mean balancing Precision and Recall)}$$

### 1.9.2 ROC Curve vs Precision-Recall (PR) Curve
* **ROC-AUC (Receiver Operating Characteristic)**: Plots True Positive Rate vs False Positive Rate across all probability thresholds.
  * *Trap*: Can be overly optimistic on heavily imbalanced datasets because a large count of $TN$ keeps False Positive Rate low.
* **PR-AUC (Precision-Recall Curve)**: Plots Precision vs Recall across all thresholds.
  * **Industry Rule**: Always evaluate fraud, medical diagnosis, and anomaly models using **PR-AUC**, not ROC-AUC.
