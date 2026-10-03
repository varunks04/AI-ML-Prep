# 2. Data Mining & Processing: Feature Engineering, Selection, and Leakage Prevention

Feature engineering transforms raw domain data into informative mathematical representations that maximize an algorithm's predictive capability.

---

## 2.1 Data Transformations

### 2.1.1 Log and Power Transforms
Many linear models and distance metrics struggle with highly right-skewed feature distributions (e.g., website pageviews, household wealth, file sizes).

```text
Skewed Distribution (Long Right Tail)          Log Transformed Distribution (Gaussian-like)
  |\                                                    /\
  | \                                                  /  \
  |  \____                                            /    \
  +-------------> Raw Values (0 to 1,000,000)        +-------------> log(x + 1) (0 to ~13.8)
```

* **Natural Logarithm**: $x' = \ln(x + 1)$ (using $\log_{1p}$ to handle zeros gracefully).
  * Compresses large values, pulls outliers closer to the distribution center, and stabilizes variance.
* **Box-Cox and Yeo-Johnson**: Parametric power transforms that mathematically search for optimal parameter $\lambda$ to convert non-normal features into near-Gaussian distributions.

### 2.1.2 Discretization and Binning
Converts continuous values into discrete categorical intervals:
* **Equal-width binning**: Divides feature range into $k$ equal-sized buckets. (Sensitive to outliers).
* **Equal-frequency (Quantile) binning**: Divides feature so each bucket contains an equal number of samples. Useful for non-linear relationships.

---

## 2.2 Feature Engineering Techniques

### 2.2.1 Temporal and Datetime Features
Raw timestamps (e.g., `2026-10-03 19:11:08`) cannot be fed directly to numerical models.
* **Decomposition**: Extract `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `quarter`.
* **Cyclical Encoding**: A model treating hour `23` and hour `0` as raw numbers sees a distance of 23, failing to understand they are consecutive 1 hour apart.
  * Solution: Map cyclical time to sine and cosine coordinates on the unit circle:

  $$x_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

### 2.2.2 Interaction Features and Domain Ratios
Combining two or more features often reveals signals that individual features hide:
* **Ratios**: $\text{Debt-to-Income} = \frac{\text{Total Debt}}{\text{Annual Income}}$.
* **Per-Unit Metrics**: $\text{Price-per-Square-Foot} = \frac{\text{House Price}}{\text{Square Footage}}$.
* **Aggregations**: Rolling window calculations (e.g., `mean_transaction_amount_last_7_days`).

---

## 2.3 Feature Selection

Feeding unnecessary, redundant, or noisy features into a model increases training time, memory consumption, risk of overfitting, and the curse of dimensionality.

```text
                             Feature Selection Strategies
                                          |
          +-------------------------------+-------------------------------+
          |                               |                               |
   Filter Methods                 Wrapper Methods                 Embedded Methods
   - Fast, model-agnostic         - Model-dependent, iterative    - Built into training
   - Pearson Correlation          - Recursive Feature             - Lasso (L1) Zeroing
   - Mutual Information             Elimination (RFE)             - Random Forest Feature
   - Chi-Square Test              - Forward / Backward Search       Importance / Gain
```

1. **Filter Methods**: Evaluate individual feature characteristics independent of the learning algorithm.
   * *Pearson Correlation Filter*:

     $$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

     If two features have $|r| > 0.85$, drop one to eliminate multicollinearity.
   * *Variance Threshold*: Drop features with near-zero variance (e.g., 99.9% identical values).
2. **Wrapper Methods**: Evaluate subsets of features by training models iteratively.
   * *Recursive Feature Elimination (RFE)*: Fits model, ranks features, drops the weakest feature, repeats. High computational cost.
3. **Embedded Methods**: Feature selection occurs naturally during optimization.
   * *Lasso Regularization*: Mathematical $L_1$ penalty drives irrelevant feature coefficients directly to zero.
   * *Tree-based Gini / Gain Importance*: Ranks features by average reduction in impurity across all splits.

---

## 2.4 Deep Dive: Data Leakage Prevention Architecture

Data leakage remains the #1 reason machine learning models achieve high scores on offline validation but catastrophically fail upon production deployment.

```text
WRONG PIPELINE (Leaky):
Full Dataset (1000 rows) ---> [ Global StandardScaler() ] ---> Split into Train (800) & Test (200)
                                        ^
                                        | (Test mean & variance leaked into Training data!)

CORRECT PIPELINE (Airlocked):
Full Dataset (1000 rows) ---> Split into Train (800) & Test (200)
                                      |
                           [ Fit StandardScaler() ]
                                      |
                    +-----------------+-----------------+
                    |                                   |
           Apply .transform()                  Apply .transform()
                    |                                   |
                    v                                   v
             Clean X_train                       Clean X_test
                    |
           [ Fit Model(X_train) ]
                    |
           [ Evaluate on X_test ]
```

### 2.4.1 Production Pipeline Implementation (Scikit-Learn Pattern)
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Define column groups
num_features = ['age', 'income', 'credit_score']
cat_features = ['city', 'employment_status']

# Preprocessing sub-pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Full airlocked pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fits scaler and encoder STRICTLY on X_train; transforms X_test without leakage
full_pipeline.fit(X_train, y_train)
accuracy = full_pipeline.score(X_test, y_test)
```

---

## 2.5 Common Data-Processing Workflows: Tabular vs LLM Ingestion

| Processing Stage | Tabular Machine Learning Pipeline | Modern LLM / RAG Document Pipeline |
| :--- | :--- | :--- |
| **Ingestion** | SQL extracts, Parquet reads, CSV streaming. | Parsing unstructured PDFs, Markdown, HTML, JSON. |
| **Cleaning** | Null imputation, outlier clipping, type casting. | Stripping HTML tags, fixing encoding bugs, OCR cleanup. |
| **Structuring** | Scalers (Z-score), Encoders (One-Hot, Target). | Text Chunking (Recursive, Semantic, Parent-Document). |
| **Representation** | Numerical feature vectors $\mathbf{x} \in \mathbb{R}^d$. | Dense vector embeddings via embedding models. |
| **Storage Destination**| Feature Store (Feast, Hopsworks) or Postgres. | Vector Database (Pinecone, Chroma, Qdrant). |
| **Serving Query** | Point lookup of entity feature vector. | Semantic similarity search ($K$-Nearest Neighbors). |
