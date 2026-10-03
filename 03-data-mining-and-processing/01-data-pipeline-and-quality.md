# 1. Data Mining & Processing: Pipelines, Quality, Drift, and Outliers

Raw data from production environments is notoriously noisy, incomplete, unstructured, and messy. A primary responsibility of an AI Engineer is transforming raw information into high-integrity, machine-usable inputs.

---

## 1.1 The Data Hierarchy: From Raw Bits to Usable Inputs

```text
[ Raw Data Sources ]  ---> Logs, APIs, Relational DBs, PDFs, Audio, User Clicks
         |
         v
[ Ingestion & Cleaning ] -> De-duplication, Outlier Removal, Imputation, Text Stripping
         |
         v
[ Structuring / ETL ]  ---> Normalization, Schema Enforcement, Chunking, Parsing
         |
         v
[ Feature / Vector ]   ---> Feature Store (Tabular) OR Vector Database (Unstructured/LLMs)
         |
         v
[ Model Consumption ]  ---> Training Set (ML), Prompt Context (LLMs), RAG Chunks
```

### 1.1.1 What is Data Mining?
Data mining is the computational process of discovering non-trivial, previously unknown, and potentially actionable patterns, correlations, and anomalies from large datasets. It bridges statistics, machine learning, and database systems.

### 1.1.2 Structured vs Semi-Structured vs Unstructured Data

| Dimension | Structured Data | Semi-Structured Data | Unstructured Data |
| :--- | :--- | :--- | :--- |
| **Format** | Strictly typed rows and columns. | Key-value pairs, nested hierarchical tags. | Free-form natural text, audio, images, video. |
| **Examples** | SQL tables, CSV files, Parquet. | JSON logs, XML, MongoDB BSON. | Customer support transcripts, PDFs, PNG images. |
| **Storage** | Relational DBs, Data Warehouses (Snowflake). | NoSQL document stores. | Data Lakes (S3, GCS), Vector Databases. |
| **AI Paradigm** | Traditional ML (XGBoost, Random Forest). | Hybrid tabular parsing. | Deep Learning & LLMs (Transformers). |

---

## 1.2 Data Collection and Ingestion Patterns

### 1.2.1 Batch Ingestion vs Streaming Ingestion
* **Batch Ingestion**: Large volumes of data collected and processed at scheduled intervals (e.g., nightly hourly cron jobs via Airflow, Spark). High throughput, high latency.
* **Streaming Ingestion**: Continuous, real-time event-by-event ingestion (e.g., Apache Kafka, AWS Kinesis). Critical for fraud detection, real-time recommendations, and agent memory updates. Low latency, requires specialized stream processing (Flink).

### 1.2.2 ETL vs ELT in Modern AI Stacks

```text
Traditional ETL (Extract, Transform, Load):
[ Sources ] ---> [ Staging Compute ] (Heavy Transformations) ---> [ Clean Data Warehouse ]

Modern ELT (Extract, Load, Transform):
[ Sources ] ---> [ Cloud Data Lake / S3 ] ---> [ In-Warehouse Compute: dbt / DuckDB / LLMs ]
```

* **ETL (Extract $\to$ Transform $\to$ Load)**: Data is cleaned, masked, and transformed in memory *before* entering storage. Used when strict privacy/PII filtering is required before persistency.
* **ELT (Extract $\to$ Load $\to$ Transform)**: Raw, untransformed data is dumped directly into scalable cloud storage (S3/Snowflake), with transformations executed downstream as needed. This preserves the original raw data so LLM ingestion pipelines can re-chunk or re-embed when models improve.

---

## 1.3 Data Quality Dimensions & Production Drift

Poor data quality produces poor models ("Garbage In, Garbage Out"). The four core dimensions to verify:

1. **Completeness**: Are required attributes populated? (e.g., percentage of nulls in critical foreign keys).
2. **Accuracy & Ground Truth Validity**: Do the recorded values match objective reality? (e.g., customer age $= 250$ indicates invalid data entry).
3. **Consistency**: Do data values agree across disparate internal systems? (e.g., user is marked "Active" in Stripe billing but "Churned" in Postgres user table).
4. **Timeliness (Freshness)**: How stale is the data relative to inference time? (e.g., stock price features delayed by 15 minutes).

### 1.3.1 Data Drift vs Concept Drift vs Covariate Shift

```text
                                PRODUCTION DRIFT TAXONOMY
                                            |
         +----------------------------------+----------------------------------+
         |                                  |                                  |
1. Data Drift (Covariate Shift):   2. Concept Drift:                  3. Prior Probability Shift:
   P(X) changes,                      P(y | X) changes,                  P(y) changes,
   P(y | X) remains constant.         P(X) remains constant.             P(X | y) remains constant.
   Ex: Users switch from desktop      Ex: Inflation changes what         Ex: Overall disease prevalence
       browsers to mobile apps.           qualifies as "high credit       spikes during pandemic.
                                          risk" for the same income.
```

---

## 1.4 Outlier Detection and Treatment

An outlier is an observation that lies an abnormal distance from other values in a random sample from a population.

```text
IQR BOXPLOT ANATOMY:
                     Q1 (25th)     Median (50th)     Q3 (75th)
                         +--------------+--------------+
       |-----------------|              |              |-----------------|       * (Outlier!)
                         +--------------+--------------+
                      [               IQR              ]
  Lower Fence = Q1 - 1.5 * IQR                     Upper Fence = Q3 + 1.5 * IQR
```

### 1.4.1 Detection Techniques
1. **Z-Score Method (Parametric)**:
   Measures how many standard deviations $x$ is from the mean:

   $$Z = \frac{x - \mu}{\sigma}$$

   Heuristic: $|Z| > 3$ indicates an outlier. Assumes Gaussian distribution.
2. **Interquartile Range (IQR) Method (Non-Parametric)**:

   $$\text{IQR} = Q_3 - Q_1$$

   $$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$$

   $$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

   Robust against extreme skewed values.
3. **Isolation Forest (Multivariate)**:
   Tree-based unsupervised algorithm that isolates anomalies by randomly partitioning features. Outliers require fewer recursive splits to isolate.

### 1.4.2 Treatment Strategies
* **Trimming (Removal)**: Drop rows if outliers represent corrupt sensor readings or data entry mistakes.
* **Winsorization (Capping)**: Cap extreme values at fixed percentiles (e.g., clamp all values above the 99th percentile to the 99th percentile).
* **Robust Transformations**: Apply `RobustScaler` (subtract median, divide by IQR) instead of standard scaling.

---

## 1.5 Handling Missing Data: The Imputation Playbook

```text
                           Missing Data Check
                                  |
           +----------------------+----------------------+
           |                                             |
   Is missingness < 3%                           Is missingness > 3%
   and purely MCAR?                                      |
           |                                             v
   Drop Rows (df.dropna)                +----------------+---------------+
                                        |                                |
                                Continuous Feature              Categorical Feature
                                        |                                |
                            Skewed? -------- No?                         v
                               |              |                 Impute with "Unknown"
                               v              v                 or frequent Mode
                         Impute Median   Impute Mean
                               |
                               +---> Add Binary Indicator Column: feature_is_missing (0 or 1)
```

> **Interview Golden Rule**: When imputing missing numerical values, always create a corresponding binary indicator column (e.g., `income_is_missing = 1`). This allows downstream models to learn if the *fact of missingness itself* is predictive (e.g., users who refuse to disclose income often default at higher rates).
