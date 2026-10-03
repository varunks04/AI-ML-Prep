# 1. NLP Fundamentals: From Classical Text Representation to Word Embeddings

Natural Language Processing (NLP) enables computational systems to parse, comprehend, represent, and generate human language.

---

## 1.1 The Core NLP Processing Pipeline

```text
Raw Text (" The quick brown foxes jumped! ")
   ↓
Cleaning & Normalization (Lowercasing, punctuation stripping, unicode fix)
   ↓
Tokenization (Splitting text into atomic units: words, characters, or subwords)
   ↓
Stop Words & Lemmatization (Filter noise, reduce to morphological root)
   ↓
Vector Representation (BoW, TF-IDF, Word2Vec, Dense Embeddings)
   ↓
Model Processing (Classification, NER, Vector Similarity, LLM Context)
   ↓
Prediction / Downstream Output
```

---

## 1.2 Text Preprocessing and Tokenization

### 1.2.1 Text Cleaning and Normalization
* **Lowercasing**: Mapping `"Apple"` and `"apple"` to a single token to prevent vocabulary explosion.
* **Regex Filtering**: Stripping HTML tags, markdown artifacts, malformed URLs, and non-printable Unicode characters.
* **Accent Stripping / Unicode Normalization (NFC/NFKD)**: Resolving ligature or diacritic variations (e.g., `"café"` $\to$ `"cafe"`).

### 1.2.2 Subword Tokenization Deep Dive: Byte-Pair Encoding (BPE)
Subword tokenization (used in GPT-4, LLaMA, Mistral) eliminates the Out-Of-Vocabulary (OOV) problem while maintaining compact sequence lengths.

```text
                         BYTE-PAIR ENCODING (BPE) WALKTHROUGH

Corpus:
{"low": 5, "lower": 2, "newest": 6, "widest": 3}

Step 0: Initialize vocabulary with individual characters + end-of-word marker "</w>":
Vocab = {l, o, w, e, r, n, s, t, i, d, </w>}
Represent corpus as spaced characters:
- "l o w </w>" (freq: 5)
- "l o w e r </w>" (freq: 2)
- "n e w e s t </w>" (freq: 6)
- "w i d e s t </w>" (freq: 3)

Step 1: Count most frequent adjacent symbol pair:
Pair ("e", "s") appears in "newest" (6) + "widest" (3) = 9 times. (WINNER!)
Merge ("e", "s") -> "es"
New Vocab adds "es". Corpus updated: "n e w es t </w>", "w i d es t </w>"

Step 2: Next most frequent pair:
Pair ("es", "t") appears 9 times.
Merge ("es", "t") -> "est"
New Vocab adds "est".

Step 3: Repeat until vocabulary reaches target size V (e.g., 32,000 or 128,256).
Result: Frequent words remain single tokens ("low"), while rare words decompose gracefully ("un"+"pre"+"cedented").
```

---

## 1.3 Stop Words, Stemming, and Lemmatization

```text
Input Term: "studies", "studying"

Stemming (Porter / Snowball):           Lemmatization (WordNet / spaCy):
Heuristic suffix chopping               Morphological dictionary lookup
"studies"  ---> "studi"                 "studies"  ---> "study" (Lemma)
"studying" ---> "study"                 "studying" ---> "study" (Lemma)
Fast, crude, often produces             Requires Part-of-Speech (POS) tagging;
non-words ("studi").                    slower, produces true grammatical roots.
```

### 1.3.1 Stop Words
* Extremely frequent grammatical function words (`"the"`, `"is"`, `"at"`, `"which"`, `"on"`).
* **When to remove**: In classical sparse search (TF-IDF, BM25) and simple classification to reduce matrix dimensionality.
* **When NOT to remove**: In modern Deep Learning and LLMs! In context-aware Transformers, stop words carry crucial syntactic, negation, and semantic meaning (e.g., *"to be or not to be"* loses its entire meaning if stop words are stripped).

---

## 1.4 Classical Sparse Text Representations

### 1.4.1 Bag of Words (BoW)
Represents a document as a fixed-length vector of token occurrence counts across the entire corpus vocabulary:

$$\mathbf{v}_{\text{doc}} = [\text{count}(w_1), \text{count}(w_2), \dots, \text{count}(w_V)]$$

* **Drawback**: Completely discards word order and syntax. *"Dog bites man"* and *"Man bites dog"* yield identical BoW representations.

### 1.4.2 N-Grams
Contiguous sequences of $N$ items from a given text sample:
* Unigram ($N=1$): `["machine", "learning"]`
* Bigram ($N=2$): `["machine learning"]`
* Trigram ($N=3$): `["natural language processing"]`
* **Why it matters**: Captures local word order and phrase meaning (e.g., differentiating `"not good"` from `"good"`).
* **Limitation**: Exponential vocabulary explosion ($V^N$).

### 1.4.3 TF-IDF (Term Frequency - Inverse Document Frequency)
Replaces crude raw counts with a weighting scheme that penalizes words that appear everywhere while boosting terms that uniquely distinguish a document:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

1. **Term Frequency $\text{TF}(t, d)$**: Relative frequency of term $t$ in document $d$:

   $$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\sum_{t' \in d} \text{count}(t', d)}$$

2. **Inverse Document Frequency $\text{IDF}(t, D)$**: Logarithmic penalty for words common across the entire collection of documents $D$:

   $$\text{IDF}(t, D) = \ln\left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$

```text
Corpus: 10,000 Legal Documents
- Word "the":      Appears in 10,000 docs  ---> IDF ≈ ln(10000/10000) + 1 = 1.0  ---> Low TF-IDF!
- Word "tort":     Appears in 12 docs      ---> IDF ≈ ln(10000/12) + 1    = 7.7  ---> High TF-IDF! (High Signal!)
```

* **Where TF-IDF is still used in modern AI**: Hybrid RAG search architectures (combining BM25 / TF-IDF sparse keyword matching with dense vector similarity).

---

## 1.5 Word Embeddings: Continuous Dense Semantic Spaces

Classical representations produce **sparse**, high-dimensional vectors (dimension $V \approx 50,000+$) where every word is orthogonal:

$$\mathbf{v}_{\text{hotel}} \cdot \mathbf{v}_{\text{motel}} = 0$$

Word embeddings map words into **dense**, continuous vector spaces (dimension $d \approx 100\text{ to }1536$) where geometrically close vectors share semantic meaning.

```text
Sparse One-Hot (Orthogonal, Zero Semantic Meaning):
"king"   = [ 0, 0, 1, 0, 0, 0, ... ]  (dim = 50,000)
"queen"  = [ 0, 0, 0, 0, 1, 0, ... ]  (dim = 50,000)
dot("king", "queen") = 0.00

Dense Vector Embeddings (Continuous Semantic Space):
"king"   = [  0.72,  0.89, -0.34,  0.12, ... ] (dim = 300)
"queen"  = [  0.70,  0.87,  0.65,  0.15, ... ] (dim = 300)
dot("king", "queen") = 0.88  (High Semantic Affinity!)
```

### 1.5.1 Word2Vec (Mikolov et al., 2013)

```text
CBOW (Continuous Bag of Words):                Skip-Gram:
Surrounding Context Words                       Target Center Word
[ "the", "cat", "sat", "on" ]                          [ "cat" ]
             \                                             /
              v                                           v
       [ Projection ]                              [ Projection ]
              |                                           |
              v                                           v
     Predict Target Word                        Predict Context Words
           [ "mat" ]                            [ "the", "sat", "on" ]
(Faster; better on frequent words)             (Superior for rare words and small datasets)
```

### 1.5.2 Negative Sampling (Why Word2Vec Trains Fast)
Computing full Softmax over vocabulary $V = 100,000$ at every training step is computationally prohibitive.
* **Negative Sampling** converts multi-class classification into $k$ independent binary logistic regressions:

$$\mathcal{L} = \log \sigma(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}) \right]$$

Maximizes probability of true context pair $(w_I, w_O)$ while minimizing probability of $k$ randomly sampled "negative" noise words ($w_i$).

### 1.5.3 Vector Arithmetic
Dense embeddings demonstrate emergent linear compositionality:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

$$\mathbf{v}_{\text{Paris}} - \mathbf{v}_{\text{France}} + \mathbf{v}_{\text{Japan}} \approx \mathbf{v}_{\text{Tokyo}}$$

### 1.5.4 Static vs Contextual Embeddings (The Bridge to Transformers)
* **Static Embeddings (Word2Vec, GloVe, FastText)**: Each token has a single invariant vector regardless of context.
  * *Failure Mode*: In *"The bank of the river"* and *"The bank approved the loan"*, the word *"bank"* receives the exact same vector.
* **Contextual Embeddings (BERT, GPT, Modern LLMs)**: Every token representation is computed dynamically based on all surrounding tokens in the sequence via multi-head self-attention.
