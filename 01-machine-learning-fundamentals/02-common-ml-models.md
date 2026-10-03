# 2. Machine Learning Fundamentals: Common Models Guide

Every model is structured using the standard interview framework:
**What is it? $\rightarrow$ How does it work? $\rightarrow$ Why is it used? $\rightarrow$ When would you use it? $\rightarrow$ Limitations $\rightarrow$ Top Interview Questions**

---

## 2.1 Linear Regression

### 2.1.1 What is it?
A foundational parametric algorithm for modeling the linear relationship between a continuous target variable $y$ and an explanatory feature vector $\mathbf{x}$.

### 2.1.2 How does it work?
Assumes a linear mapping:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = \mathbf{w}^T \mathbf{x}$$

The optimization minimizes the **Residual Sum of Squares (RSS)**:

$$\mathcal{L}_{\text{OLS}}(\mathbf{w}) = \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$$

```text
y
^                      * Actual Data Points (y_i)
|                    *  |
|                  *    | e_i = (y_i - y_hat_i) [Residual]
|                *-----[x] Regression Line: y_hat = w^T x
|             *
|           *
+----------------------------------------------------> x
```

* **Analytical Solution (Normal Equation)**:
  $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
* **Iterative Solution (Gradient Descent)**:
  $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \left( -\frac{2}{N} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w}_t) \right)$$

### 2.1.3 Why is it used?
* Direct interpretability: $w_j$ represents the marginal change in target $y$ per unit increase in $x_j$, holding all other features constant.
* Fast training and sub-millisecond scoring.

### 2.1.4 When would you use it?
* Baseline regression benchmarks.
* Regulated domains (credit underwriting, drug dosage, pricing elasticity) requiring mathematical feature attribution.

### 2.1.5 Regularization: Ridge ($L_2$) vs Lasso ($L_1$)

```text
Lasso (L1) Constraint Surface (Diamond)        Ridge (L2) Constraint Surface (Circle)
             w2                                             w2
             ^                                              ^
             |                                              |
        /\   |                                           .-----.
       /  \  |                                          /       \
<-----+----#-+-----> w1                        <-------+----+----+-------> w1
       \  /  |                                          \       /
        \/   |                                           '-----'
             v                                              v
Corner hit driven exactly to w1 = 0!            Smooth shrinkage towards origin.
Automated Feature Selection.                    Handles Multicollinearity.
```

* **Ridge ($L_2$)**: $\mathcal{L}_{\text{Ridge}} = \text{RSS} + \lambda \sum w_j^2$. Solves singularity when $\mathbf{X}^T\mathbf{X}$ is non-invertible.
* **Lasso ($L_1$)**: $\mathcal{L}_{\text{Lasso}} = \text{RSS} + \lambda \sum |w_j|$. Drives non-essential weights to exact zeros.

---

## 2.2 Logistic Regression

### 2.2.1 What is it?
A linear parametric model for **classification** that maps continuous linear combinations of features to calibrated probabilities between 0 and 1.

### 2.2.2 How does it work?
Computes linear logit $z = \mathbf{w}^T \mathbf{x} + b$ and maps it through the **Sigmoid function**:

$$P(y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

```text
Probability P(y=1)
  1.0 +                                       .-------- (Saturates near 1)
      |                                     /
  0.5 + - - - - - - - - - - - - - - - - - -+ (Threshold z = 0)
      |                                   /
  0.0 +--------.                         /
      +--------+-------------------------+---------> Linear Score z = w^T x + b
              -4                         0         4
```

Optimized using **Binary Cross-Entropy Loss (Log Loss)**:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 2.2.3 Common Interview Questions
* **Q: Why is Log Loss convex for Logistic Regression while MSE is non-convex?**
  * *Answer*: Substituting the non-linear Sigmoid into the quadratic MSE loss creates a non-convex surface riddled with flat plateaus and local minima where $\nabla \approx 0$. Taking the negative log likelihood (Log Loss) cancels the exponential in the sigmoid numerator, yielding a strictly convex Hessian matrix $\mathbf{H} = \mathbf{X}^T \mathbf{S} \mathbf{X}$ (positive semi-definite everywhere), guaranteeing gradient descent converges to the global minimum.

---

## 2.3 Decision Trees

### 2.3.1 What is it?
A non-parametric supervised algorithm that recursively partitions feature space into axis-aligned hyper-rectangles using hierarchical if-else rules.

### 2.3.2 How does it work? (Recursive Binary Splitting)

```text
Full Dataset (D): 100 samples (50 Positive, 50 Negative) -> Gini = 0.500
                         |
           Candidate Split: [ Income > $60k? ]
                         |
         +---------------+---------------+
         |                               |
    YES (Branch 1)                  NO (Branch 2)
    40 Pos, 10 Neg                  10 Pos, 40 Neg
    Gini_1 = 1 - (0.8^2 + 0.2^2)    Gini_2 = 1 - (0.2^2 + 0.8^2)
           = 0.32                          = 0.32
    Weighted Gini = (50/100)*0.32 + (50/100)*0.32 = 0.32
    Information Gain = 0.500 - 0.32 = +0.180 (High Gain -> Split Selected!)
```

### 2.3.3 Impurity Metrics
* **Gini Impurity**: $I_G(p) = 1 - \sum_{k=1}^K p_k^2$ (computationally faster; no logarithms).
* **Entropy**: $H(p) = -\sum_{k=1}^K p_k \log_2(p_k)$ (information-theoretic measure of uncertainty).

### 2.3.4 Strengths and Limitations
* *Strengths*: Invariant to feature scaling; handles categorical data; transparent white-box logic.
* *Limitations*: High variance; prone to over-splitting and memorizing sample noise if unpruned.

---

## 2.4 Random Forest

### 2.4.1 What is it?
An ensemble learning technique that trains hundreds of unpruned, randomized decision trees in parallel using **Bagging (Bootstrap Aggregation)** and **Feature Subsampling**.

### 2.4.2 Architectural Mechanics

```text
Training Dataset D (N rows, P features)
       |
       +=======================================================================+
       |                                                                       |
       v (Bootstrap Sample 1)           v (Bootstrap Sample 2)                  v (Bootstrap Sample B)
[ N rows sampled w/ replacement ]  [ N rows sampled w/ replacement ]       [ N rows sampled w/ replacement ]
       |                                |                                       |
       v (Random sqrt(P) features)      v (Random sqrt(P) features)             v (Random sqrt(P) features)
[ Deep Decision Tree 1 ]          [ Deep Decision Tree 2 ]                [ Deep Decision Tree B ]
       |                                |                                       |
       v                                v                                       v
 Prediction 1                     Prediction 2                            Prediction B
       \                                |                                       /
        +-------------------------------+--------------------------------------+
                                        |
                        [ AGGREGATION & VOTING ]
                        Classification: Majority Vote
                        Regression: Arithmetic Mean
                                        |
                                        v
                             Final Ensemble Output
```

### 2.4.3 Out-of-Bag (OOB) Error
* When drawing $N$ samples with replacement from $N$ rows, the probability of any given row *not* being picked is:
  $$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 36.8\%$$
* These 36.8% unpicked rows form the **Out-of-Bag (OOB)** set for that tree.
* Evaluating each tree on its unseen OOB samples provides an **unbiased validation score built directly into the training process** without needing an external validation split.

---

## 2.5 Gradient Boosting & XGBoost

### 2.5.1 What is it?
A sequential ensemble method where each successive weak decision tree is trained to predict the **pseudo-residuals (negative gradients of the loss)** of the existing ensemble.

### 2.5.2 How Gradient Boosting Learns Step-by-Step

```text
Step 0: Initialize with constant base prediction (e.g., mean of y):
        F_0(x) = y_bar
              |
              v
Step 1: Compute Residuals: r_i1 = y_i - F_0(x_i)
        Train Tree 1 to predict r_i1
        Update Ensemble: F_1(x) = F_0(x) + η * Tree_1(x)
              |
              v
Step 2: Compute New Residuals: r_i2 = y_i - F_1(x_i)
        Train Tree 2 to predict r_i2
        Update Ensemble: F_2(x) = F_1(x) + η * Tree_2(x)
              |
              v
Step M: Final Model: F_M(x) = F_0(x) + η * sum_{m=1}^M Tree_m(x)
```

Where $\eta \in (0, 1]$ is the **learning rate (shrinkage factor)** that scales down the contribution of each tree to prevent overfitting.

### 2.5.3 What makes XGBoost (Extreme Gradient Boosting) the Industry Standard?
1. **Second-Order Taylor Approximation**: While standard GBM uses only first derivatives (gradients $g_i$), XGBoost incorporates both first derivatives ($g_i$) and second derivatives (**Hessians** $h_i$):
   $$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
2. **Explicit Regularization ($\Omega$)**: Directly penalizes tree complexity with leaf count penalty $\gamma T$ and $L_2$ leaf weight regularization $\frac{1}{2}\lambda \sum w_j^2$.
3. **Sparsity-Aware Splitting**: Automatically learns the optimal default branch direction for missing values.
4. **Histogram-based Quantization**: Bins continuous features into discrete buckets, speeding up candidate split evaluations by $10\times$.

---

## 2.6 Support Vector Machines (SVM)

### 2.6.1 What is it?
A supervised algorithm that finds an optimal linear hyperplane that maximizes the geometric **margin** separating two classes.

```text
                               SUPPORT VECTOR MACHINE
x2
 ^
 |             Class +1 (Positive)
 |               *        *
 |                  *   [+] <--- Support Vector (lies exactly on margin)
 |      - - - - - - - - - - - - - - - Margin Hyperplane: w^T x + b = +1
 |                    /     ^
 |                   /      | Margin Width = 2 / ||w||
 |                  /       v
 |-----------------+----------------- Decision Boundary: w^T x + b = 0
 |                /
 |     - - - - - / - - - - - - - - - Margin Hyperplane: w^T x + b = -1
 |             [-] <------------ Support Vector
 |           o       o
 |        o      o        Class -1 (Negative)
 +------------------------------------------------------------> x1
```

### 2.6.2 The Kernel Trick: Conquering Non-Linearity
When data is linearly inseparable in the input space $\mathbb{R}^2$, the **Kernel Trick** maps data into a higher-dimensional feature space $\mathbb{R}^3$ where a linear hyperplane can separate the classes:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$$

* **Radial Basis Function (RBF / Gaussian Kernel)**:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$
  Calculates inner products in an **infinite-dimensional Hilbert space** without explicitly computing high-dimensional coordinates.

---

## 2.7 K-Means Clustering

### 2.7.1 What is it?
An unsupervised partitioning algorithm that clusters $N$ observations into $K$ spherical, non-overlapping clusters by minimizing within-cluster variance.

### 2.7.2 Centroid Update Mechanics (Lloyd's Algorithm)

```text
Iteration 1: Random Centroids       Iteration 2: Voronoi Assignment    Iteration 3: Convergence
  o    o                             o--\                           \
    (C1)                                \  (C1)                      \   (C1)*
  o        *    *                    o--/     \                  o   o\  *   *
         (C2)                                  \   *   *           o   \ *  (C2)*
      *        *                                \ (C2)                  \  *
Centroids placed arbitrarily.        Points assigned to nearest.   Centroids shift to cluster means.
```

### 2.7.3 Objective Function (Inertia)
Minimizes the Within-Cluster Sum of Squares (WCSS):

$$\mathcal{J} = \sum_{k=1}^K \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \mathbf{\mu}_k\|^2$$

* **Determining $K$**:
  * **Elbow Method**: Plot Inertia vs $K$; select the point where rate of decrease abruptly flattens.
  * **Silhouette Analysis**: Measures cohesion vs separation ($s \in [-1, 1]$). Score near 1 indicates clean clustering.

---

## 2.8 Principal Component Analysis (PCA)

### 2.8.1 What is it?
An unsupervised linear transformation that projects high-dimensional data onto orthogonal axes of **maximum variance**, minimizing reconstruction error.

```text
x2
 ^                             PC1: 1st Principal Component (Captures 85% Variance)
 |                           /
 |                         /  *
 |                  *    /  *
 |                     /  *
 |                   /  *
 |              *  /
 |               /             PC2: 2nd Principal Component (Orthogonal, 15% Variance)
 |             /  \
 |           /      \
 +---------/----------\---------------------------------------> x1
```

### 2.8.2 Mathematical Steps
1. Center feature matrix to zero mean: $\mathbf{X}_c = \mathbf{X} - \mathbf{\mu}$.
2. Compute $D \times D$ Covariance Matrix: $\mathbf{\Sigma} = \frac{1}{N} \mathbf{X}_c^T \mathbf{X}_c$.
3. Compute Eigenvalues ($\lambda_j$) and Eigenvectors ($\mathbf{v}_j$) via Singular Value Decomposition (SVD):
   $$\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j$$
4. Sort eigenvectors in descending order of $\lambda_j$. The top $k$ eigenvectors form the projection matrix $\mathbf{W}_k$.
5. Project to reduced space: $\mathbf{X}_{\text{reduced}} = \mathbf{X}_c \mathbf{W}_k$.

---

## 2.9 Neural Network Foundations: The Artificial Neuron

```text
Inputs (x_j)       Weights (w_j)       Summation & Bias          Activation Function (f)
x_1 --------------> [ * w_1 ] ---\
                                  \
x_2 --------------> [ * w_2 ] -----> [ z = w^T x + b ] -----> [ a = f(z) ] -----> Output (y_hat)
                                  /
x_d --------------> [ * w_d ] ---/
                      Bias (b) -/
```

### 2.9.1 The Activation Suite

```text
1. Sigmoid: σ(z) = 1 / (1 + e^-z)       2. ReLU: f(z) = max(0, z)          3. GELU: f(z) = z * Φ(z)
      1.0 +         .---                      ^                                  ^
          |        /                          |      /                           |      /
      0.5 + - - - + - -                       |     /                            |     /
          |      /                            |    /                             |    /
      0.0 +-----+-------> z                   +---+-----------> z                +---+-----------> z
         -4     0     4                          0                                  /  (Smooth dip)
      Squashes to (0, 1).                     Zeroes negatives; no gradient      Default activation in
      Vanishing gradient at tails.            saturation for positive values.    BERT, GPT-3, LLaMA.
```

### 2.9.2 Backpropagation: The Chain Rule Computational Graph

```text
Forward Pass:   x ---> [ Linear: z = w*x + b ] ---> [ Activation: a = f(z) ] ---> [ Loss: L ]
                               |                                  |                     |
Backward Pass:  dL/dw <--------+----------- dL/dz <---------------+----------- dL/da <--+
```

$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_j} = \left(\frac{\partial \mathcal{L}}{\partial a} \cdot f'(z)\right) \cdot x_j$$
