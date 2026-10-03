# 2. Essential Mathematics: Probability, Statistics, and Calculus for AI

This guide covers the core statistical and calculus concepts necessary for an AI Engineer to reason about model behavior, loss functions, gradient propagation, and optimization.

---

## 2.1 Basic Statistics

### 2.1.1 Measures of Central Tendency
* **Mean ($\mu$ or $\bar{x}$)**: Arithmetic average:

  $$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$

  Sensitive to extreme outliers.
* **Median**: The middle value of a sorted list. Robust to outliers (ideal for income distributions or API response latency percentiles).
* **Mode**: The most frequently occurring value (useful for categorical frequency distributions).

### 2.1.2 Measures of Dispersion
* **Variance ($\sigma^2$)**: The average squared deviation from the mean:

  $$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2$$

* **Standard Deviation ($\sigma$)**: The square root of variance, expressing spread in the original feature units:

  $$\sigma = \sqrt{\sigma^2}$$

> **AI Engineering Context**: When monitoring LLM API latency, tracking mean latency is misleading because LLM response times are heavy-tailed. Production systems monitor **p50 (median)**, **p95**, and **p99** tail latencies.

---

## 2.2 Probability Fundamentals

### 2.2.1 Independent and Conditional Probability
* **Independent Events**: Occurrence of $A$ has no influence on $B$: $P(A \cap B) = P(A) \cdot P(B)$.
* **Conditional Probability**: The probability of $A$ given that event $B$ has already occurred:

  $$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

### 2.2.2 Bayes' Theorem
Relates conditional and marginal probabilities, forming the basis for probabilistic inference:

$$P(\text{Hypothesis} \mid \text{Evidence}) = \frac{P(\text{Evidence} \mid \text{Hypothesis}) \cdot P(\text{Hypothesis})}{P(\text{Evidence})}$$

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

* **Prior $P(H)$**: Initial belief before seeing data.
* **Likelihood $P(E \mid H)$**: Probability of observing the evidence given the hypothesis.
* **Posterior $P(H \mid E)$**: Updated belief after observing the evidence.

---

## 2.3 Probability Distributions in Modern AI

```text
Normal (Gaussian) Distribution                 Softmax (Categorical) Distribution
             ^                                              ^
           /   \                                    [0.02]  |
          /  |  \                                   [0.78]  | ====> Next Token: "apple" (78%)
        /    |    \                                 [0.15]  | ====> Next Token: "banana" (15%)
    ---/-----|-----\---                             [0.05]  |
           Mean (μ)                                         +------------------------------
   -3σ     -1σ  +1σ   +3σ                                       Discrete Vocabulary Tokens
```

### 2.3.1 Normal (Gaussian) Distribution
Continuous probability distribution symmetric about the mean:

$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

* **68-95-99.7 Rule**: 68.2% of data falls within $\pm 1\sigma$, 95.4% within $\pm 2\sigma$, 99.7% within $\pm 3\sigma$.
* **Why it matters**: Neural network weight initializations (He and Xavier initialization) and latent spaces in Variational Autoencoders (VAEs) and Diffusion Models are modeled as Gaussian distributions.

### 2.3.2 The Categorical Distribution via Softmax
In an LLM, the model outputs raw unnormalized real numbers (logits $\mathbf{z} \in \mathbb{R}^{V}$) across the entire vocabulary of size $V$ (e.g., $V = 128,256$ tokens).

The **Softmax** function maps logits into a valid categorical probability distribution:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}$$

Where $T$ is the sampling **temperature**. Every value is strictly positive ($> 0$), and all probabilities sum to 1.

---

## 2.4 Calculus for AI: Derivatives and Gradients

### 2.4.1 The Derivative Intuition
The derivative $\frac{df}{dx}$ measures the instantaneous rate of change of a function $f(x)$ at point $x$.
* If $\frac{df}{dx} > 0$: Increasing $x$ increases $f(x)$.
* If $\frac{df}{dx} < 0$: Increasing $x$ decreases $f(x)$.
* If $\frac{df}{dx} = 0$: Stationary point (local minimum, local maximum, or saddle point).

### 2.4.2 The Gradient Vector ($\nabla$)
For a multivariable function $f(\mathbf{w})$ where $\mathbf{w} \in \mathbb{R}^d$, the gradient $\nabla_{\mathbf{w}} f$ is a vector containing all partial derivatives:

$$\nabla_{\mathbf{w}} f = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_d} \right]^T$$

> **Key Intuition**: The gradient points in the direction of the **steepest ascent** of the function. Therefore, moving in the negative gradient direction ($-\nabla$) moves toward the **steepest descent** (lowest loss).

---

## 2.5 Loss Functions: Quantifying Error

A loss function translates model mistakes into a scalar penalty.

### 2.5.1 Mean Squared Error (MSE)
For continuous numerical regression targets:

$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 2.5.2 Cross-Entropy Loss (Log Loss)
For classification and **LLM Next-Token Prediction**:

$$\mathcal{L}_{\text{CE}}(y, \hat{y}) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

In an LLM, the ground truth target is a one-hot vector where only the true next token $t^*$ has $y_{t^*} = 1$. The formula simplifies to:

$$\mathcal{L}_{\text{LLM}} = -\log P(w_{\text{target}})$$

```text
Scenario 1 (Good prediction): Model assigns 0.90 probability to true token.
Loss = -ln(0.90) = 0.105 (Small penalty)

Scenario 2 (Bad prediction): Model assigns 0.01 probability to true token.
Loss = -ln(0.01) = 4.605 (Heavy penalty)
```

### 2.5.3 Why Cross-Entropy Beats MSE for Classification: The Gradient Proof
* If we use **MSE with Sigmoid activation** ($\hat{y} = \sigma(z)$):

  $$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = (\hat{y} - y) \cdot \sigma'(z) = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y})$$

  When the model is confidently wrong ($\hat{y} \approx 1$ while $y = 0$), the term $\hat{y}(1 - \hat{y}) \to 0$. The gradient **vanishes**, meaning the model cannot learn to correct bad mistakes!
* If we use **Cross-Entropy with Sigmoid/Softmax**:

  $$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = \hat{y} - y$$

  The gradient is directly proportional to the prediction error with no vanishing derivative factor! The model learns aggressively when it makes large mistakes.

---

## 2.6 Gradient Descent and Modern Optimizers

### 2.6.1 Gradient Descent Update Rule

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}_t)$$

Where $\alpha > 0$ is the **learning rate**.

```text
                           LEARNING RATE DYNAMICS
                           
   Too Small (α = 0.00001)        Optimal (α = 0.01)           Too Large / Divergent (α = 2.0)
             \       /                   \       /                       \       /
              \ .-. /                     \ .-. /                         \ .-. /
               '---'                       '---'                           '---'
   Takes millions of steps;     Smooth, fast convergence      Oscillates wildly and explodes
   gets trapped in saddles.     to minimum.                   to infinity (NaN loss).
```

### 2.6.2 Adam and AdamW Optimizers

Modern LLMs and neural networks rarely use plain SGD. They use **Adam** (Adaptive Moment Estimation) or **AdamW** (Loshchilov & Hutter, 2017).

#### Mathematical Formulation:
At optimization step $t$ with gradient $g_t = \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}_t)$:

1. **First Moment (Momentum)**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   *(Typically $\beta_1 = 0.9$)*

2. **Second Moment (Uncentered Variance)**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   *(Typically $\beta_2 = 0.95$ for LLMs or $0.999$)*

3. **Bias Corrections** (accounts for initialization bias towards zero in early steps):
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

4. **Weight Update**:
   * **Standard Adam with $L_2$ Regularization** adds weight decay directly into $g_t$, which mistakenly causes weights with large past gradients to experience less regularization:
     $$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
   * **AdamW (Decoupled Weight Decay)** separates weight shrinkage from the adaptive gradient scale:
     $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \lambda \mathbf{w}_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
     where $\lambda$ is the weight decay rate.

> **Why AdamW is standard for Transformers**: Standard Adam fails to regularize parameters with frequent, large gradients because $\frac{1}{\sqrt{\hat{v}_t}}$ shrinks the regularization penalty. AdamW applies true proportional weight decay $\alpha \lambda \mathbf{w}_t$ directly to all weights, preventing hidden layer representation collapse and weight explosion.
