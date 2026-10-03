# 1. LLM Application Engineering: Production APIs, Streaming, Latency, and Cost

Moving a prototype from a notebook to an enterprise-grade production software system requires robust API management, error handling, caching, observability, and cost/latency optimization.

---

## 1.1 Anatomy of Inference Latency: TTFT vs Inter-Token Latency

In production LLM applications, user-perceived speed is governed by two fundamentally different hardware phases on the GPU:

```text
Request Dispatched
        |
        +======== PHASE 1: PREFILL PHASE (Compute-Bound) ============================+
        | - Model processes the entire input prompt (e.g., 2,000 prompt tokens).     |
        | - Highly parallelized matrix multiplications on Tensor Cores.              |
        | - Generates initial KV Cache.                                              |
        | - Output: Emits the VERY FIRST token.                                      |
        +============================================================================+
        |
        v
Time To First Token (TTFT) ~ 300ms - 1,200ms
        |
        +======== PHASE 2: DECODING PHASE (Memory Bandwidth-Bound) ==================+
        | - Model generates output tokens autoregressively ONE BY ONE.               |
        | - Must reload ALL model weights from GPU HBM to SRAM for EVERY single token|
        | - Output: Subsequent tokens streamed to user.                              |
        +============================================================================+
        |
        v
Inter-Token Latency (Time Per Output Token, TPOT) ~ 20ms - 50ms per token
```

---

## 1.2 Rate Limits, Exponential Backoff, and Full Jitter

When traffic spikes exceed provider quotas (RPM/TPM), APIs return `HTTP 429 Too Many Requests`.

```text
WITHOUT JITTER (Thundering Herd Problem):
Client 1: [ Sleep 2s ] --------> Hammers API at t=2.000s (Collides & Fails!)
Client 2: [ Sleep 2s ] --------> Hammers API at t=2.000s (Collides & Fails!)
Client 3: [ Sleep 2s ] --------> Hammers API at t=2.000s (Collides & Fails!)

WITH FULL JITTER (Smooth Uniform Distribution):
Client 1: [ Sleep 1.2s ] ------> Succeeds at t=1.2s!
Client 2: [ Sleep 2.1s ] ------> Succeeds at t=2.1s!
Client 3: [ Sleep 0.8s ] ------> Succeeds at t=0.8s!
```

$$\text{Sleep Delay} = \text{random}(0, \min(M, \text{base} \cdot 2^{\text{attempt}}))$$

---

## 1.3 LLM Caching Architectures: Exact vs Semantic vs Prefix

```text
                                  LLM CACHING TAXONOMY
                                            |
         +----------------------------------+----------------------------------+
         |                                  |                                  |
1. Exact Key Caching (Redis)       2. Semantic Caching (GPTCache)     3. Prefix Prompt Caching
   SHA256(Prompt + Model + Temp)      Embed Query -> Vector Search       Provider-side KV cache
   - Hits if string is 100% exact.    - Hits if Cosine_Sim > 0.96.       - Hits if prompt prefix
   - Latency: 2ms. Cost: $0.00.       - Handles synonym phrasing.          matches past requests.
                                                                         - Cuts cost by 50%-90%.
```

---

## 1.4 The Enterprise LLM Gateway Pattern

In production architectures, applications never call external LLM APIs directly. They route through an **LLM Gateway** (e.g., LiteLLM, Portkey):

```text
[ Web App / Mobile / Agents ]
              |
              v
+===============================================================================+
|                             ENTERPRISE LLM GATEWAY                            |
|                                                                               |
|  [ Unified API Schema ]  ---> OpenAI / Anthropic / Local vLLM normalized      |
|  [ Rate Limiter ]        ---> Enforces tenant token quotas                    |
|  [ Fallback Router ]     ---> If Claude 3.5 fails (500), auto-route to GPT-4o |
|  [ Semantic Cache ]      ---> Checks Redis cache before external call         |
|  [ Audit & Telemetry ]   ---> Logs spend, latency, and PII to Langfuse        |
+===============================================================================+
              |
       +------+-----------------------------+
       |                                    |
       v                                    v
[ Anthropic API ]                    [ OpenAI API ]                    [ Private vLLM Cluster ]
```
