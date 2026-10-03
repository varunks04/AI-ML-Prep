# 1. AI Agents: Architectures, Reasoning Loops, Multi-Agent Teams, and State Graphs

An AI Agent is an autonomous software system where a foundation model is paired with planning capabilities, persistent memory, tool execution, and environment feedback to achieve multi-step goals with minimal human intervention.

---

## 1.1 The ReAct Reasoning and Execution Loop

Proposed by Yao et al. (2022), the **ReAct (Reason + Act)** pattern interleaves verbal reasoning ("Thought") with concrete execution ("Action") and sensory feedback ("Observation"):

```text
                                 THE ReAct EXECUTION LOOP
                                 
                                     [ User Goal ]
                                           |
                                           v
+=====================================> [ THOUGHT ] <===================================+
|                                  (LLM verbal reasoning                                |
|                                   over current state)                                 |
|                                          |                                            |
|                                          v                                            |
|                                      [ ACTION ]                                       |
|                                  (Emits structured                                    |
|                                   tool call JSON)                                     |
|                                          |                                            |
|                                          v                                            |
|                                 [ TOOL EXECUTION ]                                    |
|                                  (Runs Python, SQL,                                   |
|                                   or API call)                                        |
|                                          |                                            |
|                                          v                                            |
|                                   [ OBSERVATION ]                                     |
|                                  (Receives runtime output                             |
|                                   from environment)                                   |
|                                          |                                            |
|                                          v                                            |
|                              [ GOAL SATISFACTION CHECK ]                              |
|                                          |                                            |
|                     Is goal met? --------+-------- Goal not met?                      |
|                          |                                 |                          |
|                          v                                 +--------------------------+
+===================> [ FINAL ANSWER ]
```

---

## 1.2 Multi-Agent Architectural Patterns

When a task exceeds the context capacity or specialized capabilities of a single agent, systems deploy **Multi-Agent Architectures**:

```text
1. SUPERVISOR - WORKER PATTERN (Hierarchical Orchestration):
                          [ Supervisor / Router Agent ]
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
     [ Research Worker Agent ]                     [ Code Synthesis Worker ]
     Tools: WebSearch, Arxiv                       Tools: Python REPL, GitHub
                \                                             /
                 +---------------------+---------------------+
                                       |
                                       v
                     [ Critic / QA Validation Agent ]

2. DEBATE / REFLEXION PATTERN:
   Agent A generates solution ---> Agent B audits & critiques flaws ---> Agent A revises.
```

---

## 1.3 State Graphs: DAGs vs Cyclical State Machines (LangGraph Pattern)

```text
1. Traditional Chain / DAG (Linear Workflow):
   Node A ---> Node B ---> Node C ---> End
   * Cannot loop back upon tool errors; brittle.

2. State Graph (Cyclical Agentic Loop):
   [ State: { messages: [], iterations: 0 } ]
          |
          v
      [ Agent ] <-------------------+
          |                         |
          v                         |
    [ Tool Node ] -- (Error?) ------+  (Loop back with error trace!)
          |
     (Success?)
          v
       [ END ]
```

* **State**: A centralized typed schema (e.g., Pydantic or TypedDict) passed across all graph nodes.
* **Conditional Edges**: Dynamic routing functions that inspect the state (e.g., checking if the model output contains a `tool_calls` payload or a final text response).
