# 2. AI Agents & MCP: The Model Context Protocol Architecture and Transports

The **Model Context Protocol (MCP)**, open-sourced by Anthropic in late 2024, is the universal open standard for connecting AI systems to external tools, databases, filesystems, and context repositories.

---

## 2.1 The $N \times M$ Integration Problem

```text
THE OLD FRAGMENTED WAY (N x M Custom Integrations):
[ Claude Desktop ] -----\ /----- [ GitHub API ]
[ Cursor IDE     ] ------X------ [ Postgres DB ]  ===> Fragile, non-standard wrappers
[ Custom Agent   ] -----/ \----- [ Slack Bot   ]

THE MCP STANDARD (Open Universal Protocol):
[ Claude Desktop ] ---\                     /---> [ GitHub MCP Server ]   ---> GitHub
[ Cursor IDE     ] ----+---> [ MCP JSON ] ------> [ Postgres MCP Server ] ---> DB
[ Custom Agent   ] ---/      [ -RPC 2.0 ]   \---> [ Slack MCP Server ]    ---> Slack
```

---

## 2.2 End-to-End Sequence Diagram

```text
User           Host App / IDE      MCP Client            Foundation LLM         MCP Server         Database
 |                  |                  |                        |                    |                 |
 |--- "Top users" ->|                  |                        |                    |                 |
 |                  |-- tools/list --->|                        |                    |                 |
 |                  |                  |--------------------------------- tools/list>|                 |
 |                  |                  |<--------------------------------- [schemas] |                 |
 |                  |                  |                        |                    |                 |
 |                  |-- Send Prompt + Tool Schemas ------------>|                    |                 |
 |                  |                  |                        |                    |                 |
 |                  |<-- Emit Tool Call: query_db(sql="...") ---|                    |                 |
 |                  |                  |                        |                    |                 |
 |                  |-- tools/call --->|                                             |                 |
 |                  |                  |-------------------------------- tools/call >|                 |
 |                  |                  |                                             |-- Execute SQL ->|
 |                  |                  |                                             |<- Rows: Alice --|
 |                  |                  |<------------------------------ Tool Result -|                 |
 |                  |                  |                        |                    |                 |
 |                  |-- Forward Result to LLM ----------------->|                    |                 |
 |                  |                  |                        |                    |                 |
 |                  |<-- Final Synthesized Answer --------------|                    |                 |
 |<-- "Alice..." ---|                  |                        |                    |                 |
```

---

## 2.3 MCP Transport Layers: `stdio` vs `SSE`

MCP defines two standard communication transports based on JSON-RPC 2.0:

```text
1. Standard Input/Output (stdio Transport - Local Execution):
   [ Host App / Client ]  <=== Child Process Pipes (stdin / stdout) ===>  [ MCP Server ]
   - Runs as a local subprocess on the developer's machine.
   - Zero network overhead; blazing fast IPC; isolated process memory.
   - Ideal for: Desktop tools, local SQLite DBs, local filesystem access, git.

2. Server-Sent Events (SSE Transport - Remote Cloud Execution):
   [ Host App / Client ]  <=== HTTP POST (Requests) & SSE (Responses) ===> [ Remote MCP Server ]
   - Operates over standard web infrastructure (HTTPS).
   - Server streams responses and push events back to the client over an open SSE connection.
   - Ideal for: Enterprise cloud microservices, shared corporate databases, SaaS APIs.
```

---

## 2.4 The Three MCP Primitives: Tools, Resources, and Prompts

* **Tools (Actions)**: Executable functions invoked by the model to effect changes or query live systems. (e.g., `create_jira_ticket`, `run_sql_query`).
* **Resources (Context Attachments)**: Passive read-only data attached to the model's context, identified by URI (e.g., `postgres://prod/users/schema` or `file:///app.log`).
* **Prompts (Slash Workflows)**: Pre-engineered, reusable prompt templates exposed by the server to guide user interactions (e.g., `/analyze-code pr=42`).

---

## 2.5 Security: Human-in-the-Loop Approval Pattern

Because MCP servers execute real actions on filesystems and databases, the protocol mandates strict client-side confirmation hooks:

```text
LLM decides to call: delete_table(name="test_runs")
                         |
                         v
        +-----------------------------------+
        |      HOST PERMISSION MODAL        |
        | Tool: delete_table                |
        | Server: postgres-mcp              |
        | Parameter: name = "test_runs"     |
        | [ Approve Action ]  [ Deny Call ] |
        +-----------------------------------+
                         |
           (Only executes upon user click)
```
