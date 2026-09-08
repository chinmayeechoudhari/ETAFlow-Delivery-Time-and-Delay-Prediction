---
trigger: always_on
description: Consult the graphify knowledge graph at graphify-out/ for codebase and architecture questions.
---

## graphify — Dedicated Context Knower

Graphify serves exclusively as the project's **Context Knower** (codebase knowledge oracle and relationship map).

### Role and Boundaries
- **Passive Knowledge Provider Only**: Graphify's sole job is to hold and deliver ground-truth context about project architecture, component relationships, data flows, and code structure. It does NOT direct tasks, orchestrate workflows, or make implementation decisions.
- **When Context Is Needed / Forgotten**: If any model or agent forgets codebase structure, file locations, dependency chains, or component interactions, it queries Graphify first to recover complete, accurate context.
- **Implementation Handoff**: Once the context is retrieved from Graphify, the model implements the code and fixes directly using standard tools.
- **Context Lookup Priority**:
  - When `graphify-out/graph.json` exists:
    - Run `graphify query "<question>"` (CLI) or `query_graph` (MCP) to resolve architectural or contextual questions.
    - Run `graphify path "<A>" "<B>"` for understanding relationships/connections.
    - Run `graphify explain "<concept>"` for deep dives into specific modules or logic.
  - If `graphify-out/wiki/index.md` or `graphify-out/GRAPH_REPORT.md` exists, consult them as needed to restore context.
- **Syncing Context**: After modifying code files, run `graphify update .` to keep the graph and context up-to-date (AST-only, zero API cost).
