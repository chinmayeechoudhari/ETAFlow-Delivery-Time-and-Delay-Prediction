---
name: graphify
description: Turn any folder of files into a navigable knowledge graph
---

# Workflow: graphify — Refresh Context Knower

Builds or refreshes the repository knowledge graph so models can consult it as the "Context Knower".

Follow the graphify skill to extract or update the knowledge graph.
If no path argument is given, use `.` (current directory).
Once generated, models consult the resulting graph/wiki for architectural and code context before implementing changes.
