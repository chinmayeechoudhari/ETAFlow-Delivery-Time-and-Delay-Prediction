# Graph Report - ETAFlow — Intelligent Delivery ETA & Delay Prediction Platform  (2026-09-08)

## Corpus Check
- 4 files · ~8,134 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 41 nodes · 65 edges · 10 communities (3 shown, 7 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `56d48a60`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- graphify — Dedicated Context Knower
- workflows/graphify.md
- generate_dataset.py
- DataFrame
- ._generate_geography
- .generate
- LogisticsDatasetGenerator
- ._generate_operational_and_environment
- ._generate_temporal_scaffold
- ._inject_controlled_missingness

## God Nodes (most connected - your core abstractions)
1. `LogisticsDatasetGenerator` - 13 edges
2. `main()` - 6 edges
3. `validate_dataset()` - 5 edges
4. `create_dataset_metadata()` - 5 edges
5. `vectorized_haversine()` - 4 edges
6. `create_data_dictionary()` - 4 edges
7. `parse_args()` - 3 edges
8. `graphify — Dedicated Context Knower` - 2 edges
9. `Computes great-circle distances in kilometers between origin and destination…` - 1 edges
10. `Production-grade synthetic dataset generator for ETAFlow. Leverages NumPy…` - 1 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `LogisticsDatasetGenerator`  [EXTRACTED]
  scripts/generate_dataset.py → scripts/generate_dataset.py  _Bridges community 6 → community 2_
- `create_dataset_metadata()` --references--> `DataFrame`  [EXTRACTED]
  scripts/generate_dataset.py →   _Bridges community 3 → community 2_

## Import Cycles
- None detected.

## Communities (10 total, 7 thin omitted)

### Community 2 - "generate_dataset.py"
Cohesion: 0.33
Nodes (8): Any, Namespace, create_dataset_metadata(), main(), parse_args(), Generates detailed JSON metadata documenting generation run and statistical…, Performs rigorous data validation against business rules and integrity…, validate_dataset()

### Community 3 - "DataFrame"
Cohesion: 0.29
Nodes (5): DataFrame, create_data_dictionary(), Constructs the comprehensive data dictionary matching ETAFlow standards., Generates package weights, volumes, counts, categories, and financial values., Computes realistic non-linear transit times, promised SLA days, actual delivery…

### Community 4 - "._generate_geography"
Cohesion: 0.29
Nodes (5): ndarray, Computes great-circle distances in kilometers between origin and destination…, Generates realistic origin/destination hubs, coordinates, and road-circuity…, Assigns transport mode, compatible vehicles, carriers, warehouse, and service…, vectorized_haversine()

## Knowledge Gaps
- **2 isolated node(s):** `Role and Boundaries`, `Workflow: graphify — Refresh Context Knower`
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LogisticsDatasetGenerator` connect `LogisticsDatasetGenerator` to `generate_dataset.py`, `DataFrame`, `._generate_geography`, `.generate`, `._generate_operational_and_environment`, `._generate_temporal_scaffold`, `._inject_controlled_missingness`?**
  _High betweenness centrality (0.244) - this node is a cross-community bridge._
- **Why does `validate_dataset()` connect `generate_dataset.py` to `DataFrame`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `create_dataset_metadata()` connect `generate_dataset.py` to `DataFrame`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **What connects `Role and Boundaries`, `Workflow: graphify — Refresh Context Knower` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._