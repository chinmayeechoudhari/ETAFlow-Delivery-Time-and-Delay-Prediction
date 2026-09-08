# Graph Report - ETAFlow — Intelligent Delivery ETA & Delay Prediction Platform  (2026-09-08)

## Corpus Check
- 7 files · ~34,888 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 105 nodes · 163 edges · 13 communities (7 shown, 6 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b63af911`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- graphify — Dedicated Context Knower
- workflows/graphify.md
- generate_dataset.py
- LogisticsDatasetGenerator
- ETAFlow — Comprehensive Machine Learning Readiness Audit Report
- .run
- DataFrame
- DatasetAuditor
- 20. Categorized Recommendations & Next Steps
- audit_dataset.py
- 14. Data Leakage & Operational Horizon Analysis
- 9. Target Variable Audit (Regression & Classification)
- 19. ML Readiness Scorecard (17 Dimensions)

## God Nodes (most connected - your core abstractions)
1. `DatasetAuditor` - 27 edges
2. `ETAFlow — Comprehensive Machine Learning Readiness Audit Report` - 22 edges
3. `LogisticsDatasetGenerator` - 13 edges
4. `main()` - 6 edges
5. `validate_dataset()` - 5 edges
6. `create_dataset_metadata()` - 5 edges
7. `vectorized_haversine()` - 4 edges
8. `create_data_dictionary()` - 4 edges
9. `20. Categorized Recommendations & Next Steps` - 4 edges
10. `parse_args()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `DatasetAuditor`  [EXTRACTED]
  scripts/audit_dataset.py → scripts/audit_dataset.py  _Bridges community 7 → community 9_
- `main()` --calls--> `LogisticsDatasetGenerator`  [EXTRACTED]
  scripts/generate_dataset.py → scripts/generate_dataset.py  _Bridges community 3 → community 2_

## Import Cycles
- None detected.

## Communities (13 total, 6 thin omitted)

### Community 2 - "generate_dataset.py"
Cohesion: 0.23
Nodes (11): ndarray, create_dataset_metadata(), main(), parse_args(), Any, Namespace, Generates detailed JSON metadata documenting generation run and statistical…, Computes great-circle distances in kilometers between origin and destination… (+3 more)

### Community 3 - "LogisticsDatasetGenerator"
Cohesion: 0.14
Nodes (14): create_data_dictionary(), LogisticsDatasetGenerator, DataFrame, Constructs the comprehensive data dictionary matching ETAFlow standards., Production-grade synthetic dataset generator for ETAFlow. Leverages NumPy…, Executes the full vectorized generation pipeline., Generates order dates and calendar features with Indian holiday matching., Generates realistic origin/destination hubs, coordinates, and road-circuity… (+6 more)

### Community 4 - "ETAFlow — Comprehensive Machine Learning Readiness Audit Report"
Cohesion: 0.11
Nodes (18): 10. Target Consistency & Mathematical Coupling, 11. Feature-Target Relationships & Correlation, 12. Non-linear Relationship Evidence, 13. Categorical-Target Slices & Operational Dynamics, 15. Temporal Distribution & Seasonality, 16. Geographical & Network Route Integrity, 17. Multicollinearity & Feature Redundancy, 18. Synthetic Data Realism & ML Usability (+10 more)

### Community 8 - "20. Categorized Recommendations & Next Steps"
Cohesion: 0.50
Nodes (4): 20. Categorized Recommendations & Next Steps, ACCEPTABLE FOR NOW, MUST FIX BEFORE ML, SHOULD CONSIDER (Engineering Guidelines for Next Stage)

### Community 9 - "audit_dataset.py"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Namespace

### Community 10 - "14. Data Leakage & Operational Horizon Analysis"
Cohesion: 0.67
Nodes (3): 14. Data Leakage & Operational Horizon Analysis, Prediction Horizon Point A: Order Booking Time (Pre-Fulfillment), Prediction Horizon Point B: Vehicle Departure Time (Post-Dispatch)

### Community 11 - "9. Target Variable Audit (Regression & Classification)"
Cohesion: 0.67
Nodes (3): 9. Target Variable Audit (Regression & Classification), Classification Target: `is_delayed`, Regression Target: `actual_delivery_days`

## Knowledge Gaps
- **27 isolated node(s):** `Role and Boundaries`, `Workflow: graphify — Refresh Context Knower`, `1. Executive Summary`, `2. Dataset Overview`, `3. Audit Methodology` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ETAFlow — Comprehensive Machine Learning Readiness Audit Report` connect `ETAFlow — Comprehensive Machine Learning Readiness Audit Report` to `20. Categorized Recommendations & Next Steps`, `14. Data Leakage & Operational Horizon Analysis`, `9. Target Variable Audit (Regression & Classification)`, `19. ML Readiness Scorecard (17 Dimensions)`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `DatasetAuditor` connect `DatasetAuditor` to `audit_dataset.py`, `.run`, `DataFrame`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `LogisticsDatasetGenerator` connect `LogisticsDatasetGenerator` to `generate_dataset.py`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **What connects `Role and Boundaries`, `Workflow: graphify — Refresh Context Knower`, `1. Executive Summary` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `LogisticsDatasetGenerator` be split into smaller, more focused modules?**
  _Cohesion score 0.14130434782608695 - nodes in this community are weakly interconnected._
- **Should `ETAFlow — Comprehensive Machine Learning Readiness Audit Report` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._