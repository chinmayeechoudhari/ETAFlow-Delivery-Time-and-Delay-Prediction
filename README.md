# ETAFlow — Intelligent Delivery ETA & Delay Prediction Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-EB5424.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-2E8B57.svg)](https://lightgbm.readthedocs.io/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioned-945DD6.svg?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Code Quality: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ETAFlow** is an enterprise-grade Machine Learning platform designed to predict shipment delivery times (Continuous ETA in days) and delay risks (Binary Probability & Operational Delay Days) across complex multimodal supply chain networks in India.

Built around **Two-Horizon Operational Decision Points** (Order Booking vs. Vehicle Dispatch), ETAFlow completely eliminates data leakage while providing real-time operational recalibration across 28 major logistics hubs, 8 carriers, and 4 line-haul transit modes.

---

## 📌 Table of Contents

- [Key Architecture & Dual-Horizon Design](#-key-architecture--dual-horizon-design)
- [Project Directory Structure](#-project-directory-structure)
- [Dataset & ML-Readiness Highlights](#-dataset--ml-readiness-highlights)
- [Preprocessing & Feature Engineering Layer](#-preprocessing--feature-engineering-layer)
- [Milestone 6-A: Baseline ML & Benchmarking](#-milestone-6-a--baseline-ml-development--benchmarking)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Dataset Generation & Versioning](#dataset-generation--versioning)
  - [Running the Preprocessing Pipeline](#running-the-preprocessing-pipeline)
  - [Running Unit & Integration Tests](#running-unit--integration-tests)
- [API Serving Layer](#-api-serving-layer)
- [MLOps & Observability](#-mlops--observability)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Key Architecture & Dual-Horizon Design

Traditional ETA models suffer from **temporal feature leakage** by conditioning predictions on post-dispatch operational information during initial checkout/booking. ETAFlow solves this by decoupling prediction into two distinct operational horizons:

```
                                  SHIPMENT LIFECYCLE
                                  
  [ Order Booking ] ─────────────► [ Staging & Dispatch ] ─────────────► [ Final Delivery ]
         │                                   │                                    ▲
         ▼                                   ▼                                    │
  ┌───────────────────────┐           ┌───────────────────────┐                   │
  │  HORIZON A: BOOKING   │           │  HORIZON B: DISPATCH  │                   │
  │  Available:           │           │  Available:           │                   │
  │  • Origin / Dest Hubs │           │  • All Horizon A data │                   │
  │  • Distance / Weight  │           │  • Loading duration   │                   │
  │  • Carrier / Priority │           │  • Warehouse lag      │                   │
  │  • Scheduled SLA Days │           │  • Live Congestion    │                   │
  │  • Calendar/Seasonality│          │  • Weather Risk Score │                   │
  └──────────┬────────────┘           └──────────┬────────────┘                   │
             │                                   │                                │
             ▼                                   ▼                                │
  ┌───────────────────────┐           ┌───────────────────────┐                   │
  │ Initial ETA & Risk    │           │ Recalibrated ETA &    │                   │
  │ Promising (Point A)   │           │ Dynamic Alert (Point B)│──────────────────┘
  └───────────────────────┘           └───────────────────────┘
```

### Mathematical Consistency Guarantee
The data and modeling pipelines strictly enforce target coupling:
$$\text{delivery\_delay\_days} = \max(\text{actual\_delivery\_days} - \text{promised\_delivery\_days}, 0.0)$$
$$\text{is\_delayed} = 1 \iff \text{delivery\_delay\_days} > 0$$

---

## 📁 Project Directory Structure

```text
ETAFlow/
├── .agents/                    # Multi-agent autonomous workflow rules & graph context
├── .dvc/                       # DVC configuration & cache pointers
├── .github/                    # CI/CD GitHub Actions workflows
├── backend/                    # FastAPI production serving microservice
│   └── app/
│       ├── api/                # API routes & versioned endpoints
│       ├── core/               # Configuration, logging, security settings
│       ├── models/             # ORM / database domain models
│       ├── schemas/            # Pydantic v2 request & response schemas
│       └── services/           # Inference services, pipeline runners, cache
├── configs/                    # Declarative YAML feature specifications
│   ├── features_booking.yaml   # Prediction Point A (Booking Horizon) spec
│   └── features_dispatch.yaml  # Prediction Point B (Dispatch Horizon) spec
├── data/                       # DVC-managed data storage
│   ├── raw/                    # Raw shipments dataset (DVC tracked)
│   ├── processed/              # Engineered feature matrices & joblib transformers
│   └── reference/              # Data dictionary & schema metadata (Git tracked)
├── docker/                     # Containerization manifests (API, MLflow, Prometheus)
├── docs/                       # Architecture blueprints, data cards, model cards
├── frontend/                   # Interactive dashboard (Next.js / Vite UI)
├── ml/                         # Core Machine Learning codebase
│   ├── evaluation/             # Metrics calculation (MAE, RMSE, ROC-AUC, PR-AUC)
│   ├── explainability/         # SHAP attribution & global/local explanation generators
│   ├── features/               # Domain feature extractors (geospatial, temporal)
│   ├── preprocessing/          # Scikit-learn pipelines & custom transformers
│   │   ├── data_loader.py      # Validated data loading & schema conformance
│   │   ├── feature_engineering.py # Leakage-safe vector transformations
│   │   ├── pipeline.py         # Sklearn ColumnTransformer pipeline factory
│   │   ├── transformers.py     # Custom Scikit-learn BaseEstimator classes
│   │   └── validation.py       # Pydantic/pandera dataset validation suite
│   └── training/               # Model training harnesses (XGBoost, LightGBM, CatBoost)
├── models/                     # Serialized model checkpoints & production artifacts
├── monitoring/                 # Continuous MLOps Observability
│   ├── evidently/              # Data & concept drift test suites
│   ├── grafana/                # Grafana dashboards for latency & drift tracking
│   └── prometheus/             # Prometheus metrics exporters & alerts
├── notebooks/                  # Exploratory Data Analysis & experimentation
├── pipelines/                  # Orchestration DAGs (Apache Airflow / Prefect)
├── reports/                    # Generated audit reports, figures & evaluation logs
│   └── dataset_audit/          # 17-point ML-readiness statistical audit & visualizations
├── scripts/                    # Automation scripts
│   ├── audit_dataset.py        # 17-dimensional non-destructive dataset audit
│   ├── generate_dataset.py     # High-fidelity synthetic Indian logistics generator
│   └── preprocess_dataset.py   # Dual-horizon preprocessing execution script
├── tests/                      # Automated test suite
│   ├── api/                    # Endpoint & integration tests
│   ├── unit/                   # Transformer & pipeline unit tests
│   └── conftest.py             # Pytest fixtures & sample data generators
├── .env.example                # Environment variables template
├── .gitattributes              # Git line endings & merge drivers
├── .gitignore                  # Production-grade Git ignore file
├── requirements.txt            # Python production & development dependencies
└── README.md                   # Platform documentation
```

---

## 📊 Dataset & ML-Readiness Highlights

The platform includes a specialized generator (`scripts/generate_dataset.py`) producing 100,000 realistic shipment trajectories across India:

- **Volume**: 100,000 shipments × 55 attributes covering 2023–2025.
- **Geographic Network**: 28 verified Indian commercial hubs (Delhi, Mumbai, Bengaluru, Kolkata, Chennai, Hyderabad, etc.) with real Haversine & highway circuity bounds ($1.22\times$ to $1.45\times$).
- **Realistic Delay Rate**: Natural **26.27%** delay incidence without synthetic oversampling artifacts.
- **Statistical Audit**: Fully verified across 17 dimensions with **0 target violations**, strict chronological ordering ($t_{\text{order}} \le t_{\text{pickup}} \le t_{\text{dispatch}} \le t_{\text{delivery}}$), and audited sensor missingness mechanisms.

Full audit results are accessible in [`reports/dataset_audit/dataset_audit_report.md`](reports/dataset_audit/dataset_audit_report.md).

---

## ⚙️ Preprocessing & Feature Engineering Layer

The preprocessing subsystem (`ml/preprocessing/`) standardizes transformations into reusable, serializable Scikit-Learn pipelines:

| Horizon | Config File | Input Features | Transformed Features | Serialized Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Point A (Booking)** | [`configs/features_booking.yaml`](configs/features_booking.yaml) | 38 features | **81 features** | `data/processed/booking_pipeline.joblib` |
| **Point B (Dispatch)** | [`configs/features_dispatch.yaml`](configs/features_dispatch.yaml) | 49 features | **92 features** | `data/processed/dispatch_pipeline.joblib` |

### Custom Scikit-Learn Transformers
1. **`HolidaySanitizer`**: Converts structural nulls in seasonal holiday features into explicit `"No_Holiday"` categories.
2. **`MissingIndicatorAdder`**: Generates informative binary flags (`*_missing`) for operational sensor dropouts (congestion, weather).
3. **`TemporalFeatureExtractor`**: Decomposes ISO timestamps into cyclical order hours, day of week, and seasonal indices.

---

## 🔬 Milestone 6-A — Baseline ML Development & Benchmarking

Milestone 6-A establishes an empirical baseline for delivery-time regression (`actual_delivery_days`) and binary delay risk classification (`is_delayed`) across both operational horizons.

### 1. Model Families Benchmarked
- **Level 0 (Heuristic)**: `DummyRegressor` (mean), `DummyClassifier` (prior)
- **Level 1 (Linear)**: `Ridge Regression`, `Logistic Regression` (StandardScaler conditioned)
- **Level 2 (Bagging)**: `RandomForestRegressor`, `RandomForestClassifier`
- **Level 3 (Boosting)**: `LightGBM Regressor / Classifier`, `XGBoost Regressor / Classifier`

### 2. Leakage-Safe Chronological Split
Records are strictly partitioned chronologically by `order_date` without temporal lookahead:
- **Train (70%)**: 70,000 shipments (`2023-01-01` to `2025-02-05`)
- **Validation (15%)**: 15,000 shipments (`2025-02-05` to `2025-07-18`)
- **Test (15%)**: 15,000 shipments (`2025-07-18` to `2025-12-30`)
- **Leakage Prevention**: All preprocessing transformers are fitted **strictly on the Train partition** before transforming Validation and Test sets.

### 3. Empirical Benchmark Results (Test Set)

#### Delivery-Time Regression (`actual_delivery_days` in days):
| Horizon | Model | MAE | RMSE | R² | Within ±1 Day | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Booking** | **LightGBM** | **0.517** | **0.806** | **0.738** | **87.6%** | 0.46s |
| Booking | XGBoost | 0.519 | 0.810 | 0.736 | 87.7% | 0.65s |
| Booking | Random Forest | 0.545 | 0.847 | 0.711 | 86.3% | 10.45s |
| Booking | Ridge | 0.605 | 0.913 | 0.664 | 86.0% | 0.44s |
| Booking | Dummy | 1.176 | 1.582 | -0.010 | 49.5% | <0.01s |
| **Dispatch** | **LightGBM** | **0.153** | **0.278** | **0.969** | **98.7%** | 0.55s |
| Dispatch | XGBoost | 0.157 | 0.285 | 0.967 | 98.6% | 0.69s |
| Dispatch | Random Forest | 0.222 | 0.365 | 0.946 | 97.7% | 13.37s |
| Dispatch | Ridge | 0.345 | 0.563 | 0.872 | 93.7% | 0.45s |
| Dispatch | Dummy | 1.176 | 1.582 | -0.010 | 49.5% | <0.01s |

#### Delay Risk Classification (`is_delayed` binary):
| Horizon | Model | F1 Score | PR-AUC | ROC-AUC | Precision | Recall | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Booking** | **XGBoost** | **0.8245** | **0.9274** | **0.9632** | 0.8737 | 0.7806 | 0.60s |
| Booking | LightGBM | 0.8235 | 0.9278 | 0.9635 | 0.8657 | 0.7852 | 0.48s |
| Booking | Logistic Regression | 0.8177 | 0.9167 | 0.9557 | 0.8519 | 0.7861 | 2.88s |
| Booking | Random Forest | 0.7643 | 0.9130 | 0.9532 | 0.9256 | 0.6508 | 0.96s |
| Booking | Dummy | 0.0000 | 0.2917 | 0.5000 | 0.0000 | 0.0000 | <0.01s |
| **Dispatch** | **LightGBM** | **0.9476** | **0.9888** | **0.9939** | 0.9623 | 0.9333 | 0.59s |
| Dispatch | XGBoost | 0.9466 | 0.9879 | 0.9937 | 0.9640 | 0.9298 | 0.72s |
| Dispatch | Logistic Regression | 0.9375 | 0.9827 | 0.9913 | 0.9458 | 0.9294 | 2.67s |
| Dispatch | Random Forest | 0.8738 | 0.9712 | 0.9857 | 0.9702 | 0.7948 | 1.21s |
| Dispatch | Dummy | 0.0000 | 0.2917 | 0.5000 | 0.0000 | 0.0000 | <0.01s |

### 4. Booking vs. Dispatch Comparative Takeaways
1. **Error Reduction**: Dispatch information achieves a **70.37% relative reduction in ETA MAE** (dropping from 0.517 to 0.153 days).
2. **Within-Day Reliability**: Consignments arriving within ±1 day of predicted ETA increases from **87.6% to 98.7%**.
3. **Delay Discrimination**: F1 score jumps by **+14.92%** (from 0.8245 to 0.9476) and PR-AUC reaches **0.9888**.
4. **Key Influencers**: Dispatch-stage telematics (congestion index, weather risk score, loading durations) provide the critical delta separating high-risk shipments from on-time consignments.

Full benchmark report and feature importance rankings: [`reports/model_benchmark/model_benchmark_report.md`](reports/model_benchmark/model_benchmark_report.md).

---

## 🚀 Getting Started

### Prerequisites
- **Python**: `3.10` or higher (`3.11` / `3.12` / `3.13` supported)
- **Git**: `2.30+`
- **DVC**: Version `3.0+`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chinmayeechoudhari/ETAFlow-Delivery-Time-and-Delay-Prediction.git
   cd "ETAFlow — Intelligent Delivery ETA & Delay Prediction Platform"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Environment Configuration

Copy the example environment configuration:
```bash
cp .env.example .env
```
Update `.env` with your PostgreSQL database credentials, MLflow tracking URI, and application settings.

### Dataset Generation & Versioning

If starting from scratch without downloaded data, generate the 100,000-row synthetic dataset:
```bash
python scripts/generate_dataset.py
```

Track the raw dataset using DVC:
```bash
dvc add data/raw/shipments.csv
git add data/raw/shipments.csv.dvc .gitignore
```

### Running the Preprocessing Pipeline

Run the dual-horizon preprocessing engine to generate training matrices and serialized pipelines:
```bash
python scripts/preprocess_dataset.py
```

Outputs produced in `data/processed/`:
- `booking_features.parquet` & `booking_pipeline.joblib` (Point A)
- `dispatch_features.parquet` & `dispatch_pipeline.joblib` (Point B)
- `targets.parquet` (Coupled regression & classification ground truths)
- `feature_manifest.json` (Full cryptographic & schema lineage manifest)

### Running Baseline Model Benchmarking

Execute the full baseline model suite (Dummy, Ridge/LogisticRegression, Random Forest, LightGBM, XGBoost) with chronological splitting:
```bash
# Benchmark all prediction horizons (Booking & Dispatch)
python scripts/train_baseline.py --all

# Or benchmark an individual horizon
python scripts/train_baseline.py --horizon booking
python scripts/train_baseline.py --horizon dispatch
```

Outputs produced in `models/baseline/` and `reports/model_benchmark/`:
- `models/baseline/{booking,dispatch}/regression/*.joblib`
- `models/baseline/{booking,dispatch}/classification/*.joblib`
- `models/baseline/{booking,dispatch}/feature_pipeline.joblib`
- `reports/model_benchmark/regression_results.csv`
- `reports/model_benchmark/classification_results.csv`
- `reports/model_benchmark/benchmark_summary.json`
- `reports/model_benchmark/model_benchmark_report.md`

### Running Unit & Integration Tests

Execute the automated test suite with pytest:
```bash
pytest -v
```

Run test suite with coverage report:
```bash
pytest --cov=ml --cov-report=term-missing tests/
```

---

## 🌐 API Serving Layer

ETAFlow delivers high-throughput, low-latency inference using FastAPI.

### Start the Local API Server
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8000/docs`.

### Prediction Endpoints

#### 1. Point A: Booking Horizon Prediction
```http
POST /api/v1/predict/booking
Content-Type: application/json

{
  "origin_city": "Mumbai",
  "destination_city": "Bengaluru",
  "distance_km": 984.5,
  "package_weight_kg": 3.2,
  "transport_mode": "Road",
  "carrier": "Delhivery",
  "priority_level": "High",
  "scheduled_sla_days": 3.0
}
```

**Response:**
```json
{
  "horizon": "booking",
  "predicted_eta_days": 2.45,
  "delay_probability": 0.14,
  "risk_level": "LOW",
  "confidence_interval_95": [2.10, 2.80]
}
```

#### 2. Point B: Post-Dispatch Recalibration
```http
POST /api/v1/predict/dispatch
Content-Type: application/json

{
  "shipment_id": "SHP-2025-08492",
  "booking_features": { ... },
  "warehouse_processing_hours": 9.4,
  "loading_time_hours": 3.1,
  "congestion_index": 0.84,
  "weather_risk_score": 0.72
}
```

---

## 📈 MLOps & Observability

- **Data Version Control (DVC)**: Guarantees reproducibility of raw datasets and intermediate matrices without storing heavy binary blobs in Git.
- **MLflow**: Tracks hyperparameter tuning experiments, metrics (MAE, RMSE, Log-Loss, F1), and model artifacts.
- **Evidently AI**: Daily batch monitoring jobs compare production inference data against baseline training distributions to detect covariate drift and target shift.
- **Prometheus & Grafana**: Live service metrics (P99 inference latency, request throughput, error rates) exposed at `/metrics`.

---

## 🤝 Contributing

Contributions are welcomed! Follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'feat: add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Please ensure all tests pass (`pytest`) and code conforms to Ruff formatting before submitting PRs.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
