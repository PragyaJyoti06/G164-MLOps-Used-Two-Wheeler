# G164 - MLOps Mini Project

## Project Information

- Course: SCSE3040 - MLOps
- Group ID: G164
- Problem ID: MP233
- Project Title: Used Two-Wheeler Price Prediction
- Location: Guwahati

## Group Members

| Name | Enrollment Number |
|---|---|
| Abhishek Raj | S24CSEU2115 |
| Dikshant | S24CSEU2138 |
| Pratik Mishra | S24CSEU2116 |
| Pragya Jyoti | S24CSEU2122 |

## Project Objective

Develop an end-to-end MLOps pipeline to predict the resale price of used two-wheelers in Guwahati using machine learning.

The project covers data versioning, data cleaning, model training, experiment tracking, model registration, API development, containerization, automated testing, Kubernetes configuration, monitoring, model retraining and holdout prediction.

## Target Variable and Evaluation Metric

- Target: `resale_thousand_rs`
- Evaluation metric: Mean Absolute Error (MAE)
- Prediction unit: Thousand Indian Rupees

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- MLflow with SQLite backend
- DVC
- FastAPI
- Docker
- Kubernetes
- Pytest
- GitHub Actions

## Dataset

The project uses three datasets:

| Dataset | Records | Purpose |
|---|---:|---|
| train.csv | 1,033 | Initial training data |
| later.csv | 300 | Temporal monitoring and retraining |
| holdout_features.csv | 250 | Final prediction generation |

The original training dataset SHA-256 is logged in MLflow runs for data provenance.

## Data Cleaning and Investigation

The cleaning pipeline performs the following operations:

- Removes exact duplicate rows.
- Converts the `service_records` sentinel value `-1` into missing data.
- Handles suspicious repeated `km_driven` values.
- Corrects identified target-value decimal-place errors.
- Removes `final_price_from_rc_transfer` because it is a potential target leakage column.
- Preserves the remaining missing values for pipeline-based imputation.

### Data Quality Findings

| Issue | Finding |
|---|---|
| Potential leakage column | `final_price_from_rc_transfer` |
| Sentinel values | 33 |
| Exact duplicate rows | 33 |
| Repeated `km_driven` sequence | 129 consecutive rows |
| Suspicious target values | 11 |
| Unit mix-up | Not detected |
| Spelling variants | Not detected |

The cleaned training dataset contains 1,000 rows.

Detailed reports are available in the `reports/` directory.

## Model Training and Experiment Tracking

The training pipeline reads its configuration from `params.yaml`.

The model uses a preprocessing pipeline with:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical features.
- Random Forest regression.

MLflow tracks experiments, parameters, metrics, data SHA-256 and model artifacts.

The registered model is:

`mp233_g164_used_two_wheeler`

The model registry uses the `champion` alias to identify the production candidate.

## Model Input Features

The API and model use these eight features:

1. `age_years`
2. `km_driven`
3. `engine_cc`
4. `owners`
5. `service_records`
6. `accidents`
7. `brand_tier`
8. `city_zone`

The target and potential leakage column are not used as model inputs.

## API

The prediction service is implemented using FastAPI.

| Endpoint | Purpose |
|---|---|
| `/health` | Health check |
| `/predict` | Predict used two-wheeler resale price |

Input validation rejects invalid or missing feature values.

## Docker and Kubernetes

The application includes:

- A Dockerfile for containerization.
- A `.dockerignore` file.
- Kubernetes Deployment and Service manifests in `k8s/`.

No live Kubernetes cluster deployment is claimed as part of this project.

## Automated Testing and CI

Pytest covers API behavior, input validation, data quality and model registry checks.

The latest local test run completed successfully:

- Total tests: 12
- Passed: 12
- Failed: 0

GitHub Actions runs the test suite through the configured CI workflow.

## Model Monitoring

The monitoring pipeline compares the cleaned training data with the later dataset using Population Stability Index (PSI).

The PSI calculation uses 10 quantile-based bins for numerical features.

### Observed PSI Results

| Feature | PSI |
|---|---:|
| age_years | 5.783528 |
| km_driven | 1.505996 |
| service_records | 0.389983 |
| engine_cc | 0.092787 |
| owners | 0.004121 |
| accidents | 0.003140 |

The most drifted feature was `age_years`.

### V1 Performance by City Zone

| City Zone | V1 MAE |
|---|---:|
| Central | 23.881079 |
| South | 16.023676 |
| North | 15.225275 |

Central had the highest observed MAE.

## Chronological Retraining and Model Promotion

The later dataset was sorted chronologically and divided into:

- First 70%: 210 rows for V2 training.
- Last 30%: 90 rows for evaluation.

The promotion threshold is configured in `params.yaml` and requires at least an 8% reduction in MAE.

| Metric | Result |
|---|---:|
| V1 check MAE | 18.766689 |
| V2 check MAE | 11.586422 |
| MAE improvement | 38.26% |
| Required improvement | 8% |

Since V2 exceeded the required improvement, it was promoted as the MLflow champion version 2.

The exported champion model is stored at `models/model.joblib`.

## Holdout Predictions

The promoted V2 model generates predictions for the 250 records in `data/raw/holdout_features.csv`.

Output:

`predictions/holdout_predictions.csv`

The output contains:

- `record_id`
- `date`
- `predicted_resale_thousand_rs`

All 250 records received predictions, with no missing predictions or duplicate record IDs.

## Project Structure

```text
G164-MLOps-Used-Two-Wheeler/
|-- app/
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|-- predictions/
|-- reports/
|-- src/
|   |-- mp233/
|-- tests/
|-- k8s/
|-- .github/
|   |-- workflows/
|-- Dockerfile
|-- .dockerignore
|-- dvc.yaml
|-- dvc.lock
|-- .dvc/
|-- mlflow.db
|-- params.yaml
|-- requirements.txt
|-- README.md
```

## Reproducing the Project

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the project scripts:

```powershell
python src/mp233/cleaning.py
python src/mp233/train.py
python src/mp233/monitor.py
python src/mp233/retrain.py
python src/mp233/predict_holdout.py
python -m pytest -v
```

Start the API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000

```
