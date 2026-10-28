# Credit Risk Assessment Using Machine Learning and SHAP

A machine-learning application that estimates loan default risk from applicant and loan information. It combines a trained classification model, a configurable decision threshold, and a FastAPI web service. The project repository also includes a Jupyter notebook for model development and SHAP-oriented analysis.

> **Disclaimer:** This is an educational/demo project, not a substitute for a regulated credit-scoring process or a lending decision made by a qualified institution.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
- [Input Features](#input-features)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run Locally](#run-locally)
- [Deployment](#deployment)
- [Explainability with SHAP](#explainability-with-shap)
- [Limitations and Responsible Use](#limitations-and-responsible-use)

## Overview

Credit risk assessment estimates the likelihood that a borrower will fail to repay a loan. This project uses applicant and loan attributes to estimate default probability and classify an application as **High Risk** or **Low Risk**.

The backend loads two serialized artifacts at startup:

- `credit_risk_model.pkl` — trained prediction model.
- `best_threshold.pkl` — decision threshold used to convert the estimated probability into a binary prediction.

The repository includes a Jupyter notebook (`Credit_Risk.ipynb`), a dataset (`credit_risk_dataset.csv`), a static frontend (`static/`), and a FastAPI backend (`main.py`).

## Key Features

- Predicts loan-default probability from applicant information.
- Converts the probability into a binary risk classification using a saved threshold.
- Validates prediction inputs with Pydantic.
- Serves predictions through a FastAPI `POST /predict` endpoint.
- Serves static frontend files from the same application.
- Includes a notebook for data analysis and model development.
- Includes SHAP-oriented explainability analysis in the project workflow.
- Includes a Render deployment configuration (`render.yaml`).

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas |
| Machine learning | Scikit-learn, XGBoost |
| Model serialization | Joblib |
| API framework | FastAPI |
| Request validation | Pydantic |
| Notebook environment | Jupyter Notebook |
| Explainable AI | SHAP |
| Frontend serving | FastAPI StaticFiles |
| Deployment configuration | Render |

The exact pinned dependencies are listed in `requirements.txt`.

## How It Works

1. Applicant and loan information is submitted to the API.
2. FastAPI validates the request against the `LoanApplication` schema.
3. The input is converted into a Pandas DataFrame.
4. The trained model estimates the probability of default.
5. The probability is compared with the saved decision threshold.
6. The API returns the probability, binary prediction, threshold, and risk label.

```text
Applicant / Loan Information
            |
            v
    FastAPI Request
        Validation
            |
            v
   Pandas Input DataFrame
            |
            v
   Trained ML Classifier
            |
            v
    Default Probability
            |
            v
 Compare with Saved Threshold
            |
            v
     High Risk / Low Risk
            |
            v
       JSON Response
```

## Input Features

The `POST /predict` endpoint expects these fields:

| Field | Type | Description |
|---|---|---|
| `person_age` | Integer | Applicant's age |
| `person_income` | Number | Applicant's income |
| `person_home_ownership` | String | Home ownership category |
| `person_emp_length` | Number | Employment length |
| `loan_intent` | String | Purpose of the loan |
| `loan_grade` | String | Loan grade |
| `loan_amnt` | Number | Requested loan amount |
| `loan_int_rate` | Number | Loan interest rate |
| `loan_percent_income` | Number | Loan amount as a proportion of income |
| `cb_person_default_on_file` | String | Previous-default indicator |
| `cb_person_cred_hist_length` | Integer | Length of credit history |

Use categories and value formats consistent with those used during model training. The request schema defines field types, but it does not by itself guarantee that every category or numeric value is valid for the trained model.

## API Reference

### `POST /predict`

Accepts applicant information and returns the estimated probability and risk classification.

**Example request**

```json
{
  "person_age": 28,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

This is an illustrative request only. Ensure the categories and value ranges match the dataset and preprocessing used to train the saved model.

**Example response**

```json
{
  "default_probability": 0.23,
  "default_prediction": 0,
  "threshold": 0.35,
  "Result": "Low Risk"
}
```

The response values above are examples, not measured output from the model. The actual probability and threshold depend on the saved model artifacts and the submitted input.

- `default_probability`: model-estimated probability of default.
- `default_prediction`: `1` when the probability is greater than or equal to the threshold; otherwise `0`.
- `threshold`: the loaded decision threshold.
- `Result`: `High Risk` when the prediction is `1`; otherwise `Low Risk`.

The backend also mounts the `static/` directory at `/`, allowing the application to serve its frontend files.

## Project Structure

```text
Credit-Risk-Assesment-using-SHAP/
├── static/                    # Frontend assets served by FastAPI
├── Credit_Risk.ipynb          # Notebook for analysis/model development
├── credit_risk_dataset.csv    # Credit-risk dataset
├── credit_risk_model.pkl      # Serialized trained model
├── best_threshold.pkl         # Saved classification threshold
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── runtime.txt                # Runtime configuration
├── render.yaml                # Render deployment configuration
└── .gitignore
```

## Installation

### Prerequisites

- Python version compatible with the pinned dependencies.
- Git.
- The model artifacts and dataset included in the repository.

### 1. Clone the repository

```bash
git clone https://github.com/tanishq-latent/Credit-Risk-Assesment-using-SHAP.git
cd Credit-Risk-Assesment-using-SHAP
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally

From the project root, start the FastAPI application:

```bash
uvicorn main:app --reload
```

By default, Uvicorn serves the application at `http://127.0.0.1:8000`.

- Frontend: http://127.0.0.1:8000/
- Interactive API documentation: http://127.0.0.1:8000/docs
- Alternative API documentation: http://127.0.0.1:8000/redoc
- Prediction endpoint: `POST http://127.0.0.1:8000/predict`

The model and threshold files must be available at the paths expected by `main.py` when the app starts. If startup fails, confirm that both `.pkl` files exist and can be loaded in your environment.

### Run the notebook

Open the notebook in Jupyter:

```bash
pip install notebook
jupyter notebook Credit_Risk.ipynb
```

Run the notebook cells in order. Review the notebook for its exact data preparation, model-training, evaluation, and SHAP analysis steps.

## Deployment

The repository contains a `render.yaml` file for Render deployment. Review that configuration before deploying and verify that:

- The configured build/install command installs `requirements.txt`.
- The start command runs the FastAPI application using the correct module path.
- The model artifacts are included in the deployment environment.
- Any required frontend assets are present.
- The deployed service can load the model and threshold at startup.

Do not assume that a deployment is active merely because a deployment configuration file exists.

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) is a method for understanding how input features contribute to a machine-learning prediction. It can be used to study overall feature importance and explain individual predictions.

This repository is named for SHAP-based credit-risk assessment and includes a notebook for model analysis. The inspected `POST /predict` implementation returns the probability, binary prediction, threshold, and risk label; it does **not** return SHAP values in its API response. To provide explanations in the live application, a SHAP explainer and a response/UI integration would need to be implemented and tested.

## Limitations and Responsible Use

- A risk label is a model output, not a definitive statement about a person's ability or willingness to repay.
- Model quality depends on the data, preprocessing, validation approach, and how well the training data represents future applicants.
- Decision thresholds affect false positives and false negatives; they should be selected with appropriate validation and business context.
- SHAP explanations describe model behavior; they do not prove causation or establish that a model is fair.
- Credit models require careful testing for bias, privacy, security, calibration, and compliance with applicable law.
- Do not use this demo as the sole basis for real lending decisions.

## Repository

Original project: https://github.com/tanishq-latent/Credit-Risk-Assesment-using-SHAP
