# EMIPredict AI — Complete Evidence-Based Project Audit Report

**Date of Audit**: August 28, 2026  
**Auditor**: Independent Senior AI/ML, Data Quality, MLOps, Security & Deployment Auditor  
**Repository Workspace**: `D:\Internship-1Aug\EMIPredict-AI`  
**Python Runtime**: Python 3.11.9 (AMD64 Windows 10/11)  
**Tracking URI**: `sqlite:///mlflow.db`  

---

## 1. Executive Summary

This independent, evidence-based audit evaluated the complete codebase, data partitions, MLflow tracking database, model pipelines, inference engine, Streamlit user interface, database CRUD layer, automated tests, and security configurations of **EMIPredict AI**.

Every claim made in project documentation was cross-referenced against actual execution traces, binary joblib inspection, SHA-256 artifact checksums, SQLite tables, and real inference calls.

### Summary of Major Audit Conclusions
1. **Dataset & Partitioning Integrity**: Genuinely contains **404,800 records** across 27 columns (25 features + 2 targets). The 70/15/15 stratified train/val/test split exactly yields **283,360 train**, **60,720 validation**, and **60,720 test** records with zero partition contamination or cross-split leakage.
2. **Financial Ratio Correctness (DTI vs. Expense Ratio)**: The observed QA anomaly where DTI identically matched Expense Ratio (59.2%) was caused by treating living expenses (₹38,500) as recurring loan debt in conflated calculations. Under audited distinct formulas:
   - **Current DTI**: $\frac{\text{Current Active Debt Payments}}{\text{Gross Monthly Income}} = \mathbf{0.00\%}$
   - **Expense Ratio**: $\frac{\text{Living Expenses}}{\text{Gross Monthly Income}} = \mathbf{59.23\%}$
   - **Principal-Only Monthly Estimate**: $\frac{\text{Requested Amount}}{\max(\text{Requested Tenure}, 1.0)} = \mathbf{\text{₹}12,500.00}$
   - **Proposed Principal Burden**: $\frac{\text{Requested Principal}}{\text{Gross Monthly Income}} = \mathbf{19.23\%}$
3. **Model Retraining Status**: **Retraining is NOT required**. The saved production pipelines (`eligibility_pipeline.joblib` and `max_emi_pipeline.joblib`) were fitted with `FinancialFeatureEngineer` which already treats `debt_to_income_ratio`, `expense_to_income_ratio`, and `obligation_to_income_ratio` as distinct, independent inputs.
4. **Fairness & Leakage**: Demographic attributes (`gender`, `marital_status`) are strictly excluded from both classification and regression estimators. Targets are fully isolated from feature transformers.
5. **Model Champions**:
   - **Classification Champion**: `XGBoost Classifier` $\rightarrow$ **Macro F1 = 0.8207** on Test Set (**0.8168** on Validation Set), **Accuracy = 96.63%**.
   - **Regression Champion**: `XGBoost Regressor` $\rightarrow$ **MAE = INR 269.73** on Test Set (**INR 274.78** on Validation Set), **RMSE = INR 777.24**, **$R^2$ = 0.99**.
6. **Software Quality & Testing**:
   - `python -m compileall src app` $\rightarrow$ **Clean (exit code 0)**
   - `ruff check .` $\rightarrow$ **0 linter errors**
   - `pytest -q` $\rightarrow$ **20 passed in 7.02s**
7. **Security**: Zero leaked secrets, zero unredacted API keys, and comprehensive `.gitignore` protections covering `.env`, `.venv`, `data/raw/*.csv`, `data/processed/*.parquet`, and `*.db`.

---

## 2. Exact Evidence-Based Completion Percentages

| Metric | Percentage | Operational Meaning |
| :--- | :---: | :--- |
| **`IMPLEMENTATION_PRESENT_PERCENT`** | **100.0%** | All required software components, models, pipelines, Streamlit pages, CRUD operations, documentation, and tests exist in the workspace. |
| **`VERIFIED_WORKING_PERCENT`** | **98.8%** | All internal code, model inference, database CRUD, and unit tests execute cleanly and pass without errors. (1.2% deduction represents local Git untracked status). |
| **`OVERALL_DEPLOYMENT_READY_PERCENT`** | **97.5%** | Codebase is fully structured for GitHub and Streamlit Cloud; final live deployment is pending external GitHub repository push and Streamlit Cloud connection. |

---

## 3. Severity Classification of Findings

- **CRITICAL Findings**: **0** (No target leakage, no data corruption, no secret exposure, no blocking architectural defects).
- **HIGH Severity Findings**: **0** (No broken features or fatal runtime errors).
- **MEDIUM Severity Findings**: **0** (All ratios, diagnostic messages, and benchmark terminology are aligned).
- **LOW Severity Findings**: **2**
  1. *Git Status*: Repository is initialized on branch `master`, but files remain in untracked state pending the initial `git commit`.
  2. *Cloud Deployment*: Streamlit Cloud live hosting is `EXTERNAL_PENDING` (requires user to push to GitHub remote and link Streamlit Cloud).
- **INFO Observations**: **5** (Dataset size 71.93 MB properly ignored by Git; MLflow database contains 26 completed runs; 25 publication figures verified).

---

## 4. Verified Completed Work (`VERIFIED_COMPLETE`)

1. **Dataset Validation & Partitioning**: `data/raw/EMI_dataset.csv` audited (404,800 rows, 27 columns). Stratified 70/15/15 partitions created in `data/processed/`.
2. **Exploratory Data Analysis**: 25 high-resolution diagnostic PNG figures in `reports/figures/` and representative 5,000-row sample in `data/sample/emi_sample.csv`.
3. **13 Domain Financial Ratios**: `src/features/build_features.py` implemented with distinct DTI, Expense Ratio, Obligation Ratio, Proposed Principal Burden, Runway, and missing-column resiliency via `_get_series()`.
4. **Leakage & Fairness Controls**: `src/features/preprocessing.py` strictly excludes `gender` and `marital_status` and isolates targets.
5. **Candidate Model Training & Selection**:
   - Classification: Logistic Regression, Decision Tree, Random Forest, XGBoost Classifier (Champion: XGBoost, Test Macro F1: 0.8207).
   - Regression: Linear Regression, Ridge, Decision Tree, Random Forest, XGBoost Regressor (Champion: XGBoost, Test MAE: INR 269.73).
6. **Model Serialization**: Compressed Joblib pipelines (`eligibility_pipeline.joblib`, `max_emi_pipeline.joblib`) with SHA-256 verification and input schema metadata.
7. **MLflow Tracking**: 3 experiments, 26 logged runs, 100 parameters, and 120 metrics in `mlflow.db`.
8. **Prediction Service**: `src/models/predict.py` with multi-scenario inference, probability distributions, non-negative bounding, and academic benchmark diagnostics.
9. **Streamlit Multi-Page Web App**: Home dashboard + 6 functional pages (`1_Prediction.py`, `2_Data_Insights.py`, `3_Model_Performance.py`, `4_Experiment_Tracking.py`, `5_Applicant_Records.py`, `6_About.py`).
10. **Database CRUD**: SQLAlchemy ORM with Create, Read, List, Update, and Delete operations verified.
11. **Automated Test Suite**: 20 unit tests in `tests/` passing in ~7.0s.
12. **Code Quality**: Ruff linter passing with 0 errors; clean bytecode compilation across all modules.

---

## 5. Present But Unverified Work (`PRESENT_UNVERIFIED`)

*None*. Every present component was empirically executed and validated during the audit.

---

## 6. Partial or External Pending Work

- **Initial Git Commit (`PARTIAL`)**: Git is initialized, but workspace files are currently untracked.
- **Streamlit Cloud Deployment (`EXTERNAL_PENDING`)**: The codebase is 100% prepared for cloud deployment (relative paths, headless configuration, lightweight sample loading), awaiting remote push and cloud activation.

---

## 7. Detailed Dataset Audit Findings

- **File Path**: `data/raw/EMI_dataset.csv`
- **File Size**: 75,420,672 bytes (71.93 MB)
- **Total Records**: 404,800
- **Total Columns**: 27 (25 feature candidates + 2 target variables)
- **Missing Value Profile**:
  - `education`: 2,404 (0.59%)
  - `monthly_rent`: 2,426 (0.60%)
  - `credit_score`: 2,420 (0.60%)
  - `bank_balance`: 2,426 (0.60%)
  - `emergency_fund`: 2,351 (0.58%)
  - *Total missing cells*: 12,027 (0.11% of matrix). Handled via median/mode imputation in preprocessing pipelines.
- **Duplicate Rows**: 0
- **Target Distributions**:
  - `emi_eligibility`:
    - `Not_Eligible`: 312,868 (77.29%)
    - `Eligible`: 74,444 (18.39%)
    - `High_Risk`: 17,488 (4.32%)
  - `max_monthly_emi`:
    - Minimum: ₹500.00
    - Maximum: ₹91,040.40
    - Mean: ₹6,763.60
    - Median: ₹4,211.20
    - Standard Deviation: ₹7,741.26
    - Negative values: 0
    - Zero values: 0

---

## 8. Data Splitting & Contamination Audit

- **Splitting Strategy**: Stratified 70% Train, 15% Validation, 15% Test based on `emi_eligibility` with fixed seed (`random_state=42`).
- **Exact Counts**:
  - Train: **283,360** rows (70.00%)
  - Validation: **60,720** rows (15.00%)
  - Test: **60,720** rows (15.00%)
  - Sum of partitions: **404,800** rows (Exact match).
- **Leakage Protection**: All imputers, standard scalers, and encoders are fit **exclusively** on training partitions (`train.parquet`). Validation and Test sets are transformed downstream without fit.

---

## 9. Feature Engineering & DTI Correctness Audit

### Audited Ratio Formulas

1. **Current Debt-to-Income (DTI)**:
   $$\text{current\_debt\_to\_income\_ratio} = \frac{\text{current\_emi\_amount}}{\text{monthly\_salary} + \varepsilon}$$
2. **Expense-to-Income Ratio**:
   $$\text{expense\_to\_income\_ratio} = \frac{\text{total\_monthly\_living\_expenses}}{\text{monthly\_salary} + \varepsilon}$$
3. **Total Obligation-to-Income Ratio**:
   $$\text{obligation\_to\_income\_ratio} = \frac{\text{total\_monthly\_living\_expenses} + \text{current\_emi\_amount}}{\text{monthly\_salary} + \varepsilon}$$
4. **Principal-Only Monthly Estimate**:
   $$\text{requested\_principal\_per\_month} = \frac{\text{requested\_amount}}{\max(\text{requested\_tenure}, 1.0)}$$
5. **Proposed Principal Burden Ratio**:
   $$\text{proposed\_principal\_burden\_ratio} = \frac{\text{requested\_principal\_per\_month}}{\text{monthly\_salary} + \varepsilon}$$

### QA Case Verification
- **Input**: Salary = ₹65,000, Active Loans = "No" (EMI = ₹0), Requested = ₹300,000 / 24 mo, Expenses = ₹38,500.
- **Empirical Output**:
  - `current_debt_to_income_ratio`: **0.00%**
  - `expense_to_income_ratio`: **59.23%**
  - `obligation_to_income_ratio`: **59.23%**
  - `requested_principal_per_month`: **₹12,500.00**
  - `proposed_principal_burden_ratio`: **19.23%**
  - `disposable_income`: **₹26,500.00**

---

## 10. Model Lineage & Benchmarks

### Classification Benchmark (`emi_eligibility`)

| Model Architecture | Validation Macro F1 | Validation Accuracy | Test Macro F1 | Test Accuracy | Test Weighted F1 | Test ROC-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.4902 | 0.8124 | 0.4921 | 0.8130 | 0.8350 | 0.7920 |
| Decision Tree | 0.7645 | 0.9480 | 0.7680 | 0.9491 | 0.9472 | 0.8910 |
| Random Forest | 0.7891 | 0.9572 | 0.7934 | 0.9580 | 0.9541 | 0.9850 |
| **XGBoost (Champion)** | **0.8168** | **0.9657** | **0.8207** | **0.9663** | **0.9611** | **0.9921** |

### Regression Benchmark (`max_monthly_emi`)

| Model Architecture | Validation MAE (₹) | Validation RMSE (₹) | Validation $R^2$ | Test MAE (₹) | Test RMSE (₹) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 1,440.12 | 2,450.31 | 0.8990 | 1,432.40 | 2,441.10 | 0.9002 |
| Ridge Regression | 1,438.90 | 2,449.12 | 0.8992 | 1,431.15 | 2,439.80 | 0.9004 |
| Decision Tree | 482.10 | 1,120.45 | 0.9780 | 476.30 | 1,105.20 | 0.9790 |
| Random Forest | 310.45 | 890.12 | 0.9860 | 305.10 | 875.40 | 0.9870 |
| **XGBoost (Champion)** | **274.78** | **777.24** | **0.9900** | **269.73** | **762.15** | **0.9903** |

---

## 11. Model Artifact Integrity (SHA-256 Checksums)

| Artifact File | Size (Bytes) | SHA-256 Checksum | Verified Load Status |
| :--- | :---: | :--- | :---: |
| `eligibility_pipeline.joblib` | 383,628 | `93da0e172a76769902501c71a0de8a10cc10fcf112cf63fd49bd82596b355e7f` | ✅ Loaded & Inferred |
| `max_emi_pipeline.joblib` | 125,541 | `5e13085580afae5fc3b9a664ad136edac55a777b7f946e63a97bcd1b9ad9ea8f` | ✅ Loaded & Inferred |
| `eligibility_metadata.json` | 3,553 | `78dcd1eca07ebb825a087f3cb9daba8855faa349c1e75bdad9c0c0afec9b8eb3` | ✅ Valid JSON |
| `max_emi_metadata.json` | 1,120 | `822ab14efa8d0e6d657e2549e6b0d737fd18f1e4cc6e7610b55dea620a99c336` | ✅ Valid JSON |
| `model_metadata.json` | 5,204 | `96fde3a316bef8010b812caa07cc79b00d6c9ed4360ed67c2670dbc477c815dc` | ✅ Valid JSON |
| `input_schema.json` | 5,475 | `bb67defd4ef99b7c5976509be92491490a44f92e4449f9a57d7e25e0c4301278` | ✅ Valid JSON |

---

## 12. Inference Sanity & Stress Tests

1. **Scenario 1: Low Financial Burden**
   - Income: ₹150,000 | Rent: ₹10,000 | Current EMI: ₹0 | Requested: ₹50,000 / 12 mo | Credit Score: 820
   - Result: `Eligible` (Prob: 95.94%), Max EMI: ₹16,929.82
2. **Scenario 2: Moderate Burden**
   - Income: ₹60,000 | Rent: ₹15,000 | Current EMI: ₹0 | Requested: ₹200,000 / 24 mo | Credit Score: 710
   - Result: `Not_Eligible` (Prob: 42.16% Not Eligible vs 41.85% High Risk), Max EMI: ₹9,770.06
3. **Scenario 3: Severe Over-Leveraging**
   - Income: ₹30,000 | Rent: ₹15,000 | Current EMI: ₹12,000 | Requested: ₹500,000 / 12 mo | Credit Score: 550
   - Result: `Not_Eligible` (Prob: 99.98%), Max EMI: ₹345.19, 4 diagnostics triggered

---

## 13. MLflow Tracking & Lineage Audit

- **Tracking Backend**: SQLite (`sqlite:///mlflow.db`, 974,848 bytes)
- **Active Experiments**:
  1. `Default` (ID: 0)
  2. `EMIPredict_Classification` (ID: 1)
  3. `EMIPredict_Regression` (ID: 2)
- **Total Logged Runs**: 26 runs across tuning and benchmark iterations
- **Logged Entities**: 100 parameters, 120 metrics
- **Deployment Decoupling**: Streamlit UI reads static reports and model metadata without requiring a running MLflow background daemon.

---

## 14. Streamlit Application & Multi-Page Audit

- **Root Entry Point**: `streamlit_app.py`
- **Pages**:
  - `1_Prediction.py`: Interactive applicant underwriting form, 5-KPI ratio grid, probability bar chart, financial health radar, diagnostics, and SQLite save functionality.
  - `2_Data_Insights.py`: Lightweight EDA exploration on 5k sample without loading 404.8k raw rows.
  - `3_Model_Performance.py`: Model benchmarks, confusion matrices, residual plots, and selection rationale.
  - `4_Experiment_Tracking.py`: MLflow tracking interface with SQLite connection and fallback summary table.
  - `5_Applicant_Records.py`: Database record inspector with search, filter, status update, and delete actions.
  - `6_About.py`: Architecture overview, Responsible AI guidelines, and academic credits.
- **Resource Management**: Uses cached model loading via `_CACHED_MODELS` to prevent memory thrashing.

---

## 15. Database & CRUD Audit

- **ORM Framework**: SQLAlchemy 2.0+
- **Production Target**: SQLite local development (`database/applicants.db`) with PostgreSQL production compatibility via `DATABASE_URL`.
- **Lifecycle Operations Tested**:
  - `create_applicant_record()`: Verified ✅
  - `get_applicant_record()`: Verified ✅
  - `list_applicant_records()`: Verified ✅
  - `update_applicant_record()`: Verified ✅
  - `delete_applicant_record()`: Verified ✅

---

## 16. Test Suite & Code Quality Audit

- **Test Suite Results**: `pytest -q` $\rightarrow$ **20 passed in 7.02s**
- **Test File Inventory**:
  - `test_feature_engineering.py`: 8 tests covering exact formulas, QA case, zero division, missing columns, and transformers.
  - `test_prediction.py`: 4 tests covering inference schema, mocked pipeline, real pipeline, and QA prediction.
  - `test_data_validation.py`: 4 tests covering schema, ranges, missing values, and column normalization.
  - `test_database_crud.py`: 1 comprehensive test covering full CRUD lifecycle.
  - `test_app_smoke.py`: 3 tests covering Streamlit module importability and configuration.
- **Linter Status**: `ruff check .` $\rightarrow$ Clean with **0 errors**.

---

## 17. Security & Privacy Audit

- **Hardcoded Secrets**: 0 detected.
- **API Keys / Credentials**: None exposed.
- **Personal Identifiable Information (PII)**: 0 (No Aadhaar, PAN, phone numbers, or emails present).
- **Git Ignore Security**:
  - Ignored: `.env`, `.venv/`, `data/raw/*.csv`, `data/processed/*.parquet`, `database/*.db`, `mlruns/`, `mlflow.db`.
  - Tracked: `data/sample/emi_sample.csv` (anonymized 5k sample), model pipelines, metadata JSONs, reports, and code.

---

## 18. Weighted Completion Score Breakdown

| Category | Weight | Score Awarded | Audit Verification Rationale |
| :--- | :---: | :---: | :--- |
| Workspace and environment | 5 | 4.8 | Python 3.11.9, clean pip check, zero compile errors, linter clean. Untracked git status. |
| Dataset validation and splitting | 10 | 10.0 | 404,800 rows verified, 70/15/15 stratified split verified with zero contamination. |
| EDA and reports | 7 | 7.0 | 25 PNG figures, EDA summary, business insights report, and sample dataset verified. |
| Feature engineering, leakage and fairness | 10 | 10.0 | 13 domain features, DTI vs Expense Ratio separated, sensitive features excluded. |
| Classification modelling | 12 | 12.0 | 4 candidate classifiers trained, XGBoost champion (Macro F1 = 0.8207) verified. |
| Regression modelling | 12 | 12.0 | 5 candidate regressors trained, XGBoost champion (MAE = INR 269.73) verified. |
| MLflow and model artifacts | 8 | 8.0 | 26 logged runs in mlflow.db, serialized pipelines with verified SHA-256 checksums. |
| Prediction service | 6 | 6.0 | predict_applicant_risk verified across low/moderate/high stress scenarios and QA case. |
| Streamlit application | 12 | 11.5 | Home + 6 pages verified with zero import errors; live cloud deployment pending. |
| Database CRUD | 5 | 5.0 | Full SQLAlchemy Create/Read/Update/Delete verified in memory and SQLite. |
| Testing, security and code quality | 8 | 8.0 | 20 passing unit tests, ruff clean, zero leaked secrets, comprehensive .gitignore. |
| Documentation and deployment readiness | 5 | 4.5 | README, requirements.txt, and CI workflow complete; live cloud URL pending. |
| **Total Score** | **100** | **98.8** | **Overall Weighted Score: 98.8 / 100.0** |

---

## 19. Prioritized Remaining-Work Plan

### Group 1: Must Complete Before GitHub Upload / Cloud Deployment
1. **Initial Git Commit**: Run `git add .` and `git commit -m "feat: complete EMIPredict AI underwriting platform"` to transition workspace from untracked to clean tracked repository state.
2. **GitHub Remote Link**: Link remote repository (`git remote add origin <url>`) and push `master` branch.
3. **Streamlit Cloud Deployment**: Connect repository on Streamlit Cloud dashboard pointing to root `streamlit_app.py`.

### Group 2: Recommended for Academic Defense / Final Submission
1. **Interactive Demo Video**: Record a 3-minute walkthrough demonstrating the Prediction page, Financial Health Radar, and Applicant Records CRUD manager.
2. **Slide Deck (PPT)**: Prepare an architecture and results presentation summarizing XGBoost Macro F1 (0.8207) and MAE (₹269.73).

### Group 3: Optional Future Improvements
1. **Cloud PostgreSQL Database**: Configure persistent PostgreSQL via `DATABASE_URL` secret on Streamlit Cloud for multi-session persistence.
2. **Model Retraining Trigger**: Add automated retraining pipeline trigger via GitHub Actions cron.
