# EMIPredict AI — Model Card

## Model Details
- **Organization**: EMIPredict AI Open Source Project
- **Model Date**: August 2026
- **Model Version**: 1.0.0
- **Model Type**: Scikit-Learn Pipeline combining Feature Engineering + ColumnTransformer + Ensemble Classifiers/Regressors
- **License**: MIT

## Intended Use
- **Primary Use Case**: Assisting financial institutions and loan officers with preliminary applicant risk profiling and monthly repayment affordability estimation.
- **Out of Scope**: Automatic, unsupervised loan rejection or granting without human underwriter review.

## Training & Evaluation Data
- **Source**: `data/raw/EMI_dataset.csv` (404,800 records across 25 input variables and 2 target variables).
- **Partitioning**: 70% Training, 15% Validation, 15% Test (Stratified Split).

## Ethical Considerations
- **Fairness**: Sensitive demographic features (`gender`, `marital_status`) are omitted from input features.
- **Non-Negativity Constraint**: Regression predictions are bounded at INR 0 to avoid impossible negative repayment commitments.
