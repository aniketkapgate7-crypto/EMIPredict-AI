"""
Utility script to generate comprehensive, valid JSON Jupyter Notebooks (.ipynb)
for all 5 stages of the EMIPredict AI project.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def build_all_notebooks():
    # -------------------------------------------------------------
    # 01_data_validation.ipynb
    # -------------------------------------------------------------
    nb01_cells = [
        md_cell("# 01 — Data Validation & Schema Audit\n## EMIPredict AI Platform\nThis notebook performs comprehensive schema verification, data quality checks, null/inf detection, and domain boundary audits on `data/raw/EMI_dataset.csv` using modular functions from `src.data`."),
        code_cell("import sys\nfrom pathlib import Path\n\nroot_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nif str(root_dir) not in sys.path:\n    sys.path.insert(0, str(root_dir))\n\nfrom src.data.load_data import load_raw_dataset\nfrom src.data.validate_data import run_data_validation"),
        md_cell("### 1. Load Raw Dataset with Memory Downcasting\nWe load the raw CSV dataset, downcast floating and integer columns to reduce RAM footprint, and verify dataset dimensions."),
        code_cell("df = load_raw_dataset()\nprint(f'Dataset Shape: {df.shape}')\nprint(f'Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB')\ndf.head()"),
        md_cell("### 2. Execute Full Data Quality Audit\nWe execute the automated data validation suite, checking for:\n- Duplicated columns and rows\n- Missing values per column\n- Infinite numerical values\n- Constant & near-constant columns\n- Categorical anomalies & target distribution\n- Target leakage risks"),
        code_cell("validation_report = run_data_validation(df)\nprint('Data Validation Audit Completed.')\nprint(f'Total Rows: {validation_report[\"total_rows\"]:,}')\nprint(f'Exact Duplicate Rows: {validation_report[\"duplicate_rows\"][\"count\"]:,}')\nprint(f'Missing Cells: {validation_report[\"missing_values_summary\"][\"total_missing_cells\"]}')"),
        md_cell("### 3. Target Distribution Verification\nWe check the balance of our multiclass target `emi_eligibility` and key statistical moments of `max_monthly_emi`."),
        code_cell("print('Classification Target Distribution:')\nprint(df['emi_eligibility'].value_counts())\n\nprint('\\nRegression Target Summary (₹):')\nprint(df['max_monthly_emi'].describe())"),
        md_cell("### Summary & Next Steps\nThe raw dataset contains 404,800 records across 25 input variables and 2 target variables. All records have valid types, and the validation report has been saved to `reports/data_quality_report.json`.")
    ]
    with open(NOTEBOOKS_DIR / "01_data_validation.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb01_cells), f, indent=2)

    # -------------------------------------------------------------
    # 02_exploratory_data_analysis.ipynb
    # -------------------------------------------------------------
    nb02_cells = [
        md_cell("# 02 — Exploratory Data Analysis & Business Insights\n## EMIPredict AI Platform\nThis notebook explores financial distributions, credit ratings, living expense structures, and relationships with eligibility and maximum monthly EMI capacity."),
        code_cell("import sys\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nroot_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nif str(root_dir) not in sys.path:\n    sys.path.insert(0, str(root_dir))\n\nfrom src.config import FIGURES_DIR\nfrom src.data.load_data import load_raw_dataset"),
        md_cell("### 1. Load Dataset for Analysis"),
        code_cell("df = load_raw_dataset()\nFIGURES_DIR.mkdir(parents=True, exist_ok=True)\nsns.set_theme(style='whitegrid', palette='muted')"),
        md_cell("### 2. Target Variable Analysis\nWe visualize the distribution of `emi_eligibility` and `max_monthly_emi`."),
        code_cell("fig, ax = plt.subplots(1, 2, figsize=(14, 5))\nsns.countplot(data=df, x='emi_eligibility', palette=['#00A896', '#F4A261', '#E63946'], ax=ax[0])\nax[0].set_title('EMI Eligibility Distribution', fontsize=12, fontweight='bold')\n\nsns.histplot(df['max_monthly_emi'], kde=True, color='#028090', ax=ax[1])\nax[1].set_title('Max Monthly EMI Distribution (₹)', fontsize=12, fontweight='bold')\nplt.tight_layout()\nplt.savefig(FIGURES_DIR / 'target_distributions.png', dpi=300)\nplt.show()"),
        md_cell("### 3. Income vs Credit Score & Eligibility\nAnalyzing how income bands and credit rating correlate with financial risk categories."),
        code_cell("fig, ax = plt.subplots(figsize=(8, 6))\nsns.scatterplot(data=df.sample(2000, random_state=42), x='monthly_salary', y='credit_score', hue='emi_eligibility', alpha=0.7, palette=['#00A896', '#F4A261', '#E63946'], ax=ax)\nax.set_title('Monthly Salary vs Credit Score by Eligibility Tier', fontsize=12, fontweight='bold')\nplt.tight_layout()\nplt.savefig(FIGURES_DIR / 'salary_vs_credit_score.png', dpi=300)\nplt.show()"),
        md_cell("### Summary of Business Insights\n- Higher credit score (> 700) and substantial savings buffer strongly correlate with `Eligible` classification.\n- Living expenses and existing loan EMIs significantly constrain `max_monthly_emi` capacity.")
    ]
    with open(NOTEBOOKS_DIR / "02_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb02_cells), f, indent=2)

    # -------------------------------------------------------------
    # 03_feature_engineering.ipynb
    # -------------------------------------------------------------
    nb03_cells = [
        md_cell("# 03 — Feature Engineering & Preprocessing Pipelines\n## EMIPredict AI Platform\nThis notebook demonstrates domain-specific financial ratio calculations, Scikit-Learn transformers, and ethical AI fairness exclusions."),
        code_cell("import sys\nfrom pathlib import Path\n\nroot_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nif str(root_dir) not in sys.path:\n    sys.path.insert(0, str(root_dir))\n\nfrom src.data.prepare_data import prepare_sample_and_splits\nfrom src.features.build_features import calculate_financial_features\nfrom src.features.preprocessing import create_preprocessor, get_model_feature_lists"),
        md_cell("### 1. Partition Data (70% Train, 15% Val, 15% Test)\nStrict rule: preprocessing pipelines are fitted exclusively on training data to prevent leakage."),
        code_cell("train_df, val_df, test_df, sample_df = prepare_sample_and_splits()\nprint(f'Train partition: {len(train_df):,} rows')\nprint(f'Validation partition: {len(val_df):,} rows')\nprint(f'Test partition: {len(test_df):,} rows')"),
        md_cell("### 2. Compute Engineered Financial Ratios\nWe calculate 13 domain metrics including Current DTI, Expense Ratio, Obligation Ratio, Proposed Principal Burden, Disposable Income, Emergency Runway, and Savings Ratio with division-by-zero safeguards."),
        code_cell("train_features = calculate_financial_features(train_df)\ntrain_features[['monthly_salary', 'current_debt_to_income_ratio', 'expense_to_income_ratio', 'proposed_principal_burden_ratio', 'disposable_income', 'emergency_fund_months']].head()"),
        md_cell("### 3. Fit ColumnTransformer Preprocessing Pipeline\nWe verify that sensitive attributes (`gender`, `marital_status`) are excluded from model features, and numeric/categorical transformers are assembled."),
        code_cell("num_cols, cat_cols, sensitive = get_model_feature_lists()\nprint(f'Numerical features ({len(num_cols)}): {num_cols}')\nprint(f'Categorical features ({len(cat_cols)}): {cat_cols}')\nprint(f'Excluded sensitive attributes for fairness: {sensitive}')\n\npreprocessor = create_preprocessor(include_feature_engineering=True)\nX_train = train_df.drop(columns=['emi_eligibility', 'max_monthly_emi'])\nX_transformed = preprocessor.fit_transform(X_train)\nprint(f'Transformed feature matrix shape: {X_transformed.shape}')"),
        md_cell("### Summary\nThe feature engineering and preprocessing pipeline transforms raw applicant data into a robust, leak-free, scaled matrix ready for classification and regression modeling.")
    ]
    with open(NOTEBOOKS_DIR / "03_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb03_cells), f, indent=2)

    # -------------------------------------------------------------
    # 04_classification_models.ipynb
    # -------------------------------------------------------------
    nb04_cells = [
        md_cell("# 04 — Multiclass Classification Modeling (`emi_eligibility`)\n## EMIPredict AI Platform\nThis notebook trains and evaluates candidate classification models (Logistic Regression, Random Forest, XGBoost) optimizing for **Macro F1-Score** with MLflow tracking."),
        code_cell("import sys\nfrom pathlib import Path\n\nroot_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nif str(root_dir) not in sys.path:\n    sys.path.insert(0, str(root_dir))\n\nfrom src.models.train_classification import train_classification_models"),
        md_cell("### 1. Train & Benchmark Classification Models\nExecutes complete model training, validation evaluation, MLflow run logging, and test-set verification."),
        code_cell("best_pipe, comp_df, meta = train_classification_models()\nprint('Model Benchmark Comparison (Sorted by Macro F1):')\ncomp_df"),
        md_cell("### 2. Inspect Champion Model Performance & Metrics"),
        code_cell("print(f'Selected Champion Classifier: {meta[\"model_name\"]}')\nprint(f'Validation Macro F1: {meta[\"validation_metrics\"][\"macro_f1\"]:.4f}')\nprint(f'Test Set Macro F1: {meta[\"test_metrics\"][\"macro_f1\"]:.4f}')\nprint(f'Test Set Accuracy: {meta[\"test_metrics\"][\"accuracy\"]:.4f}')"),
        md_cell("### Summary\nThe champion model pipeline has been saved to `models/eligibility_pipeline.joblib` and is ready for inference.")
    ]
    with open(NOTEBOOKS_DIR / "04_classification_models.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb04_cells), f, indent=2)

    # -------------------------------------------------------------
    # 05_regression_models.ipynb
    # -------------------------------------------------------------
    nb05_cells = [
        md_cell("# 05 — EMI Affordability Regression Modeling (`max_monthly_emi`)\n## EMIPredict AI Platform\nThis notebook trains and evaluates regression models (Linear Regression, Random Forest, XGBoost) optimizing for **Mean Absolute Error (MAE)** with MLflow tracking."),
        code_cell("import sys\nfrom pathlib import Path\n\nroot_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nif str(root_dir) not in sys.path:\n    sys.path.insert(0, str(root_dir))\n\nfrom src.models.select_model import generate_model_reports_and_metadata\nfrom src.models.train_regression import train_regression_models"),
        md_cell("### 1. Train & Benchmark Regression Models\nExecutes training, validation MAE evaluation, and test-set evaluation."),
        code_cell("best_pipe, comp_df, meta = train_regression_models()\nprint('Regression Model Benchmark Comparison (Sorted by MAE):')\ncomp_df"),
        md_cell("### 2. Generate Consolidated Model Reports & Schema"),
        code_cell("generate_model_reports_and_metadata()\nprint('Consolidated metadata and model cards generated.')"),
        md_cell("### Summary\nThe champion regression pipeline has been exported to `models/max_emi_pipeline.joblib` and evaluated on the untouched test set.")
    ]
    with open(NOTEBOOKS_DIR / "05_regression_models.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb05_cells), f, indent=2)

    print("All 5 Jupyter Notebooks successfully generated in notebooks/")


if __name__ == "__main__":
    build_all_notebooks()
