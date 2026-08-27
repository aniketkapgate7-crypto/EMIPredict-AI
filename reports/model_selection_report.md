# EMIPredict AI — Model Selection & Architecture Report

## Executive Summary
This report documents the empirical evaluation, trade-off analysis, and selection of production machine learning models for the EMIPredict AI platform.

## 1. Classification Model Selection (`emi_eligibility`)
- **Primary Selection Metric**: **Macro F1-Score** (to ensure robust, balanced performance across all risk classes).
- **Selected Champion**: `XGBoost`
- **Validation Macro F1**: 0.8168
- **Validation Accuracy**: 0.9657
- **Test Set Macro F1**: 0.8207

## 2. Regression Model Selection (`max_monthly_emi`)
- **Primary Selection Metric**: **Mean Absolute Error (MAE)** (measures expected rupee error directly in INR).
- **Selected Champion**: `XGBoost`
- **Validation MAE**: INR 274.78
- **Validation RMSE**: INR 777.24
- **Validation R2**: 0.99
- **Test Set MAE**: INR 269.73

## 3. Responsible AI & Fairness Decisions
- Demographic attributes (`gender`, `marital_status`) were strictly excluded from model feature inputs to prevent discriminatory lending bias and comply with ethical AI principles.
- Preprocessing pipelines were strictly fitted only on training partitions to prevent data leakage.
