# Exploratory Data Analysis (EDA) Summary

## 1. Overview
- Dataset size: 404,800 rows and 27 features.
- Dual targets: `emi_eligibility` (Multiclass) and `max_monthly_emi` (Continuous).

## 2. Key Statistical Findings
- **Classification Target**: Well-represented across Eligible, High Risk, and Not Eligible classes.
- **Regression Target**: Mean max monthly EMI of ₹6,763.60 with standard deviation of ₹7,741.26.
- **Credit Score**: Strong discriminator between Eligible (> 700 median) and Not Eligible (< 620 median).
- **Current Debt-to-Income (DTI)**: Measures active debt servicing burden against gross income. Applicants with active DTI > 40% (academic benchmark) demonstrate elevated credit risk.
- **Expense-to-Income Ratio**: Distinctly tracks living expenses (rent, utilities, school fees) relative to gross income.
