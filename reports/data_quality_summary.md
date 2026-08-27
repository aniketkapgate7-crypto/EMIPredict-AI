# Data Quality & Schema Audit Summary

- **Total Records**: 404,800
- **Total Columns**: 27 (25 input features + 2 targets)
- **Exact Duplicate Rows**: 0 (0.00%)
- **Total Missing Cells**: 12,041

## Data Corrections & Sanitization (Transparently Recorded)
- **Column `age`**: 3 values with multi-dot formatting ('58.0.0', '38.0.0', '32.0.0') sanitized to integer/float.
- **Column `monthly_salary`**: 1,993 values with multi-dot formatting (e.g. '64300.0.0') sanitized to valid decimal float.
- **Column `bank_balance`**: 1,952 values with multi-dot formatting (e.g. '270700.0.0') sanitized to valid decimal float.

## Target Variable Distributions

### 1. Classification Target (`emi_eligibility`)
- **Not_Eligible**: 312,868 (77.29%)
- **Eligible**: 74,444 (18.39%)
- **High_Risk**: 17,488 (4.32%)

### 2. Regression Target (`max_monthly_emi`)
- **min**: 500.00
- **max**: 91,040.40
- **mean**: 6,763.60
- **median**: 4,211.20
- **std**: 7,741.26
- **negative_values_count**: 0
- **zero_values_count**: 0

## Data Quality Audit Flags
- **Infinite Values**: 0 columns
- **Constant Columns**: 0
- **Near-Constant Columns**: 0
- **Potential Identifiers**: 0
- **Range Flags**: 1 columns inspected
