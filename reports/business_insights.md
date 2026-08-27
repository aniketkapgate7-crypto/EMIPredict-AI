# Underwriting & Business Insights Report

## 1. Executive Summary
Analysis of the 404,800-record lending dataset reveals that applicant solvency is primarily driven by three core pillars: Net Disposable Margin, Current Debt-to-Income Ratio, and Credit Track Record.

## 2. Risk Driver Analysis
1. **Debt-Servicing Burden**: Existing active loan EMIs represent recurring debt obligations. High current DTI restricts capacity for incremental debt.
2. **Cost-of-Living Overhead**: Living expenses (housing, utilities, tuition) directly determine net disposable cash flow.
3. **Liquidity Buffer**: Applicants with > 3 months emergency runway exhibit robust resilience against financial distress.
4. **Affordability Ceiling**: The model-predicted `max_monthly_emi` serves as an empirical upper bound to prevent borrower over-indebtedness.
