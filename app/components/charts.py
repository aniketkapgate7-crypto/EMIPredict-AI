"""
Plotly interactive visualization components for EMIPredict AI Streamlit UI.
"""

from typing import Dict

import plotly.graph_objects as go
import streamlit as st


def plot_probability_distribution(prob_dict: Dict[str, float]) -> None:
    """Renders an interactive horizontal bar chart of multiclass prediction probabilities."""
    labels = list(prob_dict.keys())
    values = [prob_dict[k] * 100 for k in labels]

    color_map = {
        "Eligible": "#00A896",
        "High_Risk": "#F4A261",
        "Not_Eligible": "#E63946",
    }
    colors = [color_map.get(lbl, "#3A86FF") for lbl in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Model Prediction Confidence (%)",
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        yaxis=dict(title=""),
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_financial_health_radar(
    dti: float,
    savings_ratio: float,
    emergency_months: float,
    credit_score: float,
    disposable_ratio: float
) -> None:
    """Renders a 5-axis normalized radar chart illustrating financial health."""
    # Normalize components to a 0-100 scale
    score_credit = min(100.0, max(0.0, (credit_score - 300) / 600 * 100))
    score_dti = max(0.0, 100.0 - (dti * 100.0))  # Lower DTI is better
    score_savings = min(100.0, savings_ratio * 100.0)
    score_runway = min(100.0, (emergency_months / 6.0) * 100.0)
    score_disp = min(100.0, max(0.0, disposable_ratio * 100.0))

    categories = [
        "Credit Score",
        "Debt Capacity (Low DTI)",
        "Savings Cushion",
        "Emergency Runway",
        "Disposable Margin",
    ]
    values = [score_credit, score_dti, score_savings, score_runway, score_disp]
    # Close polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(0, 168, 150, 0.3)",
            line=dict(color="#00A896", width=2),
            name="Applicant Profile",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#A0AEC0"),
            angularaxis=dict(color="#FFFFFF"),
            bgcolor="rgba(28, 37, 65, 0.4)",
        ),
        showlegend=False,
        title="Applicant Financial Health Radar",
        margin=dict(l=40, r=40, t=40, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
    )
    st.plotly_chart(fig, use_container_width=True)
