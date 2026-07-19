"""
========================================================
Module: UI Charts
Purpose: Handles the creation of interactive Plotly figures
         and gauges for the budget analytics dashboard and
         cost allocations.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render_budget_donut(cost: float, mode: str) -> go.Figure:
    """Generate Donut Chart for typical travel cost breakdown."""
    is_premium = mode in ("Flight", "Luxury")
    travel_weight = 0.35 if is_premium else 0.20
    hotel_weight = 0.40 if is_premium else 0.35

    categories = ["Transit (Travel)", "Hotel (Stay)", "Food & Dining", "Sightseeing/Local", "Misc/Shopping"]
    values = [
        cost * travel_weight,
        cost * hotel_weight,
        cost * 0.20,
        cost * 0.15,
        cost * 0.10,
    ]

    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        hole=0.45,
        marker=dict(colors=["#2563EB", "#7C3AED", "#06B6D4", "#10B981", "#F59E0B"]),
        textinfo="percent",
        hoverinfo="label+value",
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9", family="Inter"),
        height=260,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def render_accuracy_gauge(score: float) -> go.Figure:
    """Generate accuracy gauge indicator chart."""
    val = score * 100 if score <= 1.0 else score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": "%", "font": {"color": "#F1F5F9", "family": "Outfit"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
            "bar": {"color": "#2563EB"},
            "bgcolor": "#1E293B",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [0, 60], "color": "rgba(239, 68, 68, 0.1)"},
                {"range": [60, 85], "color": "rgba(245, 158, 11, 0.1)"},
                {"range": [85, 100], "color": "rgba(16, 185, 129, 0.1)"},
            ],
            "threshold": {"line": {"color": "#7C3AED", "width": 4}, "thickness": 0.8, "value": val},
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9", family="Inter"),
        height=180,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    return fig


def render_dashboard_destinations(top_dest: List[Dict[str, Any]]) -> go.Figure:
    """Create chart of the most searched destinations from query logs."""
    df = pd.DataFrame(top_dest)
    fig = px.bar(
        df,
        x="destination",
        y="search_count",
        color="avg_predicted_cost",
        color_continuous_scale="Viridis",
        labels={"search_count": "Searches", "destination": "Destination", "avg_predicted_cost": "Avg Budget (₹)"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F1F5F9",
        font_family="Inter",
        height=280,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig


def render_dashboard_ratings(highest_rated: pd.DataFrame) -> go.Figure:
    """Create bar chart of the highest rated destinations."""
    fig = px.bar(
        highest_rated,
        x="Place",
        y="Overall_Score",
        color="Overall_Score",
        color_continuous_scale="Tealgrn",
        labels={"Overall_Score": "Rating (1-5)", "Place": "Destination"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F1F5F9",
        font_family="Inter",
        height=280,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig


def render_dashboard_seasons(season_counts: pd.DataFrame) -> go.Figure:
    """Create pie chart of the most visited seasons."""
    fig = px.pie(
        season_counts,
        names="Season",
        values="Count",
        color_discrete_sequence=px.colors.sequential.Bluered_r,
        hole=0.4
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F1F5F9",
        font_family="Inter",
        height=280,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig


def render_dashboard_experiences(exp_counts: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of the most popular experiences."""
    fig = px.bar(
        exp_counts,
        x="Count",
        y="Experience",
        orientation="h",
        color="Count",
        color_continuous_scale="Purples",
        labels={"Count": "Travellers Count", "Experience": "Preferred Style"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F1F5F9",
        font_family="Inter",
        height=280,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig


def render_dashboard_budgets(df: pd.DataFrame) -> go.Figure:
    """Create budget distribution histogram."""
    fig = px.histogram(
        df,
        x="Cost",
        nbins=30,
        color_discrete_sequence=["#06B6D4"],
        labels={"Cost": "Trip Budget (₹)"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F1F5F9",
        font_family="Inter",
        height=280,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig


def render_dashboard_durations(duration_df: pd.DataFrame) -> go.Figure:
    """Create stay duration bar chart."""
    fig = px.bar(
        duration_df,
        x="Place",
        y="Days",
        color="Days",
        color_continuous_scale="Plasma",
        labels={"Days": "Average Days", "Place": "Destination"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F1F5F9",
        font_family="Inter",
        height=280,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig
