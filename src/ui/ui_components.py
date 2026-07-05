"""
========================================================
Module: UI Components
Purpose: Reusable Streamlit view components for displaying
         budget cards, route info, interactive donut charts,
         and data-driven historical traveller insights.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st


def load_css() -> None:
    """Load and inject styles.css styling rules into Streamlit page."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "styles.css")

    # If not found inside src/ui/, check the root directory as fallback
    if not os.path.exists(css_path):
        css_path = os.path.join(base_dir, "..", "..", "styles.css")

    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_data = f.read()
        st.markdown(f"<style>{css_data}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Styling theme could not be loaded: {e}")


def render_header() -> None:
    """Render top branding title and subtitle for TripAI."""
    st.markdown(
        """
        <div class="tripai-header">
            <h1 class="tripai-title">✈️ TripAI</h1>
            <p class="tripai-subtitle">AI-Powered Travel Intelligence & Budget Planning Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_budget_hero(amount: float, summary_text: str, confidence_score: int, confidence_level: str) -> None:
    """Display estimated budget card with trip details and confidence rating."""
    badge_class = "badge-best-value" if confidence_score >= 70 else "badge-fastest"
    st.markdown(
        f"""
        <div class="budget-hero">
            <div class="budget-label">Estimated Recommended Budget</div>
            <div class="budget-amount">₹ {int(amount):,}</div>
            <div class="budget-meta">{summary_text}</div>
            <div style="margin-top: 14px;">
                <span class="badge {badge_class}">Prediction Confidence: {confidence_score}% ({confidence_level})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_budget_tiers(tiers: Dict[str, Any]) -> None:
    """Render the minimum, recommended, comfort, and luxury budget tiers."""
    st.markdown('<p class="section-header">💰 Smart Decision Budget Tiers</p>', unsafe_allow_html=True)
    cols = st.columns(4)

    labels = {
        "minimum": ("Minimum Budget", "minimum", "Hostels & local buses"),
        "recommended": ("Recommended Budget", "recommended", "Standard hotel & dining"),
        "comfort": ("Comfort Budget", "comfort", "3-star stay & sightseeing"),
        "luxury": ("Luxury Budget", "luxury", "Premium resorts & flights"),
    }

    for index, (key, (title, css_class, desc)) in enumerate(labels.items()):
        val = int(tiers.get(key, 0))
        cols[index].markdown(
            f"""
            <div class="tier-card {css_class}">
                <div class="field-label" style="font-size: 0.7rem;">{title}</div>
                <div class="budget-amount" style="font-size: 1.8rem; margin: 8px 0;">₹{val:,}</div>
                <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f'<p style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">💡 {tiers.get("explanation")}</p>', unsafe_allow_html=True)


def render_route_info(route: Dict[str, Any]) -> None:
    """Render distance, duration, and geo-data source metrics."""
    st.markdown('<p class="section-header">📍 Route Information</p>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)

    r1.metric("📏 Distance", f"{route.get('distance_km', 0.0):,.1f} km")
    r2.metric("⏱️ Duration", route.get("duration_text", "N/A"))
    r3.metric("🚗 Travel Mode", route.get("travel_mode", "Car"))

    source_map = {
        "cache": "💾 Cache Hit",
        "offline": "📦 Offline Data",
        "google_api": "🌐 Live API",
        "estimated": "📐 Calculated",
    }
    r4.metric("📡 Data Source", source_map.get(route.get("source", "estimated"), "Calculated"))


def render_budget_analysis(smart: Dict[str, Any]) -> None:
    """Render ML prediction vs Travel Cost vs Smart Budget columns."""
    st.markdown('<p class="section-header">📊 Cost Allocation Analysis</p>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    b1.metric("🤖 ML Model Base", f"₹{int(smart.get('ml_prediction', 0)):,}", help="Base prediction from historical algorithm")
    b2.metric("🚗 Est. Transit Cost", f"₹{int(smart.get('travel_cost_estimate', 0)):,}", help="Round-trip fuel/tickets based on distance")
    b3.metric("⚡ Combined Estimate", f"₹{int(smart.get('smart_estimate', 0)):,}", help="Combined analysis prediction")


def render_donut_chart(cost: float, mode: str) -> None:
    """Render Plotly Donut Chart showing typical travel cost breakdown."""
    st.markdown('<p class="section-header">📊 Estimated Budget Split</p>', unsafe_allow_html=True)

    # Approximate cost weights based on historical splits
    is_premium = "Flight" in mode or "Luxury" in mode
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
        hole=0.4,
        marker=dict(colors=["#3B82F6", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B"]),
        textinfo="percent+label",
        hoverinfo="label+value",
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9", family="Inter"),
        height=260,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_gauge_chart(score: float) -> None:
    """Render ML Model Accuracy Gauge indicator."""
    st.markdown('<p class="section-header">🎯 Algorithm Accuracy</p>', unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100 if score <= 1.0 else score,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3B82F6"},
            "steps": [
                {"range": [0, 60], "color": "#1E293B"},
                {"range": [60, 85], "color": "#334155"},
                {"range": [85, 100], "color": "#0F172A"},
            ],
            "threshold": {"line": {"color": "#8B5CF6", "width": 4}, "thickness": 0.75, "value": score * 100 if score <= 1.0 else score},
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        height=180,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_mode_comparison(modes: Dict[str, Any], selected_mode: str) -> None:
    """Display cards comparing alternative travel modes (cheapest/fastest)."""
    st.markdown('<p class="section-header">🚗 Travel Mode Cost Comparison</p>', unsafe_allow_html=True)
    cols = st.columns(5)

    mode_list = ["Flight", "Train", "Bus", "Car", "Bike"]
    icons = {"Flight": "✈️", "Train": "🚆", "Bus": "🚌", "Car": "🚗", "Bike": "🏍️"}

    for idx, mode in enumerate(mode_list):
        if mode not in modes:
            cols[idx].markdown(
                f"""
                <div class="mode-card">
                    <div style="font-size: 1.5rem;">{icons.get(mode)}</div>
                    <div style="font-weight: 700; margin-top: 6px;">{mode}</div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">No route</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue

        data = modes[mode]
        cost = int(data.get("round_trip", 0))
        dur = data.get("duration_text", "N/A")
        selected_style = "recommended-mode" if mode.lower() == selected_mode.lower() else ""

        cols[idx].markdown(
            f"""
            <div class="mode-card {selected_style}">
                <div style="font-size: 1.5rem;">{icons.get(mode)}</div>
                <div style="font-weight: 700; margin-top: 6px;">{mode}</div>
                <div style="color: #06b6d4; font-size: 1.05rem; font-weight: 800; margin: 4px 0;">₹{cost:,}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Est. {dur}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_traveller_experience(insights: Dict[str, Any]) -> None:
    """Render historical satisfaction ratings and stats from similar dataset trips."""
    st.markdown('<p class="section-header">📈 Historical Traveller Experience</p>', unsafe_allow_html=True)
    if not insights.get("has_data", False):
        st.info("No matching historical records found for this custom query.")
        return

    st.markdown(
        f"""
        <div class="dataset-insight">
            <h4 style="margin: 0 0 16px 0; color: #06b6d4;">📊 Dataset Analytics — based on {insights['similar_count']} historical trips</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                <div>
                    <div class="field-label">Overall Satisfaction</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #10B981;">{insights['avg_satisfaction']} / 5.0 ⭐</div>
                </div>
                <div>
                    <div class="field-label">Local Transport Score</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #3B82F6;">{insights['avg_transport_rating']} / 5.0 🚌</div>
                </div>
                <div>
                    <div class="field-label">Sightseeing Quality</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #8B5CF6;">{insights['avg_sightseeing_rating']} / 5.0 ⛰️</div>
                </div>
                <div>
                    <div class="field-label">Revisit Intention</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F59E0B;">{insights['revisit_percentage']}% Yes 🔄</div>
                </div>
            </div>
            <div class="divider"></div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.85rem; color: #CBD5E1;">
                <div>💼 Preferred Style: <strong>{insights['preferred_experience']}</strong></div>
                <div>🏨 Preferred Hotel: <strong>{insights['most_preferred_hotel']}</strong></div>
                <div>🚗 Top Transit Mode: <strong>{insights['most_used_travel_mode']}</strong></div>
                <div>📅 Peak Month: <strong>{insights['most_popular_month']} ({insights['most_popular_season']})</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_related_searches(related: List[str]) -> None:
    """Render pill chips for related searches."""
    if not related:
        return
    st.markdown('<p class="field-label" style="margin-top: 16px;">🔍 Users who searched this also explored:</p>', unsafe_allow_html=True)
    chips = "".join([f'<span class="related-search-chip">{place}</span>' for place in related])
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


def render_future_features() -> None:
    """Render placeholder grid showing future features roadmap."""
    st.markdown('<p class="section-header" style="margin-top: 24px;">🚀 TripAI 2026 Future Roadmap</p>', unsafe_allow_html=True)
    cols = st.columns(3)

    features = [
        ("🎙️ Voice Assistant", "Plan hands-free with AI speech"),
        ("🗺️ Custom Itinerary", "Day-wise automated route maps"),
        ("🏨 Stays Booking", "One-click Booking.com rates"),
    ]

    for index, (title, desc) in enumerate(features):
        cols[index].markdown(
            f"""
            <div class="placeholder-card">
                <div style="font-weight: 700; font-size: 0.85rem; color: #f1f5f9;">{title}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
