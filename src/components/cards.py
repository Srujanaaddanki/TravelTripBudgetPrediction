"""
========================================================
Component: Cards
Purpose: Reusable card components for mode comparison,
         KPI metrics, and budget breakdown display.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
from typing import Any, Dict
import streamlit as st


# ── Mode metadata ────────────────────────────────────────────────────────────
_MODE_META: Dict[str, Dict[str, str]] = {
    "Train":  {"icon": "🚂", "badge": "Recommended", "badge_cls": "badge-recommended"},
    "Flight": {"icon": "✈️", "badge": "Fastest",     "badge_cls": "badge-fastest"},
    "Bus":    {"icon": "🚌", "badge": "Budget Friendly", "badge_cls": "badge-budget"},
    "Car":    {"icon": "🚗", "badge": "Flexible",    "badge_cls": "badge-flexible"},
    "Bike":   {"icon": "🏍️", "badge": "Adventurous", "badge_cls": "badge-adventure"},
}

_MODE_ORDER = ["Train", "Flight", "Bus", "Car", "Bike"]


def render_mode_comparison_cards(
    modes: Dict[str, Any],
    selected_mode: str,
    distance_km: float,
) -> None:
    """Render the 5-mode cost comparison card grid.

    Parameters
    ----------
    modes : dict
        Maps mode name → dict with 'one_way', 'round_trip', 'duration_hours'.
    selected_mode : str
        The user's chosen mode, highlighted with a border glow.
    distance_km : float
        Route distance used for duration calculation.
    """
    st.markdown(
        '<div class="section-title">🔄 Budget Comparison by Mode</div>',
        unsafe_allow_html=True,
    )

    cards_html = '<div class="mode-cards-grid">'

    for mode in _MODE_ORDER:
        meta      = _MODE_META.get(mode, {"icon": "🚀", "badge": "", "badge_cls": ""})
        mode_data = modes.get(mode, {})

        # Cost — prefer round_trip, fall back to one_way
        cost_raw  = mode_data.get("round_trip", mode_data.get("one_way", 0.0))
        cost_str  = f"₹{int(cost_raw):,}" if cost_raw else "—"

        # Duration
        dur_raw   = mode_data.get("duration_hours", 0.0)
        if dur_raw:
            h = int(dur_raw)
            m = int((dur_raw - h) * 60)
            dur_str = f"{h}h {m:02d}m"
        else:
            dur_str = "—"

        selected_cls = "selected-mode" if mode == selected_mode else ""

        cards_html += f"""
        <div class="mode-card {selected_cls}">
          <div class="mode-icon">{meta['icon']}</div>
          <div class="mode-name">{mode}</div>
          <div class="mode-price">{cost_str}</div>
          <div class="mode-time">{dur_str}</div>
          {f'<div class="mode-badge {meta["badge_cls"]}">{meta["badge"]}</div>' if meta["badge"] else ""}
        </div>
        """

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    delta: str = "",
    icon: str = "📊",
    icon_bg: str = "rgba(79,70,229,0.15)",
    delta_positive: bool = True,
) -> None:
    """Render a single analytics KPI card.

    Parameters
    ----------
    label : str
        Card label (e.g. "Total Trips").
    value : str
        Primary display value (e.g. "920").
    delta : str
        Change indicator (e.g. "+12.4% vs last year").
    icon : str
        Emoji icon for the card.
    icon_bg : str
        CSS background for the icon container.
    delta_positive : bool
        If True delta text is green, otherwise red.
    """
    delta_cls = "" if delta_positive else "negative"
    delta_html = (
        f'<div class="kpi-delta {delta_cls}">{delta}</div>'
        if delta else ""
    )

    card_html = f"""
    <div class="kpi-card">
      <div class="kpi-icon" style="background:{icon_bg};">{icon}</div>
      <div class="kpi-content">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_budget_breakdown(pred: float, travel_mode: str) -> None:
    """Render the budget breakdown card with donut-matching categories.

    Parameters
    ----------
    pred : float
        Total predicted budget in INR.
    travel_mode : str
        Travel mode — affects travel cost weight.
    """
    is_premium = travel_mode in ("Flight",)
    travel_w = 0.30 if is_premium else 0.20

    categories = [
        ("Hotel",           0.35, "#818CF8"),
        ("Travel",          travel_w, "#2563EB"),
        ("Food",            0.15, "#10B981"),
        ("Local Transport", 0.10, "#F59E0B"),
        ("Activities",      0.10, "#EC4899"),
        ("Shopping",        0.05, "#06B6D4"),
        ("Emergency",       0.05, "#94A3B8"),
    ]

    # Normalise weights to sum exactly to 1
    total_w = sum(w for _, w, _ in categories)
    categories = [(n, w / total_w, c) for n, w, c in categories]

    rows_html = ""
    for name, weight, color in categories:
        amount = int(pred * weight)
        pct    = int(weight * 100)
        rows_html += f"""
        <div class="breakdown-item">
          <div class="breakdown-dot" style="background:{color};"></div>
          <div class="breakdown-label">{name} ({pct}%)</div>
          <div class="breakdown-amount">₹{amount:,}</div>
        </div>
        """

    card_html = f"""
    <div class="breakdown-card">
      <div class="section-title">🔥 Budget Breakdown</div>
      {rows_html}
      <div class="breakdown-total">
        <span class="breakdown-total-label">Total Budget</span>
        <span class="breakdown-total-amount">₹{int(pred):,}</span>
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_feature_card(
    num: str,
    icon: str,
    title: str,
    bullets: list[str],
    delay: float = 0.0,
) -> None:
    """Render one of the 6 analytics feature overview cards.

    Parameters
    ----------
    num : str
        Card number label (e.g. "1.").
    icon : str
        Emoji icon.
    title : str
        Card title text.
    bullets : list[str]
        Bullet point lines.
    delay : float
        Animation delay in seconds.
    """
    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
    card_html = f"""
    <div class="feature-card" style="animation-delay:{delay}s;">
      <div class="feature-card-num">{icon} {num}</div>
      <div class="feature-card-title">{title}</div>
      <ul class="feature-card-bullets">
        {bullets_html}
      </ul>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
