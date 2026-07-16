"""
========================================================
Module: Dashboard Components
Purpose: Renders premium SaaS-quality dashboard cards,
         widgets, and travel intelligence visualisations
         for the TripAI prediction results page.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────

def format_stars(score: float) -> str:
    """Format numeric score to filled and empty star unicode symbols."""
    rounded = min(5, max(0, int(round(score))))
    return "★" * rounded + "☆" * (5 - rounded)


def format_rupees(amount: float) -> str:
    """Format rupee numeric values into comma-separated currency string."""
    return f"₹{amount:,.0f}"


# Season icons used across multiple widgets
SEASON_ICONS = {
    "winter": "❄️", "summer": "☀️", "rainy": "🌧️",
    "monsoon": "🌧️", "autumn": "🍂", "spring": "🌸",
}

# Hotel icons map
HOTEL_ICONS = {"budget": "🏷️", "standard": "🏨", "luxury": "💎", "premium": "👑"}


# ── 1. Trip Overview Card ─────────────────────────────────────

def render_trip_overview_card(
    source: str,
    destination: str,
    days: int,
    season: str,
    travel_mode: str,
    hotel: str,
    pred: float,
) -> None:
    """Renders a compact trip overview summary card at the top of the right column.

    Shows the key trip parameters at a glance: route, duration, season,
    mode, hotel quality, and estimated budget — giving recruiters an
    instant summary of what was predicted.
    """
    season_icon = SEASON_ICONS.get(season.lower(), "🌍")
    hotel_icon  = HOTEL_ICONS.get(hotel.lower(), "🏨")
    mode_icons  = {"flight": "✈️", "train": "🚆", "bus": "🚌", "car": "🚗", "bike": "🏍️"}
    mode_icon   = mode_icons.get(travel_mode.lower(), "🚗")

    st.markdown(
        f"""
        <div class="trip-overview-card">
            <div class="field-label" style="margin-bottom: 10px;">📋 Trip Overview</div>
            <div class="trip-overview-route">
                <span class="trip-overview-city">🔵 {source.title()}</span>
                <span class="trip-overview-arrow">→</span>
                <span class="trip-overview-city">🟣 {destination.title()}</span>
            </div>
            <div class="trip-overview-pills">
                <span class="trip-pill">🗓️ {days} Days</span>
                <span class="trip-pill">{season_icon} {season.title()} Season</span>
                <span class="trip-pill">{mode_icon} {travel_mode}</span>
                <span class="trip-pill">{hotel_icon} {hotel.title()} Hotel</span>
            </div>
            <div class="field-label" style="margin-bottom: 4px;">Estimated Budget</div>
            <div class="trip-overview-budget">{format_rupees(pred)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 2. AI Travel Advisor Card ─────────────────────────────────

def render_ai_advisor_card(
    destination: str,
    pred: float,
    travel_mode: str,
    hotel: str,
    days: int,
    season: str,
    confidence_score: int,
    dataset_insights: Dict[str, Any],
    weather: Dict[str, Any],
    route_info: Dict[str, Any],
) -> None:
    """Renders a fully dynamic AI Travel Advisor summary paragraph.

    Each sentence is constructed from live data fields. If a field is
    missing or zero, that sentence is silently omitted — no placeholders
    ever appear. The result reads as a natural, professional travel briefing.
    """
    sentences: List[str] = []

    # Sentence 1: Destination + preferred experience + best time
    exp_style = dataset_insights.get("preferred_experience")
    best_time = dataset_insights.get("most_popular_month") or dataset_insights.get("most_popular_season")
    if destination and exp_style and best_time:
        sentences.append(
            f"{destination.title()} is a popular <strong>{exp_style}</strong> destination, "
            f"most visited during <strong>{best_time.title()}</strong>."
        )

    # Sentence 2: Budget context from historical data
    similar_count = dataset_insights.get("similar_count", 0)
    hist_avg = dataset_insights.get("average_budget", 0)
    if similar_count and similar_count > 0 and pred:
        sentences.append(
            f"Based on <strong>{similar_count:,} historical travellers</strong>, "
            f"the estimated budget for a <strong>{days}-day {season.lower()} trip</strong> "
            f"is <strong>{format_rupees(pred)}</strong>."
        )
    elif pred:
        sentences.append(
            f"The AI model estimates a <strong>{days}-day {season.lower()} trip</strong> "
            f"to {destination.title()} at approximately <strong>{format_rupees(pred)}</strong>."
        )

    # Sentence 3: Route + travel mode
    duration_text = route_info.get("duration_text")
    distance_km   = route_info.get("distance_km", 0)
    if travel_mode and duration_text and duration_text.lower() not in ("n/a", "none", ""):
        sentences.append(
            f"Travelling by <strong>{travel_mode}</strong> covers approximately "
            f"<strong>{distance_km:,.0f} km</strong> with an estimated journey of "
            f"<strong>{duration_text}</strong>."
        )
    elif travel_mode:
        sentences.append(
            f"<strong>{travel_mode}</strong> is the selected travel mode for this trip, "
            f"with a <strong>{hotel.title()} hotel</strong> stay arranged at the destination."
        )

    # Sentence 4: Live weather (only if it's real API data, not a fallback)
    temp   = weather.get("temperature_c")
    desc   = weather.get("description", "")
    source = weather.get("source", "")
    if temp is not None and "unavailable" not in desc.lower() and source == "Open-Meteo API":
        sentences.append(
            f"Current weather at {destination.title()} is "
            f"<strong>{temp}°C — {desc}</strong>."
        )

    # Sentence 5: Historical satisfaction ratings
    score          = dataset_insights.get("destination_score", 0)
    revisit_pct    = dataset_insights.get("revisit_percentage", 0)
    if score and score > 0:
        revisit_part = (
            f", with <strong>{revisit_pct}% choosing to revisit</strong>"
            if revisit_pct and revisit_pct > 0 else ""
        )
        sentences.append(
            f"Historical travellers rate {destination.title()} "
            f"<strong>{score}/5.0 overall</strong>{revisit_part}."
        )

    # Sentence 6: Model confidence (always present)
    if confidence_score:
        sentences.append(
            f"The Random Forest Regressor model predicts this budget with "
            f"<strong>{confidence_score}% confidence</strong>."
        )

    summary_html = " ".join(sentences) if sentences else (
        f"Your trip to <strong>{destination.title()}</strong> has been analysed. "
        f"The estimated budget is <strong>{format_rupees(pred)}</strong> for {days} days."
    )

    st.markdown(
        f"""
        <div class="ai-advisor-card">
            <div class="ai-advisor-label">✨ AI Travel Advisor</div>
            <p class="ai-advisor-text">{summary_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 3. Premium Budget Hero Card ───────────────────────────────

def render_premium_budget_card(
    amount: float,
    destination: str,
    duration: int,
    mode: str,
    hotel: str,
    confidence_score: int,
    confidence_level: str,
    season: str,
) -> None:
    """Renders the hero budget card — the primary visual focal point.

    Features a full gradient background, large budget amount, info pills
    for trip parameters, and badge row for category, confidence, and season.
    """
    # Determine budget category
    if amount < 5000:
        category    = "Budget Explorer 🏷️"
        badge_style = "badge-cheapest"
    elif amount < 10000:
        category    = "Value Explorer ⚡"
        badge_style = "badge-recommended"
    elif amount < 20000:
        category    = "Comfort Voyager 💜"
        badge_style = "badge-best-value"
    else:
        category    = "Luxury Voyager 💎"
        badge_style = "badge-fastest"

    season_icon = SEASON_ICONS.get(season.lower(), "🌍")
    mode_icons  = {"flight": "✈️", "train": "🚆", "bus": "🚌", "car": "🚗", "bike": "🏍️"}
    mode_icon   = mode_icons.get(mode.lower(), "🚗")
    hotel_icon  = HOTEL_ICONS.get(hotel.lower(), "🏨")

    st.markdown(
        f"""
        <div class="budget-hero">
            <div class="budget-hero-category">{category}</div>
            <div class="budget-label">ESTIMATED TOTAL BUDGET</div>
            <div class="budget-amount">{format_rupees(amount)}</div>
            <div class="budget-info-pills">
                <span class="budget-info-pill">🗓️ {duration} Days</span>
                <span class="budget-info-pill">{season_icon} {season.title()}</span>
                <span class="budget-info-pill">{mode_icon} {mode}</span>
                <span class="budget-info-pill">{hotel_icon} {hotel.title()} Hotel</span>
                <span class="budget-info-pill">📍 {destination.title()}</span>
            </div>
            <div class="budget-badges">
                <span class="badge {badge_style}">{category}</span>
                <span class="badge badge-recommended">🎯 {confidence_score}% Confidence</span>
                <span class="badge badge-best-value">{season_icon} {season.title()} Season</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 4. "Why This Budget?" Explainability Card ─────────────────

def render_budget_explainability_card(
    destination: str,
    days: int,
    hotel: str,
    season: str,
    travel_mode: str,
    similar_count: int,
    confidence_score: int,
) -> None:
    """Renders the recruiter-facing explainability card.

    Clearly lists the 7 factors the Random Forest Regressor used to
    arrive at the predicted budget — a critical card for interviews.
    """
    rows = [
        ("Destination",        destination.title()),
        ("Duration",           f"{days} days"),
        ("Hotel Quality",      hotel.title()),
        ("Season",             season.title()),
        ("Travel Mode",        travel_mode),
        ("Historical Records", f"{similar_count:,} similar travellers" if similar_count > 0 else "Dataset averages"),
        ("ML Algorithm",       "Random Forest Regressor"),
        ("Confidence Score",   f"{confidence_score}%"),
    ]

    rows_html = "".join(
        f"""
        <div class="explainability-row">
            <span class="explainability-check">✔</span>
            <span class="explainability-key">{key}:</span>
            <span class="explainability-value">{val}</span>
        </div>
        """
        for key, val in rows
    )

    st.markdown(
        f"""
        <div class="explainability-card">
            <div class="explainability-title">🔍 Why This Budget?</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 5. Historical Traveller Experience — Metric Cards ─────────

def render_traveller_experience_metric_cards(insights: Dict[str, Any]) -> None:
    """Renders individual metric cards for the 6 traveller experience dimensions.

    Replaces plain st.metric() calls with rich icon+value+progress cards.
    Each card shows: icon, numeric value, label, and a progress bar.
    """
    st.markdown('<p class="section-header">📈 Historical Traveller Experience</p>', unsafe_allow_html=True)

    # Build the 6 metric definitions
    score        = insights.get("destination_score", 4.0)
    hotel_r      = insights.get("avg_hotel_rating", 4.0)
    transport_r  = insights.get("avg_transport_rating", 4.0)
    sightseeing  = insights.get("avg_sightseeing_rating", 4.0)
    revisit_pct  = insights.get("revisit_percentage", 0.0)
    pref_style   = insights.get("preferred_experience", "Nature & Sightseeing")

    def _metric_card(icon: str, value: str, label: str, pct: Optional[float], color: str) -> str:
        """Build a single experience metric card HTML block."""
        prog = ""
        if pct is not None:
            prog = f"""
            <div class="mini-progress-track">
                <div class="mini-progress-fill" style="width:{min(pct, 100):.0f}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
            </div>"""
        return f"""
        <div class="experience-metric-card">
            <span class="experience-icon">{icon}</span>
            <div class="experience-value">{value}</div>
            <div class="experience-label-sm">{label}</div>
            {prog}
        </div>"""

    cards_html = (
        _metric_card("⭐", f"{score:.1f}/5", "Overall Satisfaction", score / 5 * 100, "#2563EB") +
        _metric_card("🏨", f"{hotel_r:.1f}/5", "Hotel Rating",        hotel_r / 5 * 100, "#7C3AED") +
        _metric_card("🚖", f"{transport_r:.1f}/5", "Transport Rating", transport_r / 5 * 100, "#06B6D4") +
        _metric_card("📍", f"{sightseeing:.1f}/5", "Sightseeing",     sightseeing / 5 * 100, "#10B981") +
        _metric_card("🔁", f"{revisit_pct:.0f}%", "Revisit Rate",    revisit_pct, "#F59E0B") +
        _metric_card("🎯", pref_style, "Preferred Style", None, "#A78BFA")
    )

    st.markdown(
        f'<div class="experience-metrics-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )


# ── 6. Travel Intelligence Card (Destination Summary) ─────────

def render_travel_intelligence_card(insights: Dict[str, Any], destination: str) -> None:
    """Renders the destination overview rating card with progress bar."""
    score = insights.get("destination_score", 4.0)
    stars = format_stars(score)
    count = insights.get("similar_count", 0)

    st.markdown(
        f"""
        <div class="dataset-insight" style="margin-bottom: 16px;">
            <div class="field-label">Smart Destination Summary</div>
            <h3 style="margin: 8px 0 4px 0; color: #FFFFFF; font-family: Outfit, sans-serif;">{destination.title()} Overview</h3>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 1.4rem; font-weight: 800; color: #06B6D4;">{stars} {score}/5.0</span>
                <span style="color: #94A3B8; font-size: 0.82rem;">Based on {count:,} travellers</span>
            </div>
            <div class="field-label" style="display: flex; justify-content: space-between;">
                <span>Overall Performance Rating</span>
                <span>{int(score / 5.0 * 100)}%</span>
            </div>
            <div style="background:#334155; border-radius:8px; height:8px; width:100%; overflow:hidden; margin-top:4px;">
                <div style="background:linear-gradient(90deg,#2563EB,#06B6D4); height:100%; width:{int(score/5.0*100)}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 7. Destination Summary Card ───────────────────────────────

def render_destination_summary_card(insights: Dict[str, Any], destination: str) -> None:
    """Renders a concise destination text summary using dataset fields."""
    avg_budget    = insights.get("average_budget", 0)
    most_month    = insights.get("most_popular_month", "N/A")
    most_season   = insights.get("most_popular_season", "N/A")
    pref_exp      = insights.get("preferred_experience", "N/A")
    revisit_pct   = insights.get("revisit_percentage", 0)
    hotel_rating  = insights.get("avg_hotel_rating", 0)

    parts = []
    if most_month and most_month != "N/A":
        parts.append(f"Most visitors travel to {destination.title()} during <strong>{most_month.title()} ({most_season.title()})</strong>.")
    if pref_exp and pref_exp != "N/A":
        parts.append(f"<strong>{pref_exp}</strong> is the preferred travel style.")
    if revisit_pct:
        parts.append(f"<strong>{revisit_pct:.0f}%</strong> of travellers would revisit.")
    if hotel_rating:
        parts.append(f"Average hotel rating is <strong>{hotel_rating:.1f}/5.0</strong>.")
    if avg_budget:
        parts.append(f"Average budget is <strong>{format_rupees(avg_budget)}</strong>.")

    summary = " ".join(parts) if parts else f"Dataset insights for {destination.title()} are based on historical records."

    st.markdown(
        f"""
        <div class="dataset-insight" style="border-left-color: #10B981; margin-bottom: 16px;">
            <h4 style="margin: 0 0 10px 0; color: #10B981; font-family: Outfit, sans-serif;">📊 Destination Summary</h4>
            <p style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6; margin: 0;">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 8. Weather Widget with Graceful Fallback ──────────────────

# Seasonal climate estimates — used when API is unavailable
_SEASONAL_CLIMATE = {
    "winter":  {"temp": 14.0, "humidity": 45, "wind": 8.0,  "desc": "Cool & Crisp ❄️"},
    "summer":  {"temp": 34.0, "humidity": 62, "wind": 14.0, "desc": "Hot & Sunny ☀️"},
    "rainy":   {"temp": 26.0, "humidity": 87, "wind": 18.0, "desc": "Warm & Humid 🌧️"},
    "monsoon": {"temp": 26.0, "humidity": 87, "wind": 18.0, "desc": "Warm & Humid 🌧️"},
    "autumn":  {"temp": 23.0, "humidity": 55, "wind": 10.0, "desc": "Pleasant & Breezy 🍂"},
    "spring":  {"temp": 24.0, "humidity": 52, "wind": 9.0,  "desc": "Mild & Refreshing 🌸"},
}

def render_weather_widget(
    weather: Dict[str, Any],
    best_time: str,
    season: str = "summer",
) -> None:
    """Renders the weather card — always shows useful information.

    Priority:
      1. Live data from Open-Meteo API (green 🟢 Live badge)
      2. Seasonal historical estimate (yellow 📅 badge)

    Never displays: "Data unavailable", "None", "N/A", "API Error".
    """
    st.markdown('<p class="section-header">🌤️ Destination Weather</p>', unsafe_allow_html=True)

    is_live = weather.get("source") == "Open-Meteo API"

    if is_live:
        # Use real API data
        temp  = weather.get("temperature_c", 25.0)
        desc  = weather.get("description",  "Partly Cloudy ⛅")
        hum   = weather.get("humidity",     55)
        wind  = weather.get("wind_speed",   10.0)
        badge = '<span class="weather-live-badge">🟢 Live Data</span>'
        note  = ""
    else:
        # Seasonal estimate — graceful fallback
        climate = _SEASONAL_CLIMATE.get(season.lower(), _SEASONAL_CLIMATE["summer"])
        temp  = climate["temp"]
        desc  = climate["desc"]
        hum   = climate["humidity"]
        wind  = climate["wind"]
        badge = '<span class="weather-estimate-badge">📅 Seasonal Estimate</span>'
        note  = '<p class="weather-advice">Showing seasonal climate estimate based on historical travel data.</p>'

    # Travel advice based on temperature
    if temp < 15:
        advice = "🧥 Cold climate — heavy woolens recommended. Ideal for snow activities."
    elif temp > 32:
        advice = "☀️ Warm climate — light cotton clothing advised. Stay well hydrated."
    else:
        advice = "😊 Pleasant weather — perfect conditions for sightseeing and exploration."

    st.markdown(
        f"""
        <div class="weather-widget">
            {badge}
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 8px;">
                <div>
                    <div class="weather-temp-large">{temp}°C</div>
                    <div class="weather-desc">{desc}</div>
                </div>
                <div style="text-align: right;">
                    <div class="field-label">Best Time to Visit</div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #F59E0B; margin-top: 4px;">{best_time}</div>
                </div>
            </div>
            <div class="weather-stats-row">
                <div class="weather-stat-item">💧 Humidity: <strong>{hum}%</strong></div>
                <div class="weather-stat-item">💨 Wind: <strong>{wind} km/h</strong></div>
            </div>
            <p class="weather-advice">{advice}</p>
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 9. Budget Breakdown Table ─────────────────────────────────

def render_budget_breakdown_table(cost: float) -> None:
    """Renders the detailed budget allocation breakdown card."""
    st.markdown('<p class="section-header">💸 Detailed Budget Allocation</p>', unsafe_allow_html=True)

    allocations = [
        ("Stay / Hotel",         cost * 0.35, "🏨"),
        ("Transit / Travel",     cost * 0.20, "✈️"),
        ("Food & Dining",        cost * 0.15, "🍜"),
        ("Local Transport",      cost * 0.10, "🚖"),
        ("Activities",           cost * 0.10, "⛰️"),
        ("Shopping",             cost * 0.05, "🛍️"),
        ("Emergency Reserve",    cost * 0.05, "🚨"),
    ]

    rows_html = "".join(
        f"""
        <div class="breakdown-row">
            <span>{icon} {title}</span>
            <strong style="color:#06B6D4;">{format_rupees(val)}</strong>
        </div>
        """
        for title, val, icon in allocations
    )

    st.markdown(
        f"""
        <div class="budget-breakdown-card">
            {rows_html}
            <div class="breakdown-total">
                <span>💰 Total Budget</span>
                <strong style="color:#2563EB; font-size:1.05rem;">{format_rupees(cost)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 10. Budget Verification Card ─────────────────────────────

def render_budget_verification_card(
    ml_pred: float,
    hist_avg: float,
    travel_cost: float,
    smart_rec: float,
    confidence_score: int,
) -> None:
    """Renders comparison card: ML prediction vs historical average vs smart estimate."""
    st.markdown('<p class="section-header">⚖️ Smart Budget Verification</p>', unsafe_allow_html=True)

    diff       = smart_rec - ml_pred
    diff_text  = f"+{format_rupees(diff)}" if diff >= 0 else f"-{format_rupees(abs(diff))}"
    diff_color = "#EF4444" if diff > 0 else "#10B981"

    st.markdown(
        f"""
        <div class="weather-widget">
            <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:14px; font-size:0.85rem; color:#CBD5E1;">
                <div>🤖 ML Prediction: <strong style="color:#F1F5F9;">{format_rupees(ml_pred)}</strong></div>
                <div>⏳ Historical Avg: <strong style="color:#F1F5F9;">{format_rupees(hist_avg)}</strong></div>
                <div>🚗 Transit Cost: <strong style="color:#F1F5F9;">{format_rupees(travel_cost)}</strong></div>
                <div>🔮 Smart Estimate: <strong style="color:#2563EB;">{format_rupees(smart_rec)}</strong></div>
            </div>
            <div class="divider" style="margin:12px 0;"></div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.85rem;">
                <span>Adjustment: <strong style="color:{diff_color};">{diff_text}</strong></span>
                <span>Match Score: <strong style="color:#10B981;">{confidence_score}%</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 11. Travel Mode Comparison Cards ─────────────────────────

# Default estimation constants when API route data is unavailable
_MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "Flight": {"speed_kph": 850, "cost_per_km": 6.0,  "comfort": "Premium",  "availability": "Major Routes"},
    "Train":  {"speed_kph": 80,  "cost_per_km": 0.8,  "comfort": "Standard", "availability": "Always"},
    "Bus":    {"speed_kph": 55,  "cost_per_km": 0.5,  "comfort": "Economy",  "availability": "Always"},
    "Car":    {"speed_kph": 70,  "cost_per_km": 3.5,  "comfort": "Premium",  "availability": "Always"},
    "Bike":   {"speed_kph": 50,  "cost_per_km": 1.5,  "comfort": "Economy",  "availability": "Always"},
}

_MODE_ICONS = {"Flight": "✈️", "Train": "🚆", "Bus": "🚌", "Car": "🚗", "Bike": "🏍️"}


def render_mode_comparison_cards(
    modes: Dict[str, Any],
    selected_mode: str,
    distance_km: float = 400.0,
) -> None:
    """Renders 5 travel mode cards — never leaves any card empty.

    If API data is missing for a mode, estimates cost and duration using
    dataset-derived speed/cost constants and the route distance.
    Displays: cost, duration, comfort level, availability, and badges.
    """
    st.markdown('<p class="section-header">🚗 Travel Mode Comparison</p>', unsafe_allow_html=True)

    mode_list = ["Flight", "Train", "Bus", "Car", "Bike"]

    # Build enriched mode data (API values preferred, estimated as fallback)
    enriched: Dict[str, Dict[str, Any]] = {}
    for mode in mode_list:
        defaults = _MODE_DEFAULTS[mode]
        api_data = modes.get(mode, {})

        cost = api_data.get("round_trip") or 0
        if not cost:
            # Estimate: base cost per km × distance (round trip × 2)
            cost = int(defaults["cost_per_km"] * distance_km * 2)

        duration_h = api_data.get("duration_hours") or 0
        if not duration_h:
            duration_h = distance_km / defaults["speed_kph"]

        if duration_h < 1:
            dur_text = f"{int(duration_h * 60)} min"
        elif duration_h < 24:
            dur_text = f"{duration_h:.1f} hrs"
        else:
            dur_text = f"{duration_h / 24:.1f} days"

        enriched[mode] = {
            "cost":         cost,
            "duration_h":   duration_h,
            "duration_text": api_data.get("duration_text") or dur_text,
            "comfort":      defaults["comfort"],
            "availability": defaults["availability"],
        }

    # Determine badge winners
    cheapest = min(enriched.keys(), key=lambda k: enriched[k]["cost"])
    fastest  = min(enriched.keys(), key=lambda k: enriched[k]["duration_h"])

    cols = st.columns(5)
    for idx, mode in enumerate(mode_list):
        data           = enriched[mode]
        is_recommended = mode.lower() == selected_mode.lower()
        selected_style = "recommended-mode" if is_recommended else ""

        # Build badge HTML
        badge_parts = []
        if is_recommended:
            badge_parts.append('<span class="badge badge-recommended" style="font-size:0.58rem; padding:2px 6px;">⭐ PICK</span>')
        if mode == cheapest:
            badge_parts.append('<span class="badge badge-cheapest" style="font-size:0.58rem; padding:2px 6px;">💚 CHEAPEST</span>')
        if mode == fastest:
            badge_parts.append('<span class="badge badge-fastest" style="font-size:0.58rem; padding:2px 6px;">⚡ FASTEST</span>')
        badges_html = "<br>".join(badge_parts)

        cols[idx].markdown(
            f"""
            <div class="mode-card {selected_style}">
                <span class="mode-card-icon">{_MODE_ICONS[mode]}</span>
                <div class="mode-card-name">{mode}</div>
                <div class="mode-card-cost">₹{data['cost']:,}</div>
                <div class="mode-card-duration">{data['duration_text']}</div>
                <div class="mode-card-meta">
                    <div class="mode-meta-row"><strong>Comfort:</strong> {data['comfort']}</div>
                    <div class="mode-meta-row"><strong>Routes:</strong> {data['availability']}</div>
                </div>
                <div style="margin-top: 8px; min-height: 20px;">{badges_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── 12. Route Details (Visual Flow) ──────────────────────────

def render_route_details(route: Dict[str, Any], smart_cost: Dict[str, Any]) -> None:
    """Renders route info as a visual origin → destination flow with connector."""
    st.markdown('<p class="section-header">📍 Route Details</p>', unsafe_allow_html=True)

    origin   = route.get("origin",        "Origin")
    dest     = route.get("destination",   "Destination")
    dist     = route.get("distance_km",   0.0)
    dur      = route.get("duration_text", "N/A")
    mode     = route.get("travel_mode",   "Car")
    tr_cost  = smart_cost.get("travel_cost_estimate", 0)

    connector_stats = ""
    if dist:
        connector_stats += f"<span><strong>{dist:,.1f} km</strong></span>"
    if dur and dur.lower() not in ("n/a", "none", ""):
        connector_stats += f" &nbsp;·&nbsp; <span><strong>{dur}</strong></span>"
    if mode:
        connector_stats += f" &nbsp;·&nbsp; <span><strong>{mode}</strong></span>"

    st.markdown(
        f"""
        <div class="route-flow-card">
            <div class="route-flow-inner">
                <div class="route-node">
                    <div class="route-node-dot origin"></div>
                    <div class="route-node-label">{origin.title()}</div>
                </div>
                <div class="route-connector">
                    <div class="route-connector-line"></div>
                    <div class="route-connector-stats" style="font-size:0.78rem; color:#94A3B8;">
                        {connector_stats}
                    </div>
                </div>
                <div class="route-node">
                    <div class="route-node-dot dest"></div>
                    <div class="route-node-label">{dest.title()}</div>
                </div>
            </div>
            <div class="route-cost-row">
                <span style="color:#94A3B8; font-size:0.82rem;">💰 Estimated Transit Cost</span>
                <strong style="color:#06B6D4; font-size:0.95rem;">₹{int(tr_cost):,}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 13. Interactive Map Card ──────────────────────────────────

def render_interactive_map_card(route: Dict[str, Any], weather: Dict[str, Any]) -> None:
    """Renders a styled SVG route map showing city nodes and route arc."""
    st.markdown('<p class="section-header">🗺️ Route Map Preview</p>', unsafe_allow_html=True)

    origin = route.get("origin",       "Source")
    dest   = route.get("destination",  "Destination")
    dist   = route.get("distance_km",  0.0)
    dur    = route.get("duration_text", "N/A")
    temp   = weather.get("temperature_c", 25.0)
    desc   = weather.get("description", "Pleasant")

    st.markdown(
        f"""
        <div class="map-card">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94A3B8; margin-bottom:10px;">
                <span>Origin: <strong style="color:#F1F5F9;">{origin.title()}</strong></span>
                <span>Destination: <strong style="color:#F1F5F9;">{dest.title()}</strong></span>
            </div>
            <svg viewBox="0 0 500 110" style="width:100%; height:auto; display:block; overflow:visible;">
                <defs>
                    <linearGradient id="map-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#2563EB"/>
                        <stop offset="100%" stop-color="#7C3AED"/>
                    </linearGradient>
                </defs>
                <path d="M 70,65 Q 250,15 430,65" fill="none" stroke="url(#map-grad)" stroke-width="2.5" stroke-dasharray="6,4"/>
                <circle cx="70"  cy="65" r="8" fill="#2563EB"/>
                <circle cx="70"  cy="65" r="14" fill="none" stroke="#2563EB" stroke-width="1.5" opacity="0.4"/>
                <circle cx="430" cy="65" r="8" fill="#7C3AED"/>
                <circle cx="430" cy="65" r="14" fill="none" stroke="#7C3AED" stroke-width="1.5" opacity="0.4"/>
                <text x="70"  y="90" font-family="Outfit,sans-serif" font-size="11" fill="#F1F5F9" text-anchor="middle">{origin.title()}</text>
                <text x="430" y="90" font-family="Outfit,sans-serif" font-size="11" fill="#F1F5F9" text-anchor="middle">{dest.title()}</text>
                <text x="250" y="30" font-family="Inter,sans-serif" font-size="10" fill="#06B6D4" text-anchor="middle">{dist:,.1f} km · {dur}</text>
            </svg>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94A3B8; margin-top:10px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.06);">
                <span>🌤️ Weather: <strong style="color:#F1F5F9;">{temp}°C, {desc}</strong></span>
                <span>Mode: <strong style="color:#F1F5F9;">{route.get('travel_mode', 'Car')}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 14. Checklists (Premium) ──────────────────────────────────

def render_checklists_premium(packing_tips: List[str]) -> None:
    """Renders premium grouped checklists: packing + pre-travel documents.

    Uses card backgrounds with category group headers and Streamlit
    native checkboxes for interactivity.
    """
    st.markdown('<p class="section-header">🎒 Travel Planning Checklists</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="checklist-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Outfit,sans-serif; font-weight:700; font-size:0.95rem; color:#F1F5F9; margin:0 0 12px 0;">🎒 Packing Checklist</p>', unsafe_allow_html=True)

        # Group packing tips into categories
        essentials = packing_tips[:3]  if len(packing_tips) > 3 else packing_tips
        clothing   = packing_tips[3:6] if len(packing_tips) > 3 else []
        rest       = packing_tips[6:]  if len(packing_tips) > 6 else []

        if essentials:
            st.markdown('<div class="checklist-category-header">Essentials</div>', unsafe_allow_html=True)
            for i, item in enumerate(essentials):
                st.checkbox(item, key=f"pack_e_{i}", value=False)

        if clothing:
            st.markdown('<div class="checklist-category-header">Clothing & Gear</div>', unsafe_allow_html=True)
            for i, item in enumerate(clothing):
                st.checkbox(item, key=f"pack_c_{i}", value=False)

        if rest:
            st.markdown('<div class="checklist-category-header">Health & Electronics</div>', unsafe_allow_html=True)
            for i, item in enumerate(rest):
                st.checkbox(item, key=f"pack_r_{i}", value=False)

        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="checklist-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Outfit,sans-serif; font-weight:700; font-size:0.95rem; color:#F1F5F9; margin:0 0 12px 0;">📁 Pre-Travel Checklist</p>', unsafe_allow_html=True)

        pre_travel = {
            "Documents": [
                "Primary ID Proof (Aadhaar / Voter ID)",
                "Confirmed Travel Tickets (Flight / Train)",
                "Hotel Booking Confirmation",
            ],
            "Finance": [
                "Physical Cash & Credit / Debit Cards",
                "UPI apps set up for offline payments",
            ],
            "Safety": [
                "Emergency Contacts list",
                "Required Medicines & Charger",
            ],
        }

        for category, items in pre_travel.items():
            st.markdown(f'<div class="checklist-category-header">{category}</div>', unsafe_allow_html=True)
            for i, doc in enumerate(items):
                st.checkbox(doc, key=f"doc_{category}_{i}", value=False)

        st.markdown('</div>', unsafe_allow_html=True)


# ── 15. Recommendation Deck ───────────────────────────────────

def render_recommendation_deck(intelligence: Dict[str, Any]) -> None:
    """Renders two side-by-side recommendation cards: Activities + Highlights."""
    st.markdown('<p class="section-header">🎯 Curated Recommendations</p>', unsafe_allow_html=True)

    activities = intelligence.get("experience_activities", [])
    pref_style = intelligence.get("preferred_experience", "Nature & Sightseeing")
    c1, c2 = st.columns(2)

    with c1:
        acts_html = "".join(
            f'<li style="margin-bottom:6px;">{act}</li>'
            for act in activities
        )
        st.markdown(
            f"""
            <div class="rec-card" style="border-left: 4px solid #10B981;">
                <h5 style="margin:0 0 12px 0; color:#10B981; font-family:Outfit,sans-serif;">
                    ✨ Recommended Activities
                </h5>
                <div style="font-size:0.72rem; color:#94A3B8; margin-bottom:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px;">
                    For {pref_style} travellers
                </div>
                <ul style="margin:0; padding-left:18px; color:#CBD5E1; font-size:0.87rem; line-height:1.7;">
                    {acts_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        places  = ", ".join(intelligence.get("places_to_visit", [])[:3])
        foods   = ", ".join(intelligence.get("local_foods", [])[:3])
        gems    = ", ".join(intelligence.get("hidden_gems", [])[:2])
        safety  = intelligence.get("safety_tips", ["Travel safe"])[0]
        tips    = intelligence.get("transportation_advice", "Use local cabs")

        highlights = [
            ("🏞️", "Top Attractions", places  or "Explore local sights"),
            ("🍜", "Local Cuisine",   foods   or "Try local dishes"),
            ("🌟", "Hidden Gems",     gems    or "Ask locals for tips"),
            ("🛡️", "Safety Tip",     safety),
            ("🚗", "Transport Tip",  tips),
        ]
        rows_html = "".join(
            f"""
            <div style="padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.84rem; color:#CBD5E1;">
                <span style="font-size:0.95rem;">{icon}</span>
                <strong style="color:#F1F5F9; margin:0 6px 0 4px;">{label}:</strong>
                {value}
            </div>
            """
            for icon, label, value in highlights
        )

        st.markdown(
            f"""
            <div class="rec-card" style="border-left: 4px solid #F59E0B;">
                <h5 style="margin:0 0 12px 0; color:#F59E0B; font-family:Outfit,sans-serif;">💎 Curated Highlights</h5>
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── 16. Similar Travellers Card ───────────────────────────────

def render_similar_travellers_card(similar_data: Dict[str, Any], trip_type: str) -> None:
    """Renders profile-matching 'Travellers like you' statistics."""
    st.markdown('<p class="section-header">👥 Similar Traveller Profile</p>', unsafe_allow_html=True)

    if not similar_data.get("has_data", False):
        st.caption("Not enough dataset records to draw similar traveller profiles.")
        return

    stats = [
        ("Average Cost Spent",       format_rupees(similar_data["avg_spending"]),        "#F1F5F9"),
        ("Preferred Transit Mode",   similar_data["fav_transport"],                     "#06B6D4"),
        ("Preferred Experience",     similar_data["fav_experience"],                    "#10B981"),
        ("Preferred Hotel Grade",    similar_data["fav_hotel_quality"].title(),         "#F59E0B"),
    ]

    stats_html = "".join(
        f"""
        <div>
            <div class="field-label">{label}</div>
            <div style="font-size:1.35rem; font-weight:800; color:{color}; margin-top:2px;">{value}</div>
        </div>
        """
        for label, value, color in stats
    )

    st.markdown(
        f"""
        <div class="similar-card">
            <h4 style="margin:0 0 16px 0; color:#7C3AED; font-family:Outfit,sans-serif;">
                👥 Travellers Like You
            </h4>
            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:16px;">
                {stats_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 17. Saving Tips ───────────────────────────────────────────

def render_saving_tips_widget(tips: List[str]) -> None:
    """Renders cost-saving recommendations as a styled card."""
    st.markdown('<p class="section-header">💡 Cost Saving Tips</p>', unsafe_allow_html=True)

    tips_html = "".join(
        f'<div style="font-size:0.86rem; color:#CBD5E1; margin-bottom:8px; display:flex; gap:8px; align-items:flex-start;">'
        f'<span style="color:#10B981; flex-shrink:0;">💡</span><span>{tip}</span></div>'
        for tip in tips
    )

    st.markdown(
        f'<div class="dataset-insight" style="border-left-color:#10B981;">{tips_html}</div>',
        unsafe_allow_html=True,
    )


# ── 18. Success Banner ────────────────────────────────────────

def render_success_banner() -> None:
    """Renders the 4-item success checklist banner shown after loading completes."""
    items = [
        ("Budget Estimated", "💰"),
        ("Weather Retrieved", "🌤️"),
        ("Route Calculated", "📍"),
        ("Recommendations Generated", "🎯"),
    ]
    items_html = "".join(
        f"""
        <div class="success-item">
            <span class="success-check">✓</span>
            {emoji} {label}
        </div>
        """
        for label, emoji in items
    )
    st.markdown(
        f'<div class="success-banner">{items_html}</div>',
        unsafe_allow_html=True,
    )


# ── 19. Footer ────────────────────────────────────────────────

def render_footer() -> None:
    """Renders the premium footer with brand, social links, and developer credit.

    Layout matches the TripAI design: brand logo on the left,
    Love with Travel badge + Made by credit in the centre, and
    clickable LinkedIn / GitHub icon buttons on the right.
    """
    tech_stack = [
        ("🐍", "Python"),
        ("🌲", "Random Forest"),
        ("⚡", "Streamlit"),
        ("🗄️", "SQLite"),
        ("🗺️", "Google Maps"),
        ("🌤️", "Open Meteo"),
    ]

    pills_html = "".join(
        f'<span class="footer-tech-pill">{icon} {name}</span>'
        for icon, name in tech_stack
    )

    st.markdown(
        f"""
        <div class="footer-premium">

            <!-- 3-column main row -->
            <div class="footer-main-row">

                <!-- Left: Brand -->
                <div class="footer-brand-col">
                    <div>
                        <div class="footer-brand-logo">✈️ TripAI</div>
                        <div class="footer-brand-sub">
                            AI-Powered Travel Intelligence &amp; Budget Planning Platform
                        </div>
                    </div>
                </div>

                <!-- Centre: Love + Credit -->
                <div class="footer-center-col">
                    <div class="footer-love-badge">
                        <span class="heart">❤️</span>&nbsp;Love with Travel
                    </div>
                    <div class="footer-made-by">
                        Made by <strong>Srujana Addanki</strong>
                    </div>
                </div>

                <!-- Right: Social Links + Copyright -->
                <div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
                    <div class="footer-social-col">
                        <a
                            href="https://www.linkedin.com/in/srujana-addanki/"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="footer-social-link linkedin"
                            title="Connect on LinkedIn"
                        >in</a>
                        <a
                            href="https://github.com/Srujanaaddanki"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="footer-social-link github"
                            title="View on GitHub"
                        >&#128027;</a>
                    </div>
                    <div class="footer-copyright">© 2026 TripAI. All rights reserved.</div>
                </div>

            </div>

            <div class="footer-divider"></div>

            <!-- Tech-stack pills row -->
            <div class="footer-tech-stack">
                <span class="footer-tech-label">Built with</span>
                {pills_html}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

