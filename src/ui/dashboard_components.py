"""
========================================================
Module: Dashboard Components
Purpose: Renders modern premium SaaS dashboard components,
         visual cards, widgets, and weather indicators
         for the TripAI prediction results page.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict, List


def format_stars(score: float) -> str:
    """Format numeric score to filled and empty star unicode symbols."""
    rounded = min(5, max(0, int(round(score))))
    return "★" * rounded + "☆" * (5 - rounded)


def format_rupees(amount: float) -> str:
    """Format rupee numeric values into comma-separated currency values."""
    return f"₹{amount:,.0f}"


def render_premium_budget_card(
    amount: float,
    destination: str,
    duration: int,
    mode: str,
    hotel: str,
    confidence_score: int,
    confidence_level: str,
    season: str
) -> None:
    """Renders Feature 5: Hero budget card with modern CSS styling overlay."""
    # Categorise traveller budget grade (Feature 3)
    if amount < 5000:
        category = "Budget Explorer 🏷️"
        badge_style = "badge-cheapest"
    elif amount < 10000:
        category = "Value Explorer ⚡"
        badge_style = "badge-recommended"
    elif amount < 20000:
        category = "Comfort Voyager 💜"
        badge_style = "badge-best-value"
    else:
        category = "Luxury Voyager 💎"
        badge_style = "badge-fastest"

    st.markdown(
        f"""
        <div class="budget-hero">
            <div class="field-label" style="color: rgba(255,255,255,0.7);">{category}</div>
            <div class="budget-amount">₹{int(amount):,}</div>
            <div class="budget-meta">
                Plan: {duration} Days in {destination.title()} | Transit: {mode} | Stay: {hotel.title()}
            </div>
            <div style="margin-top: 16px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;">
                <span class="badge {badge_style}">Category: {category}</span>
                <span class="badge badge-recommended">Confidence: {confidence_score}%</span>
                <span class="badge badge-best-value">{season.title()} Season</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_budget_breakdown_table(cost: float) -> None:
    """Renders Feature 2: Detailed budget breakdown card using dataset splits."""
    st.markdown('<p class="section-header">💸 Detailed Budget Allocation</p>', unsafe_allow_html=True)
    
    allocations = [
        ("Stay/Hotel (35%)", cost * 0.35, "🏨"),
        ("Transit/Travel (20%)", cost * 0.20, "✈️"),
        ("Food & Dining (15%)", cost * 0.15, "🍜"),
        ("Local Transport (10%)", cost * 0.10, "🚖"),
        ("Activities (10%)", cost * 0.10, "⛰️"),
        ("Shopping (5%)", cost * 0.05, "🛍️"),
        ("Emergency Reserve (5%)", cost * 0.05, "🚨"),
    ]

    st.markdown('<div class="weather-widget" style="padding:16px;">', unsafe_allow_html=True)
    for title, val, icon in allocations:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding: 6px 0; font-size: 0.88rem; border-bottom: 1px solid rgba(255,255,255,0.03);">
                <span>{icon} {title}</span>
                <strong style="color: #06B6D4;">{format_rupees(val)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding: 10px 0 0 0; font-size: 0.95rem; font-weight:700;">
            <span>💰 Total Budget Allocation</span>
            <strong style="color: #2563EB;">{format_rupees(cost)}</strong>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_budget_verification_card(
    ml_pred: float,
    hist_avg: float,
    travel_cost: float,
    smart_rec: float,
    confidence_score: int
) -> None:
    """Renders Feature 1: Verification card comparing ML, historical and transit costs."""
    st.markdown('<p class="section-header">⚖️ Smart Budget Verification</p>', unsafe_allow_html=True)
    
    diff = smart_rec - ml_pred
    diff_text = f"+{format_rupees(diff)}" if diff >= 0 else f"-{format_rupees(abs(diff))}"
    diff_color = "#EF4444" if diff > 0 else "#10B981"

    st.markdown(
        f"""
        <div class="weather-widget">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; font-size: 0.85rem; color: #CBD5E1;">
                <div>🤖 ML Model Prediction: <strong>{format_rupees(ml_pred)}</strong></div>
                <div>⏳ Historical Average: <strong>{format_rupees(hist_avg)}</strong></div>
                <div>🚗 Est. Transport Cost: <strong>{format_rupees(travel_cost)}</strong></div>
                <div>🔮 Smart Recommended: <strong style="color: #2563EB;">{format_rupees(smart_rec)}</strong></div>
            </div>
            <div class="divider" style="margin:12px 0;"></div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.85rem;">
                <span>Recommended Adjustment: <strong style="color:{diff_color};">{diff_text}</strong></span>
                <span>Verification score: <strong style="color:#10B981;">{confidence_score}% Match</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_checklists_widget(packing_tips: List[str]) -> None:
    """Renders Feature 5 & 6: Interactive packing and pre-travel checklists."""
    st.markdown('<p class="section-header">🎒 Travel Planning Checklists</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.write("##### 🎒 Custom Packing List")
        for idx, item in enumerate(packing_tips):
            st.checkbox(item, key=f"pack_{idx}", value=False)

    with c2:
        st.write("##### 📁 Pre-Travel Documents")
        docs = [
            "Primary ID Proof (Aadhaar/Voter ID)",
            "Confirmed Travel Tickets (Flight/Train)",
            "Hotel Booking Confirmation",
            "Physical Cash & Credit/Debit Cards",
            "Emergency Contacts list",
            "Required Medicines & Charger",
        ]
        for idx, doc in enumerate(docs):
            st.checkbox(doc, key=f"doc_{idx}", value=False)


def render_saving_tips_widget(tips: List[str]) -> None:
    """Renders Feature 7: Cost saving recommendations list."""
    st.markdown('<p class="section-header">💡 Cost Saving Recommendations</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="dataset-insight" style="border-left-color: #10B981; padding: 16px;">', unsafe_allow_html=True)
    for tip in tips:
        st.markdown(f'<div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 6px;">💡 {tip}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_travel_intelligence_card(insights: Dict[str, Any], destination: str) -> None:
    """Renders Feature 4: Destination overview rating scores and progress bar."""
    score = insights.get("destination_score", 4.0)
    stars = format_stars(score)
    count = insights.get("similar_count", 0)

    st.markdown(
        f"""
        <div class="dataset-insight" style="margin-bottom: 20px;">
            <div class="field-label">Smart Destination Summary</div>
            <h3 style="margin: 8px 0; color: #FFFFFF; font-family: Outfit, sans-serif;">{destination.title()} Overview</h3>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 1.6rem; font-weight: 800; color: #06B6D4;">{stars} {score}/5.0</span>
                <span style="color: #94A3B8; font-size: 0.85rem;">(Based on {count} previous travellers)</span>
            </div>
            <div style="margin: 16px 0;">
                <div class="field-label" style="display: flex; justify-content: space-between;">
                    <span>Overall Performance Rating</span>
                    <span>{int(score / 5.0 * 100)}%</span>
                </div>
                <div style="background-color: #334155; border-radius: 8px; height: 10px; width: 100%; overflow: hidden; margin-top: 4px;">
                    <div style="background: linear-gradient(90deg, #2563EB, #06B6D4); height: 100%; width: {int(score / 5.0 * 100)}%;"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_traveller_experience_widget(insights: Dict[str, Any]) -> None:
    """Renders Feature 4: Aggregated dataset traveller experience metrics."""
    st.markdown('<p class="section-header">📈 Historical Traveller Experience</p>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    cols[0].metric("Overall Rating ⭐", f"{insights.get('destination_score', 4.0)}/5.0")
    cols[1].metric("Hotel Stay 🏨", f"{insights.get('avg_hotel_rating', 4.0)}/5.0")
    cols[2].metric("Local Transit 🚖", f"{insights.get('avg_transport_rating', 4.0)}/5.0")
    cols[3].metric("Sightseeing 📍", f"{insights.get('avg_sightseeing_rating', 4.0)}/5.0")

    st.markdown(
        f"""
        <div class="dataset-insight" style="margin-top: 15px; border-left-color: #7C3AED;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.85rem; color: #CBD5E1;">
                <div>💼 Preferred Style: <strong>{insights.get('preferred_experience', 'N/A')}</strong></div>
                <div>🏨 Preferred Hotel: <strong>{insights.get('most_preferred_hotel', 'N/A')}</strong></div>
                <div>🚗 Top Transit Mode: <strong>{insights.get('most_used_travel_mode', 'N/A')}</strong></div>
                <div>🔁 Revisit Intention: <strong>{insights.get('revisit_percentage', 0.0)}% Yes</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_destination_summary_card(insights: Dict[str, Any], destination: str) -> None:
    """Renders Feature 4: Dedicated destination overview text card."""
    summary_text = (
        f"Most visitors travel to {destination.title()} during "
        f"{insights.get('most_popular_month')} ({insights.get('most_popular_season')}). "
        f"'{insights.get('preferred_experience')}' is the preferred style. "
        f"{insights.get('revisit_percentage')}% would revisit. "
        f"Average hotel rating is {insights.get('avg_hotel_rating')}/5.0. "
        f"Average budget is {format_rupees(insights.get('average_budget'))}."
    )
    st.markdown(
        f"""
        <div class="dataset-insight" style="border-left-color: #10B981;">
            <h4 style="margin: 0 0 10px 0; color: #10B981; font-family: Outfit, sans-serif;">Smart Summary Summary</h4>
            <p style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5; margin: 0;">{summary_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_weather_widget(weather: Dict[str, Any], best_time: str) -> None:
    """Renders Feature 10: Weather card and travel advisor."""
    st.markdown('<p class="section-header">🌤️ Destination Weather Widget</p>', unsafe_allow_html=True)
    temp = weather.get("temperature_c", 25.0)
    desc = weather.get("description", "Clear Sky ☀️")
    
    if temp < 15.0:
        advice = "Cold Climate — Heavy woollens recommended. Best for snow activities."
    elif temp > 30.0:
        advice = "Warm Climate — Keep light cotton clothes and stay hydrated."
    else:
        advice = "Pleasant weather — Perfect for sightseeing and walking tours."

    st.markdown(
        f"""
        <div class="weather-widget">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size: 2.2rem; font-weight: 800; font-family: Outfit, sans-serif; color: #FFFFFF;">{temp}°C</div>
                    <div style="font-size: 0.95rem; color: #06B6D4; font-weight: 600; margin-top: 4px;">{desc}</div>
                </div>
                <div style="text-align: right;">
                    <div class="field-label">Best Time to Visit</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: #F59E0B; margin-top: 4px;">{best_time}</div>
                </div>
            </div>
            <div class="divider" style="margin: 15px 0 10px 0;"></div>
            <div style="display: flex; gap: 20px; font-size: 0.85rem; color: #94A3B8;">
                <div>💧 Humidity: <strong style="color:#F1F5F9;">{weather.get('humidity', 50)}%</strong></div>
                <div>💨 Wind: <strong style="color:#F1F5F9;">{weather.get('wind_speed', 10.0)} km/h</strong></div>
            </div>
            <p style="font-size: 0.8rem; color: #CBD5E1; margin: 10px 0 0 0; font-style: italic;">💡 {advice}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_route_details(route: Dict[str, Any], smart_cost: Dict[str, Any]) -> None:
    """Renders Feature 11: Route detail metric cards."""
    st.markdown('<p class="section-header">📍 Route Details & Cost Analysis</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    c1.metric("📏 Distance", f"{route.get('distance_km', 0.0):,.1f} km", help="Geographical distance between cities")
    c2.metric("⏱️ Travel Time", route.get("duration_text", "N/A"), help="Calculated travel duration")
    c3.metric("💰 Est. Transit Cost", f"₹{int(smart_cost.get('travel_cost_estimate', 0)):,}", help="Transit fuel/tickets projection")


def render_mode_comparison_cards(modes: Dict[str, Any], selected_mode: str) -> None:
    """Renders Feature 12: Transport mode comparisons with hover styles."""
    st.markdown('<p class="section-header">🚗 Travel Mode Speed & Cost Options</p>', unsafe_allow_html=True)
    cols = st.columns(5)

    mode_list = ["Flight", "Train", "Bus", "Car", "Bike"]
    icons = {"Flight": "✈️", "Train": "🚆", "Bus": "🚌", "Car": "🚗", "Bike": "🏍️"}

    valid_modes = {m: modes[m] for m in mode_list if m in modes}
    cheapest = min(valid_modes.keys(), key=lambda k: valid_modes[k].get("round_trip", 999999)) if valid_modes else "Train"
    fastest = min(valid_modes.keys(), key=lambda k: valid_modes[k].get("duration_hours", 999)) if valid_modes else "Flight"

    for idx, mode in enumerate(mode_list):
        if mode not in modes:
            cols[idx].markdown(
                f"""
                <div class="mode-card"><div style="font-size: 1.4rem;">{icons[mode]}</div>
                <div style="font-weight: 700; margin-top: 6px; font-size:0.85rem;">{mode}</div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 6px;">No route</div></div>
                """, unsafe_allow_html=True
            )
            continue

        data = modes[mode]
        cost = int(data.get("round_trip", 0))
        selected_style = "recommended-mode" if mode.lower() == selected_mode.lower() else ""
        badge = ""
        if mode == cheapest:
            badge = '<span class="badge badge-cheapest" style="padding: 2px 6px; font-size: 0.6rem; display:block; margin: 4px auto 0 auto; width: fit-content;">CHEAPEST</span>'
        elif mode == fastest:
            badge = '<span class="badge badge-fastest" style="padding: 2px 6px; font-size: 0.6rem; display:block; margin: 4px auto 0 auto; width: fit-content;">FASTEST</span>'

        cols[idx].markdown(
            f"""
            <div class="mode-card {selected_style}">
                <div style="font-size: 1.4rem;">{icons[mode]}</div>
                <div style="font-weight: 700; margin-top: 6px; font-size:0.85rem;">{mode}</div>
                <div style="color: #06b6d4; font-size: 1.05rem; font-weight: 800; margin: 4px 0;">₹{cost:,}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Est. {data.get("duration_text", "N/A")}</div>
                {badge}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_similar_travellers_card(similar_data: Dict[str, Any], trip_type: str) -> None:
    """Renders Feature 5: Profile matching 'Travellers like you' stats."""
    st.markdown('<p class="section-header">👥 Similar Traveller Profile Analysis</p>', unsafe_allow_html=True)
    if not similar_data.get("has_data", False):
        st.caption("Not enough dataset records to draw similar traveller profiles.")
        return

    st.markdown(
        f"""
        <div class="dataset-insight" style="border-left-color: #7C3AED;">
            <h4 style="margin: 0 0 16px 0; color: #7C3AED; font-family: Outfit, sans-serif;">👥 Travellers like you (Historical matches)</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                <div><div class="field-label">Average Cost Spent</div><div style="font-size: 1.5rem; font-weight: 800; color: #F1F5F9;">{format_rupees(similar_data['avg_spending'])}</div></div>
                <div><div class="field-label">Preferred Transit Mode</div><div style="font-size: 1.5rem; font-weight: 800; color: #06B6D4;">{similar_data['fav_transport']}</div></div>
                <div><div class="field-label">Preferred Experience Style</div><div style="font-size: 1.5rem; font-weight: 800; color: #10B981;">{similar_data['fav_experience']}</div></div>
                <div><div class="field-label">Preferred Stay Grade</div><div style="font-size: 1.5rem; font-weight: 800; color: #F59E0B;">{similar_data['fav_hotel_quality']}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_deck(intelligence: Dict[str, Any]) -> None:
    """Renders Feature 4 & 9: Curated experience options deck of cards."""
    st.markdown('<p class="section-header">🎯 Recommended Curated Highlights</p>', unsafe_allow_html=True)
    
    activities = intelligence.get("experience_activities", [])
    pref_style = intelligence.get("preferred_experience", "Nature & Sightseeing")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="dataset-insight" style="border-left-color: #10B981; height: 100%;">
                <h5 style="margin: 0 0 10px 0; color: #10B981; font-family: Outfit, sans-serif;">✨ Recommended activities for '{pref_style}'</h5>
                <ul style="margin: 0; padding-left: 20px; color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
                    {"".join([f'<li style="margin-bottom: 5px;">{act}</li>' for act in activities])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="dataset-insight" style="border-left-color: #F59E0B; height: 100%;">
                <h5 style="margin: 0 0 10px 0; color: #F59E0B; font-family: Outfit, sans-serif;">💎 Curated Highlights</h5>
                <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
                    <div style="margin-bottom: 8px;">🏞️ <strong>Top Attractions:</strong> {", ".join(intelligence.get("places_to_visit", [])[:3])}</div>
                    <div style="margin-bottom: 8px;">🍜 <strong>Local Delicacies:</strong> {", ".join(intelligence.get("local_foods", [])[:3])}</div>
                    <div style="margin-bottom: 8px;">🌟 <strong>Hidden Gems:</strong> {", ".join(intelligence.get("hidden_gems", [])[:2])}</div>
                    <div>🛡️ <strong>Safety Advisory:</strong> {intelligence.get("safety_tips", ["Travel safe"])[0]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_interactive_map_card(route: Dict[str, Any], weather: Dict[str, Any]) -> None:
    """Renders Feature 15: Styled premium SVG mockup map showing coordinates, city nodes, and route line."""
    st.markdown('<p class="section-header">🗺️ Interactive Route Map Preview</p>', unsafe_allow_html=True)
    
    origin = route.get("origin", "Source")
    dest = route.get("destination", "Destination")
    dist = route.get("distance_km", 0.0)
    dur = route.get("duration_text", "N/A")
    temp = weather.get("temperature_c", 25.0)
    desc = weather.get("description", "Pleasant")

    st.markdown(
        f"""
        <div class="weather-widget" style="padding: 20px; background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px;">
                <span>Origin: <strong>{origin}</strong></span>
                <span>Destination: <strong>{dest}</strong></span>
            </div>
            <svg viewBox="0 0 500 120" style="width: 100%; height: auto; display: block; overflow: visible;">
                <defs>
                    <linearGradient id="map-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#2563EB" />
                        <stop offset="100%" stop-color="#7C3AED" />
                    </linearGradient>
                </defs>
                <path d="M 60,60 Q 250,10 440,60" fill="none" stroke="url(#map-grad)" stroke-width="3" stroke-dasharray="6,4" />
                <circle cx="60" cy="60" r="8" fill="#2563EB" />
                <circle cx="60" cy="60" r="14" fill="none" stroke="#2563EB" stroke-width="2" opacity="0.4" />
                <circle cx="440" cy="60" r="8" fill="#7C3AED" />
                <circle cx="440" cy="60" r="14" fill="none" stroke="#7C3AED" stroke-width="2" opacity="0.4" />
                <text x="60" y="90" font-family="Outfit, sans-serif" font-size="12" fill="#F1F5F9" text-anchor="middle">{origin}</text>
                <text x="440" y="90" font-family="Outfit, sans-serif" font-size="12" fill="#F1F5F9" text-anchor="middle">{dest}</text>
                <text x="250" y="30" font-family="Inter, sans-serif" font-size="11" fill="#06B6D4" text-anchor="middle">{dist:,.1f} km (~{dur})</text>
            </svg>
            <div class="divider" style="margin: 15px 0 10px 0;"></div>
            <div style="font-size: 0.8rem; color: #CBD5E1; display:flex; justify-content: space-between;">
                <span>🌤️ weather at destination: <strong>{temp}°C, {desc}</strong></span>
                <span>Recommended mode: <strong>{route.get('travel_mode', 'Car')}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
