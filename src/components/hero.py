"""
========================================================
Component: Hero Budget Card
Purpose: Renders the large gradient hero card that displays
          the ML-predicted budget, per-day budget, and
          trip summary chips.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import streamlit as st


def render_budget_hero_card(
    amount: float,
    days: int,
    season: str,
    mode: str,
    hotel: str,
    popularity_label: str = "✦ Smart Travel Intelligence",
) -> None:
    """Render the premium hero budget card with total and per-day rate.

    Parameters
    ----------
    amount : float
        Smart-calculated trip budget in INR.
    days : int
        Trip duration in days.
    season : str
        Travel season (e.g. "summer", "winter").
    mode : str
        Primary travel mode (e.g. "Train", "Flight").
    hotel : str
        Hotel quality (e.g. "Standard", "Luxury").
    popularity_label : str
        Popularity badge label (e.g. "💎 Hidden Gem", "🔥 Popular Choice").
    """
    # Format amount with Indian-style comma separation
    formatted = f"₹{int(amount):,}"
    
    # Calculate per-day rate
    per_day_val = amount / max(days, 1)
    per_day_formatted = f"₹{int(per_day_val):,}/day"

    # Season emoji map
    season_icons = {
        "summer": "☀️",
        "winter": "❄️",
        "rainy":  "monsoon",
        "monsoon": "🌧️",
        "autumn":  "🍂",
        "spring":  "spring",
    }
    season_icon = season_icons.get(season.lower(), "🌤️")
    if season_icon == "monsoon":
        season_icon = "🌧️"
    elif season_icon == "spring":
        season_icon = "🌸"
    season_label = season.title()

    # Mode emoji
    mode_icons = {
        "Train":  "🚂",
        "Flight": "✈️",
        "Bus":    "🚌",
        "Car":    "🚗",
        "Bike":   "🏍️",
    }
    mode_icon = mode_icons.get(mode, "🚀")

    # Hotel emoji
    hotel_icons = {
        "Budget":   "🏠",
        "Standard": "🏨",
        "Homestay": "🏡",
        "Premium":  "🏩",
        "Luxury":   "💎",
    }
    hotel_icon = hotel_icons.get(hotel.title(), "🏨")

    hero_html = f"""
    <div class="hero-budget-card">
      <div class="hero-label">Estimated Budget</div>
      <div style="display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;">
        <div class="hero-amount">{formatted}</div>
        <div style="font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 600; color: rgba(255, 255, 255, 0.7); margin-bottom: 16px;">
          ({per_day_formatted})
        </div>
      </div>
      <div class="hero-badge">{popularity_label}</div>

      <div class="hero-chips">
        <div class="hero-chip">
          <span class="hero-chip-icon">📅</span>
          <span class="hero-chip-value">{days} Days</span>
          <span class="hero-chip-label">Trip Duration</span>
        </div>
        <div class="hero-chip">
          <span class="hero-chip-icon">{season_icon}</span>
          <span class="hero-chip-value">{season_label}</span>
          <span class="hero-chip-label">Travel Season</span>
        </div>
        <div class="hero-chip">
          <span class="hero-chip-icon">{mode_icon}</span>
          <span class="hero-chip-value">{mode}</span>
          <span class="hero-chip-label">Travel Mode</span>
        </div>
        <div class="hero-chip">
          <span class="hero-chip-icon">{hotel_icon}</span>
          <span class="hero-chip-value">{hotel.title()}</span>
          <span class="hero-chip-label">Hotel Quality</span>
        </div>
      </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
