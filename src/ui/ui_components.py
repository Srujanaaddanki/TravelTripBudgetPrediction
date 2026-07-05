"""
========================================================
Module: UI Components
Purpose: Renders modern top-level landing pages, sticky navbars,
         search forms, and future ready roadmap metrics.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
import streamlit as st


def load_css() -> None:
    """Load and inject styles.css styling rules into Streamlit page."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "styles.css")
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


def render_landing_hero() -> bool:
    """Renders Feature 1: Modern Premium Landing Page section.

    Returns
    -------
    bool
        True if the user clicked the 'Predict Budget' CTA button.
    """
    st.markdown(
        """
        <div class="landing-hero" style="text-align: center; padding: 60px 40px; background: linear-gradient(135deg, rgba(37,99,235,0.05) 0%, rgba(124,58,237,0.05) 100%); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 30px;">
            <div style="font-size: 3rem; font-weight: 800; font-family: Outfit, sans-serif; color: #FFFFFF; line-height: 1.2;">Plan Smarter. Travel Better.</div>
            <p style="font-size: 1.15rem; color: #94A3B8; margin-top: 15px; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                AI-powered travel budget prediction and travel intelligence platform. Discover optimal routes, expense breakdowns, and traveller ratings.
            </p>
            <div style="margin-top: 30px; display: flex; gap: 16px; justify-content: center;">
                <span style="font-size: 0.95rem; padding: 12px 28px; background: linear-gradient(135deg, #2563EB, #7C3AED); border-radius: 12px; font-weight: 700; color: #FFFFFF; cursor: pointer; box-shadow: 0 4px 14px rgba(37,99,235,0.4);">
                    🔮 Try Predict Budget
                </span>
                <span style="font-size: 0.95rem; padding: 12px 28px; background: #1E293B; border-radius: 12px; font-weight: 700; color: #94A3B8; border: 1px solid rgba(255,255,255,0.05); cursor: pointer;">
                    🌎 Explore Destinations
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return True


def render_future_features() -> None:
    """Render Feature 10: Placeholder grid showing future features roadmap."""
    st.markdown('<p class="section-header" style="margin-top: 24px;">🚀 TripAI Future Roadmap</p>', unsafe_allow_html=True)
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


def render_related_searches(related: List[str]) -> None:
    """Render a list of related destinations as styled chips.

    Parameters
    ----------
    related : List[str]
        List of matching destinations to display.
    """
    if not related:
        return
    st.markdown('<p class="section-header">🔍 Similar Destinations Searched</p>', unsafe_allow_html=True)
    chips_html = "".join([
        f'<span class="related-search-chip">🌎 {place.title()}</span>'
        for place in related
    ])
    st.markdown(f'<div style="margin-top: 10px;">{chips_html}</div>', unsafe_allow_html=True)
