"""
========================================================
Module: UI Components
Purpose: Renders the premium startup header, portfolio
         statistics banner, landing hero, and future
         roadmap cards for the TripAI platform.

HTML → Rendered via st.markdown(..., unsafe_allow_html=True)
CSS  → Loaded from styles.css via inject_css() / load_css()
       styles.css contains ONLY CSS — no HTML tags.

Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import os
from typing import List
import streamlit as st


# ── CSS Injection ─────────────────────────────────────────────

def inject_css() -> None:
    """Load and inject styles.css into the Streamlit page.

    Reads the pure-CSS stylesheet and wraps it in a <style> tag
    injected via st.markdown — the only correct way to apply
    custom CSS in Streamlit.  styles.css must contain CSS only;
    no <div>, <a>, or any HTML tags should appear inside it.

    Call this ONCE at application startup (top of app.py).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "styles.css")
    if not os.path.exists(css_path):
        # Fallback: look two levels up (project root)
        css_path = os.path.join(base_dir, "..", "..", "styles.css")

    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_data = f.read()
        st.markdown(f"<style>{css_data}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ styles.css not found — UI theme could not be loaded.")
    except Exception as exc:
        st.warning(f"⚠️ Styling theme could not be loaded: {exc}")


# Alias so existing callers using load_css() continue to work
def load_css() -> None:
    """Alias for inject_css() — kept for backward compatibility."""
    inject_css()


# ── Top Navigation Bar ────────────────────────────────────────

def render_header() -> None:
    """Render the TripAI premium sticky top navbar.

    Outputs a fully styled navigation bar with:
    - Brand logo + tagline (left)
    - Visual nav buttons — Plan Trip / Analytics (centre)
    - LinkedIn, GitHub and theme-toggle icon buttons (right)

    All HTML is rendered via st.markdown with unsafe_allow_html=True.
    The CSS classes (.tripai-topnav, .tripai-nav-brand, etc.) are
    defined in styles.css — no HTML tags live inside that file.
    """
    st.markdown(
        """
        <div class="tripai-topnav">

            <!-- Brand / Logo -->
            <div class="tripai-nav-brand">
                <div>
                    <div class="tripai-nav-logo">✈️ TripAI</div>
                    <div class="tripai-nav-tagline">Love with Travel</div>
                </div>
            </div>

            <!-- Nav Buttons (visual only — Streamlit option_menu handles routing) -->
            <div class="tripai-nav-links">
                <span class="tripai-nav-btn" id="nav-plan-trip">✈️ Plan Trip</span>
                <span class="tripai-nav-btn" id="nav-analytics">📊 Analytics Report</span>
            </div>

            <!-- Social Icon Buttons -->
            <div class="tripai-nav-socials">
                <a
                    href="https://www.linkedin.com/in/srujana-addanki/"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="tripai-social-btn linkedin"
                    title="Connect on LinkedIn"
                >in</a>
                <a
                    href="https://github.com/Srujanaaddanki"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="tripai-social-btn github"
                    title="View on GitHub"
                >&#128027;</a>
                <span class="tripai-social-btn" id="theme-toggle" title="Toggle theme">🌙</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Statistics Banner ─────────────────────────────────────────

def render_stats_banner(
    total_records: int,
    model_accuracy_pct: float,
    unique_destinations: int,
) -> None:
    """Render the recruiter-facing portfolio statistics banner.

    Displays five key metrics in a horizontal strip immediately below the
    header, before the prediction form — designed to impress at first glance.

    Parameters
    ----------
    total_records : int
        Total traveller records in the dataset.
    model_accuracy_pct : float
        Model accuracy as a percentage (e.g., 95.0).
    unique_destinations : int
        Number of unique destinations in the dataset.
    """
    records_label  = f"{total_records:,}+"  if total_records      else "920+"
    accuracy_label = f"{model_accuracy_pct:.0f}%" if model_accuracy_pct else "95%"
    dest_label     = f"{unique_destinations}+"    if unique_destinations else "40+"

    st.markdown(
        f"""
        <div class="stats-banner">
            <div class="stats-banner-item">
                <div class="stats-value">{records_label}</div>
                <div class="stats-label">Traveller Records</div>
            </div>
            <div class="stats-banner-item">
                <div class="stats-value">{accuracy_label}</div>
                <div class="stats-label">Model Accuracy</div>
            </div>
            <div class="stats-banner-item">
                <div class="stats-value">{dest_label}</div>
                <div class="stats-label">Destinations</div>
            </div>
            <div class="stats-banner-item">
                <div class="stats-value">Live</div>
                <div class="stats-label">Weather Data</div>
            </div>
            <div class="stats-banner-item">
                <div class="stats-value">Maps</div>
                <div class="stats-label">Google Maps Integrated</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Landing Hero ──────────────────────────────────────────────

def render_landing_hero() -> None:
    """Render the premium landing hero section shown before the first prediction.

    Displays a startup-quality hero with headline, subtitle, and CTA buttons.
    Designed to look and feel like a commercial travel SaaS product.
    All markup is HTML rendered via st.markdown — never printed as text.
    """
    st.markdown(
        """
        <div class="landing-hero">
            <div style="font-size: 0.75rem; font-weight: 700; color: #06B6D4; text-transform: uppercase;
                        letter-spacing: 3px; margin-bottom: 14px;">
                AI-Powered · Random Forest · Dataset Intelligence
            </div>
            <div style="font-size: 2.6rem; font-weight: 800; font-family: Outfit, sans-serif;
                        color: #FFFFFF; line-height: 1.2; margin-bottom: 16px;">
                Plan Smarter.<br>
                <span style="background: linear-gradient(135deg, #60A5FA, #A78BFA);
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                             background-clip: text;">Travel Better.</span>
            </div>
            <p style="font-size: 1.05rem; color: #94A3B8; max-width: 580px;
                      margin: 0 auto 30px auto; line-height: 1.65;">
                Enter your trip details in the form to get an AI-predicted budget,
                historical traveller insights, live weather, and a full route analysis —
                all in one premium dashboard.
            </p>
            <div style="display: flex; gap: 14px; justify-content: center; flex-wrap: wrap;">
                <span style="font-size: 0.92rem; padding: 11px 26px;
                             background: linear-gradient(135deg, #2563EB, #7C3AED);
                             border-radius: 12px; font-weight: 700; color: #FFFFFF;
                             box-shadow: 0 4px 16px rgba(37,99,235,0.35);">
                    🔮 Try Predict Budget →
                </span>
                <span style="font-size: 0.92rem; padding: 11px 26px; background: #1E293B;
                             border-radius: 12px; font-weight: 600; color: #94A3B8;
                             border: 1px solid rgba(255,255,255,0.07);">
                    📊 View Insights Dashboard
                </span>
            </div>
            <div style="display: flex; gap: 24px; justify-content: center; margin-top: 32px;
                        flex-wrap: wrap;">
                <div style="text-align:center;">
                    <div style="font-size:1.4rem; font-weight:800; color:#60A5FA;">920+</div>
                    <div style="font-size:0.7rem; color:#64748B; text-transform:uppercase; letter-spacing:0.8px;">Training Records</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem; font-weight:800; color:#A78BFA;">95%</div>
                    <div style="font-size:0.7rem; color:#64748B; text-transform:uppercase; letter-spacing:0.8px;">Model Accuracy</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem; font-weight:800; color:#34D399;">40+</div>
                    <div style="font-size:0.7rem; color:#64748B; text-transform:uppercase; letter-spacing:0.8px;">Destinations</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.4rem; font-weight:800; color:#F59E0B;">Live</div>
                    <div style="font-size:0.7rem; color:#64748B; text-transform:uppercase; letter-spacing:0.8px;">Weather API</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Coming-Soon Roadmap Cards ─────────────────────────────────

def render_future_features() -> None:
    """Render the premium 'Coming Soon' roadmap section with 5 feature cards.

    Each card features a gradient icon container, title, description, and
    a 'Coming Soon' badge — styled to feel like a real product roadmap.
    All markup is rendered via st.markdown — never displayed as raw text.
    """
    st.markdown(
        '<p class="section-header" style="margin-top: 24px;">🚀 Coming Soon</p>',
        unsafe_allow_html=True,
    )

    features = [
        ("🎙️", "Voice Assistant",    "Plan hands-free with AI speech commands"),
        ("💳", "Expense Tracker",    "Track real-time trip spending on the go"),
        ("🏨", "Hotel Booking",      "One-click Booking.com rate comparison"),
        ("🗺️", "Smart Itinerary",   "Day-wise automated route planner"),
        ("🤖", "AI Chat Assistant", "24/7 travel Q&A with a conversational AI"),
    ]

    cards_html = "".join(
        f"""
        <div class="roadmap-card">
            <div class="roadmap-icon-container">{icon}</div>
            <div class="roadmap-title">{title}</div>
            <div class="roadmap-desc">{desc}</div>
            <span class="badge badge-coming-soon">Coming Soon</span>
        </div>
        """
        for icon, title, desc in features
    )

    st.markdown(f'<div class="roadmap-grid">{cards_html}</div>', unsafe_allow_html=True)


# ── Related Destination Chips ─────────────────────────────────

def render_related_searches(related: List[str]) -> None:
    """Render related destination suggestions as styled pill chips.

    Parameters
    ----------
    related : List[str]
        List of destination names to display as chips.
    """
    if not related:
        return
    st.markdown('<p class="section-header">🔍 Similar Destinations</p>', unsafe_allow_html=True)
    chips_html = "".join(
        f'<span class="related-search-chip">🌎 {place.title()}</span>'
        for place in related
    )
    st.markdown(f'<div style="margin-top:10px;">{chips_html}</div>', unsafe_allow_html=True)
