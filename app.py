# -*- coding: utf-8 -*-
"""
========================================================
Module: TripAI — Application Entry Point (v2 — Self-Learning)
Purpose: Initialises Streamlit page config, loads CSS,
         bootstraps all backend services, handles theme
         toggle via session_state (preserves predictions),
         and delegates to single Plan Trip page.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import os
import sys
from typing import Any

import joblib
import streamlit as st
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# Ensure project root is on sys.path so all src.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.search_tracker import SearchTracker
from src.data.maps_service import MapsService
from src.intelligence.dataset_intelligence import DatasetIntelligence
from src.services.travel_intelligence import TravelIntelligenceEngine
from src.components.navbar import render_navbar
from src.components.footer import render_footer
from src.pages.plan_trip import render_plan_trip_page


# ── 1. Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TraWell — AI Travel Intelligence Platform",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── 2. Theme State ────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Preserve all prediction state keys across reruns
for _key in [
    "last_report", "last_form_data", "gemini_intel", "packing_tips",
    "last_display_dest", "last_is_known", "last_weather",
    "validation_failed", "ai_suggestion", "failed_input_destination",
    "fuzzy_pending", "gemini_pending", "destination_override",
]:
    if _key not in st.session_state:
        st.session_state[_key] = None if "report" in _key or "data" in _key else (
            {} if _key in ("gemini_intel", "last_weather", "ai_suggestion") else
            [] if _key == "packing_tips" else
            False if _key in ("validation_failed", "fuzzy_pending", "gemini_pending") else
            ""
        )


# ── 3. CSS Injection ──────────────────────────────────────────────────────────
def _inject_css() -> None:
    """Load and inject all CSS files into Streamlit."""
    css_files = [
        os.path.join("src", "styles", "main.css"),
        os.path.join("src", "styles", "responsive.css"),
    ]
    combined = ""
    for path in css_files:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                combined += f.read() + "\n"

    # Light theme CSS variables & high-contrast overrides
    if st.session_state.get("theme") == "light":
        combined += """
        :root {
          --bg-primary:    #F8FAFC !important;
          --bg-card:       #FFFFFF !important;
          --bg-card-alt:   #F1F5F9 !important;
          --bg-sidebar:    #F1F5F9 !important;
          --bg-input:      #FFFFFF !important;
          --text-primary:  #0F172A !important;
          --text-secondary:#1E293B !important;
          --text-muted:    #334155 !important;
          --border-color:  rgba(0,0,0,0.15) !important;
          --border-hover:  rgba(0,0,0,0.3) !important;
        }
        .stApp { background-color: #F8FAFC !important; color: #0F172A !important; }
        .stTextInput > div > div > input { background-color: #FFFFFF !important; color: #0F172A !important; border: 1.5px solid rgba(0,0,0,0.2) !important; }
        .stTextInput > div > div > input::placeholder { color: #64748B !important; }
        [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #0F172A !important; border: 1.5px solid rgba(0,0,0,0.2) !important; }
        [data-baseweb="select"] span { color: #0F172A !important; }
        .form-label { color: #0F172A !important; font-weight: 700 !important; }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span { color: #0F172A !important; }
        div[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 800 !important; }
        div[data-testid="stMetricLabel"] { color: #334155 !important; font-weight: 600 !important; }
        .checklist-title, .checklist-text, .section-title { color: #0F172A !important; }
        .hero-budget-card, .map-card, .checklist-card { background-color: #FFFFFF !important; border: 1px solid rgba(0,0,0,0.12) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important; }
        .trawell-navbar { background: rgba(248, 250, 252, 0.95) !important; border-bottom: 1px solid rgba(0,0,0,0.1) !important; }
        """

    if combined:
        st.markdown(f"<style>{combined}</style>", unsafe_allow_html=True)


_inject_css()

# ── Splash Screen — exact recreation of reference image ───────────────────────
if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = True
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&display=swap');
    #tw-splash {
      position: fixed; inset: 0; z-index: 99999;
      background: linear-gradient(135deg, #f9a05a 0%, #f7762e 35%, #ee4e2a 100%);
      display: flex; align-items: center; justify-content: center;
      font-family: 'Outfit', 'Inter', sans-serif;
      animation: twFade 3.2s ease forwards;
      overflow: hidden;
    }
    @keyframes twFade {
      0%,72% { opacity:1; }
      100%    { opacity:0; pointer-events:none; visibility:hidden; }
    }
    /* dot grid — right-side only */
    .tw-dots {
      position: absolute; right: 0; top: 0; width: 55%; height: 100%;
      background-image: radial-gradient(circle, rgba(220,80,50,0.6) 1.5px, transparent 1.5px);
      background-size: 30px 30px;
    }
    .tw-inner {
      position: relative; z-index: 2;
      display: flex; align-items: center;
      width: 92%; max-width: 1050px; gap: 0;
    }
    /* ── LEFT: illustrated suitcase ── */
    .tw-bag-wrap {
      flex: 0 0 42%;
      display: flex; align-items: flex-end; justify-content: center;
      animation: bagRock 2s ease-in-out infinite alternate;
    }
    @keyframes bagRock {
      from { transform: rotate(-14deg) translateY(0px);   }
      to   { transform: rotate(-9deg)  translateY(-18px); }
    }
    /* ── RIGHT: text ── */
    .tw-text { flex: 1; padding-left: 32px; }
    .tw-headline {
      font-size: clamp(30px, 4.5vw, 58px);
      font-weight: 900; color: #fff;
      line-height: 1.12; margin-bottom: 18px;
      text-shadow: 0 3px 24px rgba(0,0,0,0.18);
    }
    .tw-sub {
      font-size: clamp(14px, 1.6vw, 20px);
      color: rgba(255,255,255,0.82);
      line-height: 1.65; font-weight: 400;
    }
    </style>
    <div id="tw-splash">
      <div class="tw-dots"></div>
      <div class="tw-inner">

        <!-- Illustrated suitcase (SVG — matches image 1) -->
        <div class="tw-bag-wrap">
          <svg viewBox="0 0 320 400" width="320" height="400" xmlns="http://www.w3.org/2000/svg">
            <!-- dark navy shadow base -->
            <ellipse cx="155" cy="395" rx="130" ry="18" fill="#0f1a35" opacity="0.45"/>
            <!-- main body -->
            <rect x="20" y="90" width="270" height="285" rx="22" fill="#f5c98a"/>
            <!-- body border / frame -->
            <rect x="20" y="90" width="270" height="285" rx="22" fill="none" stroke="#1b2d5b" stroke-width="9"/>
            <!-- horizontal strap seam top -->
            <rect x="20" y="205" width="270" height="18" rx="0" fill="#1b2d5b"/>
            <!-- horizontal strap seam bottom -->
            <rect x="20" y="230" width="270" height="8" rx="0" fill="#f5c98a"/>
            <!-- body vertical lines (luggage texture) -->
            <line x1="80"  y1="100" x2="80"  y2="205" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <line x1="130" y1="100" x2="130" y2="205" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <line x1="180" y1="100" x2="180" y2="205" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <line x1="230" y1="100" x2="230" y2="205" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <line x1="80"  y1="238" x2="80"  y2="365" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <line x1="130" y1="238" x2="130" y2="365" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <line x1="180" y1="238" x2="180" y2="365" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <line x1="230" y1="238" x2="230" y2="365" stroke="#d4a86a" stroke-width="5" stroke-linecap="round"/>
            <!-- small tag/label -->
            <rect x="35" y="112" width="36" height="22" rx="4" fill="#1b2d5b"/>
            <line x1="35" y1="120" x2="71" y2="120" stroke="#f5c98a" stroke-width="2"/>
            <!-- left wheel -->
            <circle cx="62"  cy="373" r="18" fill="#1b2d5b"/>
            <circle cx="62"  cy="373" r="10" fill="#0a1020"/>
            <circle cx="58"  cy="369" r="3"  fill="#2d4080"/>
            <!-- right wheel -->
            <circle cx="248" cy="373" r="18" fill="#1b2d5b"/>
            <circle cx="248" cy="373" r="10" fill="#0a1020"/>
            <circle cx="244" cy="369" r="3"  fill="#2d4080"/>
            <!-- telescopic handle rails -->
            <rect x="102" y="5"  width="14" height="92" rx="7" fill="#1b2d5b"/>
            <rect x="194" y="5"  width="14" height="92" rx="7" fill="#1b2d5b"/>
            <!-- handle grip bar -->
            <rect x="96"  y="5"  width="118" height="18" rx="9" fill="#1b2d5b"/>
            <!-- handle grip detail -->
            <rect x="105" y="8"  width="100" height="10" rx="5" fill="#2d4080"/>
          </svg>
        </div>

        <!-- Text block -->
        <div class="tw-text">
          <div class="tw-headline">
            Discover places.<br>
            Predict budgets.<br>
            Travel smarter.
          </div>
          <div class="tw-sub">
            AI-powered travel intelligence<br>for modern explorers.
          </div>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)


# ── 4. ML Resource Loading ────────────────────────────────────────────────────
@st.cache_resource
def load_ml_resources() -> tuple[Any, Any, float]:
    """Load the trained Random Forest model, encoders, and accuracy score."""
    model_path    = os.path.join("models", "final_model.pkl")
    encoders_path = os.path.join("models", "encoders.pkl")
    accuracy_path = os.path.join("models", "model_accuracy.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        return None, None, 0.0
    try:
        model    = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        acc      = joblib.load(accuracy_path) if os.path.exists(accuracy_path) else 0.95
        return model, encoders, float(acc)
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None, None, 0.0


model, encoders, accuracy_score = load_ml_resources()


# ── 5. Backend Service Initialisation ────────────────────────────────────────
@st.cache_resource
def init_services_v3():
    """Initialise all backend services (cached across reruns)."""
    tracker       = SearchTracker()
    maps_service  = MapsService()
    dataset_intel = DatasetIntelligence()
    travel_engine = TravelIntelligenceEngine(
        maps_service, tracker._db, dataset_intel
    )
    return tracker, maps_service, dataset_intel, travel_engine


tracker, maps_service, dataset_intel, travel_engine = init_services_v3()


# ── 6. Navigation Bar & Theme Toggle ─────────────────────────────────────────
render_navbar(theme=st.session_state.get("theme", "dark"))

# Theme toggle button (icon only: 🌙 for dark mode, ☀️ for light mode)
_curr_theme = st.session_state.get("theme", "dark")
_theme_icon = "🌙" if _curr_theme == "dark" else "☀️"

_theme_container = st.container()
with _theme_container:
    if st.button(_theme_icon, key="btn_theme_toggle", help="Switch Theme"):
        st.session_state["theme"] = "light" if _curr_theme == "dark" else "dark"
        st.rerun()


# ── 7. Render Plan Trip Page (single page only) ───────────────────────────────
render_plan_trip_page(
    model=model,
    encoders=encoders,
    maps_service=maps_service,
    travel_engine=travel_engine,
    tracker=tracker,
)


# ── 8. Footer ─────────────────────────────────────────────────────────────────
render_footer()