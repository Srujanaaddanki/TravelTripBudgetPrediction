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

# ── Favicon & Branding ────────────────────────────────────────────────────────
from src.components.branding import logo_uri, splash_uri

_logo_img   = logo_uri()
_splash_img = splash_uri()

# Inject real favicon via link tag
if _logo_img:
    st.markdown(
        f'<link rel="icon" type="image/jpeg" href="{_logo_img}">',
        unsafe_allow_html=True,
    )

# ── Splash Screen (Pure Image — splash.png) ───────────────────────────────────
if "splash_shown" not in st.session_state and _splash_img:
    st.session_state["splash_shown"] = True
    st.markdown(f"""
    <style>
    /* Block scroll while splash is visible */
    body.tw-splashing {{
      overflow: hidden !important;
    }}
    /* Hide Streamlit header/toolbar during splash */
    body.tw-splashing [data-testid="stHeader"],
    body.tw-splashing [data-testid="stToolbar"],
    body.tw-splashing [data-testid="stAppViewContainer"] > section:not(:first-child) {{
      opacity: 0 !important;
      pointer-events: none !important;
    }}

    #tw-splash {{
      position: fixed;
      inset: 0;
      z-index: 2147483647;
      margin: 0;
      padding: 0;
      background: url("{_splash_img}") center center / cover no-repeat;
      animation: twSplashFade 4s ease forwards;
    }}

    @keyframes twSplashFade {{
      0%   {{ opacity: 1; visibility: visible; pointer-events: all; }}
      75%  {{ opacity: 1; visibility: visible; pointer-events: all; }}
      99%  {{ opacity: 0; visibility: visible; pointer-events: none; }}
      100% {{ opacity: 0; visibility: hidden;  pointer-events: none; }}
    }}
    </style>
    <div id="tw-splash"></div>
    <script>
    (function() {{
      var body = document.body;
      body.classList.add('tw-splashing');
      var splash = document.getElementById('tw-splash');
      if (splash) {{
        splash.addEventListener('animationend', function() {{
          body.classList.remove('tw-splashing');
          splash.style.display = 'none';
        }});
      }}
    }})();
    </script>
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