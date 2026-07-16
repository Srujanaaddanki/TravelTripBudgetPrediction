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
    page_title="TripAI — AI Travel Intelligence Platform",
    page_icon="✈️",
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

    # Light theme CSS variables
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
          --text-muted:    #475569 !important;
          --border-color:  rgba(0,0,0,0.15) !important;
          --border-hover:  rgba(0,0,0,0.3) !important;
        }
        .stApp { background-color: #F8FAFC !important; color: #0F172A !important; }
        .stTextInput > div > div > input { background-color: #FFFFFF !important; color: #0F172A !important; border: 1.5px solid rgba(0,0,0,0.15) !important; }
        .stTextInput > div > div > input::placeholder { color: #64748B !important; }
        [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #0F172A !important; border: 1.5px solid rgba(0,0,0,0.15) !important; }
        [data-baseweb="select"] span { color: #0F172A !important; }
        .form-label { color: #0F172A !important; }
        [data-testid="stMarkdownContainer"] { color: #0F172A !important; }
        div[data-testid="stMetricValue"] { color: #0F172A !important; }
        .checklist-title, .checklist-text { color: #0F172A !important; }
        .hero-budget-card, .map-card, .checklist-card { background-color: #FFFFFF !important; border: 1px solid rgba(0,0,0,0.1) !important; }
        """

    if combined:
        st.markdown(f"<style>{combined}</style>", unsafe_allow_html=True)


_inject_css()


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
def init_services():
    """Initialise all backend services (cached across reruns)."""
    tracker       = SearchTracker()
    maps_service  = MapsService()
    dataset_intel = DatasetIntelligence()
    travel_engine = TravelIntelligenceEngine(
        maps_service, tracker._db, dataset_intel
    )
    return tracker, maps_service, dataset_intel, travel_engine


tracker, maps_service, dataset_intel, travel_engine = init_services()


# ── 6. Navigation Bar & Theme Toggle ─────────────────────────────────────────
render_navbar(theme=st.session_state.get("theme", "dark"))

# Invisible overlay button for the theme toggle icon in the navbar
_spacer, _theme_col = st.columns([11.2, 0.8])
with _theme_col:
    if st.button("Theme", key="btn_theme", use_container_width=True, type="secondary"):
        # Toggle theme — do NOT clear prediction state
        st.session_state["theme"] = (
            "light" if st.session_state["theme"] == "dark" else "dark"
        )
        st.rerun()

# Overlay style to place Streamlit's native button exactly over the navbar toggle
st.markdown("""
<style>
  [data-testid="column"]:nth-child(2) [data-testid="stBaseButton-secondary"] {
    position: absolute;
    opacity: 0;
    pointer-events: auto;
    height: 38px;
    width: 38px;
    right: 32px;
    top: -52px;
    cursor: pointer;
    z-index: 9999;
    border-radius: 10px;
  }
  [data-testid="column"]:nth-child(2) {
    position: relative;
    overflow: visible;
  }
</style>
""", unsafe_allow_html=True)


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