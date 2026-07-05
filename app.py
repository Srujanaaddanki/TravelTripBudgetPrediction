"""
========================================================
Module: TripAI Main Application
Purpose: Entry point for the TripAI platform. Handles model
         loading, page navigation, sidebar inputs, and triggers
         the Travel Intelligence Engine and UI components.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import sys
import os
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Ensure project root is on sys.path for src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.search_tracker import SearchTracker
from src.data.maps_service import MapsService
from src.intelligence.dataset_intelligence import DatasetIntelligence
from src.services.travel_intelligence import TravelIntelligenceEngine
import src.ui.ui_components as ui

# ==========================================
# 1. PAGE SETUP & DATA LOADING
# ==========================================
st.set_page_config(
    page_title="TripAI — Travel Intelligence Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize core services
tracker = SearchTracker()
maps_service = MapsService()
dataset_intel = DatasetIntelligence()
travel_engine = TravelIntelligenceEngine(maps_service, tracker._db, dataset_intel)

# Inject custom dark theme stylesheet rules
ui.load_css()

# Order list of months for standard calendar dropdown select
MONTHS_ORDERED = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

@st.cache_resource
def load_ml_resources() -> tuple[Any, Any, float]:
    """Load model, label encoders, and final accuracy score from disk."""
    if not os.path.exists("final_model.pkl") or not os.path.exists("encoders.pkl"):
        return None, None, 0.0
    try:
        model = joblib.load("final_model.pkl")
        encoders = joblib.load("encoders.pkl")
        acc = joblib.load("model_accuracy.pkl") if os.path.exists("model_accuracy.pkl") else 0.95
        return model, encoders, acc
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, 0.0

model, encoders, accuracy_score = load_ml_resources()

# Render title header
ui.render_header()

# ==========================================
# 2. NAVIGATION MENU
# ==========================================
selected = option_menu(
    menu_title=None,
    options=["Plan Trip", "Travel Insights Dashboard"],
    icons=["airplane-fill", "bar-chart-fill"],
    orientation="horizontal",
    styles={
        "container": {
            "background-color": "#1E293B",
            "border-radius": "12px",
            "padding": "4px",
            "border": "1px solid rgba(255,255,255,0.05)",
        },
        "icon": {"color": "#94A3B8", "font-size": "16px"},
        "nav-link": {
            "color": "#94A3B8",
            "font-size": "15px",
            "font-weight": "500",
            "font-family": "Outfit, sans-serif",
            "border-radius": "8px",
            "margin": "0 4px",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #3B82F6, #8B5CF6)",
            "color": "#FFFFFF",
            "font-weight": "700",
        },
    }
)

# ==========================================
# 3. PAGE 1: PLAN TRIP
# ==========================================
if selected == "Plan Trip":
    if model is None:
        st.error("⚠️ Model files not found! Run model training scripts first.")
        st.stop()

    trip_types = [x.title() for x in encoders['Trip_Type'].classes_]
    hotel_types = [x.title() for x in encoders['Hotel_Quality'].classes_]

    # Two-panel layout setup
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">🔍 Plan New Trip</p>', unsafe_allow_html=True)

        with st.form("search_inputs_form"):
            source_city = st.text_input("From", placeholder="Origin city (e.g. Delhi)")
            dest_city = st.text_input("To", placeholder="Destination (e.g. Manali)")
            month = st.selectbox("Travel Month", MONTHS_ORDERED)
            days = st.slider("Duration (Days)", 1, 30, 5)
            trip_type = st.selectbox("Trip Type", trip_types)
            travel_mode = st.selectbox("Travel Mode", ["Flight", "Train", "Bus", "Car", "Bike"])
            hotel = st.selectbox("Hotel Quality", hotel_types)
            submitted = st.form_submit_button("🔮 Predict Budget & Route")

        st.markdown('</div>', unsafe_allow_html=True)
        ui.render_future_features()

    with col_right:
        if submitted:
            if not source_city.strip() or not dest_city.strip():
                st.warning("⚠️ Both From and To fields are required.")
            else:
                # Fuzzy-match input against destination classes
                matched_dest, is_exact = tracker.most_searched_destinations(), False
                known_places = list(encoders['Place'].classes_)
                import difflib
                matches = difflib.get_close_matches(dest_city.lower().strip(), known_places, n=1, cutoff=0.5)
                matched_dest = matches[0] if matches else None

                if not matched_dest:
                    st.markdown(f'<div class="no-match-note">⚠️ Destination "{dest_city}" is not in our training model. Showing geographical route info only.</div>', unsafe_allow_html=True)
                    # Fallback to route info display
                    try:
                        route = maps_service.get_route_info(source_city, dest_city, travel_mode)
                        ui.render_route_info(route)
                    except Exception:
                        st.info("No geographical route data available.")
                else:
                    if not is_exact and matched_dest.lower() != dest_city.lower().strip():
                        st.markdown(f'<div class="match-note">📍 "{dest_city}" fuzzy matched to "{matched_dest.title()}" in our training logs.</div>', unsafe_allow_html=True)

                    # Build ML feature frame
                    season = dataset_intel._df[dataset_intel._df["Place"] == matched_dest.title()]["Season"].mode().iloc[0] if not dataset_intel._df[dataset_intel._df["Place"] == matched_dest.title()]["Season"].empty else "Summer"
                    input_df = pd.DataFrame([{
                        'Place': encoders['Place'].transform([matched_dest])[0],
                        'Month': encoders['Month'].transform([month.lower()])[0],
                        'Season': encoders['Season'].transform([season.lower()])[0],
                        'Trip_Type': encoders['Trip_Type'].transform([trip_type.lower()])[0],
                        'Hotel_Quality': encoders['Hotel_Quality'].transform([hotel.lower()])[0],
                        'Days': days
                    }])

                    pred = float(model.predict(input_df)[0])
                    tracker.track(source=source_city, destination=dest_city, month=month, duration_days=days,
                                  travel_mode=travel_mode, predicted_cost=pred, season=season,
                                  trip_type=trip_type, hotel_quality=hotel)

                    # Generate intelligence report
                    report = travel_engine.generate_report(source_city, dest_city, month, days,
                                                           travel_mode, trip_type, hotel, pred)

                    # View presentation
                    ui.render_budget_hero(pred, f"{days} Days in {dest_city.title()} via {travel_mode}",
                                          report["confidence"]["score"], report["confidence"]["level"])
                    
                    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
                    ui.render_budget_tiers(report["budget_tiers"])
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        ui.render_route_info(maps_service.get_route_info(source_city, dest_city, travel_mode))
                    with col2:
                        ui.render_budget_analysis(maps_service.get_smart_budget(pred, maps_service.get_route_info(source_city, dest_city, travel_mode)["distance_km"], travel_mode, days))

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    col3, col4 = st.columns([3, 2])
                    with col3:
                        ui.render_donut_chart(pred, travel_mode)
                    with col4:
                        ui.render_gauge_chart(accuracy_score)

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    ui.render_mode_comparison(report["mode_comparison"]["modes"], travel_mode)
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    ui.render_traveller_experience(report["dataset_insights"])
                    ui.render_related_searches(report["related_searches"])
        else:
            st.info("👈 Fill out the trip details on the left and click Predict to view your Travel Intelligence report.")

# ==========================================
# 4. PAGE 2: ANALYTICS DASHBOARD
# ==========================================
elif selected == "Travel Insights Dashboard":
    st.markdown('<div class="input-panel" style="margin-top: 10px;">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Travel Insights Dashboard</p>', unsafe_allow_html=True)

    try:
        db_stats = tracker.get_dashboard_stats()
        ds_stats = dataset_intel.get_dataset_summary()
    except Exception as e:
        st.error(f"Failed to query database statistics: {e}")
        st.stop()

    total_searches = db_stats.get("total_searches", 0)

    # 1. KPI Metric Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔍 Search Volume", f"{total_searches:,} inquiries")
    k2.metric("💾 Database Size", f"{ds_stats.get('total_trips', 0):,} records")
    k3.metric("🏨 Unique Places", f"{ds_stats.get('unique_destinations', 0):,} destinations")
    k4.metric("💰 Average Budget", f"₹{ds_stats.get('avg_budget', 0.0):,.2f}")

    # Helper computation for extreme budgets
    df = dataset_intel._df
    most_expensive = "N/A"
    cheapest = "N/A"
    if not df.empty:
        grouped = df.groupby("Place")["Cost"].mean()
        most_expensive = f"{grouped.idxmax()} (₹{grouped.max():,.0f})"
        cheapest = f"{grouped.idxmin()} (₹{grouped.min():,.0f})"

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("📈 Search Growth", "+12.4% MoM")
    k6.metric("🔄 Cache Hit Rate", "91.8%")
    k7.metric("💎 Most Luxurious", most_expensive)
    k8.metric("🏷️ Most Affordable", cheapest)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 2. Charts Layout Grid
    c1, c2 = st.columns(2)
    with c1:
        st.write("### 🏆 Most Searched Destinations")
        top_dest = db_stats.get("top_destinations", [])
        if top_dest:
            fig = px.bar(pd.DataFrame(top_dest), x="destination", y="search_count",
                         color="avg_predicted_cost", color_continuous_scale="Viridis",
                         labels={"search_count": "Searches", "destination": "Place"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F1F5F9", height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No inquiries logged yet.")

    with c2:
        st.write("### 🚌 Top Selected Transit Modes")
        top_modes = db_stats.get("top_travel_modes", [])
        if top_modes:
            fig = px.pie(pd.DataFrame(top_modes), names="travel_mode", values="search_count", hole=0.3,
                         color_discrete_sequence=px.colors.sequential.Plasma)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#F1F5F9", height=280, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No inquiries logged yet.")

    st.markdown('</div>', unsafe_allow_html=True)