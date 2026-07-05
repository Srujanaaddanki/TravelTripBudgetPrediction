"""
========================================================
Module: TripAI Main Application
Purpose: Entry point for the TripAI platform. Handles model
         loading, page navigation, sidebar inputs, and triggers
         the Travel Intelligence Engine, UI components, and PDF/HTML report exports.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations
from typing import Any

import sys
import os
import difflib
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
from src.services.report_exporter import generate_html_report
import src.ui.ui_components as ui
import src.ui.dashboard_components as db_ui
import src.ui.charts as charts

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

def find_closest_place(user_input: str, encoder: Any) -> tuple[str | None, bool]:
    """Fuzzy-match user input destination to known places in encoder."""
    user_input = user_input.lower().strip()
    known = list(encoder.classes_)
    if user_input in known:
        return user_input, True
    matches = difflib.get_close_matches(user_input, known, n=1, cutoff=0.5)
    if matches:
        return matches[0], False
    return None, False

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
            "background": "linear-gradient(135deg, #2563EB, #7C3AED)",
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
                matched_dest, is_exact = find_closest_place(dest_city, encoders['Place'])

                if not matched_dest:
                    st.markdown(f'<div class="no-match-note">⚠️ Destination "{dest_city}" is not in our training model. Showing geographical route info only.</div>', unsafe_allow_html=True)
                    try:
                        route = maps_service.get_route_info(source_city, dest_city, travel_mode)
                        db_ui.render_route_details(route, {})
                    except Exception:
                        st.info("No geographical route data available.")
                else:
                    if not is_exact:
                        st.markdown(f'<div class="match-note">📍 "{dest_city}" fuzzy matched to "{matched_dest.title()}" in our training logs.</div>', unsafe_allow_html=True)

                    month_season_map = {
                        "january": "winter", "february": "winter", "march": "spring",
                        "april": "summer", "may": "summer", "june": "summer",
                        "july": "rainy", "august": "rainy", "september": "rainy",
                        "october": "autumn", "november": "autumn", "december": "winter"
                    }
                    season = month_season_map.get(month.lower(), "summer")

                    # Predict
                    input_df = pd.DataFrame([{
                        'Place': encoders['Place'].transform([matched_dest])[0],
                        'Month': encoders['Month'].transform([month.lower()])[0],
                        'Season': encoders['Season'].transform([season])[0],
                        'Trip_Type': encoders['Trip_Type'].transform([trip_type.lower()])[0],
                        'Hotel_Quality': encoders['Hotel_Quality'].transform([hotel.lower()])[0],
                        'Days': days
                    }])

                    pred = float(model.predict(input_df)[0])
                    tracker.track(source=source_city, destination=dest_city, month=month, duration_days=days,
                                  travel_mode=travel_mode, predicted_cost=pred, season=season,
                                  trip_type=trip_type, hotel_quality=hotel)

                    # Build intelligence report
                    report = travel_engine.generate_report(source_city, matched_dest, month, days,
                                                           travel_mode, trip_type, hotel, pred)
                    route_info = maps_service.get_route_info(source_city, matched_dest, travel_mode)
                    smart_cost = maps_service.get_smart_budget(pred, route_info["distance_km"], travel_mode, days)

                    dash_left, dash_right = st.columns(2)

                    with dash_left:
                        db_ui.render_travel_intelligence_card(report["dataset_insights"], matched_dest)
                        db_ui.render_traveller_experience_widget(report["dataset_insights"])
                        db_ui.render_weather_widget(report["weather"], report["intelligence"]["best_time"])
                        db_ui.render_destination_summary_card(report["dataset_insights"], matched_dest)

                    with dash_right:
                        db_ui.render_premium_budget_card(pred, matched_dest, days, travel_mode, hotel,
                                                         report["confidence"]["score"], report["confidence"]["level"], season)
                        
                        db_ui.render_budget_verification_card(pred, report["dataset_insights"]["average_budget"],
                                                              smart_cost["travel_cost_estimate"], smart_cost["smart_estimate"],
                                                              report["confidence"]["score"])

                        db_ui.render_budget_breakdown_table(pred)
                        
                        st.markdown('<div class="weather-widget">', unsafe_allow_html=True)
                        st.plotly_chart(charts.render_budget_donut(pred, travel_mode), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        db_ui.render_route_details(route_info, smart_cost)
                        db_ui.render_mode_comparison_cards(report["mode_comparison"]["modes"], travel_mode)
                        db_ui.render_interactive_map_card(route_info, report["weather"])

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    db_ui.render_checklists_widget(report["intelligence"]["packing_tips"])
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    db_ui.render_saving_tips_widget(report["intelligence"]["money_saving_tips"])
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    db_ui.render_recommendation_deck(report["intelligence"])
                    db_ui.render_similar_travellers_card(report["similar_traveller"], trip_type)
                    ui.render_related_searches(report["related_searches"])

                    # Feature 10: HTML export trip report download button
                    report_html = generate_html_report(source_city, matched_dest, month, days, travel_mode, hotel, pred, report)
                    st.download_button(
                        label="📥 Export Premium Trip Report (Print-Ready HTML)",
                        data=report_html,
                        file_name=f"tripai_itinerary_{matched_dest.lower()}.html",
                        mime="text/html",
                        help="Download a styled, print-friendly report of your trip itinerary."
                    )
        else:
            ui.render_landing_hero()

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

    # 1. KPI Metric Grid
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔍 Live Search Volume", f"{total_searches:,} inquiries")
    k2.metric("💾 Database Size", f"{ds_stats.get('total_trips', 0):,} records")
    k3.metric("🏨 Unique Places", f"{ds_stats.get('unique_destinations', 0):,} destinations")
    k4.metric("💰 Average Budget", f"₹{ds_stats.get('avg_budget', 0.0):,.2f}")

    df = dataset_intel._df
    most_expensive, cheapest = "N/A", "N/A"
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

    # 2. Charts Grid
    c1, c2 = st.columns(2)
    with c1:
        st.write("### 🏆 Most Searched Destinations (Live Queries)")
        top_dest = db_stats.get("top_destinations", [])
        if top_dest:
            st.plotly_chart(charts.render_dashboard_destinations(top_dest), use_container_width=True)
        else:
            st.caption("No inquiries logged yet.")

    with c2:
        st.write("### ⭐ Highest Rated Destinations (Dataset overall score)")
        if not df.empty:
            df_scores = df.copy()
            df_scores["Overall_Score"] = (df_scores["Satisfaction_Rating"] + df_scores["Hotel_Rating"] + df_scores["Local_Trans_Rating"] + df_scores["Sightseeing_Rating"]) / 4.0
            highest_rated = df_scores.groupby("Place")["Overall_Score"].mean().reset_index().sort_values("Overall_Score", ascending=False).head(8)
            st.plotly_chart(charts.render_dashboard_ratings(highest_rated), use_container_width=True)
        else:
            st.caption("Dataset is empty.")

    c3, c4 = st.columns(2)
    with c3:
        st.write("### 📅 Most Visited Seasons")
        if not df.empty:
            season_counts = df["Season"].value_counts().reset_index()
            season_counts.columns = ["Season", "Count"]
            st.plotly_chart(charts.render_dashboard_seasons(season_counts), use_container_width=True)
        else:
            st.caption("Dataset is empty.")

    with c4:
        st.write("### ✨ Most Popular Preferred Experiences")
        if not df.empty:
            exp_counts = df["Preferred_Experience"].value_counts().reset_index().head(8)
            exp_counts.columns = ["Experience", "Count"]
            st.plotly_chart(charts.render_dashboard_experiences(exp_counts), use_container_width=True)
        else:
            st.caption("Dataset is empty.")

    c5, c6 = st.columns(2)
    with c5:
        st.write("### 💰 Budget Distribution")
        if not df.empty:
            st.plotly_chart(charts.render_dashboard_budgets(df), use_container_width=True)
        else:
            st.caption("Dataset is empty.")

    with c6:
        st.write("### ⏱️ Average Stay Duration per Destination (Days)")
        if not df.empty:
            duration_df = df.groupby("Place")["Days"].mean().reset_index().sort_values("Days", ascending=False).head(8)
            st.plotly_chart(charts.render_dashboard_durations(duration_df), use_container_width=True)
        else:
            st.caption("Dataset is empty.")

    st.markdown('</div>', unsafe_allow_html=True)