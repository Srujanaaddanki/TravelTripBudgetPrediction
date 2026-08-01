"""
========================================================
Component: Route Map (Plotly-based, always works)
Purpose: Renders an interactive route map using Plotly
         Scattergeo — no folium, no external tile servers,
         no API tokens. Works 100% on Streamlit Cloud.
Author: Srujana Addanki
Project: TraWell — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st


# ── City coordinate map: ALWAYS (lat, lon) ────────────────────────────────────
_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    # North India
    "delhi":              (28.6139, 77.2090),
    "new delhi":          (28.6139, 77.2090),
    "chandigarh":         (30.7333, 76.7794),
    "amritsar":           (31.6340, 74.8723),
    "agra":               (27.1767, 78.0081),
    "varanasi":           (25.3176, 82.9739),
    "lucknow":            (26.8467, 80.9462),
    "haridwar":           (29.9457, 78.1642),
    "rishikesh":          (30.0869, 78.2676),
    "dehradun":           (30.3165, 78.0322),
    "manali":             (32.2396, 77.1887),
    "shimla":             (31.1048, 77.1734),
    "srinagar":           (34.0837, 74.7973),
    "leh":                (34.1526, 77.5771),
    "leh ladakh":         (34.1526, 77.5771),
    "ladakh":             (34.1526, 77.5771),
    "kedarnath":          (30.7352, 79.0669),
    "badrinath":          (30.7433, 79.4938),
    "mussoorie":          (30.4598, 78.0664),
    "nainital":           (29.3919, 79.4542),
    "jaipur":             (26.9124, 75.7873),
    "jodhpur":            (26.2389, 73.0243),
    "udaipur":            (24.5854, 73.7125),
    "bikaner":            (28.0229, 73.3119),
    "jaisalmer":          (26.9157, 70.9083),
    "pushkar":            (26.4899, 74.5511),
    "patna":              (25.6093, 85.1376),
    "ranchi":             (23.3441, 85.3096),
    "bhopal":             (23.2599, 77.4126),
    "indore":             (22.7196, 75.8577),
    "nagpur":             (21.1458, 79.0882),
    "gorakhpur":          (26.7605, 83.3732),
    "ayodhya":            (26.7922, 82.1998),
    "dharamshala":        (32.2190, 76.3234),

    # West India
    "mumbai":             (19.0760, 72.8777),
    "pune":               (18.5204, 73.8567),
    "goa":                (15.2993, 74.1240),
    "panaji":             (15.4909, 73.8278),
    "gokarna":            (14.5479, 74.3188),
    "ahmedabad":          (23.0225, 72.5714),
    "surat":              (21.1702, 72.8311),
    "vadodara":           (22.3072, 73.1812),
    "rajkot":             (22.3039, 70.8022),
    "aurangabad":         (19.8762, 75.3433),

    # South India
    "bangalore":          (12.9716, 77.5946),
    "bengaluru":          (12.9716, 77.5946),
    "hyderabad":          (17.3850, 78.4867),
    "chennai":            (13.0827, 80.2707),
    "kochi":              (9.9312,  76.2673),
    "cochin":             (9.9312,  76.2673),
    "thiruvananthapuram": (8.5241,  76.9366),
    "trivandrum":         (8.5241,  76.9366),
    "mysore":             (12.2958, 76.6394),
    "coimbatore":         (11.0168, 76.9558),
    "madurai":            (9.9252,  78.1198),
    "ooty":               (11.4102, 76.6950),
    "munnar":             (10.0889, 77.0595),
    "alleppey":           (9.4981,  76.3388),
    "alappuzha":          (9.4981,  76.3388),
    "varkala":            (8.7334,  76.7157),
    "kovalam":            (8.4004,  76.9785),
    "wayanad":            (11.6854, 76.1320),
    "thekkady":           (9.5996,  77.1700),
    "pondicherry":        (11.9416, 79.8083),
    "puducherry":         (11.9416, 79.8083),
    "coorg":              (12.3375, 75.8069),
    "dharmasthala":       (12.9560, 75.3740),

    # Andhra Pradesh
    "tirupati":           (13.6288, 79.4192),
    "vijayawada":         (16.5062, 80.6480),
    "visakhapatnam":      (17.6868, 83.2185),
    "vizag":              (17.6868, 83.2185),
    "guntur":             (16.3067, 80.4365),
    "nellore":            (14.4426, 79.9865),
    "kurnool":            (15.8281, 78.0373),
    "anantapur":          (14.6819, 77.6006),
    "kakinada":           (16.9891, 82.2475),
    "rajahmundry":        (17.0005, 81.8040),
    "eluru":              (16.7107, 81.0952),
    "ongole":             (15.5057, 80.0499),
    "amaravati":          (16.5730, 80.3578),

    # Telangana
    "warangal":           (17.9784, 79.5941),
    "karimnagar":         (18.4386, 79.1288),
    "nizamabad":          (18.6725, 78.0941),
    "khammam":            (17.2473, 80.1514),

    # East India
    "kolkata":            (22.5726, 88.3639),
    "bhubaneswar":        (20.2961, 85.8245),
    "puri":               (19.8135, 85.8312),
    "guwahati":           (26.1445, 91.7362),
    "darjeeling":         (27.0360, 88.2627),
    "gangtok":            (27.3314, 88.6138),
    "shillong":           (25.5788, 91.8933),
    "imphal":             (24.8170, 93.9368),

    # International
    "paris":              (48.8566,  2.3522),
    "london":             (51.5074, -0.1278),
    "dubai":              (25.2048, 55.2708),
    "singapore":          (1.3521,  103.8198),
    "bangkok":            (13.7563, 100.5018),
    "new york":           (40.7128, -74.0060),
    "tokyo":              (35.6762, 139.6503),
    "sydney":             (-33.8688, 151.2093),
    "rome":               (41.9028, 12.4964),
    "amsterdam":          (52.3676,  4.9041),
    "barcelona":          (41.3851,  2.1734),
    "zurich":             (47.3769,  8.5417),
}

_MODE_EMOJI: Dict[str, str] = {
    "Flight": "✈️", "Train": "🚂", "Bus": "🚌", "Car": "🚗", "Bike": "🏍️",
}


def _resolve_coords(city: str) -> Optional[Tuple[float, float]]:
    """Resolve city name to (lat, lon). Tries local dict then maps_service."""
    norm = city.strip().lower()
    if norm in _CITY_COORDS:
        return _CITY_COORDS[norm]
    try:
        from src.data.maps_service import CITY_COORDINATES
        norm2 = norm.replace(" ", "_")
        if norm2 in CITY_COORDINATES:
            coords = CITY_COORDINATES[norm2]
            lat, lon = coords[0], coords[1]
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
    except Exception:
        pass
    return None


def render_map_placeholder() -> None:
    """Render the empty-state placeholder before prediction."""
    st.markdown("""
    <div style="
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 40px 24px;
      text-align: center;
      margin-bottom: 16px;
    ">
      <div style="font-size:40px;margin-bottom:14px;">🗺️</div>
      <div style="font-family:'Outfit',sans-serif;font-weight:700;color:#F1F5F9;font-size:15px;margin-bottom:8px;">
        Interactive Route Map
      </div>
      <div style="font-size:12px;color:#94A3B8;line-height:1.6;">
        Predict a trip to visualize your journey on an interactive map.
      </div>
      <div style="font-size:11px;color:#475569;margin-top:10px;font-style:italic;">
        "The world is a book, and those who do not travel read only one page."
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_route_map(route_info: Dict[str, Any], travel_mode: str = "Car") -> None:
    """Render an interactive Plotly Scattergeo route map.
    Works everywhere — no folium, no external tile servers, no API tokens.
    """
    st.markdown(
        '<div id="interactive-route-map" style="font-family:\'Outfit\',sans-serif;font-size:15px;font-weight:700;'
        'color:var(--text-primary);display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        '🗺️ Interactive Route Map</div>',
        unsafe_allow_html=True,
    )

    source      = route_info.get("source", "")
    destination = route_info.get("destination", "")

    # Indirect route bars (keep existing logic)
    try:
        from src.intelligence.destination_rules import get_indirect_route_bars
        indirect_bars = get_indirect_route_bars(source, destination)
    except Exception:
        indirect_bars = None

    if indirect_bars:
        bars_html = f"""
        <div style="padding:12px 16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
             border-radius:10px;margin-bottom:10px;display:flex;flex-direction:column;gap:8px;">
          <div style="font-size:11px;font-weight:700;color:#A78BFA;text-transform:uppercase;letter-spacing:0.05em;">
            🗺️ Route Accessibility (Indirect Route)
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
            <div style="padding:6px 10px;background:rgba(255,255,255,0.02);border:1.5px solid rgba(255,255,255,0.08);border-radius:6px;display:flex;justify-content:space-between;">
              <span style="font-size:11px;font-weight:700;color:#F1F5F9;">🚂 Train</span>
              <span style="font-size:11px;font-weight:600;color:#34D399;">{indirect_bars.get('Train','')}</span>
            </div>
            <div style="padding:6px 10px;background:rgba(255,255,255,0.02);border:1.5px solid rgba(255,255,255,0.08);border-radius:6px;display:flex;justify-content:space-between;">
              <span style="font-size:11px;font-weight:700;color:#F1F5F9;">🚌 Bus</span>
              <span style="font-size:11px;font-weight:600;color:#34D399;">{indirect_bars.get('Bus','')}</span>
            </div>
            <div style="padding:6px 10px;background:rgba(255,255,255,0.02);border:1.5px solid rgba(255,255,255,0.08);border-radius:6px;display:flex;justify-content:space-between;">
              <span style="font-size:11px;font-weight:700;color:#F1F5F9;">✈️ Flight</span>
              <span style="font-size:11px;font-weight:600;color:#34D399;">{indirect_bars.get('Flight','')}</span>
            </div>
          </div>
        </div>
        """
        st.markdown(bars_html, unsafe_allow_html=True)

    # Resolve coordinates
    src_coords  = route_info.get("source_coords") or _resolve_coords(source)
    dest_coords = route_info.get("dest_coords")   or _resolve_coords(destination)

    if not src_coords or not dest_coords:
        st.info(
            f"Map coordinates not found for **{source}** → **{destination}**. "
            "Budget figures above are still accurate."
        )
        return

    # Build Plotly Scattergeo map — no external dependencies
    mode_emoji = _MODE_EMOJI.get(travel_mode, "📍")

    # Determine map scope
    lat_avg = (src_coords[0] + dest_coords[0]) / 2
    lon_avg = (src_coords[1] + dest_coords[1]) / 2

    if -10 <= lat_avg <= 40 and 65 <= lon_avg <= 100:
        scope = "asia"
    elif 35 <= lat_avg <= 72 and -10 <= lon_avg <= 40:
        scope = "europe"
    else:
        scope = "world"

    fig = go.Figure()

    # Route line
    fig.add_trace(go.Scattergeo(
        lat=[src_coords[0], dest_coords[0]],
        lon=[src_coords[1], dest_coords[1]],
        mode="lines",
        line=dict(width=3, color="#7C3AED"),
        name="Route",
        hoverinfo="skip",
    ))

    # Origin marker
    fig.add_trace(go.Scattergeo(
        lat=[src_coords[0]],
        lon=[src_coords[1]],
        mode="markers+text",
        marker=dict(size=14, color="#10B981", symbol="circle",
                    line=dict(width=2, color="#ffffff")),
        text=[f"{mode_emoji} {source.title()}"],
        textposition="top center",
        textfont=dict(size=11, color="#F1F5F9"),
        name=f"From: {source.title()}",
        hovertemplate=f"<b>Origin</b><br>{source.title()}<extra></extra>",
    ))

    # Destination marker
    fig.add_trace(go.Scattergeo(
        lat=[dest_coords[0]],
        lon=[dest_coords[1]],
        mode="markers+text",
        marker=dict(size=14, color="#EF4444", symbol="circle",
                    line=dict(width=2, color="#ffffff")),
        text=[f"📍 {destination.title()}"],
        textposition="top center",
        textfont=dict(size=11, color="#F1F5F9"),
        name=f"To: {destination.title()}",
        hovertemplate=f"<b>Destination</b><br>{destination.title()}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(5,8,22,0.95)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            scope=scope,
            showland=True,
            landcolor="#0f172a",
            showocean=True,
            oceancolor="#0a0f2e",
            showlakes=True,
            lakecolor="#0a0f2e",
            showcountries=True,
            countrycolor="rgba(255,255,255,0.15)",
            showcoastlines=True,
            coastlinecolor="rgba(255,255,255,0.1)",
            bgcolor="rgba(5,8,22,0.95)",
            projection_type="natural earth",
            center=dict(lat=lat_avg, lon=lon_avg),
        ),
        height=300,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
        font=dict(color="#94A3B8", family="Inter"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
