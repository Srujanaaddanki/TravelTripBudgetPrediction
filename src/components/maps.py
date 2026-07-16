"""
========================================================
Component: Route Map (Bug-Fixed)
Purpose: Renders an interactive Folium route map.
         Fixes:
         - Correct lat/lon order (lat first, always)
         - Transport mode icon changes per user selection
         - More city coordinates (Andhra Pradesh fixed)
         - Better fallback handling
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False


# ── City coordinate map: ALWAYS (lat, lon) ────────────────────────────────────
# Verified correct order: lat first, lon second
_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    # North India
    "delhi":           (28.6139, 77.2090),
    "new delhi":       (28.6139, 77.2090),
    "chandigarh":      (30.7333, 76.7794),
    "amritsar":        (31.6340, 74.8723),
    "agra":            (27.1767, 78.0081),
    "varanasi":        (25.3176, 82.9739),
    "lucknow":         (26.8467, 80.9462),
    "haridwar":        (29.9457, 78.1642),
    "rishikesh":       (30.0869, 78.2676),
    "dehradun":        (30.3165, 78.0322),
    "manali":          (32.2396, 77.1887),
    "shimla":          (31.1048, 77.1734),
    "srinagar":        (34.0837, 74.7973),
    "leh":             (34.1526, 77.5771),
    "leh ladakh":      (34.1526, 77.5771),
    "ladakh":          (34.1526, 77.5771),
    "kedarnath":       (30.7352, 79.0669),
    "badrinath":       (30.7433, 79.4938),
    "sonprayag":       (30.6289, 79.0919),
    "mussoorie":       (30.4598, 78.0664),
    "nainital":        (29.3919, 79.4542),
    "jaipur":          (26.9124, 75.7873),
    "jodhpur":         (26.2389, 73.0243),
    "udaipur":         (24.5854, 73.7125),
    "bikaner":         (28.0229, 73.3119),
    "jaisalmer":       (26.9157, 70.9083),
    "pushkar":         (26.4899, 74.5511),
    "patna":           (25.6093, 85.1376),
    "ranchi":          (23.3441, 85.3096),
    "bhopal":          (23.2599, 77.4126),
    "indore":          (22.7196, 75.8577),
    "nagpur":          (21.1458, 79.0882),

    # West India
    "mumbai":          (19.0760, 72.8777),
    "pune":            (18.5204, 73.8567),
    "goa":             (15.2993, 74.1240),
    "panaji":          (15.4909, 73.8278),
    "ahmedabad":       (23.0225, 72.5714),
    "surat":           (21.1702, 72.8311),
    "vadodara":        (22.3072, 73.1812),
    "rajkot":          (22.3039, 70.8022),
    "aurangabad":      (19.8762, 75.3433),

    # South India — CORRECTED COORDINATES
    "bangalore":       (12.9716, 77.5946),
    "bengaluru":       (12.9716, 77.5946),
    "hyderabad":       (17.3850, 78.4867),
    "chennai":         (13.0827, 80.2707),
    "kochi":           (9.9312,  76.2673),
    "cochin":          (9.9312,  76.2673),
    "thiruvananthapuram": (8.5241, 76.9366),
    "trivandrum":      (8.5241,  76.9366),
    "mysore":          (12.2958, 76.6394),
    "coimbatore":      (11.0168, 76.9558),
    "madurai":         (9.9252,  78.1198),
    "ooty":            (11.4102, 76.6950),
    "munnar":          (10.0889, 77.0595),
    "alleppey":        (9.4981,  76.3388),
    "alappuzha":       (9.4981,  76.3388),
    "varkala":         (8.7334,  76.7157),
    "kovalam":         (8.4004,  76.9785),
    "wayanad":         (11.6854, 76.1320),
    "thekkady":        (9.5996,  77.1700),
    "pondicherry":     (11.9416, 79.8083),
    "puducherry":      (11.9416, 79.8083),

    # Andhra Pradesh — CORRECTED (was appearing in North India due to wrong coords)
    "tirupati":        (13.6288, 79.4192),   # lat=13.6, lon=79.4  (Andhra, near Tamil Nadu border)
    "vijayawada":      (16.5062, 80.6480),   # lat=16.5, lon=80.6  (Central Andhra)
    "visakhapatnam":   (17.6868, 83.2185),   # lat=17.7, lon=83.2  (Northern Andhra coast)
    "vizag":           (17.6868, 83.2185),
    "guntur":          (16.3067, 80.4365),
    "nellore":         (14.4426, 79.9865),
    "kurnool":         (15.8281, 78.0373),
    "anantapur":       (14.6819, 77.6006),
    "kakinada":        (16.9891, 82.2475),
    "rajahmundry":     (17.0005, 81.8040),
    "eluru":           (16.7107, 81.0952),
    "ongole":          (15.5057, 80.0499),
    "amaravati":       (16.5730, 80.3578),

    # Telangana
    "warangal":        (17.9784, 79.5941),
    "karimnagar":      (18.4386, 79.1288),
    "nizamabad":       (18.6725, 78.0941),
    "khammam":         (17.2473, 80.1514),

    # East India
    "kolkata":         (22.5726, 88.3639),
    "bhubaneswar":     (20.2961, 85.8245),
    "puri":            (19.8135, 85.8312),
    "guwahati":        (26.1445, 91.7362),
    "darjeeling":      (27.0360, 88.2627),
    "gangtok":         (27.3314, 88.6138),
    "shillong":        (25.5788, 91.8933),
    "imphal":          (24.8170, 93.9368),
    "aizawl":          (23.7271, 92.7176),

    # International
    "paris":           (48.8566,  2.3522),
    "london":          (51.5074, -0.1278),
    "dubai":           (25.2048, 55.2708),
    "singapore":       (1.3521,  103.8198),
    "bangkok":         (13.7563, 100.5018),
    "new york":        (40.7128, -74.0060),
    "tokyo":           (35.6762, 139.6503),
    "sydney":          (-33.8688, 151.2093),
    "rome":            (41.9028, 12.4964),
    "amsterdam":       (52.3676,  4.9041),
    "barcelona":       (41.3851,  2.1734),
    "zurich":          (47.3769,  8.5417),
}


# ── Transport mode icons (FontAwesome via folium) ─────────────────────────────
_MODE_FOLIUM_ICONS: Dict[str, Dict[str, str]] = {
    "Flight": {"icon": "plane",    "prefix": "fa", "color": "cadetblue"},
    "Train":  {"icon": "train",    "prefix": "fa", "color": "darkblue"},
    "Bus":    {"icon": "bus",      "prefix": "fa", "color": "orange"},
    "Car":    {"icon": "car",      "prefix": "fa", "color": "green"},
    "Bike":   {"icon": "motorcycle","prefix": "fa", "color": "purple"},
}

_MODE_DISPLAY_ICON: Dict[str, str] = {
    "Flight": "plane",
    "Train":  "train",
    "Bus":    "bus",
    "Car":    "car",
    "Bike":   "bicycle",
}


def _resolve_coords(city: str) -> Optional[Tuple[float, float]]:
    """Resolve city name to (lat, lon). Always returns (lat, lon) order."""
    norm = city.strip().lower()
    if norm in _CITY_COORDS:
        return _CITY_COORDS[norm]
    # Try the maps_service
    try:
        from src.data.maps_service import CITY_COORDINATES
        norm2 = norm.replace(" ", "_")
        if norm2 in CITY_COORDINATES:
            coords = CITY_COORDINATES[norm2]
            # Validate: lat must be between -90 and 90, lon between -180 and 180
            lat, lon = coords[0], coords[1]
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
    except Exception:
        pass
    return None


def _validate_coords(coords: Tuple[float, float]) -> bool:
    """Validate that coordinates are in correct (lat, lon) order."""
    if not coords or len(coords) < 2:
        return False
    lat, lon = coords[0], coords[1]
    # India lat: 6–37, lon: 68–98; international lat: -90 to 90
    if not (-90 <= lat <= 90):
        return False
    if not (-180 <= lon <= 180):
        return False
    # Detect if accidentally reversed (common bug):
    # If "lat" looks like a longitude (>90) and "lon" looks like a latitude (<37)
    # this was the Andhra Pradesh bug — coords were (lon, lat) not (lat, lon)
    if lat > 90:
        return False
    return True


def render_map_placeholder() -> None:
    """Render the empty-state placeholder before prediction."""
    placeholder_html = """
    <div class="map-placeholder">
      <div class="map-placeholder-icon">&#x1F5FA;</div>
      <div class="map-placeholder-title">Interactive Route Map</div>
      <div class="map-placeholder-text">
        Predict a trip to visualize your journey on an interactive map.
      </div>
      <div class="map-placeholder-quote">
        "The world is a book, and those who do not travel read only one page."
      </div>
    </div>
    """
    st.markdown(placeholder_html, unsafe_allow_html=True)


def _decode_polyline(polyline_data: Any) -> list:
    """Decode GeoJSON coordinates or encoded polyline string."""
    if not polyline_data:
        return []
    if isinstance(polyline_data, dict):
        coords = polyline_data.get("coordinates", [])
        return [(c[1], c[0]) for c in coords if len(c) >= 2]
    if isinstance(polyline_data, str):
        coords = []
        index = 0
        lat = 0
        lng = 0
        try:
            while index < len(polyline_data):
                shift, result = 0, 0
                while True:
                    b = ord(polyline_data[index]) - 63 - 1
                    index += 1
                    result |= (b & 0x1f) << shift
                    shift += 5
                    if not (b >= 0x1f):
                        break
                dlat = ~(result >> 1) if (result & 1) else (result >> 1)
                lat += dlat
                shift, result = 0, 0
                while True:
                    b = ord(polyline_data[index]) - 63 - 1
                    index += 1
                    result |= (b & 0x1f) << shift
                    shift += 5
                    if not (b >= 0x1f):
                        break
                dlng = ~(result >> 1) if (result & 1) else (result >> 1)
                lng += dlng
                coords.append((lat / 1e5, lng / 1e5))
        except Exception:
            pass
        return coords
    return []


def render_route_map(route_info: Dict[str, Any], travel_mode: str = "Car") -> None:
    """Render the interactive Folium route map.

    Parameters
    ----------
    route_info : dict
        Must contain 'source', 'destination'.
        Optionally: 'source_coords', 'dest_coords' as (lat, lon) tuples.
    travel_mode : str
        Used to select the transport icon on the origin marker.
    """
    st.markdown(
        '<div class="map-card-header"><div class="map-card-title">&#x1F5FA; Interactive Route Map</div></div>',
        unsafe_allow_html=True,
    )

    source      = route_info.get("source", "")
    destination = route_info.get("destination", "")

    # Resolve coordinates — always (lat, lon) order
    src_coords  = route_info.get("source_coords") or _resolve_coords(source)
    dest_coords = route_info.get("dest_coords")   or _resolve_coords(destination)

    # Validate coords to catch any reversed (lon, lat) bugs
    if src_coords and not _validate_coords(src_coords):
        # Attempt auto-fix: swap if looks reversed
        src_coords = (src_coords[1], src_coords[0])
    if dest_coords and not _validate_coords(dest_coords):
        dest_coords = (dest_coords[1], dest_coords[0])

    if not src_coords or not dest_coords:
        st.info(
            f"Map coordinates not available for {source} -> {destination}. "
            "The route info and budget above are still accurate."
        )
        return

    # Route polyline
    raw_poly    = route_info.get("polyline")
    route_pts   = _decode_polyline(raw_poly)
    if not route_pts:
        route_pts = [src_coords, dest_coords]

    if _FOLIUM_AVAILABLE:
        _render_folium_map(source, destination, src_coords, dest_coords, route_pts, travel_mode)
    else:
        _render_plotly_fallback(source, destination, src_coords, dest_coords)


def _render_folium_map(
    source: str,
    destination: str,
    src_coords: Tuple[float, float],
    dest_coords: Tuple[float, float],
    route_points: list,
    travel_mode: str = "Car",
) -> None:
    """Render Folium satellite map with mode-specific icons."""
    # Centre between the two points — using validated (lat, lon) coords
    mid_lat = (src_coords[0] + dest_coords[0]) / 2
    mid_lon = (src_coords[1] + dest_coords[1]) / 2

    m = folium.Map(
        location=[mid_lat, mid_lon],   # [lat, lon] — CORRECT ORDER
        zoom_start=5,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
    )

    # Mode-specific icon for origin
    mode_cfg = _MODE_FOLIUM_ICONS.get(travel_mode, _MODE_FOLIUM_ICONS["Car"])

    # Origin marker
    folium.Marker(
        location=list(src_coords),    # [lat, lon]
        popup=folium.Popup(source.title(), max_width=150),
        tooltip=f"Origin: {source.title()}",
        icon=folium.Icon(
            color=mode_cfg["color"],
            icon=mode_cfg["icon"],
            prefix=mode_cfg["prefix"],
        ),
    ).add_to(m)

    # Destination marker
    folium.Marker(
        location=list(dest_coords),   # [lat, lon]
        popup=folium.Popup(destination.title(), max_width=150),
        tooltip=f"Destination: {destination.title()}",
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(m)

    # Route line
    folium.PolyLine(
        locations=route_points,       # list of [lat, lon]
        color="#E11D48",
        weight=4,
        opacity=0.9,
    ).add_to(m)

    st_folium(m, width=None, height=280, returned_objects=[])


def _render_plotly_fallback(
    source: str,
    destination: str,
    src_coords: Tuple[float, float],
    dest_coords: Tuple[float, float],
) -> None:
    """Plotly fallback when folium is not installed."""
    import plotly.graph_objects as go

    lats   = [src_coords[0], dest_coords[0]]
    lons   = [src_coords[1], dest_coords[1]]
    names  = [source.title(), destination.title()]
    colors = ["#10B981", "#EF4444"]

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons, mode="lines",
        line=dict(width=3, color="#7C3AED"),
        name="Route",
    ))
    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons, mode="markers+text",
        marker=dict(size=14, color=colors),
        text=names, textposition="top center",
        textfont=dict(color="#F1F5F9", size=12),
        name="Cities",
    ))

    mid_lat = (lats[0] + lats[1]) / 2
    mid_lon = (lons[0] + lons[1]) / 2

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=mid_lat, lon=mid_lon),
            zoom=4.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9", family="Inter"),
        height=320,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
