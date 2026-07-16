"""
========================================================
Module: Route Alternatives Engine
Purpose: When no direct route is available, generates
         a realistic multi-stop alternate route chain
         with waypoints, cost estimates, and travel times.
         Also provides transport mode availability flags
         to prevent misleading bars in comparison charts.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ── City coordinate reference (lat, lon) ─────────────────────────────────────
_COORDS: Dict[str, Tuple[float, float]] = {
    "hyderabad":    (17.3850, 78.4867),
    "delhi":        (28.6139, 77.2090),
    "new delhi":    (28.6139, 77.2090),
    "haridwar":     (29.9457, 78.1642),
    "rishikesh":    (30.0869, 78.2676),
    "dehradun":     (30.3165, 78.0322),
    "sonprayag":    (30.6289, 79.0919),
    "gaurikund":    (30.6550, 79.1175),
    "kedarnath":    (30.7352, 79.0669),
    "badrinath":    (30.7433, 79.4938),
    "mumbai":       (19.0760, 72.8777),
    "bangalore":    (12.9716, 77.5946),
    "chennai":      (13.0827, 80.2707),
    "kolkata":      (22.5726, 88.3639),
    "jaipur":       (26.9124, 75.7873),
    "manali":       (32.2396, 77.1887),
    "goa":          (15.2993, 74.1240),
    "shimla":       (31.1048, 77.1734),
    "leh":          (34.1526, 77.5771),
    "ladakh":       (34.1526, 77.5771),
    "varanasi":     (25.3176, 82.9739),
    "amritsar":     (31.6340, 74.8723),
    "agra":         (27.1767, 78.0081),
    "pune":         (18.5204, 73.8567),
    "ahmedabad":    (23.0225, 72.5714),
    "chandigarh":   (30.7333, 76.7794),
    "vijayawada":   (16.5062, 80.6480),
    "visakhapatnam":(17.6868, 83.2185),
    "kochi":        (9.9312,  76.2673),
    "srinagar":     (34.0837, 74.7973),
    "paris":        (48.8566,  2.3522),
    "dubai":        (25.2048, 55.2708),
    "london":       (51.5074, -0.1278),
    "singapore":    (1.3521,  103.8198),
    "tirupati":     (13.6288, 79.4192),
    "nagpur":       (21.1458, 79.0882),
    "indore":       (22.7196, 75.8577),
    "lucknow":      (26.8467, 80.9462),
    "patna":        (25.6093, 85.1376),
    "bhubaneswar":  (20.2961, 85.8245),
    "guwahati":     (26.1445, 91.7362),
    "darjeeling":   (27.0360, 88.2627),
    "gangtok":      (27.3314, 88.6138),
    "shillong":     (25.5788, 91.8933),
    "ranthambore":  (26.0173, 76.5026),
    "jim corbett":  (29.5300, 78.7747),
    "munnar":       (10.0889, 77.0595),
    "ooty":         (11.4102, 76.6950),
    "alleppey":     (9.4981,  76.3388),
    "jodhpur":      (26.2389, 73.0243),
    "udaipur":      (24.5854, 73.7125),
    "mysore":       (12.2958, 76.6394),
    "pondicherry":  (11.9416, 79.8083),
    "coimbatore":   (11.0168, 76.9558),
}

# ── Known route chains: (src_keyword, dst_keyword) → waypoints ───────────────
# Each waypoint: {"city": str, "mode": str, "notes": str}
_KNOWN_ROUTES: List[Dict[str, Any]] = [

    # Hyderabad → Kedarnath (no direct train/flight to Kedarnath)
    {
        "src_keys": ["hyderabad"],
        "dst_keys": ["kedarnath"],
        "segments": [
            {"from": "Hyderabad", "to": "Delhi", "mode": "Flight", "km": 1490, "notes": "Direct flight ~2h"},
            {"from": "Delhi", "to": "Haridwar", "mode": "Train", "km": 215, "notes": "Shatabdi/Jan Shatabdi ~4.5h"},
            {"from": "Haridwar", "to": "Rishikesh", "mode": "Bus", "km": 24, "notes": "Shared taxi/bus ~45min"},
            {"from": "Rishikesh", "to": "Sonprayag", "mode": "Bus", "km": 220, "notes": "Govt bus / shared jeep ~8h"},
            {"from": "Sonprayag", "to": "Gaurikund", "mode": "Bus", "km": 5, "notes": "Short bus ~15min"},
            {"from": "Gaurikund", "to": "Kedarnath", "mode": "Trek", "km": 16, "notes": "Trek ~6-8h OR Helicopter from Phata"},
        ],
    },

    # Hyderabad → Ladakh / Leh
    {
        "src_keys": ["hyderabad"],
        "dst_keys": ["leh", "ladakh"],
        "segments": [
            {"from": "Hyderabad", "to": "Delhi", "mode": "Flight", "km": 1490, "notes": "Direct flight ~2h"},
            {"from": "Delhi", "to": "Leh", "mode": "Flight", "km": 700, "notes": "Direct flight ~1.5h (IndiGo/Air India)"},
        ],
    },

    # Delhi → Kedarnath
    {
        "src_keys": ["delhi", "new delhi"],
        "dst_keys": ["kedarnath"],
        "segments": [
            {"from": "Delhi", "to": "Haridwar", "mode": "Train", "km": 215, "notes": "Shatabdi ~4.5h"},
            {"from": "Haridwar", "to": "Rishikesh", "mode": "Bus", "km": 24, "notes": "Shared taxi ~45min"},
            {"from": "Rishikesh", "to": "Sonprayag", "mode": "Bus", "km": 220, "notes": "Shared jeep ~8h"},
            {"from": "Sonprayag", "to": "Gaurikund", "mode": "Bus", "km": 5, "notes": "Short bus ~15min"},
            {"from": "Gaurikund", "to": "Kedarnath", "mode": "Trek", "km": 16, "notes": "Trek 6-8h OR Helicopter"},
        ],
    },

    # Hyderabad → Paris (international)
    {
        "src_keys": ["hyderabad"],
        "dst_keys": ["paris", "france"],
        "segments": [
            {"from": "Hyderabad", "to": "Delhi", "mode": "Flight", "km": 1490, "notes": "Domestic connection ~2h"},
            {"from": "Delhi", "to": "Paris", "mode": "Flight", "km": 6700, "notes": "Direct Air France / IndiGo ~8.5h"},
        ],
    },

    # Bangalore → Paris
    {
        "src_keys": ["bangalore"],
        "dst_keys": ["paris", "france"],
        "segments": [
            {"from": "Bangalore", "to": "Delhi", "mode": "Flight", "km": 1750, "notes": "Domestic ~2.5h"},
            {"from": "Delhi", "to": "Paris", "mode": "Flight", "km": 6700, "notes": "Direct ~8.5h"},
        ],
    },

    # Mumbai → Paris
    {
        "src_keys": ["mumbai"],
        "dst_keys": ["paris", "france"],
        "segments": [
            {"from": "Mumbai", "to": "Paris", "mode": "Flight", "km": 7200, "notes": "Direct Air France ~9.5h"},
        ],
    },

    # Hyderabad → Dubai
    {
        "src_keys": ["hyderabad"],
        "dst_keys": ["dubai", "uae"],
        "segments": [
            {"from": "Hyderabad", "to": "Dubai", "mode": "Flight", "km": 2450, "notes": "Direct flight ~3.5h (IndiGo/Air Arabia)"},
        ],
    },

    # Hyderabad → Manali (no direct train)
    {
        "src_keys": ["hyderabad"],
        "dst_keys": ["manali"],
        "segments": [
            {"from": "Hyderabad", "to": "Chandigarh", "mode": "Flight", "km": 1550, "notes": "Via Delhi / direct ~2h"},
            {"from": "Chandigarh", "to": "Manali", "mode": "Bus", "km": 310, "notes": "HRTC overnight bus ~10h"},
        ],
    },

    # Hyderabad → Darjeeling
    {
        "src_keys": ["hyderabad"],
        "dst_keys": ["darjeeling"],
        "segments": [
            {"from": "Hyderabad", "to": "Bagdogra (NJP)", "mode": "Flight", "km": 1650, "notes": "Nearest airport ~2h"},
            {"from": "Bagdogra", "to": "Darjeeling", "mode": "Car", "km": 88, "notes": "Shared jeep / taxi ~3.5h"},
        ],
    },

    # Hyderabad → Lakshadweep (sea route)
    {
        "src_keys": ["hyderabad"],
        "dst_keys": ["lakshadweep"],
        "segments": [
            {"from": "Hyderabad", "to": "Kochi", "mode": "Flight", "km": 1300, "notes": "Direct ~1.5h"},
            {"from": "Kochi", "to": "Agatti Island", "mode": "Flight", "km": 459, "notes": "Lakshadweep helicopter service"},
        ],
    },
]


# ── Destination mode availability ─────────────────────────────────────────────
# Some destinations have no practical direct access for certain transport modes.

_MODE_RESTRICTIONS: Dict[str, Dict[str, str]] = {
    "kedarnath": {
        "Flight": "unavailable",  # No airport at Kedarnath
        "Train":  "unavailable",  # No train to Kedarnath
        "Bus":    "partial",      # Bus only to Sonprayag/Gaurikund
        "Car":    "partial",      # Car only to Sonprayag
        "Bike":   "partial",      # Bike only to Gaurikund (last 16km trek)
    },
    "badrinath": {
        "Flight": "unavailable",
        "Train":  "unavailable",
        "Bus":    "partial",
        "Car":    "partial",
        "Bike":   "partial",
    },
    "manali": {
        "Flight": "partial",      # Nearest: Bhuntar (Kullu) ~50km
        "Train":  "partial",      # Nearest: Chandigarh / Pathankot
        "Bus":    "available",
        "Car":    "available",
        "Bike":   "available",
    },
    "leh": {
        "Flight": "available",
        "Train":  "unavailable",  # No railway to Leh
        "Bus":    "partial",      # Only via Manali–Leh highway (seasonal)
        "Car":    "partial",      # Manali–Leh highway (seasonal)
        "Bike":   "partial",      # Popular biker route (seasonal)
    },
    "ladakh": {
        "Flight": "available",
        "Train":  "unavailable",
        "Bus":    "partial",
        "Car":    "partial",
        "Bike":   "partial",
    },
    "darjeeling": {
        "Flight": "partial",      # Nearest: Bagdogra ~88km
        "Train":  "partial",      # Nearest: NJP / Siliguri
        "Bus":    "available",
        "Car":    "available",
        "Bike":   "available",
    },
    "goa": {
        "Flight": "available",
        "Train":  "available",
        "Bus":    "available",
        "Car":    "available",
        "Bike":   "available",
    },
    "paris": {
        "Flight": "available",
        "Train":  "unavailable",   # No train from India to Paris
        "Bus":    "unavailable",
        "Car":    "unavailable",
        "Bike":   "unavailable",
    },
    "dubai": {
        "Flight": "available",
        "Train":  "unavailable",
        "Bus":    "unavailable",
        "Car":    "unavailable",
        "Bike":   "unavailable",
    },
    "london": {
        "Flight": "available",
        "Train":  "unavailable",
        "Bus":    "unavailable",
        "Car":    "unavailable",
        "Bike":   "unavailable",
    },
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _mode_cost_per_km(mode: str) -> float:
    return {
        "Flight": 8.0, "Train": 1.5, "Bus": 1.8,
        "Car": 6.0, "Bike": 2.5, "Trek": 0.0,
    }.get(mode, 3.0)


def _mode_speed_kmh(mode: str) -> float:
    return {
        "Flight": 800.0, "Train": 55.0, "Bus": 40.0,
        "Car": 60.0, "Bike": 40.0, "Trek": 3.0,
    }.get(mode, 50.0)


def get_mode_availability(destination: str) -> Dict[str, str]:
    """Return availability of each transport mode for a destination.

    Returns
    -------
    dict: mode → "available" | "partial" | "unavailable"
    """
    dest_lower = destination.strip().lower()
    for key, restrictions in _MODE_RESTRICTIONS.items():
        if key in dest_lower or dest_lower in key:
            return restrictions
    # Default: all modes available
    return {m: "available" for m in ["Flight", "Train", "Bus", "Car", "Bike"]}


def get_alternate_route(
    source: str,
    destination: str,
    preferred_mode: str = "Train",
) -> Optional[Dict[str, Any]]:
    """Return an alternate multi-stop route when no direct route exists.

    Parameters
    ----------
    source, destination : str
        City names.
    preferred_mode : str
        User's preferred travel mode.

    Returns
    -------
    dict | None
        Keys: segments (list of dicts), total_km, total_cost_inr,
              total_time_h, description
        Returns None if destination has direct route available.
    """
    src_lower = source.strip().lower()
    dst_lower = destination.strip().lower()

    # Check if destination generally has direct access
    avail = get_mode_availability(dst_lower)
    if avail.get(preferred_mode) == "available":
        return None  # Direct route exists — no alternate needed

    # Search known route chains
    for route in _KNOWN_ROUTES:
        src_match = any(k in src_lower for k in route["src_keys"])
        dst_match = any(k in dst_lower for k in route["dst_keys"])
        if src_match and dst_match:
            segments = route["segments"]
            total_km   = sum(s["km"] for s in segments)
            total_cost = sum(
                s["km"] * _mode_cost_per_km(s["mode"])
                for s in segments
            )
            total_time = sum(
                s["km"] / _mode_speed_kmh(s["mode"])
                for s in segments
            )
            return {
                "segments":        segments,
                "total_km":        round(total_km, 1),
                "total_cost_inr":  round(total_cost, -1),
                "total_time_h":    round(total_time, 1),
                "description":     f"Indirect route via intermediate hubs",
                "source":          "Route Intelligence Engine",
            }

    # Generic fallback: build a route via nearest hub
    hub = _find_nearest_hub(src_lower, dst_lower)
    if hub:
        src_coords = _COORDS.get(src_lower)
        hub_coords = _COORDS.get(hub)
        dst_coords = _COORDS.get(dst_lower)
        if src_coords and hub_coords and dst_coords:
            km1 = _haversine(*src_coords, *hub_coords)
            km2 = _haversine(*hub_coords, *dst_coords)
            segments = [
                {"from": source.title(), "to": hub.title(), "mode": "Flight",
                 "km": round(km1, 0), "notes": "Nearest hub connection"},
                {"from": hub.title(), "to": destination.title(), "mode": "Bus/Car",
                 "km": round(km2, 0), "notes": "Local transfer"},
            ]
            total_km   = km1 + km2
            total_cost = km1 * 8 + km2 * 3
            total_time = km1 / 800 + km2 / 50
            return {
                "segments":       segments,
                "total_km":       round(total_km, 1),
                "total_cost_inr": round(total_cost, -1),
                "total_time_h":   round(total_time, 1),
                "description":    "Auto-generated indirect route via nearest hub",
                "source":         "Haversine Fallback",
            }
    return None


def _find_nearest_hub(source: str, destination: str) -> Optional[str]:
    """Find the nearest major hub city to act as a connecting stop."""
    major_hubs = ["delhi", "mumbai", "bangalore", "hyderabad", "chennai", "kolkata",
                  "chandigarh", "haridwar", "dehradun", "bagdogra"]
    dst_coords = _COORDS.get(destination)
    if not dst_coords:
        return None
    best_hub, best_dist = None, float("inf")
    for hub in major_hubs:
        if hub == source:
            continue
        hub_coords = _COORDS.get(hub)
        if hub_coords:
            d = _haversine(*dst_coords, *hub_coords)
            if d < best_dist:
                best_dist = d
                best_hub  = hub
    return best_hub


def format_alternate_route_html(route: Dict[str, Any]) -> str:
    """Render the alternate route as an HTML string for Streamlit."""
    if not route:
        return ""

    segments_html = ""
    for i, seg in enumerate(route["segments"]):
        mode_icon = {
            "Flight": "✈️", "Train": "🚂", "Bus": "🚌",
            "Car": "🚗", "Bike": "🏍️", "Trek": "🥾",
        }.get(seg["mode"], "🔀")

        segments_html += f"""
        <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">
          <div style="font-size:20px;min-width:28px;text-align:center;">{mode_icon}</div>
          <div>
            <div style="font-weight:700;color:#F1F5F9;font-size:13px;">
              {seg['from']} → {seg['to']}
            </div>
            <div style="color:#94A3B8;font-size:11px;margin-top:2px;">
              {seg['mode']} | ~{seg['km']:.0f} km | {seg.get('notes','')}
            </div>
          </div>
        </div>
        """
        if i < len(route["segments"]) - 1:
            segments_html += '<div style="width:2px;height:12px;background:rgba(147,51,234,0.4);margin-left:14px;margin-bottom:4px;"></div>'

    total_h = route["total_time_h"]
    total_h_int = int(total_h)
    total_m_int = int((total_h % 1) * 60)

    return f"""
    <div style="background:rgba(79,70,229,0.08);border:1px solid rgba(79,70,229,0.25);
                border-radius:12px;padding:16px;margin:12px 0;">
      <div style="font-family:'Outfit',sans-serif;font-weight:700;color:#A78BFA;
                  font-size:14px;margin-bottom:14px;display:flex;align-items:center;gap:8px;">
        🔀 Alternate Route (No Direct Access)
      </div>
      {segments_html}
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.07);
                  display:flex;gap:20px;flex-wrap:wrap;">
        <div style="font-size:11px;color:#94A3B8;">
          <span style="color:#F1F5F9;font-weight:600;">Total Distance:</span>
          {route['total_km']:.0f} km
        </div>
        <div style="font-size:11px;color:#94A3B8;">
          <span style="color:#F1F5F9;font-weight:600;">Est. Cost:</span>
          ₹{int(route['total_cost_inr']):,}
        </div>
        <div style="font-size:11px;color:#94A3B8;">
          <span style="color:#F1F5F9;font-weight:600;">Est. Time:</span>
          {total_h_int}h {total_m_int:02d}m
        </div>
      </div>
    </div>
    """
