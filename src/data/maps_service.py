"""
TripAI - Google Maps Integration Service
=========================================

Provides distance, duration, coordinates, and route information
between Indian cities using a three-tier lookup strategy:

1. SQLite Cache  - check database first (fastest, zero cost)
2. Offline Fallback - 100+ hardcoded Indian cities (zero API cost)
3. Google Maps API - live call only on cache + fallback miss

Usage
-----
>>> from src.data.maps_service import MapsService
>>> maps = MapsService()
>>> info = maps.get_route_info("Delhi", "Manali", "Car")
>>> print(info["distance_km"], info["duration_hours"])
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("tripai.maps_service")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAVEL_MODES: Dict[str, Dict[str, Any]] = {
    "Flight": {"google_mode": None, "speed_kmh": 800, "distance_factor": 1.0, "cost_per_km": 5.0},
    "Train":  {"google_mode": "transit", "speed_kmh": 55, "distance_factor": 0.95, "cost_per_km": 1.5},
    "Bus":    {"google_mode": "driving", "speed_kmh": 40, "distance_factor": 1.0, "cost_per_km": 1.8},
    "Car":    {"google_mode": "driving", "speed_kmh": 60, "distance_factor": 1.0, "cost_per_km": 6.0},
    "Bike":   {"google_mode": "driving", "speed_kmh": 40, "distance_factor": 1.05, "cost_per_km": 2.5},
    "Other":  {"google_mode": "driving", "speed_kmh": 50, "distance_factor": 1.0, "cost_per_km": 4.0},
}

# ---------------------------------------------------------------------------
# City Coordinates — 130+ Indian cities  (latitude, longitude)
# ---------------------------------------------------------------------------

CITY_COORDINATES: Dict[str, tuple] = {
    # ── Major Metros & Tier-1 Cities ──────────────────────────────────────
    "delhi":              (28.6139, 77.2090),
    "new delhi":          (28.6139, 77.2090),
    "mumbai":             (19.0760, 72.8777),
    "bangalore":          (12.9716, 77.5946),
    "hyderabad":          (17.3850, 78.4867),
    "chennai":            (13.0827, 80.2707),
    "kolkata":            (22.5726, 88.3639),
    "pune":               (18.5204, 73.8567),
    "ahmedabad":          (23.0225, 72.5714),
    "jaipur":             (26.9124, 75.7873),
    "lucknow":            (26.8467, 80.9462),
    "kanpur":             (26.4499, 80.3319),
    "nagpur":             (21.1458, 79.0882),
    "indore":             (22.7196, 75.8577),
    "bhopal":             (23.2599, 77.4126),
    "visakhapatnam":      (17.6868, 83.2185),
    "patna":              (25.6093, 85.1376),
    "vadodara":           (22.3072, 73.1812),
    "ghaziabad":          (28.6692, 77.4538),
    "ludhiana":           (30.9010, 75.8573),
    "agra":               (27.1767, 78.0081),
    "nashik":             (19.9975, 73.7898),
    "faridabad":          (28.4089, 77.3178),
    "meerut":             (28.9845, 77.7064),
    "rajkot":             (22.3039, 70.8022),
    "varanasi":           (25.3176, 82.9739),
    "srinagar":           (34.0837, 74.7973),
    "aurangabad":         (19.8762, 75.3433),
    "amritsar":           (31.6340, 74.8723),
    "ranchi":             (23.3441, 85.3096),
    "coimbatore":         (11.0168, 76.9558),
    "jabalpur":           (23.1815, 79.9864),
    "gwalior":            (26.2183, 78.1828),
    "vijayawada":         (16.5062, 80.6480),
    "jodhpur":            (26.2389, 73.0243),
    "madurai":            (9.9252, 78.1198),
    "raipur":             (21.2514, 81.6296),
    "kota":               (25.2138, 75.8648),
    "guwahati":           (26.1445, 91.7362),
    "chandigarh":         (30.7333, 76.7794),
    "solapur":            (17.6599, 75.9064),
    "mysore":             (12.2958, 76.6394),
    "gurgaon":            (28.4595, 77.0266),
    "tiruchirappalli":    (10.7905, 78.7047),
    "bhubaneswar":        (20.2961, 85.8245),
    "salem":              (11.6643, 78.1460),
    "warangal":           (17.9784, 79.5941),
    "cochin":             (9.9312, 76.2673),
    "dehradun":           (30.3165, 78.0322),
    "jamshedpur":         (22.8046, 86.2029),
    "cuttack":            (20.4625, 85.8830),
    "kolhapur":           (16.7050, 74.2433),
    "ajmer":              (26.4499, 74.6399),
    "udaipur":            (24.5854, 73.7125),
    "mangalore":          (12.9141, 74.8560),
    "belgaum":            (15.8497, 74.4977),
    "jammu":              (32.7266, 74.8570),
    "siliguri":           (26.7271, 88.3953),
    "thiruvananthapuram": (8.5241, 76.9366),
    "guntur":             (16.3067, 80.4365),
    "bikaner":            (28.0229, 73.3119),
    "jhansi":             (25.4484, 78.5685),
    "gorakhpur":          (26.7606, 83.3732),
    "noida":              (28.5355, 77.3910),
    "nanded":             (19.1383, 77.3210),
    "hubli":              (15.3647, 75.1240),
    "gulbarga":           (17.3297, 76.8343),
    "jamnagar":           (22.4707, 70.0577),
    "ujjain":             (23.1765, 75.7885),
    "bathinda":           (30.2110, 74.9455),
    "rohtak":             (28.8955, 76.5796),
    "panipat":            (29.3909, 76.9635),
    "karnal":             (29.6857, 76.9905),
    "mathura":            (27.4924, 77.6737),
    "bareilly":           (28.3670, 79.4304),
    "moradabad":          (28.8386, 78.7733),
    "aligarh":            (27.8974, 78.0880),
    "saharanpur":         (29.9680, 77.5469),
    "allahabad":          (25.4358, 81.8463),
    "prayagraj":          (25.4358, 81.8463),
    "surat":              (21.1702, 72.8311),
    "thane":              (19.2183, 72.9781),
    "navi mumbai":        (19.0330, 73.0297),
    "howrah":             (22.5958, 88.2636),
    "tiruvallur":         (13.1431, 79.9082),
    "erode":              (11.3410, 77.7172),
    "tirunelveli":        (8.7139, 77.7567),
    "thanjavur":          (10.7870, 79.1378),
    "vellore":            (12.9165, 79.1325),
    "bhilai":             (21.2094, 81.3792),
    "durgapur":           (23.5204, 87.3119),
    "asansol":            (23.6889, 86.9661),
    "bokaro":             (23.6693, 86.1511),
    "dhanbad":            (23.7957, 86.4304),
    "bilaspur":           (22.0797, 82.1391),
    "korba":              (22.3595, 82.7501),

    # ── Tourist Destinations ──────────────────────────────────────────────
    "manali":             (32.2396, 77.1887),
    "shimla":             (31.1048, 77.1734),
    "mussoorie":          (30.4598, 78.0644),
    "nainital":           (29.3803, 79.4636),
    "rishikesh":          (30.0869, 78.2676),
    "haridwar":           (29.9457, 78.1642),
    "goa":                (15.4909, 73.8278),
    "ooty":               (11.4102, 76.6950),
    "kodaikanal":         (10.2381, 77.4892),
    "munnar":             (10.0889, 77.0595),
    "darjeeling":         (27.0360, 88.2627),
    "gangtok":            (27.3389, 88.6065),
    "leh":                (34.1526, 77.5771),
    "pondicherry":        (11.9416, 79.8083),
    "mahabaleshwar":      (17.9237, 73.6586),
    "lonavala":           (18.7546, 73.4062),
    "alleppey":           (9.4981, 76.3388),
    "thekkady":           (9.6005, 77.1614),
    "mount abu":          (24.5926, 72.7156),
    "tirupati":           (13.6288, 79.4192),
    "shirdi":             (19.7668, 74.4760),
    "pushkar":            (26.4900, 74.5513),
    "khajuraho":          (24.8318, 79.9199),
    "bodh gaya":          (24.6961, 84.9869),
    "hampi":              (15.3350, 76.4600),
    "kovalam":            (8.3988, 76.9782),
    "jim corbett":        (29.5300, 78.7747),
    "kasol":              (32.0100, 77.3114),
    "dalhousie":          (32.5373, 75.9710),
    "mcleod ganj":        (32.2426, 76.3213),
    "auli":               (30.5268, 79.5670),
    "coorg":              (12.4244, 75.7382),
    "wayanad":            (11.6854, 76.1320),
    "hogenakkal":         (12.1156, 77.7770),
    "puri":               (19.8135, 85.8312),
    "konark":             (19.8876, 86.0945),
    "mahabalipuram":      (12.6172, 80.1927),
    "shillong":           (25.5788, 91.8933),
    "kaziranga":          (26.5775, 93.1711),
    "digha":              (21.6275, 87.5493),
    "ayodhya":            (26.7922, 82.1998),
    "ramnagar":           (29.3954, 79.1273),
    "dharamshala":        (32.2190, 76.3234),
}

# ---------------------------------------------------------------------------
# City Aliases — bidirectional mappings for common alternate names
# ---------------------------------------------------------------------------

CITY_ALIASES: Dict[str, str] = {
    # Modern ↔ Traditional / Alternate names
    "bengaluru":          "bangalore",
    "mysuru":             "mysore",
    "mangaluru":          "mangalore",
    "belagavi":           "belgaum",
    "kochi":              "cochin",
    "trivandrum":         "thiruvananthapuram",
    "trichy":             "tiruchirappalli",
    "vizag":              "visakhapatnam",
    "gurugram":           "gurgaon",
    "puducherry":         "pondicherry",
    "panaji":             "goa",
    "alappuzha":          "alleppey",
    "madikeri":           "coorg",
    "dharamshala":        "mcleod ganj",
    "ramnagar":           "jim corbett",

    # Historical names
    "banaras":            "varanasi",
    "bombay":             "mumbai",
    "calcutta":           "kolkata",
    "madras":             "chennai",
    "poona":              "pune",

    # Convenience / UI
    "detect location...": "delhi",
}

# ---------------------------------------------------------------------------
# Known Routes — 85+ routes with real road distances (km) and durations (h)
# ---------------------------------------------------------------------------

KNOWN_ROUTES: Dict[tuple, Dict[str, float]] = {
    # ── Delhi Routes ──────────────────────────────────────────────────────
    ("delhi", "manali"):        {"distance_km": 537.0, "duration_hours_driving": 12.5},
    ("delhi", "shimla"):        {"distance_km": 342.0, "duration_hours_driving": 7.5},
    ("delhi", "jaipur"):        {"distance_km": 281.0, "duration_hours_driving": 5.0},
    ("delhi", "agra"):          {"distance_km": 233.0, "duration_hours_driving": 4.0},
    ("delhi", "mussoorie"):     {"distance_km": 290.0, "duration_hours_driving": 6.5},
    ("delhi", "rishikesh"):     {"distance_km": 240.0, "duration_hours_driving": 5.5},
    ("delhi", "haridwar"):      {"distance_km": 225.0, "duration_hours_driving": 5.0},
    ("delhi", "nainital"):      {"distance_km": 303.0, "duration_hours_driving": 7.0},
    ("delhi", "chandigarh"):    {"distance_km": 243.0, "duration_hours_driving": 4.5},
    ("delhi", "dehradun"):      {"distance_km": 255.0, "duration_hours_driving": 5.5},
    ("delhi", "amritsar"):      {"distance_km": 449.0, "duration_hours_driving": 7.5},
    ("delhi", "lucknow"):       {"distance_km": 556.0, "duration_hours_driving": 8.5},
    ("delhi", "varanasi"):      {"distance_km": 821.0, "duration_hours_driving": 12.0},
    ("delhi", "mumbai"):        {"distance_km": 1400.0, "duration_hours_driving": 22.0},
    ("delhi", "jim corbett"):   {"distance_km": 260.0, "duration_hours_driving": 6.0},

    # ── Mumbai Routes ─────────────────────────────────────────────────────
    ("mumbai", "goa"):          {"distance_km": 588.0, "duration_hours_driving": 10.0},
    ("mumbai", "pune"):         {"distance_km": 150.0, "duration_hours_driving": 2.5},
    ("mumbai", "lonavala"):     {"distance_km": 83.0, "duration_hours_driving": 1.5},
    ("mumbai", "mahabaleshwar"):{"distance_km": 263.0, "duration_hours_driving": 5.0},
    ("mumbai", "nashik"):       {"distance_km": 167.0, "duration_hours_driving": 3.5},
    ("mumbai", "shirdi"):       {"distance_km": 241.0, "duration_hours_driving": 5.0},
    ("mumbai", "ahmedabad"):    {"distance_km": 524.0, "duration_hours_driving": 8.5},
    ("mumbai", "bangalore"):    {"distance_km": 984.0, "duration_hours_driving": 16.0},
    ("mumbai", "hyderabad"):    {"distance_km": 711.0, "duration_hours_driving": 12.0},
    ("mumbai", "kolhapur"):     {"distance_km": 378.0, "duration_hours_driving": 6.5},
    ("mumbai", "aurangabad"):   {"distance_km": 331.0, "duration_hours_driving": 6.0},
    ("mumbai", "manali"):       {"distance_km": 1954.0, "duration_hours_driving": 36.0},

    # ── Bangalore Routes ──────────────────────────────────────────────────
    ("bangalore", "mysore"):        {"distance_km": 143.0, "duration_hours_driving": 3.0},
    ("bangalore", "ooty"):          {"distance_km": 270.0, "duration_hours_driving": 6.0},
    ("bangalore", "coorg"):         {"distance_km": 250.0, "duration_hours_driving": 5.5},
    ("bangalore", "goa"):           {"distance_km": 560.0, "duration_hours_driving": 10.0},
    ("bangalore", "chennai"):       {"distance_km": 346.0, "duration_hours_driving": 6.0},
    ("bangalore", "hampi"):         {"distance_km": 340.0, "duration_hours_driving": 6.5},
    ("bangalore", "pondicherry"):   {"distance_km": 310.0, "duration_hours_driving": 6.0},
    ("bangalore", "kodaikanal"):    {"distance_km": 465.0, "duration_hours_driving": 8.5},
    ("bangalore", "hyderabad"):     {"distance_km": 570.0, "duration_hours_driving": 9.0},
    ("bangalore", "mangalore"):     {"distance_km": 352.0, "duration_hours_driving": 6.5},

    # ── Chennai Routes ────────────────────────────────────────────────────
    ("chennai", "pondicherry"):     {"distance_km": 155.0, "duration_hours_driving": 3.0},
    ("chennai", "mahabalipuram"):   {"distance_km": 58.0, "duration_hours_driving": 1.5},
    ("chennai", "ooty"):            {"distance_km": 550.0, "duration_hours_driving": 10.0},
    ("chennai", "madurai"):         {"distance_km": 462.0, "duration_hours_driving": 8.0},
    ("chennai", "kodaikanal"):      {"distance_km": 527.0, "duration_hours_driving": 9.5},
    ("chennai", "tirupati"):        {"distance_km": 153.0, "duration_hours_driving": 3.0},
    ("chennai", "cochin"):          {"distance_km": 700.0, "duration_hours_driving": 12.0},
    ("chennai", "coimbatore"):      {"distance_km": 505.0, "duration_hours_driving": 8.0},

    # ── Kolkata Routes ────────────────────────────────────────────────────
    ("kolkata", "darjeeling"):      {"distance_km": 615.0, "duration_hours_driving": 11.0},
    ("kolkata", "gangtok"):         {"distance_km": 570.0, "duration_hours_driving": 11.0},
    ("kolkata", "puri"):            {"distance_km": 499.0, "duration_hours_driving": 8.0},
    ("kolkata", "digha"):           {"distance_km": 185.0, "duration_hours_driving": 4.0},
    ("kolkata", "siliguri"):        {"distance_km": 560.0, "duration_hours_driving": 10.0},
    ("kolkata", "bhubaneswar"):     {"distance_km": 440.0, "duration_hours_driving": 7.0},
    ("kolkata", "ranchi"):          {"distance_km": 400.0, "duration_hours_driving": 7.0},
    ("kolkata", "patna"):           {"distance_km": 590.0, "duration_hours_driving": 9.0},

    # ── Hyderabad Routes ──────────────────────────────────────────────────
    ("hyderabad", "warangal"):      {"distance_km": 150.0, "duration_hours_driving": 3.0},
    ("hyderabad", "vijayawada"):    {"distance_km": 272.0, "duration_hours_driving": 4.5},
    ("hyderabad", "tirupati"):      {"distance_km": 553.0, "duration_hours_driving": 9.0},
    ("hyderabad", "bangalore"):     {"distance_km": 570.0, "duration_hours_driving": 9.0},
    ("hyderabad", "nagpur"):        {"distance_km": 500.0, "duration_hours_driving": 8.0},
    ("hyderabad", "chennai"):       {"distance_km": 627.0, "duration_hours_driving": 10.0},

    # ── Jaipur Routes ─────────────────────────────────────────────────────
    ("jaipur", "udaipur"):          {"distance_km": 393.0, "duration_hours_driving": 6.5},
    ("jaipur", "jodhpur"):          {"distance_km": 332.0, "duration_hours_driving": 5.5},
    ("jaipur", "pushkar"):          {"distance_km": 146.0, "duration_hours_driving": 2.5},
    ("jaipur", "ajmer"):            {"distance_km": 131.0, "duration_hours_driving": 2.5},
    ("jaipur", "mount abu"):        {"distance_km": 490.0, "duration_hours_driving": 7.5},
    ("jaipur", "bikaner"):          {"distance_km": 334.0, "duration_hours_driving": 5.5},

    # ── Kerala Routes ─────────────────────────────────────────────────────
    ("cochin", "munnar"):               {"distance_km": 130.0, "duration_hours_driving": 3.5},
    ("cochin", "alleppey"):             {"distance_km": 53.0, "duration_hours_driving": 1.5},
    ("cochin", "thekkady"):             {"distance_km": 114.0, "duration_hours_driving": 3.0},
    ("cochin", "kovalam"):              {"distance_km": 210.0, "duration_hours_driving": 4.5},
    ("thiruvananthapuram", "kovalam"):   {"distance_km": 16.0, "duration_hours_driving": 0.5},
    ("cochin", "wayanad"):              {"distance_km": 280.0, "duration_hours_driving": 6.0},

    # ── Northeast Routes ──────────────────────────────────────────────────
    ("guwahati", "shillong"):       {"distance_km": 99.0, "duration_hours_driving": 2.5},
    ("guwahati", "kaziranga"):      {"distance_km": 193.0, "duration_hours_driving": 4.0},
    ("siliguri", "darjeeling"):     {"distance_km": 77.0, "duration_hours_driving": 2.5},
    ("siliguri", "gangtok"):        {"distance_km": 114.0, "duration_hours_driving": 4.0},

    # ── Lucknow Routes ────────────────────────────────────────────────────
    ("lucknow", "varanasi"):        {"distance_km": 300.0, "duration_hours_driving": 5.0},
    ("lucknow", "agra"):            {"distance_km": 333.0, "duration_hours_driving": 5.5},
    ("lucknow", "allahabad"):       {"distance_km": 200.0, "duration_hours_driving": 3.5},
    ("lucknow", "gorakhpur"):       {"distance_km": 273.0, "duration_hours_driving": 5.0},
    ("lucknow", "ayodhya"):         {"distance_km": 135.0, "duration_hours_driving": 3.0},

    # ── Additional Cross-Country & Popular Routes ─────────────────────────
    ("delhi", "kolkata"):           {"distance_km": 1530.0, "duration_hours_driving": 24.0},
    ("delhi", "bangalore"):         {"distance_km": 2150.0, "duration_hours_driving": 34.0},
    ("delhi", "chennai"):           {"distance_km": 2180.0, "duration_hours_driving": 35.0},
    ("delhi", "hyderabad"):         {"distance_km": 1550.0, "duration_hours_driving": 24.0},
    ("delhi", "goa"):               {"distance_km": 1850.0, "duration_hours_driving": 30.0},
    ("delhi", "leh"):               {"distance_km": 985.0, "duration_hours_driving": 22.0},
    ("delhi", "kasol"):             {"distance_km": 520.0, "duration_hours_driving": 12.0},
    ("mumbai", "chennai"):          {"distance_km": 1280.0, "duration_hours_driving": 20.0},
    ("mumbai", "kolkata"):          {"distance_km": 1960.0, "duration_hours_driving": 30.0},
    ("mumbai", "delhi"):            {"distance_km": 1400.0, "duration_hours_driving": 22.0},
    ("bangalore", "mumbai"):        {"distance_km": 984.0, "duration_hours_driving": 16.0},
    ("chennai", "bangalore"):       {"distance_km": 346.0, "duration_hours_driving": 6.0},
    ("pune", "goa"):                {"distance_km": 448.0, "duration_hours_driving": 8.0},
}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two points on Earth.

    Uses the Haversine formula with Earth radius 6 371 km.
    """
    R = 6_371.0  # Earth radius in kilometres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize(place: str) -> str:
    """Normalise a place name: lowercase, strip whitespace, resolve aliases."""
    name = place.lower().strip()
    return CITY_ALIASES.get(name, name)


def _estimate_road_distance(straight_km: float) -> float:
    """Multiply straight-line distance by 1.35 road-winding factor."""
    return straight_km * 1.35


# ---------------------------------------------------------------------------
# MapsService Class
# ---------------------------------------------------------------------------


class MapsService:
    """Google Maps integration with SQLite caching and offline fallback.

    Provides a three-tier lookup for distances, durations and coordinates:

    1. **SQLite cache** — instant, zero API cost
    2. **Offline fallback** — 130+ hardcoded Indian cities & 85+ routes
    3. **Google Maps API** — live network call (only when both above miss)

    Parameters
    ----------
    api_key : str, optional
        Google Maps Platform API key.  Falls back to the
        ``GOOGLE_MAPS_API_KEY`` environment variable if not supplied.
    db : TripDatabase, optional
        Pre-initialised database instance.  A default one is created
        automatically if omitted.
    """

    def __init__(self, api_key: Optional[str] = None, db: Optional["TripDatabase"] = None) -> None:
        self._api_key: str = api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "")

        # Import here to avoid circular imports at module level
        from src.data.database import TripDatabase

        if db is not None:
            self._db = db
        else:
            # Use the same default DB path as the rest of the app
            _BASE_DIR = Path(__file__).resolve().parent.parent.parent
            _DB_PATH = _BASE_DIR / "data" / "travel.db"
            self._db = TripDatabase(_DB_PATH)

        self._api_calls: int = 0
        self._cache_hits: int = 0
        self._offline_hits: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coordinates(self, place: str) -> Dict[str, Any]:
        """Get latitude / longitude for *place*.

        Returns
        -------
        dict
            ``{"lat": float, "lng": float, "source": str, "place": str}``

            *source* is one of ``"offline"``, ``"cache"``,
            ``"google_api"``, ``"unknown"``.  If the place cannot be
            resolved anywhere, Delhi coordinates are returned as the
            default.
        """
        normalized = _normalize(place)

        # 1. Offline lookup
        if normalized in CITY_COORDINATES:
            lat, lng = CITY_COORDINATES[normalized]
            self._offline_hits += 1
            return {"lat": lat, "lng": lng, "source": "offline", "place": normalized}

        # 2. Database geocode cache
        try:
            cached = self._db.get_geocode(normalized)
            if cached:
                self._cache_hits += 1
                return {
                    "lat": cached["latitude"],
                    "lng": cached["longitude"],
                    "source": "cache",
                    "place": normalized,
                }
        except Exception:
            pass

        # 3. Google Geocoding API
        if self._api_key:
            result = self._call_google_geocode_api(normalized)
            if result:
                return result

        # 4. Default to Delhi
        lat, lng = CITY_COORDINATES.get("delhi", (28.6139, 77.2090))
        return {"lat": lat, "lng": lng, "source": "unknown", "place": normalized}

    def get_route_info(
        self,
        origin: str,
        destination: str,
        travel_mode: str = "Car",
    ) -> Dict[str, Any]:
        """Get comprehensive route information between two places.

        Uses a three-tier lookup strategy:
        DB cache → offline fallback → Google API.

        Parameters
        ----------
        origin : str
            Starting city / place name.
        destination : str
            Ending city / place name.
        travel_mode : str
            One of the keys in :data:`TRAVEL_MODES`
            (``"Flight"``, ``"Train"``, ``"Bus"``, ``"Car"``, ``"Bike"``,
            ``"Other"``).

        Returns
        -------
        dict
            Keys: ``origin``, ``destination``, ``travel_mode``,
            ``distance_km``, ``duration_hours``, ``duration_text``,
            ``origin_coords``, ``dest_coords``, ``route_summary``,
            ``source`` (``"cache"`` / ``"offline"`` / ``"google_api"``
            / ``"estimated"``).
        """
        norm_origin = _normalize(origin)
        norm_dest = _normalize(destination)
        mode_config = TRAVEL_MODES.get(travel_mode, TRAVEL_MODES["Other"])
        google_mode = mode_config["google_mode"] or "driving"

        # 1. Check DB distance cache
        try:
            cached = self._db.get_distance(norm_origin, norm_dest, google_mode)
            if cached:
                self._cache_hits += 1
                origin_coords = self.get_coordinates(norm_origin)
                dest_coords = self.get_coordinates(norm_dest)
                return self._build_route_result(
                    norm_origin,
                    norm_dest,
                    travel_mode,
                    cached["distance_km"],
                    cached["duration_hours"],
                    origin_coords,
                    dest_coords,
                    "cache",
                )
        except Exception:
            pass

        # 2. Offline fallback
        offline = self._lookup_offline_distance(norm_origin, norm_dest)
        if offline:
            self._offline_hits += 1
            distance_km = offline["distance_km"] * mode_config["distance_factor"]

            if travel_mode == "Flight":
                # For flights use straight-line distance
                origin_c = self.get_coordinates(norm_origin)
                dest_c = self.get_coordinates(norm_dest)
                flight_dist = _haversine(
                    origin_c["lat"], origin_c["lng"],
                    dest_c["lat"], dest_c["lng"],
                )
                distance_km = flight_dist * 1.1  # 10 % overhead for flight path
                duration_hours = flight_dist / mode_config["speed_kmh"] + 1.0  # +1 h boarding / taxiing
            else:
                duration_hours = distance_km / mode_config["speed_kmh"]

            # Cache the result
            try:
                self._db.cache_distance(
                    norm_origin,
                    norm_dest,
                    distance_km,
                    duration_hours=duration_hours,
                    travel_mode=google_mode,
                )
            except Exception:
                pass

            origin_coords = self.get_coordinates(norm_origin)
            dest_coords = self.get_coordinates(norm_dest)
            return self._build_route_result(
                norm_origin,
                norm_dest,
                travel_mode,
                distance_km,
                duration_hours,
                origin_coords,
                dest_coords,
                "offline",
            )

        # 3. Google Distance Matrix API
        if self._api_key and travel_mode != "Flight":
            api_result = self._call_google_distance_api(norm_origin, norm_dest, google_mode)
            if api_result:
                try:
                    self._db.cache_distance(
                        norm_origin,
                        norm_dest,
                        api_result["distance_km"],
                        duration_hours=api_result["duration_hours"],
                        travel_mode=google_mode,
                    )
                except Exception:
                    pass
                origin_coords = self.get_coordinates(norm_origin)
                dest_coords = self.get_coordinates(norm_dest)
                return self._build_route_result(
                    norm_origin,
                    norm_dest,
                    travel_mode,
                    api_result["distance_km"],
                    api_result["duration_hours"],
                    origin_coords,
                    dest_coords,
                    "google_api",
                )

        # 4. Haversine estimate as last resort
        origin_coords = self.get_coordinates(norm_origin)
        dest_coords = self.get_coordinates(norm_dest)
        straight = _haversine(
            origin_coords["lat"], origin_coords["lng"],
            dest_coords["lat"], dest_coords["lng"],
        )

        if travel_mode == "Flight":
            distance_km = straight * 1.1
            duration_hours = straight / mode_config["speed_kmh"] + 1.0
        else:
            distance_km = _estimate_road_distance(straight) * mode_config["distance_factor"]
            duration_hours = distance_km / mode_config["speed_kmh"]

        return self._build_route_result(
            norm_origin,
            norm_dest,
            travel_mode,
            round(distance_km, 1),
            round(duration_hours, 2),
            origin_coords,
            dest_coords,
            "estimated",
        )

    def estimate_travel_cost(
        self,
        distance_km: float,
        travel_mode: str,
        duration_days: int = 1,
    ) -> Dict[str, Any]:
        """Estimate travel-only cost based on distance and mode.

        Parameters
        ----------
        distance_km : float
            One-way distance in kilometres.
        travel_mode : str
            Key from :data:`TRAVEL_MODES`.
        duration_days : int
            Trip duration in days (used to compute *per_day_travel*).

        Returns
        -------
        dict
            ``one_way_cost``, ``round_trip_cost``, ``per_day_travel``,
            ``mode``, ``cost_per_km``, ``distance_km``.
        """
        mode_config = TRAVEL_MODES.get(travel_mode, TRAVEL_MODES["Other"])
        cost_per_km = mode_config["cost_per_km"]

        one_way = round(distance_km * cost_per_km, 2)
        round_trip = round(one_way * 2, 2)
        per_day = round(round_trip / max(duration_days, 1), 2)

        return {
            "one_way_cost": one_way,
            "round_trip_cost": round_trip,
            "per_day_travel": per_day,
            "mode": travel_mode,
            "cost_per_km": cost_per_km,
            "distance_km": distance_km,
        }

    def get_smart_budget(
        self,
        ml_prediction: float,
        distance_km: float,
        travel_mode: str,
        duration_days: int,
    ) -> Dict[str, Any]:
        """Combine an ML prediction with distance-based cost analysis.

        The ML model remains the **primary** predictor.  Distance
        analysis provides supplementary travel-cost estimation that is
        used to validate and, if necessary, adjust the ML figure.

        Parameters
        ----------
        ml_prediction : float
            Budget predicted by the ML model (INR).
        distance_km : float
            One-way distance in kilometres.
        travel_mode : str
            Key from :data:`TRAVEL_MODES`.
        duration_days : int
            Trip duration in days.

        Returns
        -------
        dict
            ``ml_prediction``, ``travel_cost_estimate``,
            ``smart_estimate``, ``confidence``
            (``"high"`` / ``"medium"`` / ``"low"``),
            ``savings_tip``, ``breakdown``.
        """
        travel_costs = self.estimate_travel_cost(distance_km, travel_mode, duration_days)
        travel_cost = travel_costs["round_trip_cost"]

        # ML prediction is primary (weight ~80 %); travel cost is supplementary.
        # We do *not* simply average — instead we check whether the ML prediction
        # already accounts for travel expenses.
        #
        # If ML prediction > travel cost ⇒ ML likely includes accommodation,
        # food, activities, etc.  Smart estimate ≈ ML prediction.

        if ml_prediction < travel_cost * 0.5:
            # ML prediction seems too low — travel alone costs more
            smart = round(ml_prediction * 0.3 + (travel_cost + ml_prediction) * 0.7 / 1.5, 0)
            confidence = "low"
            savings_tip = (
                f"Travel cost alone is ~₹{travel_cost:,.0f}. "
                "Consider a more budget-friendly mode."
            )
        elif ml_prediction < travel_cost:
            # ML prediction is close to just the travel cost
            smart = round(ml_prediction * 0.8 + travel_cost * 0.2, 0)
            confidence = "medium"
            savings_tip = (
                f"Budget is tight. {travel_mode} travel costs "
                f"~₹{travel_cost:,.0f} round trip."
            )
        else:
            # ML prediction comfortably covers travel — high confidence
            smart = round(ml_prediction, 0)
            confidence = "high"
            travel_pct = (travel_cost / ml_prediction) * 100
            savings_tip = f"Travel ({travel_mode}) is ~{travel_pct:.0f}% of your budget."

        return {
            "ml_prediction": round(ml_prediction, 0),
            "travel_cost_estimate": round(travel_cost, 0),
            "smart_estimate": round(smart, 0),
            "confidence": confidence,
            "savings_tip": savings_tip,
            "breakdown": travel_costs,
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """Return a snapshot of cache / API usage counters.

        Returns
        -------
        dict
            ``api_calls``, ``cache_hits``, ``offline_hits``.
        """
        return {
            "api_calls": self._api_calls,
            "cache_hits": self._cache_hits,
            "offline_hits": self._offline_hits,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_route_result(
        self,
        origin: str,
        dest: str,
        travel_mode: str,
        distance_km: float,
        duration_hours: float,
        origin_coords: Dict[str, Any],
        dest_coords: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        """Build a standardised route-result dictionary."""
        # Format duration as human-readable text
        hrs = int(duration_hours)
        mins = int((duration_hours - hrs) * 60)
        if hrs > 0 and mins > 0:
            duration_text = f"{hrs}h {mins}m"
        elif hrs > 0:
            duration_text = f"{hrs}h"
        else:
            duration_text = f"{mins}m"

        route_summary = (
            f"{origin.title()} to {dest.title()} via {travel_mode} — "
            f"{round(distance_km, 1)} km, ~{duration_text}"
        )

        return {
            "origin": origin.title(),
            "destination": dest.title(),
            "travel_mode": travel_mode,
            "distance_km": round(distance_km, 1),
            "duration_hours": round(duration_hours, 2),
            "duration_text": duration_text,
            "origin_coords": {"lat": origin_coords["lat"], "lng": origin_coords["lng"]},
            "dest_coords": {"lat": dest_coords["lat"], "lng": dest_coords["lng"]},
            "route_summary": route_summary,
            "source": source,
        }

    def _lookup_offline_distance(
        self, origin: str, dest: str
    ) -> Optional[Dict[str, float]]:
        """Check :data:`KNOWN_ROUTES` for a hardcoded distance.

        The lookup is bidirectional — ``(origin, dest)`` and
        ``(dest, origin)`` are both tried.

        Returns
        -------
        dict or None
            ``{"distance_km": float, "duration_hours_driving": float}``
            if a match is found, else ``None``.
        """
        key = (origin, dest)
        if key in KNOWN_ROUTES:
            return KNOWN_ROUTES[key]
        # Try reverse direction
        rkey = (dest, origin)
        if rkey in KNOWN_ROUTES:
            return KNOWN_ROUTES[rkey]
        return None

    def _call_google_distance_api(
        self, origin: str, dest: str, mode: str = "driving"
    ) -> Optional[Dict[str, float]]:
        """Call the Google Distance Matrix API.

        Returns
        -------
        dict or None
            ``{"distance_km": float, "duration_hours": float}`` on
            success, ``None`` on failure.
        """
        import json as json_mod
        import urllib.parse
        import urllib.request

        try:
            params = urllib.parse.urlencode({
                "origins": origin,
                "destinations": dest,
                "mode": mode,
                "key": self._api_key,
                "units": "metric",
            })
            url = f"https://maps.googleapis.com/maps/api/distancematrix/json?{params}"

            req = urllib.request.Request(url, headers={"User-Agent": "TripAI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json_mod.loads(resp.read().decode("utf-8"))

            self._api_calls += 1

            if data.get("status") != "OK":
                log.warning("Distance Matrix API error: %s", data.get("status"))
                return None

            element = data["rows"][0]["elements"][0]
            if element.get("status") != "OK":
                log.warning("Route not found: %s -> %s", origin, dest)
                return None

            distance_m = element["distance"]["value"]  # metres
            duration_s = element["duration"]["value"]  # seconds

            return {
                "distance_km": round(distance_m / 1000, 1),
                "duration_hours": round(duration_s / 3600, 2),
            }
        except Exception as exc:
            log.error("Google Distance Matrix API call failed: %s", exc)
            return None

    def _call_google_geocode_api(
        self, place: str
    ) -> Optional[Dict[str, Any]]:
        """Call the Google Geocoding API.

        Returns
        -------
        dict or None
            ``{"lat": float, "lng": float, "source": "google_api",
            "place": str}`` on success, ``None`` on failure.
        """
        import json as json_mod
        import urllib.parse
        import urllib.request

        try:
            params = urllib.parse.urlencode({
                "address": f"{place}, India",
                "key": self._api_key,
            })
            url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"

            req = urllib.request.Request(url, headers={"User-Agent": "TripAI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json_mod.loads(resp.read().decode("utf-8"))

            self._api_calls += 1

            if data.get("status") != "OK" or not data.get("results"):
                return None

            location = data["results"][0]["geometry"]["location"]
            formatted = data["results"][0].get("formatted_address", "")

            lat = location["lat"]
            lng = location["lng"]

            # Cache the result
            try:
                self._db.cache_geocode(place, lat, lng, formatted_address=formatted)
            except Exception:
                pass

            return {"lat": lat, "lng": lng, "source": "google_api", "place": place}
        except Exception as exc:
            log.error("Google Geocoding API call failed: %s", exc)
            return None
