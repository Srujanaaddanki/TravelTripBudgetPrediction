"""
========================================================
Module: Destination Rules Engine (Upgraded)
Purpose: Returns destination-specific packing and pre-travel
         checklists based on destination keywords, month,
         weather, travel_mode, altitude, permits, and country.
         Also handles indirect route transport bars.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── City metadata mappings for dynamic resolution ───────────────────────────

_DESTINATION_PROFILES = [
    {
        "keywords": ["kedarnath", "badrinath", "gangotri", "yamunotri", "char dham"],
        "country": "India",
        "altitude_m": 3500,
        "permits": True,
        "nearest_railway": "Haridwar",
        "nearest_airport": "Dehradun",
        "nearest_road": "Sonprayag",
        "packing": [
            "Thermal wear",
            "Trekking shoes",
            "Rain jacket",
            "Power bank",
            "Altitude medicine",
            "Torch",
            "Woolen socks",
            "Waterproof bag",
        ],
        "pretravel": [
            "Char Dham Registration",
            "Medical Fitness Certificate",
            "Government ID",
            "Emergency Contacts",
            "Offline Maps",
        ]
    },
    {
        "keywords": ["amarnath", "amarnath yatra"],
        "country": "India",
        "altitude_m": 3888,
        "permits": True,
        "nearest_railway": "Jammu",
        "nearest_airport": "Srinagar",
        "nearest_road": "Baltal",
        "packing": [
            "Thermal wear",
            "Trekking shoes",
            "Rain jacket",
            "Power bank",
            "Altitude medicine",
            "Oxygen can",
            "Torch",
            "Woolen socks",
            "Waterproof bag",
        ],
        "pretravel": [
            "Amarnath Yatra Registration",
            "Medical Certificate",
            "Government ID",
            "Emergency Contacts",
            "Yatra Permit",
        ]
    },
    {
        "keywords": ["tirupati", "tirumala", "venkateswara", "balaji"],
        "country": "India",
        "altitude_m": 800,
        "permits": True,
        "nearest_railway": "Tirupati",
        "nearest_airport": "Tirupati",
        "nearest_road": "Tirupati",
        "packing": [
            "Temple dress code",
            "ID proof",
            "Cash for offerings",
        ],
        "pretravel": [
            "Darshan ticket booking",
            "Temple dress code",
            "ID proof",
            "Special entry ticket",
            "Cash for offerings",
        ]
    },
    {
        "keywords": ["ladakh", "leh", "nubra", "pangong", "spiti", "zanskar"],
        "country": "India",
        "altitude_m": 3500,
        "permits": True,
        "nearest_railway": "Jammu",
        "nearest_airport": "Leh",
        "nearest_road": "Manali",
        "packing": [
            "Winter jacket",
            "Oxygen can",
            "Altitude precautions",
            "Diamox",
        ],
        "pretravel": [
            "Inner Line Permit",
            "Altitude precautions",
            "Diamox",
        ]
    },
    {
        "keywords": ["jim corbett", "ranthambore", "bandipur", "kaziranga", "sundarbans",
                     "periyar", "nagarhole", "pench", "kanha", "tadoba", "wildlife", "safari"],
        "country": "India",
        "altitude_m": 200,
        "permits": True,
        "nearest_railway": "Ramnagar",
        "nearest_airport": "Delhi",
        "nearest_road": "Ramnagar",
        "packing": [
            "Dull/earthy clothing (khaki/olive)",
            "Insect repellent",
            "Binoculars",
            "Safari permits",
            "Forest entry passes",
        ],
        "pretravel": [
            "Safari permits",
            "Forest entry passes",
            "Government ID",
        ]
    },
    {
        "keywords": ["goa", "calangute", "baga", "anjuna", "panjim", "margao"],
        "country": "India",
        "altitude_m": 10,
        "permits": False,
        "packing": [
            "Swimwear",
            "Sunscreen SPF 50+",
            "Flip-flops",
            "Hat",
        ],
        "pretravel": [
            "Hotel bookings",
            "Scooter rental license",
        ]
    },
    {
        "keywords": ["manali", "shimla", "kufri", "kasauli", "mussoorie", "nainital", "dehradun"],
        "country": "India",
        "altitude_m": 1800,
        "permits": False,
        "nearest_railway": "Chandigarh",
        "nearest_airport": "Bhuntar",
        "nearest_road": "Manali",
        "packing": [
            "Winter jacket",
            "Thermals",
            "Woolen cap & gloves",
        ],
        "pretravel": [
            "Hotel bookings",
            "Rohtang Pass permit if driving",
        ]
    },
    {
        "keywords": ["paris", "france", "europe", "rome", "barcelona", "amsterdam",
                     "berlin", "zurich", "vienna", "prague", "switzerland"],
        "country": "France",
        "altitude_m": 30,
        "permits": True,
        "packing": [
            "Passport (valid 6 months)",
            "Universal power adapter",
            "Credit card & Euros",
        ],
        "pretravel": [
            "Schengen Visa / Entry Visa",
            "Travel Insurance",
            "Return flight tickets",
            "Hotel reservations",
        ]
    },
    {
        "keywords": ["london", "uk", "england", "united kingdom"],
        "country": "International",
        "altitude_m": 25,
        "permits": True,
        "packing": [
            "Passport (valid 6 months)",
            "Universal power adapter",
            "Credit card & Pounds",
        ],
        "pretravel": [
            "UK Visa",
            "Travel Insurance",
            "Return flight tickets",
        ]
    },
    {
        "keywords": ["dubai", "abu dhabi", "uae", "sharjah"],
        "country": "UAE",
        "altitude_m": 10,
        "permits": False,
        "packing": [
            "Light cotton clothes",
            "Sunscreen SPF 50+",
            "Sunglasses",
        ],
        "pretravel": [
            "UAE Visa / Entry permit",
            "Travel insurance",
        ]
    }
]

# ── Country / Altitude / Permits resolution helpers ───────────────────────────

def get_destination_country(destination: str) -> str:
    """Determine country of destination."""
    dest_lower = destination.strip().lower()
    for profile in _DESTINATION_PROFILES:
        for kw in profile["keywords"]:
            if kw in dest_lower or dest_lower in kw:
                return profile.get("country", "India")
    
    # Simple regex fallback
    int_keywords = ["paris", "london", "europe", "france", "dubai", "uae", "singapore", "bangkok", "thailand", "usa", "tokyo", "japan", "sydney"]
    if any(k in dest_lower for k in int_keywords):
        if "london" in dest_lower or "uk" in dest_lower:
            return "International"
        if "dubai" in dest_lower or "uae" in dest_lower:
            return "UAE"
        if "paris" in dest_lower or "france" in dest_lower:
            return "France"
        return "International"
    return "India"

def get_destination_altitude(destination: str) -> int:
    """Estimate destination altitude in meters."""
    dest_lower = destination.strip().lower()
    for profile in _DESTINATION_PROFILES:
        for kw in profile["keywords"]:
            if kw in dest_lower or dest_lower in kw:
                return profile.get("altitude_m", 0)
    return 0

def get_destination_permits_required(destination: str) -> bool:
    """Determine if special entry permits or registrations are required."""
    dest_lower = destination.strip().lower()
    for profile in _DESTINATION_PROFILES:
        for kw in profile["keywords"]:
            if kw in dest_lower or dest_lower in kw:
                return profile.get("permits", False)
    return False

def get_indirect_route_bars(source: str, destination: str) -> Optional[Dict[str, str]]:
    """Get indirect route transport bar values if direct route is unavailable."""
    dest_lower = destination.strip().lower()
    for profile in _DESTINATION_PROFILES:
        for kw in profile["keywords"]:
            if kw in dest_lower or dest_lower in kw:
                # Only show indirect route bars if there is no direct road/rail/flight all-in-one
                if profile.get("nearest_railway") and profile.get("nearest_railway") != destination:
                    rail = profile["nearest_railway"]
                    road = profile["nearest_road"]
                    air = profile["nearest_airport"]
                    return {
                        "Train": f"✓ {source.title()} → {rail.title()}",
                        "Bus": f"✓ {rail.title()} → {road.title()}",
                        "Flight": f"✓ {source.title()} → {air.title()}"
                    }
    return None

def _match_profile(destination: str) -> Optional[Dict[str, Any]]:
    """Match a destination to a profile (for compatibility)."""
    dest_lower = destination.strip().lower()
    for profile in _DESTINATION_PROFILES:
        for kw in profile["keywords"]:
            if kw in dest_lower or dest_lower in kw:
                return profile
    return None

# ── Mode-specific items ──────────────────────────────────────────────────────

_MODE_PACKING: Dict[str, List[str]] = {
    "Bike": [
        "Helmet (mandatory)",
        "Riding gloves and jacket",
        "Puncture repair kit and basic tools",
        "Fuel planning checklist",
        "Rain cover for luggage",
    ],
    "Train": [
        "Train tickets",
        "Snacks and water for journey",
        "Phone charger & power bank",
        "Valid ID proof",
    ],
    "Flight": [
        "Passport / Valid ID",
        "Boarding pass",
        "Baggage weight compliance",
        "Toiletries in 100ml bag",
    ],
    "Car": [
        "Valid driving license",
        "Car documents",
        "Emergency road kit",
    ],
    "Bus": [
        "Bus ticket",
        "Neck pillow",
        "Phone charger",
    ],
}

# ── Public API ────────────────────────────────────────────────────────────────

def get_destination_checklist(
    destination: str,
    month: str = "January",
    weather: Optional[Dict[str, Any]] = None,
    travel_mode: str = "Car",
    trip_type: str = "General",
    country: Optional[str] = None,
    altitude: Optional[float] = None,
    permits_required: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return fully customized, dynamic checklists based on the 8 required inputs.

    Returns
    -------
    dict
        Keys:
            packing   : List[str]
            pretravel : List[str]
            altitude_m: int
            permits   : bool
            matched   : bool
    """
    # 1. Resolve missing properties
    country = country or get_destination_country(destination)
    altitude = altitude if altitude is not None else get_destination_altitude(destination)
    permits_required = permits_required if permits_required is not None else get_destination_permits_required(destination)

    # 2. Look up base checklist from matched profile
    profile = _match_profile(destination)
    if profile:
        packing = list(profile["packing"])
        pretravel = list(profile["pretravel"])
        matched = True
    else:
        # Generic defaults
        packing = [
            "Comfortable walking shoes",
            "Sunscreen & Sunglasses",
            "Basic first-aid kit",
            "Power bank",
            "Water bottle",
        ]
        pretravel = [
            "Government ID",
            "Travel tickets",
            "Hotel booking confirmation",
            "Emergency Contacts",
        ]
        matched = False

    # 3. Dynamic Altitude Rules
    if altitude >= 2500:
        if "Altitude medicine" not in packing:
            packing.append("Altitude medicine")
        if "Oxygen can" not in packing:
            packing.append("Oxygen can")
        if "Altitude precautions" not in pretravel:
            pretravel.append("Altitude precautions")
        if "Medical Fitness Certificate" not in pretravel:
            pretravel.append("Medical Fitness Certificate")
        # Specific high-altitude items if not present
        if "Thermal wear" not in packing:
            packing.append("Thermal wear")
        if "Trekking shoes" not in packing:
            packing.append("Trekking shoes")
        if "Woolen socks" not in packing:
            packing.append("Woolen socks")

    # 4. Dynamic Permit Rules
    if permits_required:
        if "kedarnath" in destination.lower() and "Char Dham Registration" not in pretravel:
            pretravel.append("Char Dham Registration")
        elif ("ladakh" in destination.lower() or "leh" in destination.lower()) and "Inner Line Permit" not in pretravel:
            pretravel.append("Inner Line Permit")
        elif "safari" in destination.lower() or "wildlife" in destination.lower() or any(k in destination.lower() for k in ["corbett", "ranthambore", "bandipur"]):
            if "Safari permits" not in pretravel:
                pretravel.append("Safari permits")
            if "Forest entry passes" not in pretravel:
                pretravel.append("Forest entry passes")
        else:
            if "Local entry permit/pass" not in pretravel:
                pretravel.append("Local entry permit/pass")

    # 5. Dynamic Country Rules (International vs Domestic)
    if country.lower() != "india":
        if "Passport (valid 6 months)" not in packing:
            packing.insert(0, "Passport (valid 6 months)")
        if "Universal power adapter" not in packing:
            packing.append("Universal power adapter")
        if "Visa / e-Visa" not in pretravel:
            pretravel.insert(0, "Visa / e-Visa")
        if "Travel Insurance" not in pretravel:
            pretravel.append("Travel Insurance")
        if "Foreign currency" not in pretravel:
            pretravel.append("Foreign currency")

    # 6. Dynamic Travel Mode Rules
    mode_packing = _MODE_PACKING.get(travel_mode, [])
    for item in mode_packing:
        if item not in packing:
            packing.append(item)

    # 7. Dynamic Trip Type Rules
    if trip_type.lower() == "adventure" or trip_type.lower() == "trekking":
        for item in ["Trekking shoes", "Torch", "Waterproof bag", "Offline Maps"]:
            if item in ["Trekking shoes", "Torch", "Waterproof bag"] and item not in packing:
                packing.append(item)
            elif item == "Offline Maps" and item not in pretravel:
                pretravel.append(item)

    # 8. Dynamic Trip Type - Religious
    if trip_type.lower() == "religious" or trip_type.lower() == "spiritual":
        for item in ["Temple dress code", "Cash for offerings", "ID proof"]:
            if item in ["Temple dress code", "ID proof"] and item not in packing:
                packing.append(item)
            elif item in ["Temple dress code", "Cash for offerings", "ID proof", "Darshan ticket booking"] and item not in pretravel:
                pretravel.append(item)

    # 9. Dynamic Month/Season Rules
    month_lower = month.lower()
    if month_lower in ("june", "july", "august", "september"):
        if "Rain jacket" not in packing:
            packing.append("Rain jacket")
        if "Waterproof bag" not in packing:
            packing.append("Waterproof bag")
    if month_lower in ("november", "december", "january", "february"):
        if altitude >= 1000:
            if "Thermal wear" not in packing:
                packing.append("Thermal wear")
            if "Winter jacket" not in packing:
                packing.append("Winter jacket")

    # De-duplicate lists while preserving order
    def clean_list(lst: List[str]) -> List[str]:
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    return {
        "packing": clean_list(packing),
        "pretravel": clean_list(pretravel),
        "altitude_m": int(altitude),
        "permits": bool(permits_required),
        "matched": matched,
    }

def get_prompt_hints(destination: str) -> str:
    """Return a short hint string to inject into the Gemini prompt."""
    profile = _match_profile(destination)
    if not profile:
        return ""

    hints = []
    if profile.get("altitude_m", 0) >= 2500:
        hints.append(f"ALTITUDE: {profile['altitude_m']}m — include altitude sickness medicine, oxygen, acclimatization")
    if profile.get("permits"):
        hints.append("PERMITS REQUIRED — include permit booking in pre-travel checklist")
    if any(kw in destination.lower() for kw in ["tirupati", "balaji", "venkateswara"]):
        hints.append("TEMPLE DESTINATION — include dress code, darshan ticket booking, ID proof for entry")
    if any(kw in destination.lower() for kw in ["goa", "beach", "calangute"]):
        hints.append("BEACH DESTINATION — include swimwear, sunscreen, beach gear")
    if any(kw in destination.lower() for kw in ["paris", "london", "europe", "dubai"]):
        hints.append("INTERNATIONAL DESTINATION — include visa, passport validity, travel insurance, foreign currency")

    return " | ".join(hints)
