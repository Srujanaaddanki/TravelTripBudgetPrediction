"""
========================================================
Module: Destination Rules Engine
Purpose: Returns destination-specific packing and pre-travel
         checklists based on destination keywords, month,
         travel_mode, altitude, permits, and trip context.
         Called as fallback when Gemini is unavailable and
         as hint-injector when Gemini is building its prompt.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ── Destination profiles ───────────────────────────────────────────────────────
# Each entry:  keyword_patterns → (packing_items, pretravel_items, altitude_m, permits)

_DESTINATION_PROFILES: List[Dict[str, Any]] = [

    # ── Kedarnath / Badrinath / Char Dham ──────────────────────────────────
    {
        "keywords": ["kedarnath", "badrinath", "gangotri", "yamunotri", "char dham"],
        "altitude_m": 3500,
        "permits": True,
        "packing": [
            "Thermal inner wear (top & bottom)",
            "Heavy woolen jacket / down jacket",
            "Waterproof trekking shoes (ankle support)",
            "Rain jacket / poncho",
            "Power bank (extra capacity)",
            "Altitude sickness medicine (Diamox)",
            "Torch / headlamp with extra batteries",
            "Waterproof bag / dry bag",
            "Woolen socks (3+ pairs)",
            "Trekking pole / walking stick",
            "Glucose sachets & energy bars",
            "Sunscreen SPF 50+ & UV sunglasses",
            "Warm gloves and woolen cap",
        ],
        "pretravel": [
            "Char Dham Yatra Registration (online mandatory)",
            "Medical Fitness Certificate (for high altitude)",
            "Valid Government ID (Aadhaar / Passport)",
            "Emergency Contact List (offline copy)",
            "Download offline maps (Google Maps / Maps.me)",
            "Book helicopter ticket in advance (Phata/Sirsi)",
            "Book accommodation in Gaurikund well in advance",
            "Check weather & road conditions on IMD website",
            "Notify family of complete itinerary",
            "Travel Insurance with altitude coverage",
        ],
    },

    # ── Tirupati / Venkateswara ────────────────────────────────────────────
    {
        "keywords": ["tirupati", "tirumala", "venkateswara", "balaji"],
        "altitude_m": 800,
        "permits": True,
        "packing": [
            "Traditional / modest clothing (no shorts, sleeveless)",
            "Comfortable footwear (removed at temple entrance)",
            "Small bag / cloth bag (no plastic bags allowed)",
            "Cash for offerings and prasadam",
            "ID proof (mandatory for darshan tickets)",
            "Light cotton dupatta / stole (women)",
            "Water bottle and snacks for queue wait",
            "Rain jacket during monsoon months",
            "Sanitizer and mask",
        ],
        "pretravel": [
            "Book Darshan ticket online (tirupatibalaji.ap.gov.in)",
            "Special Entry Darshan (SED) ticket booking — book weeks ahead",
            "Follow temple dress code: traditional attire mandatory",
            "ID proof mandatory for all darshan ticket types",
            "Book accommodation at TTD Guest Houses in advance",
            "Luggage can only be kept in cloak rooms (no bags inside temple)",
            "Carry cash for offerings, prasadam, Laddu booking",
            "Check prasadam booking availability online",
            "Shave head (tonsure) reservation if planned",
        ],
    },

    # ── Amarnath ──────────────────────────────────────────────────────────
    {
        "keywords": ["amarnath", "amarnath yatra"],
        "altitude_m": 3888,
        "permits": True,
        "packing": [
            "Heavy thermal wear (top & bottom)",
            "Down jacket / heavy woolen jacket",
            "Waterproof trekking shoes",
            "Rain jacket / windproof jacket",
            "Oxygen can (portable)",
            "Altitude sickness medicine (Diamox)",
            "Trekking pole",
            "Torch / headlamp",
            "High-energy food: glucose, nuts, chocolates",
            "Waterproof bag",
            "Woolen gloves and cap",
        ],
        "pretravel": [
            "Amarnath Yatra Registration (compulsory — Shri Amarnathji Shrine Board)",
            "Medical Certificate from registered doctor",
            "Yatra Permit (obtained at Jammu / Srinagar)",
            "Valid Government ID",
            "Travel Insurance with altitude coverage",
            "Do NOT go without permit — strictly enforced",
            "Book accommodation at Pahalgam / Baltal in advance",
            "Check Army / CRPF security clearance dates",
        ],
    },

    # ── Ladakh / Leh ──────────────────────────────────────────────────────
    {
        "keywords": ["ladakh", "leh", "nubra", "pangong", "spiti", "zanskar"],
        "altitude_m": 3500,
        "permits": True,
        "packing": [
            "Heavy down jacket",
            "Thermal inner wear (multiple layers)",
            "Waterproof / windproof outer layer",
            "Altitude sickness medicine (Diamox)",
            "Portable oxygen can",
            "Sunscreen SPF 50+ (UV radiation high at altitude)",
            "UV protection sunglasses / snow goggles",
            "Woolen gloves, cap, and socks",
            "Waterproof trekking shoes / boots",
            "Power bank (charging difficult at remote camps)",
            "Offline maps (network unavailable in many areas)",
        ],
        "pretravel": [
            "Inner Line Permit (ILP) — mandatory for Nubra, Pangong, Dah-Hanu",
            "Obtain ILP at DC Office Leh or Chandigarh / Delhi",
            "Protected Area Permit (PAP) for foreigners",
            "Acclimatize at Leh for 2 full days before going higher",
            "Do NOT book Pangong/Nubra same day as arrival",
            "Book accommodation well in advance (June–September is peak)",
            "Check Manali–Leh highway status (BRO road update)",
            "Check Srinagar–Leh NH-1 status",
            "Carry sufficient cash (ATMs unreliable in remote areas)",
        ],
    },

    # ── Goa (Beach Destination) ───────────────────────────────────────────
    {
        "keywords": ["goa", "calangute", "baga", "anjuna", "panjim", "margao"],
        "altitude_m": 10,
        "permits": False,
        "packing": [
            "Swimwear / beachwear (2-3 sets)",
            "Sunscreen SPF 50+ (beach UV is intense)",
            "UV protection sunglasses",
            "Wide-brim hat / cap",
            "Flip-flops and sandals",
            "Quick-dry towels",
            "Light cotton clothing / shorts",
            "Waterproof phone pouch",
            "Insect repellent (evening mosquitoes)",
            "After-sun lotion / aloe vera gel",
        ],
        "pretravel": [
            "Book beach-side hotels / hostels in advance (peak: November–February)",
            "Check water sports package availability",
            "Check casino booking if planned",
            "Car / scooter rental available on arrival (carry driving license)",
            "Avoid Goa in heavy monsoon (June–August — sea rough, beaches closed)",
            "Carry valid ID for all night club / beach party entry",
            "Check ferry schedules for Panjim river crossing",
        ],
    },

    # ── Manali / Shimla / Hill Stations (Winter) ──────────────────────────
    {
        "keywords": ["manali", "shimla", "kufri", "kasauli", "mussoorie", "nainital", "dehradun"],
        "altitude_m": 1800,
        "permits": False,
        "packing": [
            "Heavy woolen jacket / down jacket",
            "Thermal inner wear",
            "Woolen gloves, cap, and socks",
            "Snow boots / waterproof shoes",
            "Moisturizer and lip balm (dry mountain air)",
            "Sunscreen SPF 30+",
            "Raincoat (sudden showers common)",
            "Thermos flask for hot drinks",
            "Extra charger / power bank (cold drains battery fast)",
        ],
        "pretravel": [
            "Check Rohtang Pass permit if planning to visit (HP Tourism)",
            "Book Rohtang snow activity permits online (limited slots)",
            "Book hotels well in advance in peak winter / summer season",
            "Check road conditions on HRTC / BRO bulletins",
            "Carry chains / snow socks if driving own car",
        ],
    },

    # ── Rajasthan (Desert / Heritage) ────────────────────────────────────
    {
        "keywords": ["jaipur", "jodhpur", "jaisalmer", "udaipur", "pushkar", "bikaner", "rajasthan"],
        "altitude_m": 300,
        "permits": False,
        "packing": [
            "Light cotton / linen clothing (full-sleeved for sun)",
            "Sunscreen SPF 50+",
            "Wide-brim hat or scarf",
            "Sunglasses (UV protective)",
            "Comfortable sandals + closed shoes for fort treks",
            "Electrolyte sachets (heat & dehydration)",
            "Light rain jacket (monsoon season)",
            "Modest attire for temple visits",
        ],
        "pretravel": [
            "Book palace / heritage hotel well in advance",
            "Book desert camp in Jaisalmer in advance",
            "Camel safari booking (Jaisalmer / Sam Sand Dunes)",
            "Check Pushkar Camel Fair dates if visiting in November",
            "Haggling is expected at bazaars — carry small change",
        ],
    },

    # ── Kerala / Munnar / Alleppey ────────────────────────────────────────
    {
        "keywords": ["kerala", "munnar", "alleppey", "allappuzha", "kochi", "cochin", "wayanad", "varkala", "kovalam", "thekkady"],
        "altitude_m": 50,
        "permits": False,
        "packing": [
            "Light cotton clothing (humid weather)",
            "Rain jacket / umbrella (year-round showers)",
            "Insect repellent (mosquitoes — risk of malaria)",
            "Comfortable walking sandals",
            "Modest clothing for temple visits",
            "Swimwear for backwater resorts",
            "Mosquito net for houseboat stays",
            "Camera with waterproof cover",
        ],
        "pretravel": [
            "Book houseboat stay in Alleppey well in advance",
            "Book Periyar / Wayanad wildlife safari in advance",
            "Check Athirappilly waterfall access (closed in heavy monsoon)",
            "Carry anti-malaria precautions (consult doctor)",
            "Check weather — peak rains July–August, light showers all year",
        ],
    },

    # ── Wildlife / Safari Destinations ───────────────────────────────────
    {
        "keywords": ["jim corbett", "ranthambore", "bandipur", "kaziranga", "sundarbans",
                     "periyar", "nagarhole", "pench", "kanha", "tadoba", "wildlife", "safari"],
        "altitude_m": 200,
        "permits": True,
        "packing": [
            "Dull / earthy coloured clothing (olive, khaki, brown — no bright colours)",
            "Long-sleeved shirts (protection from insects)",
            "Comfortable closed shoes / boots",
            "Insect repellent (DEET-based)",
            "Binoculars",
            "Camera with telephoto lens",
            "Sunscreen & hat",
            "Light rain jacket",
            "No perfume or strong-scented products",
        ],
        "pretravel": [
            "Safari permit booking online (forest department website)",
            "Forest entry pass and zone allocation mandatory",
            "Book guide / naturalist in advance",
            "Check park open/closed season dates",
            "No plastic allowed inside protected areas",
            "Carry valid ID proof for permit verification",
            "Book accommodation near park entrance well in advance",
        ],
    },

    # ── International: Paris / Europe ────────────────────────────────────
    {
        "keywords": ["paris", "france", "europe", "london", "rome", "barcelona", "amsterdam",
                     "berlin", "zurich", "vienna", "prague", "switzerland"],
        "altitude_m": 30,
        "permits": True,
        "packing": [
            "Passport (valid 6 months beyond travel date)",
            "Light to medium jacket (weather varies by season)",
            "Comfortable walking shoes (lots of walking on cobblestone)",
            "Universal power adapter (EU Type-C)",
            "Travel-size toiletries (100ml rule for flights)",
            "Currency: Euros (some places card-only)",
            "Scarf (churches require covered shoulders)",
            "Weather-appropriate clothing per season",
            "Portable WiFi / international SIM card",
        ],
        "pretravel": [
            "Schengen Visa application — apply 4-6 weeks in advance",
            "Valid passport (6 months validity from travel date)",
            "Travel Insurance (mandatory for Schengen visa)",
            "Confirmed hotel bookings for visa application",
            "Return flight tickets for visa application",
            "International driving permit if renting a car",
            "Notify bank of international travel to avoid card block",
            "Buy travel insurance covering medical + trip cancellation",
            "Check Euro exchange rate and carry sufficient cash",
            "Download Google Translate (French / local language)",
        ],
    },

    # ── International: Dubai / UAE ────────────────────────────────────────
    {
        "keywords": ["dubai", "abu dhabi", "uae", "sharjah"],
        "altitude_m": 10,
        "permits": False,
        "packing": [
            "Lightweight cotton clothing (extreme heat in summer)",
            "Modest clothing for malls and public areas",
            "Sunscreen SPF 50+",
            "Sunglasses and hat",
            "Comfortable walking shoes",
            "Swimwear for beach / pool",
            "Light jacket for AC interiors (very cold indoors)",
        ],
        "pretravel": [
            "Check visa on arrival / e-visa eligibility for Indian passport",
            "Apply UAE e-visa if required (processing: 3-5 days)",
            "Travel insurance recommended",
            "Carry sufficient Dirhams (AED) for local transport",
            "Notify bank of international travel",
            "Alcohol only allowed in licensed venues — research in advance",
        ],
    },

    # ── Darjeeling / Sikkim / North East India ────────────────────────────
    {
        "keywords": ["darjeeling", "gangtok", "sikkim", "shillong", "meghalaya",
                     "arunachal", "nagaland", "manipur", "assam", "northeast"],
        "altitude_m": 1500,
        "permits": True,
        "packing": [
            "Medium to heavy jacket (cool weather year-round)",
            "Thermal inner wear (for higher altitudes)",
            "Comfortable trekking shoes",
            "Rain jacket / waterproof bag (heavy rainfall region)",
            "Insect repellent",
            "Binoculars (wildlife and mountain views)",
            "Offline maps (network patchy in remote areas)",
        ],
        "pretravel": [
            "Inner Line Permit (ILP) — required for Arunachal Pradesh, Nagaland, Manipur, Mizoram",
            "Protected Area Permit (PAP) for foreigners visiting restricted zones",
            "Sikkim: Register at Rangpo check post",
            "Nathu La Pass: Permit required (Indian citizens only, not foreigners)",
            "Carry sufficient cash (ATMs scarce in remote areas)",
            "Check road conditions during monsoon (landslides common)",
        ],
    },
]


# ── Keyword → profile lookup ──────────────────────────────────────────────────

def _match_profile(destination: str) -> Optional[Dict[str, Any]]:
    """Find the best matching profile for a destination string."""
    dest_lower = destination.strip().lower()
    for profile in _DESTINATION_PROFILES:
        for kw in profile["keywords"]:
            if kw in dest_lower or dest_lower in kw:
                return profile
    return None


# ── Travel-mode specific additions ───────────────────────────────────────────

_MODE_PACKING: Dict[str, List[str]] = {
    "Bike": [
        "Helmet (mandatory)",
        "Riding gloves and jacket",
        "Puncture repair kit and basic tools",
        "Fuel planning checklist (petrol stations on route)",
        "Rain cover for luggage",
        "Bungee cords for luggage",
        "Reflective vest for night riding",
    ],
    "Train": [
        "Printed / digital train tickets",
        "Snacks and water for journey",
        "Travel blanket / light shawl",
        "Earphones and entertainment (books, offline downloads)",
        "Phone charger + power bank",
        "Valid ID proof (mandatory for ticket verification)",
        "Lock and chain for luggage (overnight trains)",
    ],
    "Flight": [
        "Passport / Valid Government ID",
        "Boarding pass (digital or printed)",
        "Baggage weight compliance (check airline limit)",
        "Toiletries in 100ml / zip-lock bag (cabin baggage)",
        "Neck pillow and eye mask (long flights)",
        "Reach airport 2 hours before domestic / 3 hours before international",
    ],
    "Car": [
        "Valid driving license",
        "Car RC book and insurance documents",
        "Roadside emergency kit (reflectors, first aid)",
        "Physical road map / offline maps",
        "Car charger for devices",
    ],
    "Bus": [
        "Bus ticket (printed or digital)",
        "Snacks and water bottle",
        "Neck pillow for overnight journeys",
        "Light blanket / shawl",
        "Phone charger / power bank",
        "Motion sickness tablets if prone",
    ],
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_destination_checklist(
    destination: str,
    month: str = "January",
    travel_mode: str = "Car",
    trip_type: str = "General",
) -> Dict[str, Any]:
    """Return destination-specific packing & pre-travel checklists.

    Parameters
    ----------
    destination : str
        Destination city / place name.
    month : str
        Travel month.
    travel_mode : str
        One of: Flight, Train, Bus, Car, Bike.
    trip_type : str
        e.g. General, Adventure, Religious, Leisure.

    Returns
    -------
    dict
        Keys:
          packing       : List[str]  — packing items
          pretravel     : List[str]  — pre-travel checklist
          altitude_m    : int        — destination altitude
          permits       : bool       — whether special permits needed
          matched       : bool       — True if a specific rule matched
    """
    profile = _match_profile(destination)

    if profile:
        packing   = list(profile["packing"])
        pretravel = list(profile["pretravel"])
        altitude  = profile.get("altitude_m", 0)
        permits   = profile.get("permits", False)
        matched   = True
    else:
        # Generic reasonable defaults
        packing   = [
            "Comfortable walking shoes",
            "Sunscreen & sunglasses",
            "Water bottle & snacks",
            "Basic first-aid kit",
            "Power bank & phone charger",
            "Travel documents (ID proof)",
            "Cash & cards",
        ]
        pretravel = [
            "Valid ID Proof (Aadhaar / Passport)",
            "Travel tickets (Train / Flight / Bus)",
            "Hotel booking confirmation",
            "Emergency contact list",
            "Sufficient cash & cards",
            "Medicines & prescriptions",
            "Travel insurance",
        ]
        altitude = 0
        permits  = False
        matched  = False

    # Add month-specific items
    month_lower = month.lower()
    if month_lower in ("june", "july", "august", "september"):
        if "Rain jacket / poncho" not in packing:
            packing.insert(1, "Rain jacket / poncho")
        if "Waterproof bag cover" not in packing:
            packing.append("Waterproof bag cover")
    if month_lower in ("november", "december", "january", "february"):
        if altitude < 500 and "Light jacket" not in packing:
            packing.insert(0, "Light to medium jacket (winter evenings are cold)")

    # Add travel-mode specific items
    mode_items = _MODE_PACKING.get(travel_mode, [])
    for item in mode_items:
        if item not in packing:
            packing.append(item)

    return {
        "packing":   packing,
        "pretravel": pretravel,
        "altitude_m": altitude,
        "permits":   permits,
        "matched":   matched,
    }


def get_prompt_hints(destination: str) -> str:
    """Return a short hint string to inject into the Gemini prompt.

    This helps Gemini generate more accurate destination-specific content
    by hinting about altitude, permits, and special requirements.
    """
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
