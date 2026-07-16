"""
========================================================
Module: Destination Aliases
Purpose: Maps common typos, alternate spellings, and
         local names → canonical destination names.
         Used as Step 2 in the 7-step resolution hierarchy
         BEFORE RapidFuzz / Geoapify / Gemini are called.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
from typing import Dict

# ── Alias Map ─────────────────────────────────────────────────────────────────
# Key:   user input (lowercase, stripped)
# Value: canonical destination name (as it appears in encoders / dataset)
DESTINATION_ALIASES: Dict[str, str] = {
    # Hyderabad
    "hydrabad": "hyderabad",
    "hyd": "hyderabad",
    "hyder": "hyderabad",

    # Tirupati
    "tirupathi": "tirupati",
    "tirumala": "tirupati",

    # Visakhapatnam / Vizag
    "vizag": "visakhapatnam",
    "vishakapatnam": "visakhapatnam",
    "vishakhapatnam": "visakhapatnam",
    "visakapatanam": "visakhapatnam",

    # Bangalore / Bengaluru
    "banglore": "bangalore",
    "bengaluru": "bangalore",
    "blr": "bangalore",
    "bangluru": "bangalore",

    # Mumbai / Bombay
    "bombay": "mumbai",
    "bbay": "mumbai",

    # Kolkata / Calcutta
    "calcutta": "kolkata",
    "calicut": "kozhikode",

    # Chennai / Madras
    "madras": "chennai",

    # Kochi / Cochin
    "cochin": "kochi",
    "ernakulam": "kochi",

    # Pondicherry
    "pondichery": "pondicherry",
    "puducherry": "pondicherry",
    "pondy": "pondicherry",

    # Delhi
    "new delhi": "delhi",
    "ncr": "delhi",
    "nct": "delhi",

    # Leh Ladakh
    "leh ladakh": "leh",
    "ladakh": "leh",
    "laddakh": "leh",

    # Ooty
    "ootty": "ooty",
    "udhagamandalam": "ooty",

    # Munnar
    "munar": "munnar",

    # Rishikesh
    "hrishikesh": "rishikesh",
    "rushikesh": "rishikesh",

    # Mussoorie
    "mussoori": "mussoorie",
    "mussorie": "mussoorie",

    # Jaisalmer
    "jaisalmar": "jaisalmer",
    "jaisalmair": "jaisalmer",

    # Pushkar
    "pushcar": "pushkar",
    "pushcker": "pushkar",

    # Amritsar
    "amristar": "amritsar",
    "amristsar": "amritsar",

    # Darjeeling
    "darjeling": "darjeeling",
    "darjilin": "darjeeling",

    # Manali
    "manal": "manali",
    "manali hp": "manali",

    # Shimla
    "simla": "shimla",

    # Agra (Taj Mahal)
    "taj mahal": "agra",

    # Varanasi / Benaras / Kashi
    "benaras": "varanasi",
    "kashi": "varanasi",
    "benares": "varanasi",

    # Jaipur
    "jaipurr": "jaipur",
    "pink city": "jaipur",

    # Udaipur
    "udaipure": "udaipur",
    "lake city": "udaipur",

    # Jodhpur
    "jodpur": "jodhpur",
    "blue city": "jodhpur",

    # Alleppey / Alappuzha (Kerala)
    "alleppey": "alappuzha",
    "alepy": "alappuzha",

    # Coorg / Kodagu
    "coorg": "madikeri",
    "kodagu": "madikeri",

    # Gokarna
    "gokarnam": "gokarna",
    "gokarn": "gokarna",

    # Hampi
    "hampy": "hampi",

    # Mysore
    "mysuru": "mysore",

    # Kukanet (specific PRD example)
    "kukunate": "kukanet",
    "kukannate": "kukanet",
    "kukanet forest": "kukanet",

    # Andaman
    "andamans": "andaman",
    "port blair": "andaman",

    # Coimbatore
    "coimbatour": "coimbatore",
    "kovai": "coimbatore",

    # Guwahati
    "guwahati": "guwahati",
    "gauhati": "guwahati",

    # Bhubaneswar
    "bhuvaneshwar": "bhubaneswar",
    "bhubaneswar": "bhubaneswar",

    # Patna
    "patna": "patna",
    "patna city": "patna",

    # Lucknow
    "luckhnow": "lucknow",
    "lucnow": "lucknow",

    # Srinagar
    "shrinagar": "srinagar",
    "kashmir": "srinagar",

    # Chandigarh
    "chandighar": "chandigarh",

    # Dehradun
    "deharadun": "dehradun",
    "dehradoon": "dehradun",
}


def resolve_alias(user_input: str) -> str | None:
    """Return the canonical destination if an alias exists, else None.

    Parameters
    ----------
    user_input : str
        Raw input from the user.

    Returns
    -------
    str | None
        Canonical destination name, or None if no alias found.
    """
    key = user_input.strip().lower()
    return DESTINATION_ALIASES.get(key)


def get_all_aliases() -> Dict[str, str]:
    """Return the full alias map (for inspection or debugging)."""
    return DESTINATION_ALIASES
