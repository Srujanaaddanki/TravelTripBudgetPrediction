"""
========================================================
Module: Destination Knowledge Base
Purpose: Curated travel information for popular Indian
         destinations loaded from a JSON configuration file.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import difflib
import json
import os
from typing import Any, Dict, List, Optional

# Month groupings for seasonal calculations
SUMMER_MONTHS: List[str] = ["April", "May", "June"]
MONSOON_MONTHS: List[str] = ["July", "August", "September"]
WINTER_MONTHS: List[str] = ["November", "December", "January", "February"]
AUTUMN_MONTHS: List[str] = ["October"]
SPRING_MONTHS: List[str] = ["March"]

# Load the hand-curated destinations data from destinations.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "destinations.json")

try:
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        DESTINATIONS: Dict[str, Any] = json.load(file)
except Exception:
    DESTINATIONS = {}


def get_destination_info(destination: str) -> Optional[Dict[str, Any]]:
    """Look up destination in knowledge base with fuzzy matching.

    Parameters
    ----------
    destination : str
        Name of destination (e.g., 'Goa' or 'Manali').

    Returns
    -------
    dict, optional
        Destination details or None if no close match is found.
    """
    dest_lower = destination.lower().strip()
    if dest_lower in DESTINATIONS:
        return DESTINATIONS[dest_lower]

    # Fuzzy match search key against known keys
    matches = difflib.get_close_matches(dest_lower, DESTINATIONS.keys(), n=1, cutoff=0.6)
    if matches:
        return DESTINATIONS[matches[0]]

    return None


def get_packing_tips(destination_type: str, month: str, temperature_c: Optional[float] = None) -> List[str]:
    """Generate packing tips based on destination type and weather.

    Parameters
    ----------
    destination_type : str
        Type of destination (e.g., 'Beach', 'Hill Station').
    month : str
        Travel month name.
    temperature_c : float, optional
        Current temperature in celsius.

    Returns
    -------
    list[str]
        Packing suggestions list.
    """
    tips = ["Comfortable walking shoes", "Sunscreen & Sunglasses", "Basic first-aid kit"]
    month_title = month.strip().title()

    # Temperature-based tips
    temp = temperature_c if temperature_c is not None else 25.0
    if temp < 15.0:
        tips.append("Heavy woollens, thermal innerwear, gloves")
    elif temp < 22.0:
        tips.append("Light jacket, sweater or cardigans")
    else:
        tips.append("Breathable cotton clothes, t-shirts")

    # Season-based tips
    if month_title in MONSOON_MONTHS:
        tips.append("Umbrella, raincoat, waterproof bag cover")

    # Type-based tips
    type_lower = destination_type.lower()
    if "beach" in type_lower:
        tips.extend(["Swimwear, flip-flops", "Quick-dry towels"])
    elif "hill" in type_lower:
        tips.extend(["Thermos flask", "Moisturizer & lip balm"])
    elif "spiritual" in type_lower or "heritage" in type_lower:
        tips.append("Modest clothing for visiting temples/monuments")

    return tips[:6]  # Return at most 6 tips


def get_crowd_level(destination: str, month: str) -> str:
    """Return crowd level ('High', 'Moderate', or 'Low') for a month.

    Parameters
    ----------
    destination : str
        Destination name.
    month : str
        Travel month.

    Returns
    -------
    str
        Crowd level.
    """
    info = get_destination_info(destination)
    month_title = month.strip().title()

    if info and "crowd_by_month" in info:
        return info["crowd_by_month"].get(month_title, "Moderate")

    # Fallback to standard season crowd levels
    if month_title in WINTER_MONTHS or month_title in SUMMER_MONTHS:
        return "High"
    if month_title in MONSOON_MONTHS:
        return "Low"
    return "Moderate"


def get_best_time_to_visit(destination: str) -> str:
    """Return a human-readable string describing the best time to visit.

    Parameters
    ----------
    destination : str
        Destination name.

    Returns
    -------
    str
        Best time summary.
    """
    info = get_destination_info(destination)
    if info and "best_months" in info:
        months = info["best_months"]
        if months:
            return f"{months[0]} to {months[-1]}"

    return "October to March"
