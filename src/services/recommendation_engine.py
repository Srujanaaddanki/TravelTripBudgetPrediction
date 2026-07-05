"""
========================================================
Module: Recommendation Engine
Purpose: Generates curated travel recommendations, packing lists,
         safety guidelines, seasonal alerts, and extracts
         session-based co-searches from SQLite history.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.intelligence.destination_knowledge import (
    get_destination_info,
    get_packing_tips,
    get_crowd_level,
    get_best_time_to_visit,
)

log = logging.getLogger("tripai.recommendation_engine")

# Fallback regional groupings to determine related destinations
REGIONAL_GROUPS = [
    ["Manali", "Shimla", "Rishikesh", "Mussoorie", "Nainital"],
    ["Kerala", "Ooty", "Coorg", "Munnar", "Pondicherry"],
    ["Goa", "Udaipur", "Jaipur", "Mount Abu", "Lonavala"],
    ["Darjeeling", "Gangtok", "Meghalaya", "Puri", "Sikkim"],
    ["Leh", "Spiti", "Kasol", "Auli", "Bir Billing"],
]

# Map preferred experiences to categories of activities (Feature 4)
EXPERIENCE_ACTIVITIES: Dict[str, List[str]] = {
    "Adventure": ["River Rafting", "Camping", "Trekking", "Paragliding", "Rock Climbing"],
    "Luxury": ["Luxury Hotels", "Spa & Wellness", "Fine Dining", "Private Guided Tours", "Helicopter rides"],
    "Nature": ["Waterfalls", "Tea Gardens", "Sunrise Point", "Forest Walks", "Wildlife Safari"],
    "Food": ["Local Cuisine tasting", "Street Food tours", "Traditional Restaurants", "Local food markets"],
    "Budget": ["Budget Stays", "Shared Transport / Local bus", "Local Markets shopping", "Free walking tours"],
    "Relaxation": ["Lakeside walking", "Luxury Resorts stay", "Scenic Parks", "Ayurvedic Spa treatments"],
    "Cultural": ["Historical Monuments", "Art Galleries", "Museum Visits", "Local festivals", "Temple Tours"]
}


class RecommendationEngine:
    """Computes destination recommendations, packing lists, and trending items."""

    def __init__(self, db: Any) -> None:
        """Initialize with SQLite database connection.

        Parameters
        ----------
        db : TripDatabase
            Direct database access instance.
        """
        self._db = db

    def build_recommendations(
        self,
        destination: str,
        month: str,
        trip_type: str,
        duration_days: int,
        weather: Dict[str, Any],
        preferred_experience: str = "Nature & Sightseeing",
    ) -> Dict[str, Any]:
        """Compile seasonal tips, packing advice, attractions, and local food.

        Parameters
        ----------
        destination : str
            Destination name.
        month : str
            Travel month.
        trip_type : str
            Stay type (Solo/Family/Friends).
        duration_days : int
            Stay duration.
        weather : dict
            Current weather parameters.
        preferred_experience : str
            Traveller style filter from the dataset.

        Returns
        -------
        dict
            Curated recommendation objects.
        """
        dest_info = get_destination_info(destination)
        crowd = get_crowd_level(destination, month)
        best_time = get_best_time_to_visit(destination)

        # Get specialized packing list
        dest_type = dest_info.get("type", "General") if dest_info else "General"
        temp = weather.get("temperature_c", 25.0)
        packing = get_packing_tips(dest_type, month, temp)

        # Match experience activities
        activities = self._get_activities_for_experience(preferred_experience)

        # Basic report structure
        rec = {
            "best_time": best_time,
            "crowd_level": crowd,
            "packing_tips": packing,
            "places_to_visit": dest_info.get("places_to_visit", ["Explore local sights"]) if dest_info else ["Explore local sights"],
            "local_foods": dest_info.get("local_foods", ["Try local dishes"]) if dest_info else ["Try local dishes"],
            "hidden_gems": dest_info.get("hidden_gems", ["Explore nearby valleys"]) if dest_info else ["Explore nearby valleys"],
            "safety_tips": dest_info.get("safety_tips", self._general_safety_tips()) if dest_info else self._general_safety_tips(),
            "hotel_area_recommendation": ", ".join(dest_info.get("hotel_areas", ["City Centre"])) if dest_info else "City Centre",
            "transportation_advice": dest_info.get("transportation", "Use local cabs") if dest_info else "Use local cabs",
            "money_saving_tips": self._generate_saving_tips(trip_type, duration_days),
            "experience_activities": activities,
            "preferred_experience": preferred_experience,
        }

        return rec

    def _get_activities_for_experience(self, exp_style: str) -> List[str]:
        """Match preferred experience style to hardcoded activities list."""
        style_clean = exp_style.lower().strip()
        for key, activities in EXPERIENCE_ACTIVITIES.items():
            if key.lower() in style_clean:
                return activities

        # Default fallback activities if match is not direct
        if "nature" in style_clean:
            return EXPERIENCE_ACTIVITIES["Nature"]
        if "food" in style_clean:
            return EXPERIENCE_ACTIVITIES["Food"]
        if "budget" in style_clean:
            return EXPERIENCE_ACTIVITIES["Budget"]
        if "relax" in style_clean:
            return EXPERIENCE_ACTIVITIES["Relaxation"]
        if "adventure" in style_clean:
            return EXPERIENCE_ACTIVITIES["Adventure"]
        if "luxury" in style_clean:
            return EXPERIENCE_ACTIVITIES["Luxury"]
        if "cultural" in style_clean or "history" in style_clean:
            return EXPERIENCE_ACTIVITIES["Cultural"]

        return ["Sightseeing", "Photography", "Souvenir shopping", "Exploring cafes"]

    def get_related_searches(self, destination: str) -> List[str]:
        """Find related searches based on shared session search history in SQLite."""
        try:
            searches = self._db.get_searches(limit=300)
            if not searches:
                return self._fallback_related(destination)

            dest_lower = destination.lower().strip()
            matching_sessions = {
                s.get("session_id") for s in searches
                if s.get("destination", "").lower().strip() == dest_lower
                if s.get("session_id")
            }

            if not matching_sessions:
                return self._fallback_related(destination)

            counts: Dict[str, int] = {}
            for s in searches:
                sid = s.get("session_id")
                place = s.get("destination", "").lower().strip()
                if sid in matching_sessions and place != dest_lower:
                    counts[place] = counts.get(place, 0) + 1

            sorted_places = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return [p.title() for p, _ in sorted_places[:4]]

        except Exception as err:
            log.warning("Failed loading related searches from DB: %s", err)
            return self._fallback_related(destination)

    def _fallback_related(self, destination: str) -> List[str]:
        """Find regional fallback recommendations if no search logs are present."""
        dest_lower = destination.lower().strip()
        for group in REGIONAL_GROUPS:
            group_lower = [item.lower() for item in group]
            if dest_lower in group_lower:
                return [item for item in group if item.lower() != dest_lower][:4]
        return ["Goa", "Manali", "Jaipur", "Kerala"]

    @staticmethod
    def _general_safety_tips() -> List[str]:
        """Provide standard travel safety guidelines."""
        return [
            "Keep emergency contact numbers handy",
            "Avoid travelling alone late at night in unfamiliar areas",
            "Secure your primary travel documents and cash",
        ]

    @staticmethod
    def _generate_saving_tips(trip_type: str, duration_days: int) -> List[str]:
        """Build money-saving strategies based on duration and travel group type."""
        tips = [
            "Reserve your accommodations at least 2 weeks early for discounts",
            "Choose public transit or share rides whenever possible",
        ]

        type_key = trip_type.lower().strip()
        if "friend" in type_key or "group" in type_key:
            tips.append("Split accommodation and local taxi fares with your group")
        elif "solo" in type_key:
            tips.append("Stay in boutique hostels to save costs and meet travellers")

        if duration_days > 5:
            tips.append("Look for weekly home stay deals or apartments with kitchens")

        return tips
