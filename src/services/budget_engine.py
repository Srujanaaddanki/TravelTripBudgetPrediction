"""
========================================================
Module: Budget Engine
Purpose: Calculates budget tiers (Minimum, Recommended,
         Comfort, Luxury) and handles adjustments for stay quality
         and travel modes.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger("tripai.budget_engine")

# Base multipliers to scale raw predictions
TIER_MULTIPLIERS = {
    "minimum": 0.70,      # Budget travel, shared stays, street food
    "recommended": 1.00,  # Matches the ML prediction exactly
    "comfort": 1.40,      # Nice hotels, dining, more activities
    "luxury": 2.00,       # 5-star hotels, luxury tours & premium meals
}

# hotel quality adjustments
HOTEL_QUALITY_FACTORS = {
    "budget": 0.85,
    "standard": 1.00,
    "homestay": 0.90,
    "premium": 1.20,
    "luxury": 1.50,
}


class BudgetEngine:
    """Calculates budget tiers and smart travel cost options."""

    def __init__(self, maps_service: Any) -> None:
        """Initialize the budget engine with the maps service instance.

        Parameters
        ----------
        maps_service : MapsService
            Service for fetching route distances and computing mode costs.
        """
        self._maps = maps_service

    def calculate_budget_tiers(
        self,
        ml_prediction: float,
        hotel_quality: str,
    ) -> Dict[str, Any]:
        """Generate budget tiers (Min, Recommended, Comfort, Luxury) from the prediction.

        Parameters
        ----------
        ml_prediction : float
            Budget prediction from Random Forest.
        hotel_quality : str
            User stay quality choice.

        Returns
        -------
        dict
            Budget tier totals and context explanation text.
        """
        quality_key = hotel_quality.lower().strip()
        quality_factor = HOTEL_QUALITY_FACTORS.get(quality_key, 1.0)

        tiers: Dict[str, float] = {}
        for tier, multiplier in TIER_MULTIPLIERS.items():
            # Apply multipliers and stay factor
            raw_tier = ml_prediction * multiplier * quality_factor
            # Round to the nearest 100 for clean UI presentation
            tiers[tier] = round(raw_tier, -2)

        # The recommended tier should always map to the ML prediction
        tiers["recommended"] = round(ml_prediction, -2)

        explanation = (
            f"Budget tiers are scaled from the Random Forest prediction "
            f"(₹{int(ml_prediction):,}), adjusted for {hotel_quality.title()} "
            f"stay quality. Minimum represents hostels and local transit; "
            f"Luxury represents premium hotels and experiences."
        )

        return {
            "minimum": tiers["minimum"],
            "recommended": tiers["recommended"],
            "comfort": tiers["comfort"],
            "luxury": tiers["luxury"],
            "explanation": explanation,
        }

    def get_mode_comparison(
        self,
        source: str,
        destination: str,
        selected_mode: str,
    ) -> Dict[str, Any]:
        """Compute and compare estimated travel costs for all modes.

        Parameters
        ----------
        source : str
            Origin city.
        destination : str
            Target city.
        selected_mode : str
            The mode chosen by the user.

        Returns
        -------
        dict
            Comparison dictionary with cost, duration, and recommendation.
        """
        try:
            route = self._maps.get_route_info(source, destination, "Car")
            distance_km = route.get("distance_km", 0.0)

            if distance_km <= 0:
                return self._empty_mode_comparison()

            # Retrieve cost matrix for all modes from the maps service
            all_costs = self._maps.get_all_mode_costs(distance_km)

            # Determine smartest value and distance recommendation
            best_val = self._find_best_value(all_costs)
            recommended = self._recommend_mode(distance_km)

            return {
                "modes": {
                    mode: all_costs[mode]
                    for mode in ["Flight", "Train", "Bus", "Car", "Bike"]
                    if mode in all_costs
                },
                "cheapest": all_costs.get("cheapest", "Train"),
                "fastest": all_costs.get("fastest", "Flight"),
                "best_value": best_val,
                "recommended": recommended,
                "distance_km": distance_km,
                "selected": selected_mode,
            }
        except Exception as err:
            log.warning("Cost modes comparison failed: %s", err)
            return self._empty_mode_comparison()

    def _find_best_value(self, all_costs: Dict[str, Any]) -> str:
        """Find the transport mode with the best cost-to-speed ratio."""
        best_mode = "Train"
        best_ratio = float("inf")

        for mode in ["Flight", "Train", "Bus", "Car", "Bike"]:
            mode_data = all_costs.get(mode)
            if not mode_data:
                continue

            cost = mode_data.get("round_trip", mode_data.get("one_way", 0.0))
            speed = mode_data.get("speed_kmh", 50)

            if speed > 0 and cost > 0:
                ratio = cost / speed
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_mode = mode

        return best_mode

    def _recommend_mode(self, distance_km: float) -> str:
        """Recommend a transportation mode based on geographic distance."""
        if distance_km < 200:
            return "Car"
        if distance_km < 600:
            return "Train"
        return "Flight"

    def _empty_mode_comparison(self) -> Dict[str, Any]:
        """Default fallback dictionary when cost matrix cannot be generated."""
        return {
            "modes": {},
            "cheapest": "N/A",
            "fastest": "N/A",
            "best_value": "N/A",
            "recommended": "N/A",
            "distance_km": 0.0,
            "selected": "N/A",
        }
