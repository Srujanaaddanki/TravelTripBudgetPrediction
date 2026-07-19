"""
========================================================
Module: Travel Intelligence Engine (Upgraded)
Purpose: Central orchestrator that coordinates:
           ML Prediction → SQLite Cache → Geo Validation
           → Route APIs → Weather → Gemini AI
           → Historical Dataset → Smart Budget Formula
           → Confidence Score → Final Report

Smart Budget Formula:
  KNOWN destination:
    Final Budget = (ML × 35%) + (Historical × 25%)
                 + (Transport × 20%) + (Duration × 20%)
  UNKNOWN destination:
    Final Budget = (ML × 50%) + (Transport × 30%)
                 + (Duration × 20%)

Multipliers applied after base formula:
  × Season Multiplier   (peak: 1.15 | off-peak: 0.90)
  × Popularity Multiplier (high: 1.15 | medium: 1.05 | low: 1.00)

Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.intelligence.dataset_intelligence import DatasetIntelligence
from src.services.weather_service import WeatherService
from src.services.budget_engine import BudgetEngine
from src.services.confidence_engine import ConfidenceEngine
from src.services.recommendation_engine import RecommendationEngine
from src.services.gemini_service import GeminiService
from src.services.route_service import RouteService
from src.services.database_service import DestinationCache
from src.services.geo_service import GeoService
from src.intelligence.destination_rules import (
    get_destination_country,
    get_destination_altitude,
    get_destination_permits_required,
)

log = logging.getLogger("tripai.travel_intelligence")

# ── Season multiplier map ──────────────────────────────────────────────────────
_PEAK_MONTHS = {"october", "november", "december", "january", "february", "march"}
_OFF_MONTHS  = {"july", "august", "september"}

# ── Hotel daily rates (INR) ────────────────────────────────────────────────────
_HOTEL_RATES = {
    "luxury":   6000.0,
    "premium":  6000.0,
    "comfort":  3000.0,
    "standard": 3000.0,
    "budget":   1200.0,
    "homestay": 1200.0,
}
_DEFAULT_HOTEL_RATE = 1200.0
_DAILY_ALLOWANCE    = 1500.0   # food + activities + local transport + emergency


class TravelIntelligenceEngine:
    """Orchestrates all services to build a unified travel intelligence report."""

    def __init__(
        self,
        maps_service: Any,
        db: Any,
        dataset_intel: DatasetIntelligence,
    ) -> None:
        self._dataset         = dataset_intel
        self._weather         = WeatherService(maps_service)
        self._budget          = BudgetEngine(maps_service)
        self._confidence      = ConfidenceEngine()
        self._recommendations = RecommendationEngine(db)
        self._gemini          = GeminiService()
        self._route           = RouteService()
        self._cache           = DestinationCache()
        self._maps            = maps_service
        self._geo             = GeoService()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_report(
        self,
        source: str,
        destination: str,
        month: str,
        duration_days: int,
        travel_mode: str,
        trip_type: str,
        hotel_quality: str,
        ml_prediction: float,
        src_coords: Optional[tuple] = None,
        dst_coords: Optional[tuple] = None,
        is_known_destination: bool = True,
    ) -> Dict[str, Any]:
        """Build the complete travel intelligence report.

        Flow
        ----
        Cache Check → Weather → Route → Dataset → Gemini
        → Smart Budget → Confidence → Cache Write → Return
        """
        season = self._get_season(month)

        # ── Step 1: SQLite Cache Check ────────────────────────────────
        cached_data = self._cache.get_cached(destination)

        if cached_data:
            log.info("Cache hit for destination: %s", destination)
            weather_info = cached_data.get("weather", {})
            if not weather_info:
                try:
                    weather_info = self._weather.get_weather(destination)
                except Exception:
                    weather_info = {}

            lat = cached_data.get("latitude", 0.0)
            lng = cached_data.get("longitude", 0.0)
            if lat != 0.0 and lng != 0.0:
                dst_coords = (lat, lng)

            route_data = self._get_route_data(
                source, destination, travel_mode, src_coords, dst_coords
            )
            distance_km    = route_data.get("distance_km", 0.0)
            transport_cost = route_data.get("transport_cost", 0.0)
            all_mode_costs = route_data.get("all_mode_costs", {})

            dataset_insights = self._dataset.get_similar_trips(
                destination=destination,
                month=month,
                trip_type=trip_type,
                hotel_quality=hotel_quality,
                days=duration_days,
            )
            historical_avg = dataset_insights.get("average_budget", 0.0)
            popularity     = dataset_insights.get("similar_count", 0)

            budget_details = self._calculate_smart_budget(
                ml_pred=ml_prediction,
                historical_avg=historical_avg,
                transport_cost=transport_cost,
                duration_days=duration_days,
                hotel_quality=hotel_quality,
                travel_mode=travel_mode,
                month=month,
                popularity=popularity,
                is_known=is_known_destination and historical_avg > 0,
            )
            smart_budget = budget_details["smart_budget"]

            gemini_intel = {
                "packing_checklist":    cached_data.get("packing", []),
                "pre_travel_checklist": cached_data.get("pretravel", []),
                "seasonal_tips":        cached_data.get("travel_tips", []),
                "health_suggestions":   [],
                "safety_tips":          [],
                "local_recommendations":[],
                "source":               "SQLite Cache (Instant)",
            }

            pref_exp = dataset_insights.get(
                "preferred_experience", "Nature & Sightseeing"
            )
            rec_info = self._recommendations.build_recommendations(
                destination=destination,
                month=month,
                trip_type=trip_type,
                duration_days=duration_days,
                weather={},
                preferred_experience=pref_exp,
            )

            if gemini_intel.get("packing_checklist"):
                rec_info["packing_tips"] = gemini_intel["packing_checklist"]

        else:
            log.info(
                "Cache miss for '%s'. Fetching via APIs.", destination
            )
            # ── Step 2: Weather ────────────────────────────
            try:
                weather_info = self._weather.get_weather(destination)
            except Exception:
                weather_info = {}

            # ── Step 3: Route ─────────────────────────────────────────
            route_data = self._get_route_data(
                source, destination, travel_mode, src_coords, dst_coords
            )
            distance_km    = route_data.get("distance_km", 0.0)
            transport_cost = route_data.get("transport_cost", 0.0)
            all_mode_costs = route_data.get("all_mode_costs", {})

            # ── Step 4: Historical Dataset ────────────────────────────
            dataset_insights = self._dataset.get_similar_trips(
                destination=destination,
                month=month,
                trip_type=trip_type,
                hotel_quality=hotel_quality,
                days=duration_days,
            )
            historical_avg = dataset_insights.get("average_budget", 0.0)
            popularity     = dataset_insights.get("similar_count", 0)

            # ── Step 5: Smart Budget ──────────────────────────────────
            budget_details = self._calculate_smart_budget(
                ml_pred=ml_prediction,
                historical_avg=historical_avg,
                transport_cost=transport_cost,
                duration_days=duration_days,
                hotel_quality=hotel_quality,
                travel_mode=travel_mode,
                month=month,
                popularity=popularity,
                is_known=is_known_destination and historical_avg > 0,
            )
            smart_budget = budget_details["smart_budget"]

            # ── Step 6: Gemini AI Intelligence ───────────────────────
            country = get_destination_country(destination)
            altitude = get_destination_altitude(destination)
            permits_required = get_destination_permits_required(destination)

            gemini_intel = self._gemini.get_destination_intelligence(
                destination=destination,
                month=month,
                travel_mode=travel_mode,
                duration_days=duration_days,
                season=season,
                trip_type=trip_type,
                weather=weather_info,
                country=country,
                altitude=altitude,
                permits_required=permits_required,
            )

            pref_exp = dataset_insights.get(
                "preferred_experience", "Nature & Sightseeing"
            )
            rec_info = self._recommendations.build_recommendations(
                destination=destination,
                month=month,
                trip_type=trip_type,
                duration_days=duration_days,
                weather={},
                preferred_experience=pref_exp,
            )

            if gemini_intel.get("packing_checklist"):
                rec_info["packing_tips"] = gemini_intel["packing_checklist"]

            # ── Step 7: Store into SQLite cache ───────────────────────
            try:
                dst_pt = self._safe_coords(
                    route_data.get("dest_coords") or dst_coords
                )
                self._cache.set_cache(destination, {
                    "actual_destination": destination,
                    "latitude":           dst_pt[0] if dst_pt else 0.0,
                    "longitude":          dst_pt[1] if dst_pt else 0.0,
                    "distance_km":        distance_km,
                    "duration_hr":        route_data.get("duration_hours", 0.0),
                    "month":              month,
                    "days":               duration_days,
                    "travel_mode":        travel_mode,
                    "hotel_quality":      hotel_quality,
                    "weather":            weather_info,
                    "travel_tips":        gemini_intel.get("seasonal_tips", []),
                    "packing":            gemini_intel.get("packing_checklist", []),
                    "pretravel":          gemini_intel.get("pre_travel_checklist", []),
                    "budget":             smart_budget,
                })
            except Exception as cache_err:
                log.warning("Cache write failed: %s", cache_err)

        # ── Budget tiers & supporting data ────────────────────────────
        budget_tiers = self._budget.calculate_budget_tiers(
            ml_prediction=smart_budget,
            hotel_quality=hotel_quality,
        )

        mode_comparison = self._build_mode_comparison(
            all_mode_costs, travel_mode, distance_km
        )

        confidence_info = self._confidence.calculate_confidence(
            dataset_insights=dataset_insights,
            mode_comparison=mode_comparison,
            ml_prediction=ml_prediction,
        )

        similar_traveller = self._dataset.get_similar_traveller_stats(
            trip_type=trip_type,
            duration_days=duration_days,
            hotel_quality=hotel_quality,
            predicted_cost=ml_prediction,
        )

        related_searches = self._recommendations.get_related_searches(destination)
        trending         = self._dataset.get_trending_destinations(top_n=5)

        return {
            "ml_prediction":     ml_prediction,
            "historical_avg":    historical_avg,
            "transport_cost":    transport_cost,
            "smart_budget":      smart_budget,
            "smart_budget_details": budget_details,
            "budget_difference": round(smart_budget - ml_prediction, 2),
            "budget_tiers":      budget_tiers,
            "confidence":        confidence_info,
            "weather":           weather_info,
            "intelligence":      rec_info,
            "gemini":            gemini_intel,
            "mode_comparison":   mode_comparison,
            "dataset_insights":  dataset_insights,
            "similar_traveller": similar_traveller,
            "related_searches":  related_searches,
            "trending":          trending,
            "route":             route_data,
            "season":            season,
            "popularity":        popularity,
            "is_known":          is_known_destination,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_route_data(
        self,
        source: str,
        destination: str,
        travel_mode: str,
        src_coords: Optional[Any],
        dst_coords: Optional[Any],
    ) -> Dict[str, Any]:
        """Resolve coordinates via Geoapify / Nominatim and call RouteService."""
        # Geocode missing coordinates
        if not src_coords:
            src_res = self._geo.validate_destination(source)
            if src_res["valid"]:
                src_coords = (src_res["lat"], src_res["lng"])

        if not dst_coords:
            dst_res = self._geo.validate_destination(destination)
            if dst_res["valid"]:
                dst_coords = (dst_res["lat"], dst_res["lng"])

        src_pt = self._safe_coords(src_coords)
        dst_pt = self._safe_coords(dst_coords)

        if src_pt and dst_pt:
            try:
                route     = self._route.get_route(
                    src_pt[0], src_pt[1],
                    dst_pt[0], dst_pt[1],
                    travel_mode,
                )
                all_costs = self._route.estimate_all_mode_costs(
                    route["distance_km"]
                )
                route["all_mode_costs"] = all_costs
                route["source_coords"]  = src_pt
                route["dest_coords"]    = dst_pt
                return route
            except Exception as route_err:
                log.warning("RouteService call failed: %s", route_err)

        # Fallback: default estimate when coordinates unavailable
        fallback_dist = 400.0
        route = {
            "distance_km":    fallback_dist,
            "duration_hours": 7.0,
            "transport_cost": 1200.0,
            "polyline":       None,
            "source":         "Default Estimate",
            "all_mode_costs": self._route.estimate_all_mode_costs(fallback_dist),
            "source_coords":  src_pt,
            "dest_coords":    dst_pt,
        }
        return route

    @staticmethod
    def _calculate_smart_budget(
        ml_pred: float,
        historical_avg: float,
        transport_cost: float,
        duration_days: int,
        hotel_quality: str,
        travel_mode: str,
        month: str = "May",
        popularity: int = 0,
        is_known: bool = True,
    ) -> Dict[str, Any]:
        """Apply the upgraded Smart Budget Formula.

        Known destination  → 35% ML + 25% Historical + 20% Transport + 20% Duration
        Unknown destination → 50% ML + 30% Transport + 20% Duration

        Then multiply by:
          × Season Multiplier   (peak months +15%, monsoon -10%)
          × Popularity Multiplier (>50 trips +15%, >20 trips +5%)
        """
        # Hotel daily rate
        hotel_key  = hotel_quality.lower().strip()
        hotel_rate = next(
            (v for k, v in _HOTEL_RATES.items() if k in hotel_key),
            _DEFAULT_HOTEL_RATE,
        )
        duration_cost = (hotel_rate + _DAILY_ALLOWANCE) * duration_days

        # Season multiplier (Disabled in simplification mode)
        season_mult = 1.00

        # Popularity multiplier (Disabled in simplification mode)
        pop_mult = 1.00

        # Budget formula
        if is_known and historical_avg > 0:
            smart = (
                ml_pred       * 0.35
                + historical_avg * 0.25
                + transport_cost * 0.20
                + duration_cost  * 0.20
            )
            formula_type = "known"
        else:
            # Unknown destination — no historical reference
            smart = (
                ml_pred       * 0.50
                + transport_cost * 0.30
                + duration_cost  * 0.20
            )
            formula_type = "unknown"

        base_budget = smart
        smart = smart * season_mult * pop_mult

        # Floor: at minimum 50% of duration cost + transport cost
        floor = duration_cost * 0.50 + transport_cost
        is_floored = floor > smart
        smart = max(smart, floor)

        final_budget = float(round(smart, -2))

        return {
            "smart_budget":      final_budget,
            "season_mult":       season_mult,
            "pop_mult":          pop_mult,
            "duration_cost":     duration_cost,
            "hotel_rate":        hotel_rate,
            "daily_allowance":   _DAILY_ALLOWANCE,
            "formula_type":      formula_type,
            "base_budget":       base_budget,
            "is_floored":        is_floored,
            "floor_value":       floor,
            "ml_part":           ml_pred * (0.35 if formula_type == "known" else 0.50),
            "hist_part":         (historical_avg * 0.25) if formula_type == "known" else 0.0,
            "trans_part":        transport_cost * (0.20 if formula_type == "known" else 0.30),
            "dur_part":          duration_cost * 0.20,
        }

    @staticmethod
    def _get_season(month: str) -> str:
        """Derive season label from month string."""
        month_lower = month.lower()
        if month_lower in ("november", "december", "january", "february"):
            return "Winter"
        if month_lower in ("july", "august", "september"):
            return "Rainy"
        if month_lower in ("march", "april"):
            return "Spring"
        if month_lower in ("october",):
            return "Autumn"
        return "Summer"

    @staticmethod
    def _safe_coords(coords: Any) -> Optional[tuple[float, float]]:
        """Safely extract (lat, lng) tuple from multiple input formats."""
        if not coords:
            return None
        if isinstance(coords, dict):
            lat = coords.get("lat", coords.get("latitude"))
            lng = coords.get("lng", coords.get("longitude"))
            if lat is not None and lng is not None:
                return float(lat), float(lng)
        if isinstance(coords, (tuple, list)) and len(coords) >= 2:
            try:
                return float(coords[0]), float(coords[1])
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def _build_mode_comparison(
        all_mode_costs: Dict[str, Any],
        selected_mode: str,
        distance_km: float,
    ) -> Dict[str, Any]:
        """Build the mode comparison dict used by existing UI components."""
        if not all_mode_costs:
            return {
                "modes":       {},
                "cheapest":    "N/A",
                "fastest":     "N/A",
                "best_value":  "N/A",
                "recommended": "N/A",
                "distance_km": distance_km,
                "selected":    selected_mode,
            }

        mode_list  = ["Flight", "Train", "Bus", "Car", "Bike"]
        costs_list = [
            (m, all_mode_costs[m]["round_trip"])
            for m in mode_list
            if m in all_mode_costs
        ]
        cheapest = min(costs_list, key=lambda x: x[1])[0] if costs_list else "Train"
        fastest  = "Flight" if distance_km > 300 else "Car"

        return {
            "modes":       all_mode_costs,
            "cheapest":    cheapest,
            "fastest":     fastest,
            "best_value":  "Train",
            "recommended": (
                "Flight" if distance_km > 700
                else "Train" if distance_km > 200
                else "Car"
            ),
            "distance_km": distance_km,
            "selected":    selected_mode,
        }
