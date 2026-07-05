"""
========================================================
Module: Travel Intelligence Engine
Purpose: Central orchestrator that coordinates weather queries,
         dataset insights, budget calculations, and confidence
         scoring to build the travel report.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.intelligence.dataset_intelligence import DatasetIntelligence
from src.services.weather_service import WeatherService
from src.services.budget_engine import BudgetEngine
from src.services.confidence_engine import ConfidenceEngine
from src.services.recommendation_engine import RecommendationEngine

log = logging.getLogger("tripai.travel_intelligence")


class TravelIntelligenceEngine:
    """Orchestrates different service components to build unified travel reports."""

    def __init__(
        self,
        maps_service: Any,
        db: Any,
        dataset_intel: DatasetIntelligence,
    ) -> None:
        """Initialize all sub-services and engines.

        Parameters
        ----------
        maps_service : MapsService
            Service used for maps and coordinates.
        db : TripDatabase
            Service used for SQLite logs.
        dataset_intel : DatasetIntelligence
            Dataset analysis module.
        """
        self._dataset = dataset_intel
        self._weather = WeatherService(maps_service)
        self._budget = BudgetEngine(maps_service)
        self._confidence = ConfidenceEngine()
        self._recommendations = RecommendationEngine(db)

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
    ) -> Dict[str, Any]:
        """Orchestrate components in sequence to create the travel report.

        Flow
        ----
        Prediction → Weather → Dataset Intelligence → Recommendations →
        Budget Engine → Confidence Engine → Final Response
        """
        # Step 1: Weather lookup
        weather_info = self._weather.get_weather(destination)

        # Step 2: Dataset Intelligence matching
        dataset_insights = self._dataset.get_similar_trips(
            destination=destination,
            month=month,
            trip_type=trip_type,
            hotel_quality=hotel_quality,
            days=duration_days,
        )

        # Step 3: Extract historical experience and build recommendations (Feature 4)
        pref_exp = dataset_insights.get("preferred_experience", "Nature & Sightseeing")
        rec_info = self._recommendations.build_recommendations(
            destination=destination,
            month=month,
            trip_type=trip_type,
            duration_days=duration_days,
            weather=weather_info,
            preferred_experience=pref_exp,
        )

        # Step 4: Budget Engine calculations (tiers and routes)
        budget_tiers = self._budget.calculate_budget_tiers(
            ml_prediction=ml_prediction,
            hotel_quality=hotel_quality,
        )
        mode_comparison = self._budget.get_mode_comparison(
            source=source,
            destination=destination,
            selected_mode=travel_mode,
        )

        # Step 5: Confidence evaluation
        confidence_info = self._confidence.calculate_confidence(
            dataset_insights=dataset_insights,
            mode_comparison=mode_comparison,
            ml_prediction=ml_prediction,
        )

        # Step 6: Similar traveller analysis (Feature 5)
        similar_traveller = self._dataset.get_similar_traveller_stats(
            trip_type=trip_type,
            duration_days=duration_days,
            hotel_quality=hotel_quality,
            predicted_cost=ml_prediction,
        )

        # Step 7: Related searches & trending destinations
        related_searches = self._recommendations.get_related_searches(destination)
        trending = self._dataset.get_trending_destinations(top_n=5)

        # Final Response construction
        return {
            "budget_tiers": budget_tiers,
            "confidence": confidence_info,
            "weather": weather_info,
            "intelligence": rec_info,
            "mode_comparison": mode_comparison,
            "dataset_insights": dataset_insights,
            "similar_traveller": similar_traveller,
            "related_searches": related_searches,
            "trending": trending,
        }
