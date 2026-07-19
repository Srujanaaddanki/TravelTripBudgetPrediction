"""
========================================================
Module: Confidence Engine
Purpose: Evaluates multi-factor confidence ratings for budget
         predictions using data availability, route checks,
         and model constraints.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

from typing import Any, Dict, List


class ConfidenceEngine:
    """Calculates multi-factor prediction confidence and descriptive factors."""

    def calculate_confidence(
        self,
        dataset_insights: Dict[str, Any],
        mode_comparison: Dict[str, Any],
        ml_prediction: float,
        is_known: bool = True,
        resolution_type: str = "known",
    ) -> Dict[str, Any]:
        """Determine final percentage score, rating level, and individual indicators."""
        factors: List[Dict[str, Any]] = []
        
        has_data = dataset_insights.get("has_data", False)
        similar_count = dataset_insights.get("similar_count", 0)
        
        if resolution_type == "known" or (is_known and has_data):
            score = 95
            if similar_count >= 5:
                score = 100
            elif similar_count >= 2:
                score = 98
            level = "Dataset Verified"
            factors.append({
                "name": "Historical Data",
                "available": True,
                "detail": f"Dataset verified: {similar_count} similar trips found",
            })
        elif resolution_type == "api_only":
            score = 65
            distance_km = mode_comparison.get("distance_km", 0.0)
            if distance_km > 0.0:
                score = 80
            level = "API Estimation"
            factors.append({
                "name": "API Resolution",
                "available": True,
                "detail": f"Geocoded via Geo APIs: {distance_km:,.1f} km",
            })
        elif resolution_type == "gemini_approx":
            score = 50
            distance_km = mode_comparison.get("distance_km", 0.0)
            if distance_km > 0.0:
                score = 70
            else:
                score = 60
            level = "AI Approximation"
            factors.append({
                "name": "AI Resolution",
                "available": True,
                "detail": "Geocoded and estimated using Gemini AI",
            })
        else:
            score = 40
            distance_km = mode_comparison.get("distance_km", 0.0)
            if distance_km > 0.0:
                score = 48
            level = "Low Confidence"
            factors.append({
                "name": "Failed Resolution",
                "available": False,
                "detail": "Failed to resolve destination, using fallback details",
            })

        # Factor 2: Route mapping verification
        distance_km = mode_comparison.get("distance_km", 0.0)
        if distance_km > 0.0:
            factors.append({
                "name": "Route Availability",
                "available": True,
                "detail": f"Route calculated successfully ({distance_km:,.1f} km)",
            })
        else:
            factors.append({
                "name": "Route Availability",
                "available": False,
                "detail": "Route geo-distance mapping unavailable",
            })

        # Factor 3: Model boundaries
        prediction_valid = 1000.0 <= ml_prediction <= 500000.0
        if prediction_valid:
            factors.append({
                "name": "Model Confidence",
                "available": True,
                "detail": "Prediction is within typical range",
            })
        else:
            factors.append({
                "name": "Model Confidence",
                "available": False,
                "detail": "Prediction falls outside regular parameters",
            })

        return {
            "score": score,
            "level": level,
            "factors": factors,
        }
