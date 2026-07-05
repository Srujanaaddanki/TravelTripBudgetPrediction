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
    ) -> Dict[str, Any]:
        """Determine final percentage score, rating level, and individual indicators.

        Parameters
        ----------
        dataset_insights : dict
            Historical trip data statistics from DatasetIntelligence.
        mode_comparison : dict
            Travel mode and distance comparison metrics.
        ml_prediction : float
            Budget predicted by ML model.

        Returns
        -------
        dict
            Keys: score (0-100), level (High/Medium/Low), and list of factor dicts.
        """
        factors: List[Dict[str, Any]] = []
        score = 0

        # Factor 1: Historical Data availability
        has_data = dataset_insights.get("has_data", False)
        similar_count = dataset_insights.get("similar_count", 0)

        if has_data and similar_count > 0:
            factors.append({
                "name": "Historical Data",
                "available": True,
                "detail": f"{similar_count} similar trips found in database",
            })
            score += 25
        else:
            factors.append({
                "name": "Historical Data",
                "available": False,
                "detail": "No matching trips in local training data",
            })

        # Factor 2: Route mapping verification
        distance_km = mode_comparison.get("distance_km", 0.0)
        if distance_km > 0.0:
            factors.append({
                "name": "Route Availability",
                "available": True,
                "detail": f"Route calculated successfully ({distance_km:,.1f} km)",
            })
            score += 25
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
                "detail": "Prediction is within typical historical range",
            })
            score += 25
        else:
            factors.append({
                "name": "Model Confidence",
                "available": False,
                "detail": "Prediction falls outside regular parameters",
            })

        # Factor 4: Dataset matching depth
        if similar_count >= 5:
            factors.append({
                "name": "Dataset Similarity",
                "available": True,
                "detail": f"Strong statistical base ({similar_count} similar entries)",
            })
            score += 25
        elif similar_count >= 2:
            factors.append({
                "name": "Dataset Similarity",
                "available": True,
                "detail": f"Moderate base ({similar_count} similar entries)",
            })
            score += 15
        else:
            factors.append({
                "name": "Dataset Similarity",
                "available": False,
                "detail": "Limited sample entries for similar parameters",
            })

        # Score level thresholds
        if score >= 75:
            level = "High Reliability"
        elif score >= 50:
            level = "Moderate Reliability"
        else:
            level = "Low Reliability"

        return {
            "score": score,
            "level": level,
            "factors": factors,
        }
