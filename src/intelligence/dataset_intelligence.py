"""
========================================================
Module: Dataset Intelligence
Purpose: Analyzes the training dataset (traveltripdata.csv)
         at runtime to extract data-driven traveller insights
         such as satisfaction ratings, transit scores,
         hotel feedback, and revisit intention.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# Map raw CSV survey headers to clean, maintainable field names
COLUMN_RENAME_MAP: Dict[str, str] = {
    "Which place did you visit recently?": "Place",
    "Which month did you travel?": "Month",
    "In Which season do you visited?": "Season",
    "What type of trip was it?": "Trip_Type",
    "How many days did the trip last?": "Days",
    "How would you describe your stay/hotel experience?": "Hotel_Quality",
    "What was your approximate total trip budget (in rupees)?": "Cost",
    "What was your primary mode of travel?": "Travel_Mode",
    "Rate the local transportation experience.": "Local_Trans_Rating",
    "How good were the sightseeing places?": "Sightseeing_Rating",
    "Overall, how satisfied were you with the trip?": "Satisfaction_Rating",
    "Would you like to revisit the same destination?": "Revisit_Intention",
    "If you travel again, what kind of experience would you prefer?": "Preferred_Experience",
}

# Minimum matching rows needed to generate specific filtered insights
_MIN_SAMPLE_SIZE: int = 3


class DatasetIntelligence:
    """Extracts aggregate statistics and traveller experience from dataset."""

    def __init__(self, csv_path: Optional[str] = None) -> None:
        """Initialize and load the travel dataset.

        Parameters
        ----------
        csv_path : str, optional
            Path to traveltripdata.csv. Resolved to root if None.
        """
        if csv_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "..", "..", "traveltripdata.csv")

        self._df: pd.DataFrame = self._load_and_clean(csv_path)

    def _clean_cost(self, value: Any) -> float:
        """Parse clean numeric cost from strings containing k, commas or ₹."""
        if isinstance(value, str):
            val_str = value.lower().replace(",", "").replace("₹", "").replace("approx", "").strip()
            if "k" in val_str:
                try:
                    return float(val_str.replace("k", "")) * 1000
                except ValueError:
                    return float("nan")
            nums = re.findall(r"\d+", val_str)
            if nums:
                return float(nums[0])
            return float("nan")
        return float(value) if value is not None else float("nan")

    def _load_and_clean(self, csv_path: str) -> pd.DataFrame:
        """Load and clean dataset columns to ensure correct types."""
        raw_df = pd.read_csv(csv_path)
        raw_df.columns = raw_df.columns.str.strip()
        df = raw_df.rename(columns=COLUMN_RENAME_MAP)

        # Standard cleanings
        if "Cost" in df.columns:
            df["Cost"] = df["Cost"].apply(self._clean_cost)
        if "Days" in df.columns:
            df["Days"] = pd.to_numeric(df["Days"], errors="coerce")

        # Convert rating columns to numeric
        rating_cols = ["Local_Trans_Rating", "Sightseeing_Rating", "Satisfaction_Rating"]
        for col in rating_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Standardise text values for consistency
        text_cols = ["Place", "Month", "Season", "Trip_Type", "Hotel_Quality",
                     "Travel_Mode", "Revisit_Intention", "Preferred_Experience"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()

        return df.dropna(subset=["Place", "Cost", "Days"])

    @staticmethod
    def _safe_mode(series: pd.Series) -> str:
        """Safe mode calculator that handles empty series gracefully."""
        if series.empty:
            return "N/A"
        modes = series.mode()
        return str(modes.iloc[0]) if not modes.empty else "N/A"

    def get_similar_trips(
        self,
        destination: str,
        month: Optional[str] = None,
        trip_type: Optional[str] = None,
        hotel_quality: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Find matching historical trips and compute summary statistics.

        Matches by destination first, then applies month, type, and quality filters.
        Falls back to destination-only if subset is smaller than _MIN_SAMPLE_SIZE.
        """
        dest_key = destination.strip().title()
        base = self._df[self._df["Place"] == dest_key]

        if base.empty:
            return self._empty_result()

        filtered = base.copy()
        if month:
            filtered = filtered[filtered["Month"] == month.strip().title()]
        if trip_type:
            filtered = filtered[filtered["Trip_Type"] == trip_type.strip().title()]
        if hotel_quality:
            filtered = filtered[filtered["Hotel_Quality"] == hotel_quality.strip().title()]

        # Fallback to destination-only baseline if sample size is too small
        if len(filtered) < _MIN_SAMPLE_SIZE:
            filtered = base

        return self._compute_statistics(filtered)

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate data to build historical insights dictionary."""
        # Mean scores
        avg_satisfaction = float(df["Satisfaction_Rating"].mean()) if "Satisfaction_Rating" in df.columns else 4.0
        avg_trans = float(df["Local_Trans_Rating"].mean()) if "Local_Trans_Rating" in df.columns else 4.0
        avg_sightseeing = float(df["Sightseeing_Rating"].mean()) if "Sightseeing_Rating" in df.columns else 4.0

        # Percentages
        revisit_yes = 0.0
        if "Revisit_Intention" in df.columns and len(df) > 0:
            revisit_yes = (df["Revisit_Intention"] == "Yes").sum() / len(df) * 100.0

        return {
            "similar_count": len(df),
            "average_budget": round(float(df["Cost"].mean()), 2),
            "min_budget": round(float(df["Cost"].min()), 2),
            "max_budget": round(float(df["Cost"].max()), 2),
            "average_duration": round(float(df["Days"].mean()), 1),
            "most_preferred_hotel": self._safe_mode(df["Hotel_Quality"]),
            "most_preferred_trip_type": self._safe_mode(df["Trip_Type"]),
            "most_used_travel_mode": self._safe_mode(df["Travel_Mode"]),
            "most_popular_month": self._safe_mode(df["Month"]),
            "most_popular_season": self._safe_mode(df["Season"]),
            "avg_satisfaction": round(avg_satisfaction, 1),
            "avg_transport_rating": round(avg_trans, 1),
            "avg_sightseeing_rating": round(avg_sightseeing, 1),
            "revisit_percentage": round(revisit_yes, 1),
            "preferred_experience": self._safe_mode(df["Preferred_Experience"]),
            "has_data": True,
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Default fallback dictionary when destination has no matches."""
        return {
            "similar_count": 0,
            "average_budget": 0.0,
            "min_budget": 0.0,
            "max_budget": 0.0,
            "average_duration": 0.0,
            "most_preferred_hotel": "N/A",
            "most_preferred_trip_type": "N/A",
            "most_used_travel_mode": "N/A",
            "most_popular_month": "N/A",
            "most_popular_season": "N/A",
            "avg_satisfaction": 0.0,
            "avg_transport_rating": 0.0,
            "avg_sightseeing_rating": 0.0,
            "revisit_percentage": 0.0,
            "preferred_experience": "N/A",
            "has_data": False,
        }

    def get_trending_destinations(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the most popular destinations grouped by count."""
        if self._df.empty:
            return []
        grouped = (
            self._df.groupby("Place")
            .agg(
                trip_count=("Place", "size"),
                avg_budget=("Cost", "mean"),
                avg_days=("Days", "mean"),
            )
            .sort_values("trip_count", ascending=False)
            .head(top_n)
            .reset_index()
        )
        return [
            {
                "destination": row["Place"],
                "trip_count": int(row["trip_count"]),
                "avg_budget": round(float(row["avg_budget"]), 2),
                "avg_days": round(float(row["avg_days"]), 1),
            }
            for _, row in grouped.iterrows()
        ]

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Compute top-level summary metrics for the dashboard."""
        if self._df.empty:
            return {}
        return {
            "total_trips": len(self._df),
            "unique_destinations": self._df["Place"].nunique(),
            "avg_budget": round(float(self._df["Cost"].mean()), 2),
            "most_popular_destination": self._safe_mode(self._df["Place"]),
            "most_popular_month": self._safe_mode(self._df["Month"]),
            "most_popular_season": self._safe_mode(self._df["Season"]),
            "most_popular_trip_type": self._safe_mode(self._df["Trip_Type"]),
            "avg_duration": round(float(self._df["Days"].mean()), 1),
        }
