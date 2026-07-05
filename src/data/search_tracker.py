"""
TripAI - Search Tracker
========================

High-level module that wraps ``TripDatabase`` to provide a simple
interface for tracking user searches and retrieving analytics.

This is the integration point used by ``app.py``.  It owns the
database lifecycle, provides one-call tracking, and exposes
analytics as Streamlit-ready data structures.

Usage in Streamlit
------------------
>>> from src.data.search_tracker import SearchTracker
>>> tracker = SearchTracker()
>>> tracker.track(source="Delhi", destination="Manali", ...)
>>> stats = tracker.get_dashboard_stats()
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.database import TripDatabase

log = logging.getLogger("tripai.search_tracker")

# ================================================================
# Singleton-style database holder so Streamlit's reruns don't open
# a new connection every cycle.
# ================================================================

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _BASE_DIR / "data" / "travel.db"

_db_instance: Optional[TripDatabase] = None


def _get_db() -> TripDatabase:
    """Return (or create) the shared database connection."""
    global _db_instance
    if _db_instance is None:
        _db_instance = TripDatabase(_DB_PATH)
        log.info("SearchTracker: database opened at %s", _DB_PATH)
    return _db_instance


# ================================================================
# SearchTracker
# ================================================================

class SearchTracker:
    """Stateless facade over :class:`TripDatabase` for search tracking.

    All methods are safe to call from a Streamlit rerun — the
    underlying database connection is reused across calls.
    """

    def __init__(self, db: Optional[TripDatabase] = None) -> None:
        self._db = db or _get_db()

    # ----------------------------------------------------------
    # Track a search
    # ----------------------------------------------------------

    def track(
        self,
        *,
        source: str,
        destination: str,
        month: str,
        duration_days: int,
        travel_mode: str,
        predicted_cost: float,
        season: Optional[str] = None,
        trip_type: Optional[str] = None,
        hotel_quality: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Log a single search event and return the row id.

        Parameters
        ----------
        source : str
            The city/state the user is travelling from.
        destination : str
            The place the user wants to visit.
        month : str
            Travel month (e.g. ``"December"``).
        duration_days : int
            Trip duration in days.
        travel_mode : str
            Preferred travel mode (e.g. ``"Flight"``, ``"Train"``).
        predicted_cost : float
            The ML-predicted budget for this search.
        season, trip_type, hotel_quality, session_id : optional
            Additional metadata.

        Returns
        -------
        int
            The database ``id`` of the newly created search row.
        """
        search_id = self._db.log_search(
            destination=destination,
            source_location=source,
            month=month,
            duration_days=duration_days,
            travel_mode=travel_mode,
            predicted_cost=predicted_cost,
            season=season,
            trip_type=trip_type,
            hotel_quality=hotel_quality,
            session_id=session_id,
        )
        log.info(
            "Tracked search #%d: %s -> %s (%s, %dd, %s, Rs.%.0f)",
            search_id, source, destination, month,
            duration_days, travel_mode, predicted_cost,
        )
        return search_id

    # ----------------------------------------------------------
    # Analytics — individual
    # ----------------------------------------------------------

    def most_searched_destinations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Top destinations by search count."""
        return self._db.most_searched_destinations(limit=limit)

    def most_searched_travel_modes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Travel modes ranked by popularity."""
        return self._db.most_searched_travel_modes(limit=limit)

    def average_predicted_budget(self) -> Dict[str, Any]:
        """Overall and per-destination budget averages."""
        return self._db.average_predicted_budget()

    def monthly_search_trends(self) -> List[Dict[str, Any]]:
        """Search volume and avg budget grouped by calendar month."""
        return self._db.monthly_search_trends()

    def recent_searches(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent search records."""
        return self._db.get_searches(limit=limit)

    def total_searches(self) -> int:
        """Total number of searches recorded."""
        stats = self._db.get_search_stats()
        return stats["total_searches"]

    # ----------------------------------------------------------
    # Dashboard — combined stats for the analytics page
    # ----------------------------------------------------------

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Return a single dict with all analytics needed by the
        Streamlit dashboard.

        Keys
        ----
        total_searches : int
        top_destinations : list
        top_travel_modes : list
        budget_stats : dict
        monthly_trends : list
        recent : list
        """
        return {
            "total_searches": self.total_searches(),
            "top_destinations": self.most_searched_destinations(limit=10),
            "top_travel_modes": self.most_searched_travel_modes(limit=10),
            "budget_stats": self.average_predicted_budget(),
            "monthly_trends": self.monthly_search_trends(),
            "recent": self.recent_searches(limit=15),
        }
