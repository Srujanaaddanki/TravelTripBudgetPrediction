"""
========================================================
Module: Route Service
Purpose: Fetches real route distance, duration, and polyline
         using OpenRouteService API.
         Falls back to haversine formula if API unavailable.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("tripai.route_service")

ORS_KEY = os.getenv("OPENROUTESERVICE_API_KEY", "")
ORS_URL  = "https://api.openrouteservice.org/v2/directions/driving-car"

# Cost per km by travel mode (INR)
MODE_COST_PER_KM: Dict[str, float] = {
    "Flight": 8.0,
    "Train":  1.5,
    "Bus":    1.8,
    "Car":    6.0,
    "Bike":   2.5,
}

# Approximate speed (km/h) per mode
MODE_SPEED: Dict[str, float] = {
    "Flight": 800.0,
    "Train":   55.0,
    "Bus":     40.0,
    "Car":     60.0,
    "Bike":    40.0,
}


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance between two lat/lng points (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


class RouteService:
    """Fetches route information between two locations."""

    def get_route(
        self,
        src_lat: float, src_lng: float,
        dst_lat: float, dst_lng: float,
        travel_mode: str = "Car",
    ) -> Dict[str, Any]:
        """Get distance, duration, and estimated cost for a route.

        Parameters
        ----------
        src_lat, src_lng : float
            Source coordinates.
        dst_lat, dst_lng : float
            Destination coordinates.
        travel_mode : str
            One of Flight / Train / Bus / Car / Bike.

        Returns
        -------
        dict
            distance_km, duration_hours, transport_cost, polyline, source
        """
        # Flight — just use straight-line distance
        if travel_mode == "Flight":
            dist_km = _haversine(src_lat, src_lng, dst_lat, dst_lng)
            dur_h   = dist_km / MODE_SPEED["Flight"]
            cost    = dist_km * MODE_COST_PER_KM["Flight"] * 2  # round trip
            return {
                "distance_km":    round(dist_km, 1),
                "duration_hours": round(dur_h, 2),
                "transport_cost": round(cost, -1),
                "polyline":       None,
                "source":         "Haversine (Flight)",
            }

        # Try OpenRouteService for road-based modes
        if ORS_KEY:
            try:
                result = self._call_ors(src_lat, src_lng, dst_lat, dst_lng)
                if result:
                    dist_km = result["distance_km"]
                    speed   = MODE_SPEED.get(travel_mode, 55.0)
                    dur_h   = dist_km / speed
                    cost    = dist_km * MODE_COST_PER_KM.get(travel_mode, 3.0) * 2
                    result["duration_hours"] = round(dur_h, 2)
                    result["transport_cost"] = round(cost, -1)
                    result["source"]         = "OpenRouteService"
                    return result
            except Exception as exc:
                log.warning("ORS call failed: %s", exc)

        # Fallback — haversine × 1.3 road-correction factor
        dist_km = _haversine(src_lat, src_lng, dst_lat, dst_lng) * 1.3
        speed   = MODE_SPEED.get(travel_mode, 55.0)
        dur_h   = dist_km / speed
        cost    = dist_km * MODE_COST_PER_KM.get(travel_mode, 3.0) * 2
        return {
            "distance_km":    round(dist_km, 1),
            "duration_hours": round(dur_h, 2),
            "transport_cost": round(cost, -1),
            "polyline":       None,
            "source":         "Haversine Fallback",
        }

    def _call_ors(
        self,
        src_lat: float, src_lng: float,
        dst_lat: float, dst_lng: float,
    ) -> Optional[Dict[str, Any]]:
        """Call OpenRouteService Directions API."""
        headers = {
            "Authorization": ORS_KEY,
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        body = {
            "coordinates": [
                [src_lng, src_lat],
                [dst_lng, dst_lat],
            ]
        }
        resp = requests.post(ORS_URL, json=body, headers=headers, timeout=8)
        if resp.status_code != 200:
            log.warning("ORS returned %s: %s", resp.status_code, resp.text[:200])
            return None

        data  = resp.json()
        route = data.get("routes", [{}])[0]
        summary = route.get("summary", {})
        dist_m  = summary.get("distance", 0)
        dur_s   = summary.get("duration", 0)

        # Extract polyline geometry
        geo = route.get("geometry", None)

        return {
            "distance_km":    round(dist_m / 1000, 1),
            "duration_hours": round(dur_s / 3600, 2),
            "transport_cost": 0,   # filled by caller
            "polyline":       geo,
        }

    def estimate_all_mode_costs(self, distance_km: float) -> Dict[str, Dict[str, Any]]:
        """Return cost + duration estimates for all 5 travel modes."""
        result = {}
        for mode, cost_per_km in MODE_COST_PER_KM.items():
            speed   = MODE_SPEED[mode]
            cost    = distance_km * cost_per_km * 2           # round-trip
            dur_h   = distance_km / speed
            h, m    = int(dur_h), int((dur_h % 1) * 60)
            result[mode] = {
                "one_way":       round(cost / 2, -1),
                "round_trip":    round(cost, -1),
                "duration_hours": round(dur_h, 2),
                "duration_str":  f"{h}h {m:02d}m",
                "speed_kmh":     speed,
            }
        return result
