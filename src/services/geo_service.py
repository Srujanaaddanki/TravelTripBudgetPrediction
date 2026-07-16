"""
========================================================
Module: Geo Service (Upgraded)
Purpose: Validates destination names and returns coordinates.
         Resolution order:
           1. Geoapify API (primary)
           2. Nominatim / OpenStreetMap (fallback)
           3. Safe unknown result
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("tripai.geo_service")

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "")
GEOAPIFY_URL     = "https://api.geoapify.com/v1/geocode/search"
NOMINATIM_URL    = "https://nominatim.openstreetmap.org/search"


class GeoService:
    """Validates and geocodes destination names.

    Uses Geoapify as the primary provider, with Nominatim
    (OpenStreetMap) as a free fallback when Geoapify fails
    or returns no results.
    """

    # ------------------------------------------------------------------
    # Primary: Geoapify
    # ------------------------------------------------------------------

    def validate_destination(self, name: str) -> Dict[str, Any]:
        """Validate a destination name and return its coordinates.

        Returns
        -------
        dict
            Keys: valid (bool), lat, lng, display_name, country,
                  state, confidence, source
        """
        if not name or not name.strip():
            return self._unknown_result(name)

        # Try Geoapify first
        result = self._call_geoapify(name)
        if result["valid"]:
            return result

        # Fallback to Nominatim
        log.info("Geoapify found nothing for '%s'. Trying Nominatim...", name)
        result = self._call_nominatim(name)
        if result["valid"]:
            return result

        return self._unknown_result(name)

    # ------------------------------------------------------------------
    # Geoapify
    # ------------------------------------------------------------------

    def _call_geoapify(self, name: str) -> Dict[str, Any]:
        """Call Geoapify geocoding API."""
        if not GEOAPIFY_API_KEY:
            log.warning("No GEOAPIFY_API_KEY — skipping Geoapify")
            return self._unknown_result(name)

        try:
            # First attempt: restrict to India
            params = {
                "text":   name,
                "apiKey": GEOAPIFY_API_KEY,
                "limit":  1,
                "lang":   "en",
                "filter": "countrycode:in",
            }
            resp = requests.get(GEOAPIFY_URL, params=params, timeout=6)
            data = resp.json()
            features = data.get("features", [])

            # Second attempt without country filter
            if not features:
                params.pop("filter", None)
                resp = requests.get(GEOAPIFY_URL, params=params, timeout=6)
                data = resp.json()
                features = data.get("features", [])

            if not features:
                return self._unknown_result(name)

            props  = features[0].get("properties", {})
            geo    = features[0].get("geometry", {})
            coords = geo.get("coordinates", [0, 0])

            return {
                "valid":        True,
                "lat":          float(coords[1]),
                "lng":          float(coords[0]),
                "display_name": props.get("formatted", name),
                "country":      props.get("country", "India"),
                "state":        props.get("state", ""),
                "confidence":   round(float(props.get("confidence", 0.5)), 2),
                "source":       "Geoapify",
            }
        except Exception as exc:
            log.warning("Geoapify call failed for '%s': %s", name, exc)
            return self._unknown_result(name)

    # ------------------------------------------------------------------
    # Fallback: Nominatim (OpenStreetMap)
    # ------------------------------------------------------------------

    def _call_nominatim(self, name: str) -> Dict[str, Any]:
        """Call Nominatim (OpenStreetMap) as a fallback geocoder.

        Nominatim requires a User-Agent header and recommends
        at least 1 second between requests to respect rate limits.
        """
        try:
            params = {
                "q":              f"{name}, India",
                "format":         "json",
                "limit":          1,
                "addressdetails": 1,
            }
            headers = {
                "User-Agent": "TripAI/2.0 (srujana.addanki@example.com)",
                "Accept-Language": "en",
            }
            resp = requests.get(
                NOMINATIM_URL, params=params, headers=headers, timeout=8
            )

            if resp.status_code != 200:
                log.warning("Nominatim returned HTTP %s", resp.status_code)
                return self._unknown_result(name)

            data = resp.json()
            if not data:
                # Retry without "India" constraint
                params["q"] = name
                resp = requests.get(
                    NOMINATIM_URL, params=params, headers=headers, timeout=8
                )
                data = resp.json()

            if not data:
                return self._unknown_result(name)

            hit  = data[0]
            addr = hit.get("address", {})
            state = (
                addr.get("state")
                or addr.get("county")
                or addr.get("region", "")
            )

            # Brief sleep to be polite to Nominatim servers
            time.sleep(0.5)

            return {
                "valid":        True,
                "lat":          float(hit["lat"]),
                "lng":          float(hit["lon"]),
                "display_name": hit.get("display_name", name),
                "country":      addr.get("country", "India"),
                "state":        state,
                "confidence":   0.70,
                "source":       "Nominatim",
            }
        except Exception as exc:
            log.warning("Nominatim call failed for '%s': %s", name, exc)
            return self._unknown_result(name)

    # ------------------------------------------------------------------
    # Autocomplete (used for suggestion widgets if needed)
    # ------------------------------------------------------------------

    def autocomplete(self, partial: str, limit: int = 5) -> list[Dict[str, Any]]:
        """Return place autocomplete suggestions via Geoapify.

        Returns
        -------
        list[dict]
            Each dict has keys: text, lat, lng, country, state.
        """
        if not GEOAPIFY_API_KEY or not partial.strip():
            return []

        try:
            params = {
                "text":   partial,
                "apiKey": GEOAPIFY_API_KEY,
                "limit":  limit,
                "lang":   "en",
                "filter": "countrycode:in",
                "type":   "city",
            }
            resp = requests.get(
                "https://api.geoapify.com/v1/geocode/autocomplete",
                params=params,
                timeout=5,
            )
            features = resp.json().get("features", [])
            suggestions = []
            for f in features:
                props = f.get("properties", {})
                coords = f.get("geometry", {}).get("coordinates", [0, 0])
                suggestions.append({
                    "text":    props.get("formatted", partial),
                    "lat":     float(coords[1]),
                    "lng":     float(coords[0]),
                    "country": props.get("country", "India"),
                    "state":   props.get("state", ""),
                })
            return suggestions
        except Exception as exc:
            log.warning("Autocomplete failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unknown_result(name: str) -> Dict[str, Any]:
        """Return a safe fallback when all geocoders are unavailable."""
        return {
            "valid":        False,
            "lat":          0.0,
            "lng":          0.0,
            "display_name": name,
            "country":      "Unknown",
            "state":        "",
            "confidence":   0.0,
            "source":       "None",
        }
