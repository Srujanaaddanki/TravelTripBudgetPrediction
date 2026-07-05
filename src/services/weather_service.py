"""
========================================================
Module: Weather Service
Purpose: Connects to the free Open-Meteo API to fetch
         real-time weather and temperature for destinations,
         handling fallbacks gracefully.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

log = logging.getLogger("tripai.weather_service")


class WeatherService:
    """Handles fetching real-time weather and temperature data for any destination."""

    def __init__(self, maps_service: Any) -> None:
        """Initialize the weather service with a maps service for coordinate lookup.

        Parameters
        ----------
        maps_service : MapsService
            Service used to fetch coordinates for a destination.
        """
        self._maps = maps_service

    def get_weather(self, destination: str) -> Dict[str, Any]:
        """Fetch current weather for a destination.

        Parameters
        ----------
        destination : str
            Name of destination.

        Returns
        -------
        dict
            Keys: temperature_c, description, humidity, feels_like, wind_speed.
        """
        try:
            # Step 1: Look up coordinates for the destination
            coords = self._maps.get_coordinates(destination)
            if coords:
                lat, lng = coords
                return self._call_open_meteo_api(lat, lng)
        except Exception as err:
            log.warning("Weather fetch failed for %s: %s", destination, err)

        return self._get_fallback_weather()

    def _call_open_meteo_api(self, lat: float, lng: float) -> Dict[str, Any]:
        """Fetch weather data from Open-Meteo API.

        Parameters
        ----------
        lat : float
            Latitude.
        lng : float
            Longitude.

        Returns
        -------
        dict
            API parsed weather properties.
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lng}"
            f"&current=temperature_2m,relative_humidity_2m,"
            f"apparent_temperature,weather_code,wind_speed_10m"
        )

        try:
            request = urllib.request.Request(url, method="GET")
            request.add_header("User-Agent", "TripAI/1.0")

            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))

            current = data.get("current", {})
            weather_code = current.get("weather_code", 0)

            return {
                "temperature_c": current.get("temperature_2m", 25.0),
                "description": self._weather_code_to_text(weather_code),
                "humidity": current.get("relative_humidity_2m", 50),
                "feels_like": current.get("apparent_temperature", 25.0),
                "wind_speed": current.get("wind_speed_10m", 10.0),
                "source": "Open-Meteo API",
            }
        except (urllib.error.URLError, json.JSONDecodeError, Exception) as err:
            log.warning("Open-Meteo API call failed: %s", err)
            return self._get_fallback_weather()

    @staticmethod
    def _weather_code_to_text(code: int) -> str:
        """Convert WMO weather code to descriptive text."""
        descriptions = {
            0: "Clear Sky ☀️",
            1: "Mainly Clear 🌤️",
            2: "Partly Cloudy ⛅",
            3: "Overcast ☁️",
            45: "Foggy 🌫️",
            48: "Icy Fog 🌫️",
            51: "Light Drizzle 🌦️",
            53: "Moderate Drizzle 🌦️",
            55: "Dense Drizzle 🌧️",
            61: "Slight Rain 🌧️",
            63: "Moderate Rain 🌧️",
            65: "Heavy Rain 🌧️",
            71: "Slight Snow ❄️",
            73: "Moderate Snow ❄️",
            75: "Heavy Snow ❄️",
            80: "Rain Showers 🌦️",
            81: "Moderate Rain Showers 🌧️",
            82: "Violent Rain Showers ⛈️",
            95: "Thunderstorm ⛈️",
            96: "Thunderstorm with Hail ⛈️",
        }
        return descriptions.get(code, "Partly Cloudy ⛅")

    def _get_fallback_weather(self) -> Dict[str, Any]:
        """Provide reasonable fallback values if API call fails."""
        return {
            "temperature_c": 28.0,
            "description": "Data unavailable — check local forecast ⛅",
            "humidity": 55,
            "feels_like": 30.0,
            "wind_speed": 12.0,
            "source": "Estimated",
        }
