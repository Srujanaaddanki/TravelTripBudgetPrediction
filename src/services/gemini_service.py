"""
========================================================
Module: Gemini Service (Upgraded)
Purpose: Uses Google Gemini API to dynamically generate:
           - Packing checklist (destination + season + mode aware)
           - Pre-travel checklist (permits, medical, etc.)
           - Seasonal / destination tips
           - Health suggestions
           - Safety tips
           - Local recommendations
           - Unknown place resolution (with lat/lng in response)
         Falls back to existing destination_knowledge.py
         when API key is missing or call fails.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("tripai.gemini_service")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class GeminiService:
    """Generates dynamic travel intelligence using Google Gemini."""

    def __init__(self) -> None:
        self._model      = None
        self._available  = False
        if GEMINI_API_KEY:
            self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialise the Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._model     = genai.GenerativeModel("gemini-1.5-flash")
            self._available = True
            log.info("Gemini API initialised successfully.")
        except Exception as exc:
            log.warning("Gemini init failed: %s", exc)
            self._available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_destination_intelligence(
        self,
        destination: str,
        month: str,
        travel_mode: str = "Car",
        duration_days: int = 5,
        season: str = "Summer",
        trip_type: str = "General",
        weather: Optional[Dict[str, Any]] = None,
        country: Optional[str] = None,
        altitude: Optional[float] = None,
        permits_required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate full destination intelligence from Gemini.

        Returns
        -------
        dict with keys:
            packing_checklist, pre_travel_checklist, seasonal_tips,
            health_suggestions, safety_tips, local_recommendations, source
        """
        if self._available:
            result = self._call_gemini(
                destination=destination,
                month=month,
                travel_mode=travel_mode,
                duration_days=duration_days,
                season=season,
                trip_type=trip_type,
                weather=weather,
                country=country,
                altitude=altitude,
                permits_required=permits_required,
            )
            if result:
                return result

        # Fallback to existing knowledge base
        return self._fallback_intelligence(
            destination=destination,
            month=month,
            weather=weather,
            travel_mode=travel_mode,
            trip_type=trip_type,
            country=country,
            altitude=altitude,
            permits_required=permits_required,
        )

    def resolve_unknown_destination_metadata(
        self, destination: str, state: str = "", country: str = ""
    ) -> Dict[str, Any]:
        """Ask Gemini for state, country, altitude, tourism category, and weather profile of a destination."""
        fallback = {
            "country": country or "India",
            "state": state or "",
            "altitude": 0.0,
            "tourism_category": "general",
            "weather_profile": "temperate",
            "population_profile": "medium",
        }
        if not self._available:
            return fallback
        
        prompt = f"""
You are a geography and travel classification assistant.
For the destination "{destination}" (State: {state or "unknown"}, Country: {country or "unknown"}), determine the following details.

Return ONLY a valid JSON object (no markdown, no extra text, no ```json wrapper):
{{
  "country": "Country name",
  "state": "State name",
  "altitude": 1200.0,
  "tourism_category": "One of: Beach, Temple, Hill Station, Metropolitan City, Rural",
  "weather_profile": "One of: Tropical, Himalayan, Temperate, Arid, Monsoon",
  "population_profile": "One of: high, medium, low (classification of population density/city type)"
}}
"""
        try:
            response = self._model.generate_content(prompt)
            raw_text = self._strip_markdown(response.text.strip())
            data = json.loads(raw_text)
            
            # Normalize keys and values
            res = {
                "country": data.get("country", country or "India"),
                "state": data.get("state", state or ""),
                "altitude": float(data.get("altitude", 0.0)),
                "tourism_category": data.get("tourism_category", "general"),
                "weather_profile": data.get("weather_profile", "temperate"),
                "population_profile": data.get("population_profile", "medium"),
            }
            return res
        except Exception as exc:
            log.warning("Failed to classify destination %s: %s", destination, exc)
            return fallback

    def suggest_alternative_destination(self, destination: str) -> Dict[str, Any]:
        """Identify an alternative / corrected destination using Gemini.

        Enhanced to return latitude + longitude so the caller can skip
        a second geocoding round-trip when Gemini identifies the place.

        Returns
        -------
        dict
            Keys: valid, suggested_destination, latitude, longitude,
                  confidence_score, explanation, village, district,
                  alternative_names
        """
        if not self._available:
            return self._unknown_suggestion()

        prompt = f"""
You are an expert Indian travel and geography advisor.

User searched for: "{destination}"

This destination could NOT be found via standard maps APIs.
Investigate and identify:
1. Is it a real place but misspelled? (e.g. "kukanet" → "Kukanet Nature Awareness Camp, Punjab")
2. A nearby village / tourist spot / forest reserve?
3. A nearby major city or district?
4. A famous destination matching this name?
5. Alternative spellings or local names?

Return ONLY a valid JSON object — no markdown, no extra text, no ```json wrapper:
{{
  "valid": true,
  "suggested_destination": "Full corrected place name with state",
  "latitude": 30.12,
  "longitude": 76.45,
  "confidence_score": 0-100,
  "explanation": "Why this suggestion was made",
  "village": "Village name if applicable",
  "district": "District name",
  "state": "State name",
  "country": "Country name",
  "altitude": 1200.0,
  "tourism_category": "One of: Beach, Temple, Hill Station, Metropolitan City, Rural",
  "weather_profile": "One of: Tropical, Himalayan, Temperate, Arid, Monsoon",
  "population_profile": "One of: high, medium, low",
  "alternative_names": ["name1", "name2"]
}}

If you cannot identify ANY real or similar place, set "valid" to false and
set latitude/longitude to 0.
"""
        try:
            response  = self._model.generate_content(prompt)
            raw_text  = response.text.strip()
            raw_text  = self._strip_markdown(raw_text)
            result    = json.loads(raw_text)
            result.setdefault("latitude",          0.0)
            result.setdefault("longitude",         0.0)
            result.setdefault("village",           "")
            result.setdefault("district",          "")
            result.setdefault("state",             "")
            result.setdefault("country",           "India")
            result.setdefault("altitude",          0.0)
            result.setdefault("tourism_category",  "general")
            result.setdefault("weather_profile",   "temperate")
            result.setdefault("population_profile", "medium")
            result.setdefault("alternative_names", [])
            return result
        except Exception as exc:
            log.warning("Gemini suggest_alternative failed for '%s': %s", destination, exc)
            return self._unknown_suggestion()

    # ------------------------------------------------------------------
    def _call_gemini(
        self,
        destination: str,
        month: str,
        travel_mode: str,
        duration_days: int,
        season: str,
        trip_type: str,
        weather: Optional[Dict[str, Any]] = None,
        country: Optional[str] = None,
        altitude: Optional[float] = None,
        permits_required: Optional[bool] = None,
    ) -> Dict[str, Any] | None:
        """Call Gemini API and parse the JSON response."""
        # Get destination-specific hints from the rules engine
        try:
            from src.intelligence.destination_rules import get_prompt_hints
            dest_hints = get_prompt_hints(destination)
        except Exception:
            dest_hints = ""

        hint_block = f"\nCRITICAL DESTINATION HINTS: {dest_hints}" if dest_hints else ""

        weather_desc = weather.get("description", "Not available") if weather else "Not available"
        weather_temp = weather.get("temperature_c", "N/A") if weather else "N/A"

        prompt = f"""You are an expert Indian and international travel advisor.
Generate HIGHLY SPECIFIC travel intelligence for the following trip.
Do NOT give generic packing items — tailor EVERYTHING to the destination.

Destination: {destination}
Country: {country or "India"}
Altitude: {f"{altitude} meters" if altitude else "Low altitude"}
Permits Required: {"Yes" if permits_required else "No"}
Travel Month: {month} (Season: {season})
Weather: {weather_desc} ({weather_temp}°C)
Duration: {duration_days} Days
Travel Mode: {travel_mode}
Trip Type: {trip_type}{hint_block}

Return ONLY a valid JSON object with these exact keys (no extra text, no markdown):
{{
  "packing_checklist": ["item1", "item2", ...],
  "pre_travel_checklist": ["item1", "item2", ...],
  "seasonal_tips": ["tip1", "tip2", ...],
  "health_suggestions": ["suggestion1", ...],
  "safety_tips": ["tip1", "tip2", ...],
  "local_recommendations": ["recommendation1", ...]
}}

Rules:
1. Packing checklist must be tailored for {destination}, {season} ({month}), {duration_days} days.
2. If travel mode is Bike: MUST include helmet, repair kit, fuel planning, rain cover.
3. If travel mode is Train: MUST include tickets, snacks, blanket, power bank, valid ID.
4. If travel mode is Flight: MUST include passport/ID, boarding pass, baggage rules.
5. Kedarnath/Badrinath/Amarnath: MUST include Yatra permit, medical certificate,
   trekking shoes, altitude sickness medicine, thermal wear, rain jacket.
6. Ladakh/Leh/Spiti: MUST include inner line permit, altitude medication (Diamox),
   oxygen can, heavy down jacket.
7. Tirupati/Balaji: MUST include temple dress code, darshan ticket booking, ID proof,
   no shorts/sleeveless, cash for offerings.
8. Goa/beach destination: MUST include swimwear, sunscreen SPF50+, flip-flops, hat.
9. Winter destinations (Manali/Shimla): MUST include thermals, heavy jacket, snow boots.
10. International destinations: MUST include visa requirements, travel insurance,
    foreign currency, passport validity.
11. Safety tips MUST include road safety if travel mode is Bike or Car.
12. Pre-travel checklist must list ACTUAL permit/booking requirements specific to destination.

IMPORTANT: Your response must be destination-specific. Do not repeat the same items
for Kedarnath and Goa. Each destination has unique requirements."""
        try:
            response = self._model.generate_content(prompt)
            raw_text = self._strip_markdown(response.text.strip())
            data     = json.loads(raw_text)

            required_keys = [
                "packing_checklist", "pre_travel_checklist", "seasonal_tips",
                "health_suggestions", "safety_tips", "local_recommendations",
            ]
            for k in required_keys:
                data.setdefault(k, [])

            data["source"] = "Gemini AI"
            return data

        except Exception as exc:
            log.warning(
                "Gemini response parse error for '%s': %s", destination, exc
            )
            return None

    def _fallback_intelligence(
        self,
        destination: str,
        month: str,
        weather: Optional[Dict[str, Any]] = None,
        travel_mode: str = "Car",
        trip_type: str = "General",
        country: Optional[str] = None,
        altitude: Optional[float] = None,
        permits_required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Fallback using destination_rules.py (destination-specific) then
        destination_knowledge.py. Always returns specific, not generic, content."""
        # Primary: use destination rules engine (destination-specific)
        try:
            from src.intelligence.destination_rules import get_destination_checklist
            rules = get_destination_checklist(
                destination=destination,
                month=month,
                weather=weather,
                travel_mode=travel_mode,
                trip_type=trip_type,
                country=country,
                altitude=altitude,
                permits_required=permits_required,
            )
            packing   = rules["packing"]
            pretravel = rules["pretravel"]

            # Supplement with seasonal tips from knowledge base if available
            try:
                from src.intelligence.destination_knowledge import get_destination_info
                info = get_destination_info(destination)
                local_recs = info.get("local_foods", ["Try local cuisine"]) if info else ["Try local cuisine"]
            except Exception:
                local_recs = ["Explore local markets", "Try regional cuisine"]

            return {
                "packing_checklist": packing,
                "pre_travel_checklist": pretravel,
                "seasonal_tips": [
                    f"Check local weather forecast before travelling in {month}.",
                    "Book accommodation well in advance during peak season.",
                    "Carry a rain jacket if travelling during monsoon months (June-September).",
                ],
                "health_suggestions": [
                    "Carry personal medicines and prescriptions.",
                    "Stay hydrated throughout your journey.",
                    "Consult a doctor for altitude precautions if visiting hill stations.",
                ],
                "safety_tips": [
                    "Keep emergency contact numbers saved offline.",
                    "Avoid travelling alone in unfamiliar areas at night.",
                    "Keep digital copies of all travel documents on cloud/email.",
                ],
                "local_recommendations": local_recs,
                "source": "Destination Rules Engine (Offline Fallback)",
            }
        except Exception as exc:
            log.warning("Destination rules fallback failed: %s", exc)
            return self._minimal_fallback()

    @staticmethod
    def _minimal_fallback() -> Dict[str, Any]:
        """Last-resort static fallback."""
        return {
            "packing_checklist": [
                "Comfortable Walking Shoes", "Sunscreen & Sunglasses",
                "Basic First-Aid Kit", "Water Bottle", "Power Bank",
                "Travel Documents",
            ],
            "pre_travel_checklist": [
                "Valid ID Proof", "Travel Tickets", "Hotel Booking",
                "Cash & Cards", "Emergency Contacts", "Medicines",
            ],
            "seasonal_tips": [
                "Check weather forecasts before packing.",
                "Book accommodation in advance.",
            ],
            "health_suggestions": [
                "Carry personal medications.", "Stay hydrated.",
            ],
            "safety_tips": [
                "Keep copies of all documents.", "Share itinerary with family.",
            ],
            "local_recommendations": [
                "Explore local markets.", "Try regional cuisine.",
            ],
            "source": "Static Fallback",
        }

    @staticmethod
    def _unknown_suggestion() -> Dict[str, Any]:
        """Return a safe empty result when Gemini is unavailable."""
        return {
            "valid":               False,
            "suggested_destination": "",
            "latitude":            0.0,
            "longitude":           0.0,
            "confidence_score":    0,
            "explanation":         "",
            "village":             "",
            "district":            "",
            "state":               "",
            "alternative_names":   [],
        }

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Strip Markdown code fences from Gemini responses."""
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
            elif len(parts) == 2:
                text = parts[1]
            if text.startswith("json"):
                text = text[4:].strip()
        return text.strip()
