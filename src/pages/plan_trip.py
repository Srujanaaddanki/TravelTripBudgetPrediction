"""
========================================================
Page: Plan Trip — Self-Learning AI Travel Intelligence
Purpose: Full 3-column dashboard.
         LEFT:   Form inputs.
         CENTER: Smart budget + metrics + map + comparison.
         RIGHT:  AI-powered checklists + tips.

Destination Resolution Hierarchy (7 Steps):
  Step 1 → SQLite cache check
  Step 2 → Destination aliases (typo correction)
  Step 3 → Exact encoder match
  Step 4 → RapidFuzz fuzzy match  (clickable suggestions)
  Step 5 → Geoapify geocode validation
  Step 6 → Nominatim fallback (via GeoService)
  Step 7 → Gemini AI resolution

Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.components.hero import render_budget_hero_card
from src.components.charts import (
    render_budget_donut,
    render_breakdown_horizontal_bar,
    render_mode_comparison_chart,
)
from src.components.maps import render_route_map, render_map_placeholder
from src.components.checklist import render_packing_checklist, render_pretravel_checklist
from src.services.geo_service import GeoService
from src.services.database_service import DestinationCache
from src.data.destination_aliases import resolve_alias
from src.services.report_exporter import generate_pdf_report
from src.intelligence.route_alternatives import (
    get_mode_availability,
    get_alternate_route,
    format_alternate_route_html,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTHS_ORDERED = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTH_SEASON_MAP = {
    "january": "winter",  "february": "winter", "march": "spring",
    "april":   "summer",  "may":      "summer",  "june":  "summer",
    "july":    "rainy",   "august":   "rainy",   "september": "rainy",
    "october": "autumn",  "november": "autumn",  "december": "winter",
}

_geo_service = GeoService()
_dest_cache  = DestinationCache()


# ── RapidFuzz helper ──────────────────────────────────────────────────────────

def _rapidfuzz_suggestions(
    query: str, known_places: List[str], limit: int = 4, threshold: float = 55.0
) -> List[Tuple[str, float]]:
    """Return the top fuzzy matches from the known encoder classes.

    Returns
    -------
    list[(place_name, score)]  sorted by score descending.
    """
    try:
        from rapidfuzz import process, fuzz
        results = process.extract(
            query,
            known_places,
            scorer=fuzz.WRatio,
            limit=limit,
        )
        return [(r[0], r[1]) for r in results if r[1] >= threshold]
    except ImportError:
        # Graceful fallback to difflib if rapidfuzz not installed
        import difflib
        matches = difflib.get_close_matches(query, known_places, n=limit, cutoff=0.55)
        return [(m, 70.0) for m in matches]


# ── Destination resolution helpers ───────────────────────────────────────────

def _resolve_destination(
    user_input: str,
    encoder_classes: List[str],
) -> Dict[str, Any]:
    """Run the 7-step destination resolution hierarchy.

    Returns
    -------
    dict with keys:
        matched_dest  : str   — best encoder-matched destination for ML
        display_dest  : str   — human-readable destination
        match_tier    : str   — one of: cache, alias, exact, fuzzy,
                                  geo_validated, nominatim, gemini, not_found
        geo_result    : dict  — Geoapify/Nominatim result if available
        suggestions   : list  — RapidFuzz suggestions if match_tier='show_suggestions'
        dst_coords    : tuple — (lat, lng) if geocoded
        is_known      : bool  — True when destination is in encoder dataset
    """
    key    = user_input.strip().lower()
    known  = [c.lower() for c in encoder_classes]

    # ── Step 1: SQLite Cache check ────────────────────────────────────
    cached = _dest_cache.get_cached(key)
    if cached:
        actual  = cached.get("actual_destination", key)
        lat     = cached.get("latitude", 0.0)
        lng     = cached.get("longitude", 0.0)
        # Check if the cached actual_destination is in the encoder
        enc_match = actual if actual in known else (
            _rapidfuzz_suggestions(actual, known, limit=1, threshold=70)
        )
        if isinstance(enc_match, list) and enc_match:
            enc_match = enc_match[0][0]
        elif not isinstance(enc_match, str):
            enc_match = known[0]
        return {
            "matched_dest": enc_match,
            "display_dest": cached.get("actual_destination", user_input).title(),
            "match_tier":   "cache",
            "geo_result":   None,
            "suggestions":  [],
            "dst_coords":   (lat, lng) if lat != 0.0 and lng != 0.0 else None,
            "is_known":     True,
        }

    # ── Step 2: Alias check ───────────────────────────────────────────
    alias = resolve_alias(user_input)
    if alias:
        alias_key = alias.lower()
        if alias_key in known:
            return {
                "matched_dest": alias_key,
                "display_dest": alias.title(),
                "match_tier":   "alias",
                "geo_result":   None,
                "suggestions":  [],
                "dst_coords":   None,
                "is_known":     True,
            }
        # Alias found but not in encoder — continue resolution with alias name
        key = alias_key

    # ── Step 3: Exact encoder match ───────────────────────────────────
    if key in known:
        return {
            "matched_dest": key,
            "display_dest": key.title(),
            "match_tier":   "exact",
            "geo_result":   None,
            "suggestions":  [],
            "dst_coords":   None,
            "is_known":     True,
        }

    # ── Step 4: RapidFuzz fuzzy match ────────────────────────────────
    fuzzy_hits = _rapidfuzz_suggestions(key, known, limit=4, threshold=62)

    if len(fuzzy_hits) == 1 and fuzzy_hits[0][1] >= 85:
        # High-confidence single match → auto-select
        return {
            "matched_dest": fuzzy_hits[0][0],
            "display_dest": fuzzy_hits[0][0].title(),
            "match_tier":   "fuzzy",
            "geo_result":   None,
            "suggestions":  [],
            "dst_coords":   None,
            "is_known":     True,
        }

    if fuzzy_hits:
        # Multiple / low-confidence matches → show suggestions to user
        return {
            "matched_dest": fuzzy_hits[0][0],
            "display_dest": fuzzy_hits[0][0].title(),
            "match_tier":   "show_suggestions",
            "geo_result":   None,
            "suggestions":  fuzzy_hits,
            "dst_coords":   None,
            "is_known":     True,
        }

    # ── Step 5: Geoapify + Step 6: Nominatim fallback ─────────────────
    geo_result = _geo_service.validate_destination(user_input)

    if geo_result["valid"]:
        # Real place — find nearest encoder proxy
        proxy_hits = _rapidfuzz_suggestions(key, known, limit=1, threshold=20)
        proxy      = proxy_hits[0][0] if proxy_hits else known[0]
        coords     = (geo_result["lat"], geo_result["lng"])
        geo_src    = geo_result.get("source", "Geoapify")
        return {
            "matched_dest": proxy,
            "display_dest": geo_result.get("display_name", user_input).title(),
            "match_tier":   "geo_validated",
            "geo_result":   geo_result,
            "suggestions":  [],
            "dst_coords":   coords,
            "is_known":     False,
        }

    # ── Step 7: Gemini AI resolution ──────────────────────────────────
    return {
        "matched_dest": known[0],
        "display_dest": user_input.title(),
        "match_tier":   "needs_gemini",
        "geo_result":   None,
        "suggestions":  [],
        "dst_coords":   None,
        "is_known":     False,
    }


# ── Form renderer ─────────────────────────────────────────────────────────────

def _render_sidebar_form(encoders: Dict[str, Any]) -> Dict[str, Any] | None:
    """Render left sidebar form with standard inputs inside st.form."""
    trip_types  = [x.title() for x in encoders["Trip_Type"].classes_]
    hotel_types = [x.title() for x in encoders["Hotel_Quality"].classes_]

    st.markdown("""
    <div style="padding:10px 16px 0;">
      <div style="
        font-family:'Outfit',sans-serif;
        font-size:18px; font-weight:700;
        color:#F1F5F9;
        display:flex; align-items:center; gap:10px;
        margin-bottom:4px;
      ">
        🗺️ Plan Your Trip
      </div>
      <div style="font-size:12px; color:#94A3B8; margin-bottom:12px;">
        Enter your trip details for AI-powered budget prediction
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("trip_search_form"):
        # FROM
        st.markdown('<div class="form-label">FROM CITY</div>', unsafe_allow_html=True)
        source_city = st.text_input(
            "from_input", label_visibility="collapsed",
            placeholder="Origin city (e.g. Vijayawada)",
            key="from_city",
        )

        # TO — read destination_override if a suggestion was selected
        _dest_override = st.session_state.pop("destination_override", None)
        if _dest_override:
            st.session_state["to_city"] = _dest_override
        st.markdown('<div class="form-label" style="margin-top:8px;">DESTINATION</div>', unsafe_allow_html=True)
        dest_city = st.text_input(
            "to_input", label_visibility="collapsed",
            placeholder="Destination (e.g. Goa, Manali)",
            key="to_city",
        )


        col_m, col_d = st.columns(2)
        with col_m:
            st.markdown('<div class="form-label">MONTH</div>', unsafe_allow_html=True)
            month = st.selectbox(
                "month_sel", MONTHS_ORDERED,
                label_visibility="collapsed", key="sel_month",
            )
        with col_d:
            st.markdown('<div class="form-label">DURATION</div>', unsafe_allow_html=True)
            days = st.selectbox(
                "duration_sel",
                [f"{i} Days" for i in range(1, 31)],
                index=4, label_visibility="collapsed", key="sel_days",
            )

        col_tt, col_tm = st.columns(2)
        with col_tt:
            st.markdown('<div class="form-label">TRIP TYPE</div>', unsafe_allow_html=True)
            trip_type = st.selectbox(
                "trip_type_sel", trip_types,
                label_visibility="collapsed", key="sel_trip_type",
            )
        with col_tm:
            st.markdown('<div class="form-label">TRAVEL MODE</div>', unsafe_allow_html=True)
            travel_mode = st.selectbox(
                "mode_sel",
                ["Train", "Flight", "Bus", "Car", "Bike"],
                label_visibility="collapsed", key="sel_mode",
            )

        st.markdown(
            '<div class="form-label" style="margin-top:8px;">HOTEL QUALITY</div>',
            unsafe_allow_html=True,
        )
        hotel = st.selectbox(
            "hotel_sel", hotel_types,
            label_visibility="collapsed", key="sel_hotel",
        )

        submitted = st.form_submit_button("Predict Budget", use_container_width=True)

    if submitted:
        days_int = int(days.split()[0])
        return {
            "source":      source_city.strip(),
            "destination": dest_city.strip(),
            "month":       month,
            "days":        days_int,
            "trip_type":   trip_type,
            "travel_mode": travel_mode,
            "hotel":       hotel,
        }
    return None


# ── Smart budget metrics ──────────────────────────────────────────────────────

def _render_smart_budget_metrics(
    ml_pred: float,
    historical_avg: float,
    transport_cost: float,
    smart_budget: float,
    confidence: Dict[str, Any],
    is_known: bool,
) -> None:
    """Render the 5-metric Smart Budget Verification Engine row."""
    st.markdown(
        '<div class="section-title">&#x1F9E0; Smart Budget Verification Engine</div>',
        unsafe_allow_html=True,
    )

    diff     = smart_budget - ml_pred
    diff_pct = (abs(diff) / ml_pred * 100) if ml_pred > 0 else 0
    conf_sc  = confidence.get("score", 0)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "ML Prediction", f"Rs.{int(ml_pred):,}",
            help=(
                "Random Forest model trained on 50,000+ real Indian trip records. "
                "Accounts for destination, month, season, hotel, trip type and duration."
            )
        )
    with col2:
        hist_label = f"Rs.{int(historical_avg):,}" if historical_avg > 0 else "N/A"
        st.metric(
            "Historical Avg", hist_label,
            help=(
                "Average actual spend by past travelers to this destination with similar settings. "
                "Used to validate and blend with the ML prediction."
            )
        )
    with col3:
        st.metric(
            "Transport Cost", f"Rs.{int(transport_cost):,}",
            help=(
                "Estimated round-trip transport cost breakdown includes: "
                "Train/Flight/Bus fare + local taxi transfers + last-mile transport. "
                "Based on distance and selected travel mode."
            )
        )
    with col4:
        formula = (
            "35% ML + 25% Historical + 20% Transport + 20% Duration"
            if is_known
            else "50% ML + 30% Transport + 20% Duration (New Destination)"
        )
        st.metric(
            "Smart Budget", f"Rs.{int(smart_budget):,}",
            delta=f"{'up' if diff >= 0 else 'down'} Rs.{int(abs(diff)):,} ({diff_pct:.1f}%)",
            delta_color="inverse" if diff > 0 else "normal",
            help=f"Final blended budget formula: {formula}",
        )
    with col5:
        level_icon = "[HI]" if conf_sc >= 75 else "[MED]" if conf_sc >= 50 else "[LOW]"
        st.metric(
            "Confidence", f"{conf_sc}%",
            delta=f"{level_icon} {confidence.get('level', '')}",
            help="Confidence is based on: data availability for this destination, route quality, and model accuracy.",
        )


# ── Travel quote ─────────────────────────────────────────────────────────────

def _render_travel_quote() -> None:
    st.markdown("""
    <div style="margin:16px; padding:16px;
      background:linear-gradient(135deg,rgba(79,70,229,0.1),rgba(147,51,234,0.05));
      border:1px solid rgba(79,70,229,0.2); border-radius:12px;">
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
        <span style="color:#E879F9;">♥</span>
        <span style="font-family:'Outfit',sans-serif;font-size:13px;font-weight:700;
          background:linear-gradient(135deg,#4F46E5,#9333EA);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;">Love with Travel</span>
      </div>
      <div style="font-size:12px;color:#94A3B8;line-height:1.6;">
        Travel is the only thing you buy that makes you
        <strong style="color:#F1F5F9;">richer.</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Right panel ───────────────────────────────────────────────────────────────

def _render_right_panel_welcome() -> None:
    """Show the 'AI Travel Assistant Ready' card before prediction."""
    st.markdown("""
    <div class="checklist-card" style="text-align:center;padding:28px 18px;margin-bottom:16px;">
      <div style="font-size:36px;margin-bottom:14px;">🤖</div>
      <div style="font-family:'Outfit',sans-serif;font-weight:700;color:#F1F5F9;font-size:15px;margin-bottom:8px;">
        AI Travel Assistant Ready
      </div>
      <div style="font-size:12px;color:#94A3B8;line-height:1.6;">
        Enter your trip details on the left to generate real-time route maps,
        smart budgets, weather forecasts, and custom AI recommendations.
      </div>
    </div>
    """, unsafe_allow_html=True)




def _render_right_panel_results(report: Dict[str, Any], form_data: Dict[str, Any]) -> None:
    gemini_intel = report.get("gemini", {})
    confidence   = report.get("confidence", {})
    weather      = report.get("weather", {})

    # 1. Confidence Summary Card
    conf_score = confidence.get("score", 0)
    level_icon = "🟢" if conf_score >= 75 else "🟡" if conf_score >= 50 else "🔴"
    st.markdown(f"""
    <div class="checklist-card" style="margin-bottom:12px; border-left: 4px solid #7C3AED; padding: 14px 16px;">
      <div class="checklist-title" style="margin-bottom: 6px;"><span>🎯</span> Destination Confidence</div>
      <div style="font-size:22px; font-weight:800; color:var(--text-primary); margin-bottom:2px; line-height:1.1;">
        {conf_score}%
      </div>
      <div style="font-size:11px; color:var(--text-secondary); font-weight:600;">
        {confidence.get('level', 'N/A')} {level_icon}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Live Weather Card
    if weather:
        temp  = weather.get("temperature_c", "N/A")
        desc  = weather.get("description", "N/A")
        humid = weather.get("humidity", "N/A")
        st.markdown(f"""
        <div class="checklist-card" style="margin-bottom:12px; padding: 14px 16px;">
          <div class="checklist-title" style="margin-bottom: 6px;"><span>🌤️</span> Live Weather</div>
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div style="font-size:22px; font-weight:800; color:var(--text-primary);">{temp}°C</div>
            <div style="font-size:12px; color:var(--text-secondary); font-weight:600; text-align:right;">{desc}</div>
          </div>
          <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">
            Humidity: {humid}% | Feels: {weather.get('feels_like', 'N/A')}°C
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Checklists
    packing = gemini_intel.get("packing_checklist", [])
    if not packing:
        packing = st.session_state.get("packing_tips", [])
    render_packing_checklist(packing)

    permits = gemini_intel.get("pre_travel_checklist", [])
    render_pretravel_checklist(permits)


def render_plan_trip_page(
    model: Any,
    encoders: Dict[str, Any],
    maps_service: Any,
    travel_engine: Any,
    tracker: Any,
) -> None:
    """Render the full Plan Trip dashboard page."""
    if model is None:
        st.error("⚠️ Model files not found. Run `python train_model.py` first.")
        st.stop()

    encoder_classes = list(encoders["Place"].classes_)

    # ── ROW 2: 3-Column Layout ─────────────────────────────────────────
    col_left, col_center, col_right = st.columns([1.1, 2.2, 1.1], gap="medium")

    # LEFT Panel: Search Form
    with col_left:
        new_form_data = _render_sidebar_form(encoders)

    # Capture execution status
    has_prediction   = st.session_state.get("last_report") is not None
    validation_failed = st.session_state.get("validation_failed", False)
    
    if has_prediction:
        form_data = st.session_state.get("last_form_data")
    else:
        form_data = None

    # CENTER Panel: Smart Budget Verification Engine
    with col_center:
        if validation_failed:
            suggestion = st.session_state.get("ai_suggestion", {})
            if suggestion and suggestion.get("valid"):
                s_dest = suggestion.get("suggested_destination", "")
                s_expl = suggestion.get("explanation", "")
                st.markdown(f"""
                <div class="checklist-card" style="padding:20px;border-left:4px solid #7C3AED;margin-bottom:16px;">
                  <div style="font-weight:700;color:#F8FAFC;font-size:15px;margin-bottom:6px;">🤖 AI Suggested Destination</div>
                  <div style="font-size:13px;color:#CBD5E1;line-height:1.5;margin-bottom:12px;">
                    Could not verify <b>"{st.session_state.get('failed_input_destination', '')}"</b> via maps.
                    Gemini suggests: <b>{s_dest}</b><br><i>Reason: {s_expl}</i>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(
                    f"👉 Use Suggested Destination: {s_dest}",
                    key="btn_use_suggestion",
                    use_container_width=True,
                ):
                    st.session_state["destination_override"] = s_dest
                    st.session_state["validation_failed"]    = False
                    st.session_state["ai_suggestion"]        = None
                    st.session_state["last_report"]          = None
                    st.rerun()
            else:
                st.error(
                    f"❌ Destination \"{st.session_state.get('failed_input_destination', '')}\" "
                    "could not be identified. Please enter a nearby major city or state."
                )
        elif not has_prediction or form_data is None:
            # Hero Placeholder Card
            st.markdown("""
            <div class="hero-budget-card" style="background: linear-gradient(135deg, #1e1b4b 0%, #31108c 100%); border: 1.5px dashed rgba(255,255,255,0.15); display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; min-height: 200px;">
              <div style="font-size: 32px; margin-bottom: 10px;">✈️</div>
              <div style="font-family: 'Outfit', sans-serif; font-size: 16px; font-weight: 700; color: var(--text-primary);">Hero Budget Card Ready</div>
              <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">Enter details and click Predict to estimate your budget</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            report         = st.session_state["last_report"]
            ml_pred        = report.get("ml_prediction", 0.0)
            historical_avg = report.get("historical_avg", 0.0)
            transport_cost = report.get("transport_cost", 0.0)
            smart_budget   = report.get("smart_budget", ml_pred)
            season         = MONTH_SEASON_MAP.get(form_data["month"].lower(), "summer")
            display_dest   = st.session_state.get("last_display_dest", form_data["destination"])
            is_known       = st.session_state.get("last_is_known", True)

            # 1. Smart Budget Verification Engine Metrics
            _render_smart_budget_metrics(
                ml_pred=ml_pred,
                historical_avg=historical_avg,
                transport_cost=transport_cost,
                smart_budget=smart_budget,
                confidence=report.get("confidence", {}),
                is_known=is_known,
            )

            # 2. Hero Card
            render_budget_hero_card(
                amount=smart_budget,
                days=form_data["days"],
                season=season,
                mode=form_data["travel_mode"],
                hotel=form_data["hotel"],
                popularity_label="✦ Smart Travel Intelligence",
            )

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

            # 3. Pie Chart and Horizontal Bar Chart side-by-side
            theme = st.session_state.get("theme", "dark")
            c_pie, c_bar = st.columns(2)
            with c_pie:
                st.plotly_chart(
                    render_budget_donut(smart_budget, form_data["travel_mode"], theme=theme),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with c_bar:
                st.plotly_chart(
                    render_breakdown_horizontal_bar(smart_budget, form_data["travel_mode"], theme=theme),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # RIGHT Panel: Checklists
    with col_right:
        if not has_prediction or form_data is None or validation_failed:
            _render_right_panel_welcome()
        else:
            _render_right_panel_results(st.session_state["last_report"], form_data)

    # ── ROW 3: Simple Spinner Loader (Main Level) ─────────────────────
    if new_form_data is not None:
        src   = new_form_data["source"]
        dst   = new_form_data["destination"]
        month = new_form_data["month"]
        days  = new_form_data["days"]
        tt    = new_form_data["trip_type"]
        mode  = new_form_data["travel_mode"]
        hotel = new_form_data["hotel"]

        if not src or not dst:
            st.warning("⚠️ Both From and To fields are required.")
            st.stop()

        resolution = _resolve_destination(dst, encoder_classes)
        match_tier = resolution["match_tier"]

        # Handle Fuzzy Suggestions Choice
        if match_tier == "show_suggestions":
            suggestions = resolution["suggestions"]
            st.markdown("""
            <div class="checklist-card" style="padding:20px;border-left:4px solid #7C3AED;margin-bottom:16px; margin-top: 16px;">
              <div style="font-weight:700;color:#F8FAFC;font-size:15px;margin-bottom:4px;">
                🔍 Did you mean one of these?
              </div>
              <div style="font-size:12px;color:#94A3B8;margin-bottom:12px;">
                We couldn't find an exact match. Select the correct destination:
              </div>
            </div>
            """, unsafe_allow_html=True)

            for i, (place, score) in enumerate(suggestions):
                col_s, col_b = st.columns([3, 1])
                with col_s:
                    st.markdown(
                        f'<div style="padding:8px 0;color:#F1F5F9;font-size:13px;">'
                        f'📍 <b>{place.title()}</b> '
                        f'<span style="color:#94A3B8;font-size:11px;">({score:.0f}% match)</span></div>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    if st.button("Select", key=f"btn_suggestion_{i}"):
                        st.session_state["destination_override"] = place
                        st.session_state["last_report"]          = None
                        st.rerun()
            st.stop()

        # Handle Gemini Recommendation
        if match_tier == "needs_gemini":
            from src.services.gemini_service import GeminiService
            gemini_svc = GeminiService()
            with st.spinner("🤖 Consulting AI Travel Assistant for suggestions..."):
                suggestion = gemini_svc.suggest_alternative_destination(dst)
            st.session_state["failed_input_destination"] = dst
            st.session_state["validation_failed"]        = True
            st.session_state["ai_suggestion"]            = suggestion
            st.session_state["last_report"]              = None
            st.rerun()

        # Normal resolution
        matched_dest = resolution["matched_dest"]
        display_dest = resolution["display_dest"]
        dst_coords   = resolution["dst_coords"]
        is_known     = resolution["is_known"]
        season       = MONTH_SEASON_MAP.get(month.lower(), "summer")

        with st.spinner("🔄 Fetching route details and predicting smart budget..."):
            input_df = pd.DataFrame([{
                "Place":         encoders["Place"].transform([matched_dest])[0],
                "Month":         encoders["Month"].transform([month.lower()])[0],
                "Season":        encoders["Season"].transform([season])[0],
                "Trip_Type":     encoders["Trip_Type"].transform([tt.lower()])[0],
                "Hotel_Quality": encoders["Hotel_Quality"].transform([hotel.lower()])[0],
                "Days":          days,
            }])
            ml_pred = float(model.predict(input_df)[0])

            tracker.track(
                source=src, destination=dst, month=month,
                duration_days=days, travel_mode=mode,
                predicted_cost=ml_pred, season=season,
                trip_type=tt, hotel_quality=hotel,
            )

            src_coords = maps_service.get_coordinates(src)
            if not dst_coords:
                dst_coords = maps_service.get_coordinates(display_dest)

            report = travel_engine.generate_report(
                source=src,
                destination=matched_dest,
                month=month,
                duration_days=days,
                travel_mode=mode,
                trip_type=tt,
                hotel_quality=hotel,
                ml_prediction=ml_pred,
                src_coords=src_coords,
                dst_coords=dst_coords,
                is_known_destination=is_known,
            )

            if not is_known:
                conf = report.get("confidence", {})
                conf["score"] = max(0, conf.get("score", 0) - 20)
                conf["level"] = "Low Reliability (Unknown Destination)"
                report["confidence"] = conf

            st.session_state["gemini_intel"]   = report.get("gemini", {})
            st.session_state["packing_tips"]   = (
                report.get("gemini", {}).get("packing_checklist")
                or report.get("intelligence", {}).get("packing_tips", [])
            )
            st.session_state["last_weather"]   = report.get("weather", {})
            st.session_state["last_report"]    = report
            st.session_state["last_form_data"] = new_form_data
            st.session_state["last_display_dest"] = display_dest
            st.session_state["last_is_known"]  = is_known
        st.rerun()

    # If prediction exists, render map, comparison, and PDF button
    if has_prediction and not validation_failed:
        report         = st.session_state["last_report"]
        display_dest   = st.session_state.get("last_display_dest", form_data["destination"])
        is_known       = st.session_state.get("last_is_known", True)
        smart_budget   = report.get("smart_budget", 0.0)
        route_info     = report.get("route", {})
        route_info["source"]      = form_data["source"]
        route_info["destination"] = display_dest
        theme = st.session_state.get("theme", "dark")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # 1. Interactive map — pass travel_mode so correct icon is shown
        render_route_map(route_info, travel_mode=form_data["travel_mode"])

        # 1b. Alternate route — show when no direct access for the selected mode
        try:
            alt_route = get_alternate_route(
                source=form_data["source"],
                destination=display_dest,
                preferred_mode=form_data["travel_mode"],
            )
            if alt_route:
                st.markdown(format_alternate_route_html(alt_route), unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # 2. Mode comparison with availability flags
        all_mode_costs = route_info.get(
            "all_mode_costs",
            report.get("mode_comparison", {}).get("modes", {}),
        )
        try:
            mode_avail = get_mode_availability(display_dest)
        except Exception:
            mode_avail = None

        st.plotly_chart(
            render_mode_comparison_chart(
                all_mode_costs,
                form_data["travel_mode"],
                theme=theme,
                mode_availability=mode_avail,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # 3. PDF Download Button
        try:
            pdf_bytes = generate_pdf_report(
                source=form_data["source"],
                destination=display_dest,
                month=form_data["month"],
                duration=form_data["days"],
                mode=form_data["travel_mode"],
                hotel=form_data["hotel"],
                report_data=report,
            )
            st.download_button(
                label="Download Trip Report (PDF)",
                data=pdf_bytes,
                file_name=f"TripAI_{display_dest.replace(' ', '_')}_{form_data['month']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"PDF export temporarily unavailable: {exc}")
