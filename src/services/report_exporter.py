"""
========================================================
Module: Report Exporter (PDF) — Single Page Fix
Purpose: Generates a professional single-page PDF trip report.
         - ONE page only (A4 portrait with compact layout)
         - No unicode/emoji characters (latin-1 safe)
         - No blank second page
         - No character encoding errors
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict


# ── Safe text helpers ─────────────────────────────────────────────────────────

def _safe(text: str) -> str:
    """Strip all non-latin-1 characters and common unicode replacements."""
    # Replace common unicode punctuation
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
        "\u20b9": "Rs.", "\u2713": "[OK]", "\u2714": "[OK]",
        "\u2715": "[X]", "\u2716": "[X]", "\u2192": "->", "\u2190": "<-",
        # Emoji replacements
        "\U0001f9f3": "[Pack]", "\U0001f4cb": "[List]", "\u2708\ufe0f": "[Flight]",
        "\U0001f697": "[Car]", "\U0001f686": "[Train]", "\U0001f68c": "[Bus]",
        "\U0001f3cd": "[Bike]", "\U0001f5fa\ufe0f": "[Map]",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    # Strip html tags
    text = re.sub(r'<[^>]*>', '', text)
    # Force encode to latin-1, replacing unknowns
    return text.encode("latin-1", "replace").decode("latin-1")


def _strip_emoji(text: str) -> str:
    """Remove emoji and non-ASCII characters, keep only printable ASCII."""
    # Remove emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "\uFE0F"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    return _safe(text).strip()


def generate_pdf_report(
    source: str,
    destination: str,
    month: str,
    duration: int,
    mode: str,
    hotel: str,
    report_data: Dict[str, Any],
) -> bytes:
    """Generate a compact single-page PDF trip report.

    Parameters
    ----------
    source, destination : str
    month : str
    duration : int
    mode : str
    hotel : str
    report_data : dict

    Returns
    -------
    bytes  — PDF file content
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return _fallback_text_report(source, destination, month, duration, mode, hotel, report_data)

    smart_budget   = report_data.get("smart_budget", report_data.get("ml_prediction", 0))
    ml_pred        = report_data.get("ml_prediction", 0)
    hist_avg       = report_data.get("historical_avg", 0)
    transport_cost = report_data.get("transport_cost", 0)
    confidence     = report_data.get("confidence", {})
    weather        = report_data.get("weather", {})
    gemini         = report_data.get("gemini", {})
    intel          = report_data.get("intelligence", {})
    route          = report_data.get("route", {})
    tiers          = report_data.get("budget_tiers", {})
    is_known       = report_data.get("is_known", hist_avg > 0)

    # Hotel rate calculation
    hotel_key = hotel.lower().strip()
    if "luxury" in hotel_key:
        hotel_rate = 6000.0
    elif "comfort" in hotel_key or "standard" in hotel_key:
        hotel_rate = 3000.0
    else:
        hotel_rate = 1200.0
    duration_cost = (hotel_rate + 1500.0) * duration

    # ── PDF setup — SINGLE PAGE, no auto page break ───────────────────────────
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False, margin=0)  # NO auto page breaks
    pdf.add_page()

    # ── Colour helpers ────────────────────────────────────────────────────────
    def set_accent():  pdf.set_text_color(79, 70, 229)
    def set_white():   pdf.set_text_color(255, 255, 255)
    def set_dark():    pdf.set_text_color(30, 41, 59)
    def set_muted():   pdf.set_text_color(100, 116, 139)
    def set_green():   pdf.set_text_color(16, 185, 129)

    # ── Header band ───────────────────────────────────────────────────────────
    pdf.set_fill_color(5, 8, 22)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_font("Helvetica", "B", 18)
    set_white()
    pdf.set_y(6)
    pdf.cell(0, 9, "TripAI - AI Travel Intelligence Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, "AI-Powered Budget & Route Intelligence Platform", ln=True, align="C")
    pdf.ln(2)
    set_dark()

    y_cur = 34  # Track vertical position manually

    def band(title: str) -> None:
        nonlocal y_cur
        pdf.set_y(y_cur)
        pdf.set_fill_color(79, 70, 229)
        pdf.set_font("Helvetica", "B", 10)
        set_white()
        pdf.cell(0, 7, "  " + _strip_emoji(title), ln=True, fill=True)
        y_cur = pdf.get_y() + 2
        set_dark()

    def row(label: str, value: str, bold_val: bool = False) -> None:
        nonlocal y_cur
        if y_cur > 272:
            return  # Do not overflow page
        pdf.set_y(y_cur)
        pdf.set_font("Helvetica", "B", 9)
        set_muted()
        pdf.cell(62, 5, _strip_emoji(label))
        if bold_val:
            pdf.set_font("Helvetica", "B", 9)
            set_accent()
        else:
            pdf.set_font("Helvetica", "", 9)
            set_dark()
        pdf.cell(0, 5, _strip_emoji(str(value)), ln=True)
        y_cur = pdf.get_y()

    def bullet(item: str, icon: str = "*") -> None:
        nonlocal y_cur
        if y_cur > 272:
            return
        pdf.set_y(y_cur)
        pdf.set_font("Helvetica", "", 8)
        set_dark()
        safe_item = _strip_emoji(str(item))
        # Truncate very long items
        if len(safe_item) > 90:
            safe_item = safe_item[:87] + "..."
        pdf.cell(6, 5, icon)
        pdf.cell(0, 5, safe_item, ln=True)
        y_cur = pdf.get_y()

    # ── Section 1: Trip Overview (two columns) ────────────────────────────────
    band("Trip Overview")
    # Left column content
    pdf.set_y(y_cur)
    left_x, right_x = 10, 110
    line_h = 5

    details = [
        ("Route:", f"{_strip_emoji(source.title())} -> {_strip_emoji(destination.title())}"),
        ("Month:", month),
        ("Duration:", f"{duration} Days"),
        ("Travel Mode:", mode),
        ("Hotel Quality:", hotel.title()),
        ("Confidence:", f"{confidence.get('score', 0)}% - {_strip_emoji(confidence.get('level', 'N/A'))}"),
    ]

    for lbl, val in details:
        if y_cur > 272:
            break
        pdf.set_y(y_cur)
        pdf.set_x(left_x)
        pdf.set_font("Helvetica", "B", 8)
        set_muted()
        pdf.cell(35, line_h, lbl)
        pdf.set_font("Helvetica", "", 8)
        set_dark()
        pdf.cell(60, line_h, _strip_emoji(val), ln=True)
        y_cur = pdf.get_y()

    y_cur += 4

    # ── Section 2: Budget Intelligence (two-column layout) ────────────────────
    band("Smart Budget Intelligence")
    pdf.set_y(y_cur)
    pdf.set_font("Helvetica", "B", 12)
    set_accent()
    pdf.cell(0, 7, f"  Recommended Budget: Rs. {int(smart_budget):,}", ln=True)
    y_cur = pdf.get_y() + 1

    # Budget rows — two per line to save space
    budget_pairs = []
    if is_known:
        budget_pairs = [
            ("ML Prediction (35%):", f"Rs. {int(ml_pred):,}"),
            ("Historical Avg (25%):", f"Rs. {int(hist_avg):,}" if hist_avg else "N/A"),
            ("Transport Cost (20%):", f"Rs. {int(transport_cost):,}"),
            ("Duration Cost (20%):", f"Rs. {int(duration_cost):,}"),
        ]
    else:
        budget_pairs = [
            ("ML Prediction (50%):", f"Rs. {int(ml_pred):,}"),
            ("Transport Cost (30%):", f"Rs. {int(transport_cost):,}"),
            ("Duration Cost (20%):", f"Rs. {int(duration_cost):,}"),
        ]

    for i in range(0, len(budget_pairs), 2):
        if y_cur > 272:
            break
        pdf.set_y(y_cur)
        # Left item
        lbl1, val1 = budget_pairs[i]
        pdf.set_x(10)
        pdf.set_font("Helvetica", "B", 8)
        set_muted()
        pdf.cell(50, 5, lbl1)
        pdf.set_font("Helvetica", "", 8)
        set_dark()
        pdf.cell(40, 5, val1)
        # Right item (if exists)
        if i + 1 < len(budget_pairs):
            lbl2, val2 = budget_pairs[i + 1]
            pdf.set_font("Helvetica", "B", 8)
            set_muted()
            pdf.cell(50, 5, lbl2)
            pdf.set_font("Helvetica", "", 8)
            set_dark()
            pdf.cell(0, 5, val2, ln=True)
        else:
            pdf.ln()
        y_cur = pdf.get_y()

    # Budget tiers on one line
    y_cur += 2
    if y_cur < 270:
        pdf.set_y(y_cur)
        pdf.set_font("Helvetica", "B", 8)
        set_dark()
        pdf.cell(20, 5, "Tiers:")
        pdf.set_font("Helvetica", "", 8)
        set_muted()
        tier_str = (
            f"Min: Rs.{int(tiers.get('minimum', 0)):,}  |  "
            f"Rec: Rs.{int(tiers.get('recommended', smart_budget)):,}  |  "
            f"Comfort: Rs.{int(tiers.get('comfort', 0)):,}  |  "
            f"Luxury: Rs.{int(tiers.get('luxury', 0)):,}"
        )
        pdf.cell(0, 5, tier_str, ln=True)
        y_cur = pdf.get_y() + 3

    # ── Section 3: Route Info ─────────────────────────────────────────────────
    band("Route Information")
    pdf.set_y(y_cur)
    dist = route.get("distance_km", "N/A")
    dur  = route.get("duration_hours", "N/A")
    pdf.set_font("Helvetica", "", 8)
    set_dark()
    pdf.set_x(10)
    pdf.cell(0, 5,
        f"Distance: {dist} km   |   Est. Travel Time: {dur} h   |   "
        f"Source: {_strip_emoji(str(route.get('source', 'Calculated')))}",
        ln=True
    )
    y_cur = pdf.get_y() + 1

    # Mode cost table (compact, single line each)
    all_mode_costs = route.get("all_mode_costs", {})
    if all_mode_costs and y_cur < 265:
        pdf.set_y(y_cur)
        pdf.set_font("Helvetica", "B", 8)
        set_muted()
        pdf.cell(0, 4, "Transport Comparison (Round Trip):", ln=True)
        y_cur = pdf.get_y()

        mode_line_parts = []
        for m in ["Flight", "Train", "Bus", "Car", "Bike"]:
            if m in all_mode_costs and y_cur < 268:
                cost = all_mode_costs[m].get("round_trip", 0)
                dur_s = all_mode_costs[m].get("duration_str", "-")
                mode_line_parts.append(f"{m}: Rs.{int(cost):,} ({dur_s})")

        if mode_line_parts and y_cur < 268:
            pdf.set_y(y_cur)
            pdf.set_font("Helvetica", "", 7)
            set_dark()
            # Print two per line
            for i in range(0, len(mode_line_parts), 3):
                if y_cur > 268:
                    break
                pdf.set_y(y_cur)
                pdf.set_x(12)
                line = "  |  ".join(mode_line_parts[i:i+3])
                pdf.cell(0, 4, line, ln=True)
                y_cur = pdf.get_y()

    y_cur += 3

    # ── Section 4: Checklists (two-column) ───────────────────────────────────
    if y_cur < 250:
        band("Packing & Pre-Travel Checklists")
        packing   = gemini.get("packing_checklist") or intel.get("packing_tips", [])
        pretravel = gemini.get("pre_travel_checklist", [])

        # Two columns side by side
        left_items  = packing[:8]
        right_items = pretravel[:8]
        max_rows    = max(len(left_items), len(right_items))

        col_w = 95
        for i in range(max_rows):
            if y_cur > 270:
                break
            pdf.set_y(y_cur)
            pdf.set_font("Helvetica", "", 7)
            set_dark()

            # Left
            pdf.set_x(10)
            left_text = _strip_emoji(str(left_items[i])) if i < len(left_items) else ""
            if len(left_text) > 50:
                left_text = left_text[:47] + "..."
            pdf.cell(col_w, 4, ("* " + left_text) if left_text else "")

            # Right
            right_text = _strip_emoji(str(right_items[i])) if i < len(right_items) else ""
            if len(right_text) > 50:
                right_text = right_text[:47] + "..."
            pdf.cell(0, 4, (">> " + right_text) if right_text else "", ln=True)
            y_cur = pdf.get_y()

    # ── Footer at bottom ──────────────────────────────────────────────────────
    footer_y = min(y_cur + 4, 284)
    pdf.set_y(footer_y)
    pdf.set_fill_color(5, 8, 22)
    pdf.rect(0, footer_y, 210, 297 - footer_y, "F")
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100, 116, 139)
    pdf.set_y(footer_y + 2)
    pdf.cell(0, 4,
        "Generated by TripAI - AI Travel Intelligence Platform  |  Made by Srujana Addanki",
        align="C", ln=True
    )
    pdf.cell(0, 4,
        "github.com/Srujanaaddanki  |  linkedin.com/in/srujana-addanki",
        align="C", ln=True
    )

    return bytes(pdf.output())


def _fallback_text_report(
    source: str, destination: str, month: str, duration: int,
    mode: str, hotel: str, report_data: Dict[str, Any]
) -> bytes:
    """Plain text fallback if fpdf2 is not installed."""
    lines = [
        "TripAI - Travel Intelligence Report",
        "=" * 45,
        f"Route:    {source} -> {destination}",
        f"Month:    {month}",
        f"Duration: {duration} Days",
        f"Mode:     {mode}",
        f"Hotel:    {hotel}",
        "",
        f"Smart Budget: Rs. {int(report_data.get('smart_budget', 0)):,}",
        f"ML Prediction: Rs. {int(report_data.get('ml_prediction', 0)):,}",
        "",
        "Generated by TripAI | Made by Srujana Addanki",
    ]
    return "\n".join(lines).encode("utf-8")


# ── Keep old HTML export available for backward compat ────────────────
def generate_html_report(
    source: str, destination: str, month: str, duration: int,
    mode: str, hotel: str, predicted_cost: float,
    report_data: Dict[str, Any],
) -> str:
    """Legacy HTML report kept for compatibility."""
    return f"<h1>TripAI Report - {destination}</h1><p>Budget: Rs. {int(predicted_cost):,}</p>"
