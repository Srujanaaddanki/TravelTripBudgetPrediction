"""
========================================================
Component: Checklists (Upgraded)
Purpose: Renders packing checklist and pre-travel checklist
         cards in the right sidebar.
         Now accepts dynamic Gemini-generated items.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
from typing import List, Optional
import streamlit as st


# ── Default packing items (used when no Gemini suggestions available) ─────────
_DEFAULT_PACKING: List[str] = [
    "Comfortable Walking Shoes",
    "Sunscreen & Sunglasses",
    "Water Bottle & Snacks",
    "Hat / Cap",
    "Power Bank",
    "Basic First-aid Kit",
    "Travel Documents",
    "Camera / Phone Charger",
]

# ── Default pre-travel checklist (static fallback) ───────────────────────────
_DEFAULT_PRETRAVEL: List[str] = [
    "ID Proof (Aadhaar / Voter ID)",
    "Travel Tickets (Train / Flight)",
    "Hotel Booking Confirmation",
    "Cash & Cards",
    "Emergency Contacts",
    "Medicines & Prescriptions",
    "Charger & Power Bank",
    "Travel Insurance",
]


def render_packing_checklist(items: Optional[List[str]] = None) -> None:
    """Render the packing checklist sidebar card."""
    display_items = items if items else _DEFAULT_PACKING

    items_html = ""
    for item in display_items[:10]:
        safe = str(item).replace("<", "&lt;").replace(">", "&gt;")
        items_html += f'<div class="checklist-item"><div class="check-icon">✓</div><div class="checklist-text">{safe}</div></div>'

    source_badge = ""
    if items:
        source_badge = '<span style="font-size:9px;color:#7C3AED;margin-left:4px;">✦ AI</span>'

    card_html = f'<div class="checklist-card" style="margin-bottom:16px;"><div class="checklist-title"><span>🧳</span> Packing Checklist{source_badge}</div>{items_html}</div>'
    st.markdown(card_html, unsafe_allow_html=True)


def render_pretravel_checklist(items: Optional[List[str]] = None) -> None:
    """Render the pre-travel checklist sidebar card."""
    display_items = items if items else _DEFAULT_PRETRAVEL

    items_html = ""
    for item in display_items[:10]:
        safe = str(item).replace("<", "&lt;").replace(">", "&gt;")
        items_html += f'<div class="checklist-item"><div class="check-icon">✓</div><div class="checklist-text">{safe}</div></div>'

    source_badge = ""
    if items:
        source_badge = '<span style="font-size:9px;color:#7C3AED;margin-left:4px;">✦ AI</span>'

    card_html = f'<div class="checklist-card"><div class="checklist-title"><span>📋</span> Pre-Travel Checklist{source_badge}</div>{items_html}</div>'
    st.markdown(card_html, unsafe_allow_html=True)
