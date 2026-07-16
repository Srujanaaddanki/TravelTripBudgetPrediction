"""
========================================================
Component: Navbar (Upgraded)
Purpose: Renders the sticky top navigation bar with logo,
         dynamic PDF download action (base64 data URI),
         LinkedIn link, GitHub link, and theme toggle.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import base64
import streamlit as st
from src.services.report_exporter import generate_pdf_report


def render_navbar(theme: str = "dark") -> None:
    """Render the top sticky navigation bar with Logo, LinkedIn, GitHub, and theme toggle."""
    theme_icon  = "☀️" if theme == "dark" else "🌙"
    theme_label = "Light Mode" if theme == "dark" else "Dark Mode"

    navbar_html = f"""
    <nav class="tripai-navbar">
      <!-- Logo -->
      <a class="navbar-logo" href="#" style="text-decoration:none;">
        <span style="font-size:36px;line-height:1;">&#x2708;&#xFE0F;</span>
        <div class="navbar-logo-text">
          <span class="navbar-logo-title" style="font-size:22px;">TripAI</span>
          <span class="navbar-logo-sub" style="font-size:11px;"><span>&#x2665;</span> Love with Travel</span>
        </div>
      </a>

      <!-- Action Icons -->
      <div class="navbar-actions">
        <a class="icon-btn"
           style="width:46px;height:46px;font-size:17px;font-weight:700;"
           href="https://www.linkedin.com/in/srujana-addanki/"
           target="_blank" title="LinkedIn">in</a>
        <a class="icon-btn"
           style="width:46px;height:46px;font-size:18px;"
           href="https://github.com/Srujanaaddanki"
           target="_blank" title="GitHub">&#x2325;</a>
        <span class="icon-btn"
              style="width:46px;height:46px;font-size:20px;"
              title="{theme_label}" id="theme-toggle">{theme_icon}</span>
      </div>
    </nav>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)
