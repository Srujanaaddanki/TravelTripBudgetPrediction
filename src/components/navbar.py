"""
========================================================
Component: Navbar (Production Polish)
Purpose: Renders the sticky top navigation bar with logo,
         official LinkedIn link, GitHub repository link,
         and theme toggle.
Author: Srujana Addanki
Project: TraWell — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import streamlit as st


def render_navbar(theme: str = "dark") -> None:
    """Render the top sticky navigation bar with Logo, LinkedIn, GitHub, Theme toggle."""
    theme_icon = "🌙" if theme == "dark" else "☀️"
    navbar_html = f"""
    <style>
    .trawell-navbar {{
      position: sticky;
      top: 0;
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 24px;
      background: rgba(10, 10, 30, 0.92);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-bottom: 1px solid rgba(255,255,255,0.07);
      font-family: 'Outfit', 'Inter', sans-serif;
    }}
    .trawell-navbar-logo {{
      display: flex;
      align-items: center;
      gap: 10px;
      text-decoration: none !important;
    }}
    .trawell-navbar-logo-text {{
      display: flex;
      flex-direction: column;
    }}
    .trawell-navbar-logo-title {{
      font-size: 20px;
      font-weight: 800;
      background: linear-gradient(135deg, #A855F7, #6366F1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: -0.3px;
    }}
    .trawell-navbar-logo-sub {{
      font-size: 10px;
      color: #94A3B8;
      font-weight: 500;
      letter-spacing: 0.3px;
    }}
    .trawell-navbar-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .trawell-nav-btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 10px;
      border: 1.5px solid rgba(255,255,255,0.1);
      background: rgba(255,255,255,0.05);
      color: #94A3B8;
      text-decoration: none !important;
      transition: all 0.25s ease;
      cursor: pointer;
      font-size: 17px;
    }}
    .trawell-nav-btn:hover {{ transform: scale(1.12) translateY(-2px); }}
    .trawell-nav-btn.li:hover {{
      color: #0A66C2 !important;
      border-color: rgba(10,102,194,0.5) !important;
      box-shadow: 0 0 14px rgba(10,102,194,0.4) !important;
      background: rgba(10,102,194,0.12) !important;
    }}
    .trawell-nav-btn.gh:hover {{
      color: #F0F6FC !important;
      border-color: rgba(240,246,252,0.4) !important;
      box-shadow: 0 0 14px rgba(255,255,255,0.3) !important;
      background: rgba(255,255,255,0.12) !important;
    }}
    .trawell-nav-btn.th:hover {{
      color: #A78BFA !important;
      border-color: rgba(124,58,237,0.5) !important;
      box-shadow: 0 0 14px rgba(124,58,237,0.4) !important;
      background: rgba(124,58,237,0.12) !important;
    }}
    /* Hide the standalone Streamlit theme button */
    div.st-key-btn_theme_toggle {{
      position: fixed !important;
      left: -9999px !important;
      visibility: hidden !important;
    }}
    </style>
    <nav class="trawell-navbar">
      <a class="trawell-navbar-logo" href="#">
        <span style="font-size:28px;line-height:1;">✈️</span>
        <div class="trawell-navbar-logo-text">
          <span class="trawell-navbar-logo-title">TraWell</span>
          <span class="trawell-navbar-logo-sub">&#9829; Love with Travel</span>
        </div>
      </a>
      <div class="trawell-navbar-actions">
        <a class="trawell-nav-btn li" href="https://www.linkedin.com/feed/update/urn:li:activity:7409296852480606208/" target="_blank" rel="noopener noreferrer" title="LinkedIn">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
        </a>
        <a class="trawell-nav-btn gh" href="https://github.com/Srujanaaddanki/TravelTripBudgetPrediction" target="_blank" rel="noopener noreferrer" title="GitHub">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/></svg>
        </a>
        <button class="trawell-nav-btn th" onclick="window.parent.document.querySelector('div.st-key-btn_theme_toggle button').click()" title="Switch Theme">{theme_icon}</button>
      </div>
    </nav>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)
