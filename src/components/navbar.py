"""
========================================================
Component: Navbar (Production Polish)
Purpose: Renders the sticky top navigation bar with logo,
         official LinkedIn link, GitHub repository link,
         Portfolio link, and theme toggle.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import streamlit as st


def render_navbar(theme: str = "dark") -> None:
    """Render the top sticky navigation bar with Logo, LinkedIn, GitHub, Portfolio, and theme toggle placeholder."""
    navbar_html = """
    <nav class="tripai-navbar">
      <!-- Logo -->
      <a class="navbar-logo" href="#" style="text-decoration:none;">
        <span style="font-size:32px;line-height:1;">✈️</span>
        <div class="navbar-logo-text">
          <span class="navbar-logo-title" style="font-size:22px;">TripAI</span>
          <span class="navbar-logo-sub" style="font-size:11px;"><span>♥</span> Love with Travel</span>
        </div>
      </a>

      <!-- Action Icons -->
      <div class="navbar-actions">
        <!-- LinkedIn -->
        <a class="social-icon-btn linkedin-btn"
           href="https://www.linkedin.com/feed/update/urn:li:activity:7409296852480606208/"
           target="_blank"
           rel="noopener noreferrer"
           title="LinkedIn">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/>
          </svg>
        </a>

        <!-- GitHub -->
        <a class="social-icon-btn github-btn"
           href="https://github.com/Srujanaaddanki/TravelTripBudgetPrediction"
           target="_blank"
           rel="noopener noreferrer"
           title="GitHub">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/>
          </svg>
        </a>

        <!-- Portfolio / Globe -->
        <a class="social-icon-btn portfolio-btn"
           href="https://github.com/Srujanaaddanki/TravelTripBudgetPrediction"
           target="_blank"
           rel="noopener noreferrer"
           title="Portfolio">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="2" y1="12" x2="22" y2="12"/>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
          </svg>
        </a>
      </div>
    </nav>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)
