"""
========================================================
Component: Footer
Purpose: Renders the bottom footer bar with logo,
         tagline, author credit, social links, copyright.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
import streamlit as st


def render_footer() -> None:
    """Render the full-width footer bar at the bottom of each page."""
    footer_html = """
    <div style="
      background: #060e1f;
      border-top: 1px solid rgba(255,255,255,0.08);
      padding: 18px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 32px;
      font-family: 'Inter', sans-serif;
    ">
      <!-- Left: Logo + Tagline -->
      <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:20px;">✈️</span>
        <div>
          <div style="
            font-family:'Outfit',sans-serif;
            font-size:15px;
            font-weight:800;
            background:linear-gradient(135deg,#4F46E5,#9333EA);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            background-clip:text;
          ">TripAI</div>
          <div style="font-size:10px; color:#475569; margin-top:1px;">
            AI-Powered Travel Intelligence &amp; Budget Planning Platform
          </div>
        </div>
      </div>

      <!-- Center: Tagline + Credit -->
      <div style="display:flex; align-items:center; gap:18px; flex-wrap:wrap;">
        <span style="font-size:12px; color:#94A3B8;">
          <span style="color:#E879F9;">♥</span> Love with Travel
        </span>
        <span style="font-size:12px; color:#475569;">|</span>
        <span style="font-size:12px; color:#94A3B8;">
          Made by
          <strong style="color:#A78BFA;">Srujana Addanki</strong>
        </span>
      </div>

      <!-- Right: Social Icons + Copyright -->
      <div style="display:flex; align-items:center; gap:10px;">
        <a href="https://www.linkedin.com/in/srujana-addanki/"
           target="_blank"
           style="
             width:30px; height:30px; border-radius:8px;
             background:rgba(255,255,255,0.06);
             border:1px solid rgba(255,255,255,0.08);
             color:#94A3B8;
             display:inline-flex; align-items:center; justify-content:center;
             font-size:12px; font-weight:700; text-decoration:none;
             transition:all 0.2s;
           ">in</a>
        <a href="https://github.com/Srujanaaddanki"
           target="_blank"
           style="
             width:30px; height:30px; border-radius:8px;
             background:rgba(255,255,255,0.06);
             border:1px solid rgba(255,255,255,0.08);
             color:#94A3B8;
             display:inline-flex; align-items:center; justify-content:center;
             font-size:13px; text-decoration:none;
             transition:all 0.2s;
           ">⌥</a>
        <span style="font-size:11px; color:#475569; margin-left:8px;">
          © 2026 TripAI. All rights reserved.
        </span>
      </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
