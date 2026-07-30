"""
========================================================
Component: Footer (Production Polish)
Purpose: Renders the bottom footer bar with logo,
         tagline, author credit, social links, copyright.
Author: Srujana Addanki
Project: TraWell — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
import streamlit as st


def render_footer() -> None:
    """Render the full-width footer bar at the bottom of each page."""
    footer_html = """
    <style>
    .trawell-footer {
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
    }
    .trawell-footer-logo {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .trawell-footer-brand {
      font-family: 'Outfit', sans-serif;
      font-size: 15px;
      font-weight: 800;
      background: linear-gradient(135deg, #4F46E5, #9333EA);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .trawell-footer-tagline {
      font-size: 10px;
      color: #475569;
      margin-top: 1px;
    }
    .trawell-footer-center {
      display: flex;
      align-items: center;
      gap: 18px;
      flex-wrap: wrap;
    }
    .trawell-footer-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .trawell-footer-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      border: 1.5px solid rgba(255,255,255,0.1);
      background: rgba(255,255,255,0.05);
      color: #94A3B8;
      text-decoration: none !important;
      transition: all 0.25s ease;
    }
    .trawell-footer-btn:hover { transform: scale(1.12) translateY(-2px); }
    .trawell-footer-btn.li:hover { color:#0A66C2!important; border-color:rgba(10,102,194,0.5)!important; box-shadow:0 0 12px rgba(10,102,194,0.4)!important; background:rgba(10,102,194,0.12)!important; }
    .trawell-footer-btn.gh:hover { color:#F0F6FC!important; border-color:rgba(240,246,252,0.4)!important; box-shadow:0 0 12px rgba(255,255,255,0.25)!important; background:rgba(255,255,255,0.1)!important; }
    </style>
    <div class="trawell-footer">
      <div class="trawell-footer-logo">
        <span style="font-size:20px;">✈️</span>
        <div>
          <div class="trawell-footer-brand">TraWell</div>
          <div class="trawell-footer-tagline">AI-Powered Travel Intelligence &amp; Budget Planning Platform</div>
        </div>
      </div>
      <div class="trawell-footer-center">
        <span style="font-size:12px; color:#94A3B8;"><span style="color:#E879F9;">&#9829;</span> Love with Travel</span>
        <span style="font-size:12px; color:#475569;">|</span>
        <span style="font-size:12px; color:#94A3B8;">Made by <strong style="color:#A78BFA;">Srujana Addanki</strong></span>
      </div>
      <div class="trawell-footer-right">
        <a class="trawell-footer-btn li" href="https://www.linkedin.com/feed/update/urn:li:activity:7409296852480606208/" target="_blank" rel="noopener noreferrer" title="LinkedIn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
        </a>
        <a class="trawell-footer-btn gh" href="https://github.com/Srujanaaddanki/TravelTripBudgetPrediction" target="_blank" rel="noopener noreferrer" title="GitHub">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/></svg>
        </a>
        <span style="font-size:11px; color:#475569; margin-left:8px;">&#169; 2026 TraWell. All rights reserved.</span>
      </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
