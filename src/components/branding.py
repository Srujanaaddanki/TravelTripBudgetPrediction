"""
========================================================
Module: branding.py
Purpose: Centralised branding assets for TraWell.
         Loads splash.png and logo.jpg from assets/,
         base64-encodes them once, and exposes them as
         data-URIs so they can be embedded in any HTML
         block without depending on static file serving.
Author: Srujana Addanki
Project: TraWell — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations
import base64
import os
from functools import lru_cache

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")


@lru_cache(maxsize=None)
def _b64(filename: str) -> str:
    """Return a base64-encoded data-URI for a file in assets/."""
    path = os.path.join(_ASSETS, filename)
    if not os.path.exists(path):
        return ""
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def logo_uri() -> str:
    """Data-URI for the TraWell luggage brand icon (logo.jpg)."""
    return _b64("logo.jpg")


def splash_uri() -> str:
    """Data-URI for the TraWell splash background image (splash.png)."""
    return _b64("splash.png")


def logo_img_tag(width: int = 32, height: int = 32, style: str = "") -> str:
    """Return a ready-to-embed <img> tag for the brand icon."""
    uri = logo_uri()
    if not uri:
        return ""
    s = f"width:{width}px;height:{height}px;object-fit:contain;{style}"
    return f'<img src="{uri}" width="{width}" height="{height}" style="{s}" alt="TraWell"/>'
