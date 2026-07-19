"""
Global Constants for TripAI Platform.
==========================================
Defines global metadata, defaults, and system-wide configurations.
"""
from __future__ import annotations

# Platform Metadata
APP_NAME = "TripAI"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Srujana Addanki"

# Default Model Parameters
DEFAULT_R2_SCORE = 0.95

# Default Cache Lifetimes (hours)
DISTANCE_CACHE_TTL_HOURS = 24 * 30      # 30 days
SUGGESTION_CACHE_TTL_HOURS = 24 * 7     # 7 days
