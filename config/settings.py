"""
Settings Configuration for TripAI Platform.
==========================================
Defines global path resolutions and system-wide settings.
"""
from __future__ import annotations
import os
from pathlib import Path

# Resolve base directories
CONFIG_DIR = Path(__file__).resolve().parent
BASE_DIR = CONFIG_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Application persistence paths
DB_PATH = DATA_DIR / "travel.db"
CSV_PATH = DATA_DIR / "traveltripdata.csv"
