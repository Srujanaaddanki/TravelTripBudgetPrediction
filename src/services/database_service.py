"""
========================================================
Module: Database Service (Upgraded)
Purpose: SQLite self-learning cache for destination
         intelligence. Full schema with all required
         columns per PRD. TTL: 7 days.
         Schema:
           id, user_input, actual_destination,
           latitude, longitude, distance_km, duration_hr,
           month, days, travel_mode, hotel_quality,
           weather, packing, tips, budget, timestamp
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

log = logging.getLogger("tripai.database_service")

# Cache TTL — refresh data after 7 days
CACHE_TTL_DAYS = 7

# SQLite file path (inside existing data/ directory)
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "travel.db"
)


class DestinationCache:
    """Manages a SQLite self-learning cache of destination intelligence.

    Schema columns
    --------------
    id               : INTEGER PRIMARY KEY AUTOINCREMENT
    user_input       : TEXT UNIQUE  — raw user query (lowercased)
    actual_destination: TEXT        — resolved canonical name
    latitude         : REAL         — destination latitude
    longitude        : REAL         — destination longitude
    distance_km      : REAL         — route distance
    duration_hr      : REAL         — route duration (hours)
    month            : TEXT         — travel month
    days             : INTEGER      — trip duration (days)
    travel_mode      : TEXT         — Flight / Train / Bus / Car / Bike
    hotel_quality    : TEXT         — Budget / Standard / Luxury
    weather          : TEXT (JSON)  — weather data dict
    packing          : TEXT (JSON)  — packing checklist list
    tips             : TEXT (JSON)  — seasonal tips list
    budget           : REAL         — recommended smart budget
    timestamp        : TEXT         — ISO-8601 datetime
    """

    # Required columns in the new schema — used to detect stale tables
    _REQUIRED_COLUMNS = {
        "user_input", "actual_destination", "latitude", "longitude",
        "distance_km", "duration_hr", "month", "days", "travel_mode",
        "hotel_quality", "weather", "packing", "tips", "pretravel", "budget", "timestamp",
    }

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._db_path = os.path.abspath(db_path)
        self._init_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_table(self) -> None:
        """Create the destination_cache table (or migrate if schema outdated)."""
        try:
            with self._connect() as conn:
                cursor = conn.execute("PRAGMA table_info(destination_cache)")
                existing_cols = {row[1] for row in cursor.fetchall()}

                # Drop old table if it is missing any required column
                if existing_cols and not self._REQUIRED_COLUMNS.issubset(existing_cols):
                    log.info(
                        "Migrating destination_cache table to new schema "
                        "(missing: %s).",
                        self._REQUIRED_COLUMNS - existing_cols,
                    )
                    conn.execute("DROP TABLE IF EXISTS destination_cache")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS destination_cache (
                        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_input         TEXT    UNIQUE NOT NULL,
                        actual_destination TEXT,
                        latitude           REAL    DEFAULT 0.0,
                        longitude          REAL    DEFAULT 0.0,
                        distance_km        REAL    DEFAULT 0.0,
                        duration_hr        REAL    DEFAULT 0.0,
                        month              TEXT    DEFAULT '',
                        days               INTEGER DEFAULT 0,
                        travel_mode        TEXT    DEFAULT '',
                        hotel_quality      TEXT    DEFAULT '',
                        weather            TEXT    DEFAULT '{}',
                        packing            TEXT    DEFAULT '[]',
                        tips               TEXT    DEFAULT '[]',
                        pretravel          TEXT    DEFAULT '[]',
                        budget             REAL    DEFAULT 0.0,
                        timestamp          TEXT    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS destination_intelligence_cache (
                        destination           TEXT PRIMARY KEY,
                        country               TEXT,
                        state                 TEXT,
                        latitude              REAL,
                        longitude             REAL,
                        weather_profile       TEXT,
                        tourism_category      TEXT,
                        population_profile    TEXT,
                        estimated_budget_type TEXT,
                        timestamp             TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS unknown_destination_history (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        searched_destination TEXT NOT NULL,
                        proxy_destination   TEXT NOT NULL,
                        country             TEXT NOT NULL,
                        coordinates         TEXT NOT NULL,
                        confidence          REAL NOT NULL,
                        prediction_source   TEXT NOT NULL,
                        resolution_source   TEXT NOT NULL,
                        proxy_score         REAL,
                        is_budget_exact     BOOLEAN,
                        timestamp           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Migrating existing table to add proxy_score and is_budget_exact if they don't exist
                try:
                    cursor = conn.execute("PRAGMA table_info(unknown_destination_history)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if "proxy_score" not in columns:
                        conn.execute("ALTER TABLE unknown_destination_history ADD COLUMN proxy_score REAL")
                        log.info("Migrated unknown_destination_history: added proxy_score column")
                    if "is_budget_exact" not in columns:
                        conn.execute("ALTER TABLE unknown_destination_history ADD COLUMN is_budget_exact BOOLEAN")
                        log.info("Migrated unknown_destination_history: added is_budget_exact column")
                except Exception as migrate_exc:
                    log.warning("Migration of unknown_destination_history columns failed: %s", migrate_exc)
                conn.commit()
                log.info("destination_cache, destination_intelligence_cache and unknown_destination_history tables ready at %s", self._db_path)
        except Exception as exc:
            log.error("Could not initialise database tables: %s", exc)

    def get_intelligence_cache(self, destination: str) -> Optional[Dict[str, Any]]:
        """Fetch cached destination intelligence if not expired (30 days TTL)."""
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM destination_intelligence_cache WHERE LOWER(destination) = ?",
                    (destination.lower().strip(),),
                )
                row = cursor.fetchone()
                if row:
                    ts_str = row["timestamp"]
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        ts = datetime.strptime(ts_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    
                    if datetime.now() - ts < timedelta(days=30):
                        return dict(row)
                    else:
                        log.info("Destination intelligence cache expired for %s", destination)
        except Exception as exc:
            log.warning("Failed to read from destination_intelligence_cache: %s", exc)
        return None

    def set_intelligence_cache(self, destination: str, data: Dict[str, Any]) -> None:
        """Store destination intelligence in the cache."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO destination_intelligence_cache 
                    (destination, country, state, latitude, longitude, weather_profile, tourism_category, population_profile, estimated_budget_type, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        destination.strip().title(),
                        data.get("country"),
                        data.get("state"),
                        float(data.get("latitude", 0.0)) if data.get("latitude") is not None else 0.0,
                        float(data.get("longitude", 0.0)) if data.get("longitude") is not None else 0.0,
                        data.get("weather_profile", "temperate"),
                        data.get("tourism_category", "general"),
                        data.get("population_profile", "medium"),
                        data.get("estimated_budget_type", "api_estimated"),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
            log.info("Stored destination intelligence cache for '%s'.", destination)
        except Exception as exc:
            log.warning("Failed to write to destination_intelligence_cache: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cached(self, destination: str) -> Optional[Dict[str, Any]]:
        """Return cached data for a destination if it is still fresh (< 7 days).

        Parameters
        ----------
        destination : str
            User input (will be lowercased and stripped for lookup).

        Returns
        -------
        dict | None
            Full cache row as a dict, or None on cache miss / expiry.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM destination_cache WHERE user_input = ?",
                    (destination.strip().lower(),),
                ).fetchone()

            if row is None:
                return None

            # Column index mapping (matches CREATE TABLE order)
            # 0:id, 1:user_input, 2:actual_destination, 3:latitude, 4:longitude,
            # 5:distance_km, 6:duration_hr, 7:month, 8:days, 9:travel_mode,
            # 10:hotel_quality, 11:weather, 12:packing, 13:tips, 14:pretravel, 15:budget, 16:timestamp
            last_updated = datetime.fromisoformat(row[16])
            if datetime.now() - last_updated > timedelta(days=CACHE_TTL_DAYS):
                log.info("Cache expired for '%s', will refresh.", destination)
                return None

            return {
                "id":                 row[0],
                "user_input":         row[1],
                "actual_destination": row[2],
                "latitude":           row[3],
                "longitude":          row[4],
                "distance_km":        row[5],
                "duration_hr":        row[6],
                "month":              row[7],
                "days":               row[8],
                "travel_mode":        row[9],
                "hotel_quality":      row[10],
                "weather":            json.loads(row[11]) if row[11] else {},
                "packing":            json.loads(row[12]) if row[12] else [],
                "travel_tips":        json.loads(row[13]) if row[13] else [],
                "pretravel":          json.loads(row[14]) if row[14] else [],
                "budget":             0.0,  # Never load final budget from cache
                "timestamp":          row[16],
            }
        except Exception as exc:
            log.warning("Cache read error for '%s': %s", destination, exc)
            return None

    def set_cache(self, destination: str, data: Dict[str, Any]) -> None:
        """Insert or update cached destination data.

        Parameters
        ----------
        destination : str
            User input key (will be lowercased and stripped).
        data : dict
            Must contain: actual_destination, latitude, longitude,
            distance_km, duration_hr, month, days, travel_mode,
            hotel_quality, weather, packing, travel_tips, pretravel, budget.
        """
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO destination_cache
                    (user_input, actual_destination, latitude, longitude,
                     distance_km, duration_hr, month, days, travel_mode,
                     hotel_quality, weather, packing, tips, pretravel, budget, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    destination.strip().lower(),
                    data.get("actual_destination", destination).strip().lower(),
                    float(data.get("latitude", 0.0)),
                    float(data.get("longitude", 0.0)),
                    float(data.get("distance_km", 0.0)),
                    float(data.get("duration_hr", 0.0)),
                    str(data.get("month", "")),
                    int(data.get("days", 0)),
                    str(data.get("travel_mode", "")),
                    str(data.get("hotel_quality", "")),
                    json.dumps(data.get("weather", {})),
                    json.dumps(data.get("packing", [])),
                    json.dumps(data.get("travel_tips", [])),
                    json.dumps(data.get("pretravel", [])),
                    0.0,  # Never cache final budget, always recalculate
                    datetime.now().isoformat(),
                ))
                conn.commit()
            log.info("Cached destination '%s' successfully.", destination)
        except Exception as exc:
            log.warning("Cache write error for '%s': %s", destination, exc)

    def list_cached_destinations(self) -> List[str]:
        """Return all user_input keys in cache, newest first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT user_input FROM destination_cache ORDER BY timestamp DESC"
                ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics for debugging."""
        try:
            with self._connect() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM destination_cache"
                ).fetchone()[0]
                fresh = conn.execute(
                    "SELECT COUNT(*) FROM destination_cache WHERE timestamp >= ?",
                    ((datetime.now() - timedelta(days=CACHE_TTL_DAYS)).isoformat(),),
                ).fetchone()[0]
            return {"total": total, "fresh": fresh, "expired": total - fresh}
        except Exception:
            return {"total": 0, "fresh": 0, "expired": 0}

    def clear_expired(self) -> int:
        """Remove expired cache entries. Returns count removed."""
        try:
            cutoff = (datetime.now() - timedelta(days=CACHE_TTL_DAYS)).isoformat()
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM destination_cache WHERE timestamp < ?", (cutoff,)
                )
                conn.commit()
            log.info("Cleared %d expired cache entries.", cur.rowcount)
            return cur.rowcount
        except Exception as exc:
            log.warning("Cache clear failed: %s", exc)
            return 0
