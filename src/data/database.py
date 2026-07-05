"""
TripAI — Production-Ready SQLite Database Layer
================================================

Provides the ``TripDatabase`` class, a single entry-point for every
database operation in the Travel Budget Prediction system.

Features
--------
* Automatic schema initialisation on first connection.
* Context-manager support (``with TripDatabase() as db: ...``).
* Full CRUD for five tables: **trip_history**, **user_searches**,
  **distance_cache**, **suggestions_cache**, **popular_destinations**.
* Built-in cache expiration (distance + suggestion caches).
* Structured logging to ``logs/database.log``.
* Thread-safe connections with WAL journal mode.

Usage
-----
>>> from src.data.database import TripDatabase
>>> db = TripDatabase()
>>> db.add_trip(destination="Manali", season="Winter", ...)
>>> db.close()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================
# Constants
# ============================================================

_BASE_DIR = Path(__file__).resolve().parent.parent.parent          # → srujan/
_DEFAULT_DB_PATH = _BASE_DIR / "data" / "travel.db"
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
_LOG_DIR = _BASE_DIR / "logs"

# Default cache lifetimes (hours)
DISTANCE_CACHE_TTL_HOURS = 24 * 30      # 30 days
SUGGESTION_CACHE_TTL_HOURS = 24 * 7     # 7 days


# ============================================================
# Logger Setup
# ============================================================

def _setup_logger() -> logging.Logger:
    """Create a file + console logger for the database layer."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("tripai.database")
    if logger.handlers:                       # avoid duplicate handlers
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s | %(funcName)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — all levels
    fh = logging.FileHandler(_LOG_DIR / "database.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — warnings and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = _setup_logger()


# ============================================================
# TripDatabase
# ============================================================

class TripDatabase:
    """Production SQLite manager for the TripAI application.

    Parameters
    ----------
    db_path : str or Path, optional
        Path to the SQLite database file.  Defaults to
        ``<project>/data/travel.db``.  Parent directories are
        created automatically.
    """

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def __init__(self, db_path: Union[str, Path, None] = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialise()

    def _initialise(self) -> None:
        """Open a connection, enable WAL mode, and create tables."""
        try:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row       # dict-like rows
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._apply_schema()
            log.info("Database initialised at %s", self.db_path)
        except sqlite3.Error as exc:
            log.critical("Failed to initialise database: %s", exc)
            raise

    def _apply_schema(self) -> None:
        """Execute the external schema.sql file to create tables."""
        if not _SCHEMA_PATH.exists():
            log.error("Schema file not found: %s", _SCHEMA_PATH)
            raise FileNotFoundError(f"Schema missing: {_SCHEMA_PATH}")
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self._conn.executescript(sql)
        self._conn.commit()
        log.debug("Schema applied successfully")

    def close(self) -> None:
        """Flush and close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            log.info("Database connection closed")

    # Context-manager protocol
    def __enter__(self) -> "TripDatabase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _execute(
        self,
        sql: str,
        params: Union[tuple, dict] = (),
        *,
        commit: bool = True,
    ) -> sqlite3.Cursor:
        """Execute a single SQL statement with logging & error handling."""
        try:
            cur = self._conn.execute(sql, params)
            if commit:
                self._conn.commit()
            return cur
        except sqlite3.Error as exc:
            log.error("SQL error — %s | params=%s | %s", sql[:120], params, exc)
            self._conn.rollback()
            raise

    def _executemany(
        self, sql: str, seq_of_params: list, *, commit: bool = True
    ) -> sqlite3.Cursor:
        """Execute a statement against a sequence of parameter sets."""
        try:
            cur = self._conn.executemany(sql, seq_of_params)
            if commit:
                self._conn.commit()
            return cur
        except sqlite3.Error as exc:
            log.error("SQL batch error — %s | %s", sql[:120], exc)
            self._conn.rollback()
            raise

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row) if row else {}

    @staticmethod
    def _rows_to_list(rows: list) -> List[Dict[str, Any]]:
        return [dict(r) for r in rows]

    # ============================================================
    # 1. TRIP HISTORY — CRUD
    # ============================================================

    def add_trip(
        self,
        destination: str,
        season: str,
        month: str,
        trip_type: str,
        hotel_quality: str,
        duration_days: int,
        predicted_cost: float,
        *,
        source_location: Optional[str] = None,
        actual_cost: Optional[float] = None,
        model_version: str = "1.0",
        satisfaction: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Insert a new trip record and return its ``id``."""
        sql = """
            INSERT INTO trip_history
                (destination, source_location, season, month, trip_type,
                 hotel_quality, duration_days, predicted_cost, actual_cost,
                 model_version, satisfaction, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cur = self._execute(sql, (
            destination, source_location, season, month, trip_type,
            hotel_quality, duration_days, predicted_cost, actual_cost,
            model_version, satisfaction, notes,
        ))
        trip_id = cur.lastrowid
        log.info("Trip #%d added → %s (%s, %dd)", trip_id, destination, season, duration_days)
        return trip_id

    def get_trip(self, trip_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single trip by ``id``."""
        cur = self._execute(
            "SELECT * FROM trip_history WHERE id = ?", (trip_id,), commit=False
        )
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_trips(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        destination: Optional[str] = None,
        season: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return trips with optional filters, newest first."""
        clauses: list[str] = []
        params: list = []

        if destination:
            clauses.append("destination LIKE ?")
            params.append(f"%{destination}%")
        if season:
            clauses.append("season = ?")
            params.append(season)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT * FROM trip_history
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self._execute(sql, tuple(params), commit=False).fetchall()
        return self._rows_to_list(rows)

    def update_trip(self, trip_id: int, **fields) -> bool:
        """Update arbitrary columns on a trip row.

        >>> db.update_trip(1, actual_cost=12000, satisfaction=4)
        """
        if not fields:
            return False
        allowed = {
            "destination", "source_location", "season", "month",
            "trip_type", "hotel_quality", "duration_days",
            "predicted_cost", "actual_cost", "model_version",
            "satisfaction", "notes",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"Invalid columns: {invalid}")

        fields["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [trip_id]
        affected = self._execute(
            f"UPDATE trip_history SET {set_clause} WHERE id = ?", tuple(vals)
        ).rowcount
        log.info("Trip #%d updated (%s)", trip_id, list(fields.keys()))
        return affected > 0

    def delete_trip(self, trip_id: int) -> bool:
        """Delete a trip row by ``id``."""
        affected = self._execute(
            "DELETE FROM trip_history WHERE id = ?", (trip_id,)
        ).rowcount
        log.info("Trip #%d deleted (affected=%d)", trip_id, affected)
        return affected > 0

    def count_trips(self) -> int:
        """Return total number of trip records."""
        cur = self._execute("SELECT COUNT(*) FROM trip_history", commit=False)
        return cur.fetchone()[0]

    # ============================================================
    # 2. USER SEARCHES — CRUD + Analytics
    # ============================================================

    def log_search(
        self,
        destination: str,
        *,
        session_id: Optional[str] = None,
        source_location: Optional[str] = None,
        season: Optional[str] = None,
        month: Optional[str] = None,
        trip_type: Optional[str] = None,
        hotel_quality: Optional[str] = None,
        duration_days: Optional[int] = None,
        travel_mode: Optional[str] = None,
        predicted_cost: Optional[float] = None,
        search_params: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        """Record a user search and return its ``id``."""
        session_id = session_id or str(uuid.uuid4())
        params_json = json.dumps(search_params) if search_params else None

        sql = """
            INSERT INTO user_searches
                (session_id, source_location, destination, season, month,
                 trip_type, hotel_quality, duration_days, travel_mode,
                 predicted_cost, search_params, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cur = self._execute(sql, (
            session_id, source_location, destination, season, month,
            trip_type, hotel_quality, duration_days, travel_mode,
            predicted_cost, params_json, ip_address, user_agent,
        ))
        search_id = cur.lastrowid
        log.debug("Search #%d logged: %s -> %s (mode=%s)", search_id, source_location, destination, travel_mode)
        return search_id

    def get_searches(
        self,
        *,
        session_id: Optional[str] = None,
        destination: Optional[str] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve search logs with optional filters."""
        clauses: list[str] = []
        params: list = []

        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if destination:
            clauses.append("destination LIKE ?")
            params.append(f"%{destination}%")
        if since:
            clauses.append("searched_at >= ?")
            params.append(since.strftime("%Y-%m-%d %H:%M:%S"))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT * FROM user_searches
            {where}
            ORDER BY searched_at DESC LIMIT ?
        """
        params.append(limit)
        rows = self._execute(sql, tuple(params), commit=False).fetchall()
        return self._rows_to_list(rows)

    def get_search_stats(self) -> Dict[str, Any]:
        """Return aggregate search statistics."""
        total = self._execute(
            "SELECT COUNT(*) FROM user_searches", commit=False
        ).fetchone()[0]
        top_dest = self._execute("""
            SELECT destination, COUNT(*) as cnt
            FROM user_searches
            GROUP BY destination ORDER BY cnt DESC LIMIT 5
        """, commit=False).fetchall()
        return {
            "total_searches": total,
            "top_destinations": self._rows_to_list(top_dest),
        }

    def delete_search(self, search_id: int) -> bool:
        """Delete a search record by ``id``."""
        affected = self._execute(
            "DELETE FROM user_searches WHERE id = ?", (search_id,)
        ).rowcount
        return affected > 0

    # ----------------------------------------------------------
    # Search Analytics
    # ----------------------------------------------------------

    def most_searched_destinations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return destinations ranked by search frequency.

        Each row: ``{destination, search_count, avg_predicted_cost,
        avg_duration, last_searched}``
        """
        sql = """
            SELECT
                destination,
                COUNT(*)                        AS search_count,
                ROUND(AVG(predicted_cost), 2)   AS avg_predicted_cost,
                ROUND(AVG(duration_days), 1)    AS avg_duration,
                MAX(searched_at)                 AS last_searched
            FROM user_searches
            GROUP BY destination
            ORDER BY search_count DESC
            LIMIT ?
        """
        rows = self._execute(sql, (limit,), commit=False).fetchall()
        log.debug("most_searched_destinations: returned %d rows", len(rows))
        return self._rows_to_list(rows)

    def most_searched_travel_modes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return travel modes ranked by search frequency.

        Each row: ``{travel_mode, search_count, avg_predicted_cost,
        percentage}``
        """
        total = self._execute(
            "SELECT COUNT(*) FROM user_searches WHERE travel_mode IS NOT NULL",
            commit=False,
        ).fetchone()[0]

        sql = """
            SELECT
                travel_mode,
                COUNT(*)                        AS search_count,
                ROUND(AVG(predicted_cost), 2)   AS avg_predicted_cost
            FROM user_searches
            WHERE travel_mode IS NOT NULL
            GROUP BY travel_mode
            ORDER BY search_count DESC
            LIMIT ?
        """
        rows = self._execute(sql, (limit,), commit=False).fetchall()
        results = self._rows_to_list(rows)
        for r in results:
            r["percentage"] = round((r["search_count"] / total) * 100, 1) if total else 0.0
        log.debug("most_searched_travel_modes: returned %d rows", len(results))
        return results

    def average_predicted_budget(self) -> Dict[str, Any]:
        """Return overall and per-destination average predicted budgets.

        Returns ``{overall_avg, min_budget, max_budget, total_searches,
        by_destination: [{destination, avg_budget, search_count}]}``
        """
        overall = self._execute("""
            SELECT
                ROUND(AVG(predicted_cost), 2) AS overall_avg,
                MIN(predicted_cost)           AS min_budget,
                MAX(predicted_cost)           AS max_budget,
                COUNT(*)                      AS total_searches
            FROM user_searches
            WHERE predicted_cost IS NOT NULL
        """, commit=False).fetchone()

        by_dest = self._execute("""
            SELECT
                destination,
                ROUND(AVG(predicted_cost), 2) AS avg_budget,
                COUNT(*)                      AS search_count
            FROM user_searches
            WHERE predicted_cost IS NOT NULL
            GROUP BY destination
            ORDER BY avg_budget DESC
        """, commit=False).fetchall()

        result = self._row_to_dict(overall) if overall else {}
        result["by_destination"] = self._rows_to_list(by_dest)
        log.debug("average_predicted_budget: overall=%.2f", result.get("overall_avg", 0) or 0)
        return result

    def monthly_search_trends(self) -> List[Dict[str, Any]]:
        """Return search volume and average budget per calendar month.

        Each row: ``{month, search_count, avg_predicted_cost,
        top_destination}``

        Results are ordered by ``search_count DESC`` so the busiest
        months appear first.
        """
        sql = """
            SELECT
                month,
                COUNT(*)                        AS search_count,
                ROUND(AVG(predicted_cost), 2)   AS avg_predicted_cost
            FROM user_searches
            WHERE month IS NOT NULL
            GROUP BY month
            ORDER BY search_count DESC
        """
        rows = self._execute(sql, commit=False).fetchall()
        results = self._rows_to_list(rows)

        # Attach the top destination per month
        for r in results:
            top = self._execute("""
                SELECT destination, COUNT(*) AS cnt
                FROM user_searches
                WHERE month = ?
                GROUP BY destination
                ORDER BY cnt DESC LIMIT 1
            """, (r["month"],), commit=False).fetchone()
            r["top_destination"] = top["destination"] if top else None

        log.debug("monthly_search_trends: returned %d months", len(results))
        return results

    # ============================================================
    # 3. DISTANCE CACHE — CRUD + Expiration
    # ============================================================

    def cache_distance(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        *,
        duration_hours: Optional[float] = None,
        travel_mode: str = "driving",
        ttl_hours: int = DISTANCE_CACHE_TTL_HOURS,
    ) -> int:
        """Insert or update a cached distance entry.

        Uses ``INSERT OR REPLACE`` so stale entries are refreshed.
        """
        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
            INSERT OR REPLACE INTO distance_cache
                (origin, destination, distance_km, duration_hours,
                 travel_mode, cached_at, expires_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        """
        cur = self._execute(sql, (
            origin.lower().strip(),
            destination.lower().strip(),
            distance_km, duration_hours, travel_mode, expires_at,
        ))
        log.debug("Distance cached: %s → %s (%.1f km, mode=%s)", origin, destination, distance_km, travel_mode)
        return cur.lastrowid

    def get_distance(
        self,
        origin: str,
        destination: str,
        travel_mode: str = "driving",
    ) -> Optional[Dict[str, Any]]:
        """Fetch a cached distance if it exists and has not expired."""
        sql = """
            SELECT * FROM distance_cache
            WHERE origin = ? AND destination = ? AND travel_mode = ?
              AND expires_at > CURRENT_TIMESTAMP
        """
        row = self._execute(
            sql,
            (origin.lower().strip(), destination.lower().strip(), travel_mode),
            commit=False,
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_distances(self, *, include_expired: bool = False) -> List[Dict[str, Any]]:
        """Return all cached distances, optionally including expired."""
        if include_expired:
            sql = "SELECT * FROM distance_cache ORDER BY cached_at DESC"
        else:
            sql = "SELECT * FROM distance_cache WHERE expires_at > CURRENT_TIMESTAMP ORDER BY cached_at DESC"
        return self._rows_to_list(self._execute(sql, commit=False).fetchall())

    def purge_expired_distances(self) -> int:
        """Delete all expired distance cache entries. Returns count removed."""
        affected = self._execute(
            "DELETE FROM distance_cache WHERE expires_at <= CURRENT_TIMESTAMP"
        ).rowcount
        log.info("Purged %d expired distance cache entries", affected)
        return affected

    def delete_distance(self, cache_id: int) -> bool:
        """Delete a specific distance cache entry."""
        affected = self._execute(
            "DELETE FROM distance_cache WHERE id = ?", (cache_id,)
        ).rowcount
        return affected > 0

    # ============================================================
    # 3b. GEOCODE CACHE — Coordinates
    # ============================================================

    def cache_geocode(
        self,
        place_name: str,
        latitude: float,
        longitude: float,
        *,
        formatted_address: Optional[str] = None,
        ttl_hours: int = DISTANCE_CACHE_TTL_HOURS,
    ) -> int:
        """Insert or update a cached geocode entry.

        Uses ``INSERT OR REPLACE`` so stale entries are refreshed.
        """
        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
            INSERT OR REPLACE INTO geocode_cache
                (place_name, latitude, longitude, formatted_address,
                 cached_at, expires_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        """
        cur = self._execute(sql, (
            place_name.lower().strip(),
            latitude, longitude, formatted_address, expires_at,
        ))
        log.debug("Geocode cached: %s (%.4f, %.4f)", place_name, latitude, longitude)
        return cur.lastrowid

    def get_geocode(self, place_name: str) -> Optional[Dict[str, Any]]:
        """Fetch cached coordinates for a place if not expired."""
        sql = """
            SELECT * FROM geocode_cache
            WHERE place_name = ?
              AND expires_at > CURRENT_TIMESTAMP
        """
        row = self._execute(
            sql, (place_name.lower().strip(),), commit=False
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def purge_expired_geocodes(self) -> int:
        """Delete all expired geocode cache entries. Returns count removed."""
        affected = self._execute(
            "DELETE FROM geocode_cache WHERE expires_at <= CURRENT_TIMESTAMP"
        ).rowcount
        log.info("Purged %d expired geocode cache entries", affected)
        return affected

    # ============================================================
    # 4. SUGGESTIONS CACHE — CRUD + Expiration
    # ============================================================

    def cache_suggestion(
        self,
        cache_key: str,
        suggestion_data: Union[dict, list, str],
        *,
        category: str = "general",
        ttl_hours: int = SUGGESTION_CACHE_TTL_HOURS,
    ) -> int:
        """Store a suggestion payload in the cache.

        If ``cache_key`` already exists it is replaced (upsert).
        """
        data_json = json.dumps(suggestion_data) if not isinstance(suggestion_data, str) else suggestion_data
        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
            INSERT OR REPLACE INTO suggestions_cache
                (cache_key, suggestion_data, category, hit_count,
                 cached_at, expires_at)
            VALUES (?, ?, ?, 0, CURRENT_TIMESTAMP, ?)
        """
        cur = self._execute(sql, (cache_key, data_json, category, expires_at))
        log.debug("Suggestion cached: key=%s, category=%s", cache_key, category)
        return cur.lastrowid

    def get_suggestion(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached suggestion by key (if not expired).

        Automatically increments ``hit_count`` on access.
        """
        sql = """
            SELECT * FROM suggestions_cache
            WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
        """
        row = self._execute(sql, (cache_key,), commit=False).fetchone()
        if not row:
            return None
        # Bump hit count
        self._execute(
            "UPDATE suggestions_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (cache_key,),
        )
        result = self._row_to_dict(row)
        # Deserialise JSON payload
        try:
            result["suggestion_data"] = json.loads(result["suggestion_data"])
        except (json.JSONDecodeError, TypeError):
            pass  # keep as string if not valid JSON
        return result

    def get_suggestions_by_category(
        self, category: str, *, include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Return all suggestions in a category."""
        expired_clause = "" if include_expired else "AND expires_at > CURRENT_TIMESTAMP"
        sql = f"""
            SELECT * FROM suggestions_cache
            WHERE category = ? {expired_clause}
            ORDER BY hit_count DESC
        """
        rows = self._execute(sql, (category,), commit=False).fetchall()
        results = self._rows_to_list(rows)
        for r in results:
            try:
                r["suggestion_data"] = json.loads(r["suggestion_data"])
            except (json.JSONDecodeError, TypeError):
                pass
        return results

    def purge_expired_suggestions(self) -> int:
        """Delete all expired suggestion cache entries."""
        affected = self._execute(
            "DELETE FROM suggestions_cache WHERE expires_at <= CURRENT_TIMESTAMP"
        ).rowcount
        log.info("Purged %d expired suggestion cache entries", affected)
        return affected

    def delete_suggestion(self, cache_key: str) -> bool:
        """Delete a specific suggestion by its cache key."""
        affected = self._execute(
            "DELETE FROM suggestions_cache WHERE cache_key = ?", (cache_key,)
        ).rowcount
        return affected > 0

    # ============================================================
    # 5. POPULAR DESTINATIONS — CRUD + Refresh
    # ============================================================

    def upsert_popular_destination(
        self,
        destination: str,
        *,
        search_count: int = 0,
        trip_count: int = 0,
        avg_predicted_cost: float = 0.0,
        avg_duration_days: float = 0.0,
        min_predicted_cost: Optional[float] = None,
        max_predicted_cost: Optional[float] = None,
        most_common_season: Optional[str] = None,
        most_common_trip_type: Optional[str] = None,
        trending_score: float = 0.0,
    ) -> int:
        """Insert or update a popular destination record."""
        sql = """
            INSERT INTO popular_destinations
                (destination, search_count, trip_count,
                 avg_predicted_cost, avg_duration_days,
                 min_predicted_cost, max_predicted_cost,
                 most_common_season, most_common_trip_type,
                 trending_score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(destination) DO UPDATE SET
                search_count       = excluded.search_count,
                trip_count         = excluded.trip_count,
                avg_predicted_cost = excluded.avg_predicted_cost,
                avg_duration_days  = excluded.avg_duration_days,
                min_predicted_cost = excluded.min_predicted_cost,
                max_predicted_cost = excluded.max_predicted_cost,
                most_common_season = excluded.most_common_season,
                most_common_trip_type = excluded.most_common_trip_type,
                trending_score     = excluded.trending_score,
                last_updated       = CURRENT_TIMESTAMP
        """
        cur = self._execute(sql, (
            destination.title(), search_count, trip_count,
            avg_predicted_cost, avg_duration_days,
            min_predicted_cost, max_predicted_cost,
            most_common_season, most_common_trip_type, trending_score,
        ))
        log.info("Popular destination upserted: %s (score=%.2f)", destination, trending_score)
        return cur.lastrowid

    def get_popular_destinations(
        self, *, limit: int = 10, min_searches: int = 0
    ) -> List[Dict[str, Any]]:
        """Return top destinations sorted by trending score."""
        sql = """
            SELECT * FROM popular_destinations
            WHERE search_count >= ?
            ORDER BY trending_score DESC
            LIMIT ?
        """
        rows = self._execute(sql, (min_searches, limit), commit=False).fetchall()
        return self._rows_to_list(rows)

    def refresh_popular_destinations(self) -> int:
        """Recompute the popular_destinations table from live data.

        Aggregates trip_history and user_searches to produce
        up-to-date rankings.  Returns the number of destinations
        written.
        """
        # Aggregate from trip_history
        trip_agg = self._execute("""
            SELECT
                destination,
                COUNT(*)              AS trip_count,
                AVG(predicted_cost)   AS avg_cost,
                AVG(duration_days)    AS avg_days,
                MIN(predicted_cost)   AS min_cost,
                MAX(predicted_cost)   AS max_cost
            FROM trip_history
            GROUP BY destination
        """, commit=False).fetchall()

        # Aggregate from user_searches
        search_agg = self._execute("""
            SELECT destination, COUNT(*) AS search_count
            FROM user_searches
            GROUP BY destination
        """, commit=False).fetchall()
        search_map = {r["destination"]: r["search_count"] for r in search_agg}

        # Most common season per destination
        season_agg = self._execute("""
            SELECT destination, season, COUNT(*) AS cnt
            FROM trip_history
            GROUP BY destination, season
            ORDER BY cnt DESC
        """, commit=False).fetchall()
        season_map: Dict[str, str] = {}
        for r in season_agg:
            if r["destination"] not in season_map:
                season_map[r["destination"]] = r["season"]

        # Most common trip type per destination
        type_agg = self._execute("""
            SELECT destination, trip_type, COUNT(*) AS cnt
            FROM trip_history
            GROUP BY destination, trip_type
            ORDER BY cnt DESC
        """, commit=False).fetchall()
        type_map: Dict[str, str] = {}
        for r in type_agg:
            if r["destination"] not in type_map:
                type_map[r["destination"]] = r["trip_type"]

        count = 0
        for row in trip_agg:
            dest = row["destination"]
            s_count = search_map.get(dest, 0)
            t_count = row["trip_count"]
            trending = (s_count * 0.6) + (t_count * 0.4)   # weighted score

            self.upsert_popular_destination(
                destination=dest,
                search_count=s_count,
                trip_count=t_count,
                avg_predicted_cost=round(row["avg_cost"], 2),
                avg_duration_days=round(row["avg_days"], 1),
                min_predicted_cost=row["min_cost"],
                max_predicted_cost=row["max_cost"],
                most_common_season=season_map.get(dest),
                most_common_trip_type=type_map.get(dest),
                trending_score=round(trending, 2),
            )
            count += 1

        log.info("Refreshed popular_destinations: %d entries", count)
        return count

    def delete_popular_destination(self, destination: str) -> bool:
        """Remove a destination from the popular list."""
        affected = self._execute(
            "DELETE FROM popular_destinations WHERE destination = ?",
            (destination.title(),),
        ).rowcount
        return affected > 0

    # ============================================================
    # Maintenance Utilities
    # ============================================================

    def purge_all_expired(self) -> Dict[str, int]:
        """Run expiration cleanup on every cache table."""
        return {
            "distance_cache": self.purge_expired_distances(),
            "geocode_cache": self.purge_expired_geocodes(),
            "suggestions_cache": self.purge_expired_suggestions(),
        }

    def get_table_stats(self) -> Dict[str, int]:
        """Return row counts for every table."""
        tables = [
            "trip_history", "user_searches", "distance_cache",
            "geocode_cache", "suggestions_cache", "popular_destinations",
        ]
        stats = {}
        for t in tables:
            cur = self._execute(f"SELECT COUNT(*) FROM {t}", commit=False)
            stats[t] = cur.fetchone()[0]
        return stats

    def vacuum(self) -> None:
        """Reclaim unused disk space."""
        self._conn.execute("VACUUM")
        log.info("Database vacuumed")

    def export_table_to_csv(self, table_name: str, output_path: Union[str, Path]) -> int:
        """Dump a table to a CSV file. Returns row count."""
        import csv

        allowed = {
            "trip_history", "user_searches", "distance_cache",
            "suggestions_cache", "popular_destinations",
        }
        if table_name not in allowed:
            raise ValueError(f"Unknown table: {table_name}")

        cur = self._execute(f"SELECT * FROM {table_name}", commit=False)
        rows = cur.fetchall()
        if not rows:
            log.warning("No rows to export from %s", table_name)
            return 0

        cols = rows[0].keys()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow(dict(r))

        log.info("Exported %d rows from %s → %s", len(rows), table_name, output_path)
        return len(rows)
