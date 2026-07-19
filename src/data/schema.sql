-- ============================================================
-- TripAI — SQLite Database Schema
-- Version: 1.0.0
-- Description: Production schema for the Travel Budget
--              Prediction system. Covers trip logging,
--              search analytics, distance/suggestion caching,
--              and popular-destination aggregation.
-- ============================================================

-- -------------------------------------------------------
-- 1. TRIP HISTORY
--    Stores every completed budget prediction with its
--    inputs, result, and optional user feedback.
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS trip_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    destination       TEXT    NOT NULL,
    source_location   TEXT,
    season            TEXT    NOT NULL,
    month             TEXT    NOT NULL,
    trip_type         TEXT    NOT NULL,
    hotel_quality     TEXT    NOT NULL,
    duration_days     INTEGER NOT NULL CHECK (duration_days > 0),
    predicted_cost    REAL    NOT NULL CHECK (predicted_cost >= 0),
    actual_cost       REAL             CHECK (actual_cost IS NULL OR actual_cost >= 0),
    model_version     TEXT    NOT NULL DEFAULT '1.0',
    satisfaction      INTEGER          CHECK (satisfaction IS NULL OR (satisfaction >= 1 AND satisfaction <= 5)),
    notes             TEXT,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trip_destination   ON trip_history(destination);
CREATE INDEX IF NOT EXISTS idx_trip_created        ON trip_history(created_at);
CREATE INDEX IF NOT EXISTS idx_trip_season          ON trip_history(season);

-- -------------------------------------------------------
-- 2. USER SEARCHES
--    Logs every prediction request for analytics
--    (most-searched destinations, peak hours, etc.).
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_searches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL,
    source_location   TEXT,
    destination       TEXT    NOT NULL,
    season            TEXT,
    month             TEXT,
    trip_type         TEXT,
    hotel_quality     TEXT,
    duration_days     INTEGER,
    travel_mode       TEXT,
    predicted_cost    REAL,
    search_params     TEXT,                        -- full input as JSON blob
    ip_address        TEXT,
    user_agent        TEXT,
    searched_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_session     ON user_searches(session_id);
CREATE INDEX IF NOT EXISTS idx_search_destination ON user_searches(destination);
CREATE INDEX IF NOT EXISTS idx_search_timestamp   ON user_searches(searched_at);
CREATE INDEX IF NOT EXISTS idx_search_travel_mode ON user_searches(travel_mode);
CREATE INDEX IF NOT EXISTS idx_search_month       ON user_searches(month);

-- -------------------------------------------------------
-- 3. DISTANCE CACHE
--    Caches origin→destination distances so repeated
--    lookups avoid redundant API calls.
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS distance_cache (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    origin            TEXT    NOT NULL,
    destination       TEXT    NOT NULL,
    distance_km       REAL    NOT NULL CHECK (distance_km >= 0),
    duration_hours    REAL             CHECK (duration_hours IS NULL OR duration_hours >= 0),
    travel_mode       TEXT    NOT NULL DEFAULT 'driving',
    cached_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at        TIMESTAMP NOT NULL,
    UNIQUE(origin, destination, travel_mode)
);

CREATE INDEX IF NOT EXISTS idx_dist_route   ON distance_cache(origin, destination);
CREATE INDEX IF NOT EXISTS idx_dist_expires ON distance_cache(expires_at);

-- -------------------------------------------------------
-- 3b. GEOCODE CACHE
--     Caches latitude/longitude coordinates per city
--     to avoid repeated Geocoding API calls.
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS geocode_cache (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    place_name        TEXT    NOT NULL UNIQUE,
    latitude          REAL    NOT NULL,
    longitude         REAL    NOT NULL,
    formatted_address TEXT,
    cached_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at        TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_geo_place   ON geocode_cache(place_name);
CREATE INDEX IF NOT EXISTS idx_geo_expires ON geocode_cache(expires_at);

-- -------------------------------------------------------
-- 4. SUGGESTIONS CACHE
--    General-purpose cache for ML-generated suggestions
--    (itinerary tips, budget breakdowns, etc.)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS suggestions_cache (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key         TEXT    NOT NULL UNIQUE,
    suggestion_data   TEXT    NOT NULL,              -- JSON payload
    category          TEXT    NOT NULL DEFAULT 'general',
    hit_count         INTEGER NOT NULL DEFAULT 0,
    cached_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at        TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sugg_key      ON suggestions_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_sugg_category ON suggestions_cache(category);
CREATE INDEX IF NOT EXISTS idx_sugg_expires  ON suggestions_cache(expires_at);

-- -------------------------------------------------------
-- 5. POPULAR DESTINATIONS
--    Materialised aggregate of search/trip data,
--    refreshed periodically for dashboard widgets.
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS popular_destinations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    destination           TEXT    NOT NULL UNIQUE,
    search_count          INTEGER NOT NULL DEFAULT 0,
    trip_count            INTEGER NOT NULL DEFAULT 0,
    avg_predicted_cost    REAL    NOT NULL DEFAULT 0.0,
    avg_duration_days     REAL    NOT NULL DEFAULT 0.0,
    min_predicted_cost    REAL,
    max_predicted_cost    REAL,
    most_common_season    TEXT,
    most_common_trip_type TEXT,
    trending_score        REAL    NOT NULL DEFAULT 0.0,
    last_updated          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pop_destination ON popular_destinations(destination);
CREATE INDEX IF NOT EXISTS idx_pop_trending    ON popular_destinations(trending_score DESC);

-- -------------------------------------------------------
-- 6. DESTINATION INTELLIGENCE CACHE
--    Caches detailed geocoding, country, state, weather profile,
--    and tourism category for destinations.
-- -------------------------------------------------------
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
);

CREATE INDEX IF NOT EXISTS idx_dest_intel_ts ON destination_intelligence_cache(timestamp);

-- -------------------------------------------------------
-- 7. UNKNOWN DESTINATION HISTORY
--    Stores unknown destination predictions and their resolved
--    proxies for future retraining.
-- -------------------------------------------------------
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
);

CREATE INDEX IF NOT EXISTS idx_unk_dest_date ON unknown_destination_history(timestamp DESC);
