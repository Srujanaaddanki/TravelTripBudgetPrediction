"""
TripAI -- Database Usage Examples
=================================

Run this file directly to exercise every database operation:

    python -m src.data.examples

It creates a temporary database, runs through all CRUD methods,
prints the results, and cleans up.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Fix Windows console encoding so the script runs on any terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

# Ensure the project root is on sys.path so ``src`` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.database import TripDatabase


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    # Use a temp file so the demo doesn't pollute the real database
    tmp_dir = _PROJECT_ROOT / "data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "example_demo.db"

    print(f"[DIR] Using database: {db_path}\n")

    with TripDatabase(db_path) as db:

        # -----------------------------------------------
        # 1. TRIP HISTORY
        # -----------------------------------------------
        separator("1. TRIP HISTORY -- CRUD")

        # CREATE
        trip1 = db.add_trip(
            destination="Manali",
            season="Winter",
            month="December",
            trip_type="Friends trip",
            hotel_quality="Excellent",
            duration_days=5,
            predicted_cost=15000.0,
            source_location="Delhi",
            model_version="1.0",
        )
        trip2 = db.add_trip(
            destination="Goa",
            season="Summer",
            month="May",
            trip_type="Couple",
            hotel_quality="Good",
            duration_days=3,
            predicted_cost=22000.0,
            source_location="Mumbai",
        )
        trip3 = db.add_trip(
            destination="Manali",
            season="Summer",
            month="June",
            trip_type="Family",
            hotel_quality="Luxury",
            duration_days=7,
            predicted_cost=45000.0,
        )
        print(f"[OK] Created trips: #{trip1}, #{trip2}, #{trip3}")

        # READ -- single
        trip = db.get_trip(trip1)
        print(f"\n[READ] Trip #{trip1}: {trip['destination']}, "
              f"Rs.{trip['predicted_cost']:,.0f}, "
              f"{trip['duration_days']} days")

        # READ -- filtered
        manali_trips = db.get_all_trips(destination="Manali")
        print(f"[READ] Trips to Manali: {len(manali_trips)}")

        # UPDATE -- add feedback
        db.update_trip(trip1, actual_cost=14200, satisfaction=5, notes="Amazing snow!")
        updated = db.get_trip(trip1)
        print(f"\n[EDIT] Updated trip #{trip1}: actual=Rs.{updated['actual_cost']:,.0f}, "
              f"satisfaction={updated['satisfaction']}*")

        # COUNT
        print(f"\n[STATS] Total trips: {db.count_trips()}")

        # DELETE
        db.delete_trip(trip2)
        print(f"[DEL] Deleted trip #{trip2} -- remaining: {db.count_trips()}")

        # -----------------------------------------------
        # 2. USER SEARCHES
        # -----------------------------------------------
        separator("2. USER SEARCHES -- Log & Query")

        s1 = db.log_search(
            destination="Shimla",
            session_id="sess-abc-001",
            season="Winter",
            month="January",
            trip_type="Solo",
            hotel_quality="Good",
            duration_days=4,
            predicted_cost=8000.0,
            search_params={"source": "Delhi", "budget_range": "economy"},
        )
        s2 = db.log_search(
            destination="Shimla",
            session_id="sess-abc-001",
            season="Summer",
            predicted_cost=6500.0,
        )
        s3 = db.log_search(
            destination="Kerala",
            session_id="sess-xyz-002",
            season="Monsoon",
            predicted_cost=18000.0,
        )
        print(f"[OK] Logged searches: #{s1}, #{s2}, #{s3}")

        # Query by session
        session_searches = db.get_searches(session_id="sess-abc-001")
        print(f"[READ] Searches in session 'sess-abc-001': {len(session_searches)}")

        # Stats
        stats = db.get_search_stats()
        print(f"[STATS] Total searches: {stats['total_searches']}")
        print(f"   Top destinations: "
              + ", ".join(f"{d['destination']}({d['cnt']})" for d in stats['top_destinations']))

        # -----------------------------------------------
        # 3. DISTANCE CACHE
        # -----------------------------------------------
        separator("3. DISTANCE CACHE -- Store & Retrieve")

        db.cache_distance("Delhi", "Manali", 537.0, duration_hours=12.5)
        db.cache_distance("Mumbai", "Goa", 588.0, duration_hours=10.0)
        db.cache_distance("Delhi", "Shimla", 342.0, duration_hours=7.5)
        print("[OK] Cached 3 distance entries")

        # Lookup
        dist = db.get_distance("Delhi", "Manali")
        if dist:
            print(f"[HIT] Delhi -> Manali: {dist['distance_km']} km, "
                  f"~{dist['duration_hours']} hrs (expires: {dist['expires_at'][:10]})")

        # Miss
        miss = db.get_distance("Chennai", "Leh")
        print(f"[MISS] Chennai -> Leh (not cached): {miss}")

        # List all
        all_dist = db.get_all_distances()
        print(f"[STATS] Active distance cache entries: {len(all_dist)}")

        # Purge (nothing expired yet)
        purged = db.purge_expired_distances()
        print(f"[PURGE] Purged expired distances: {purged}")

        # -----------------------------------------------
        # 4. SUGGESTIONS CACHE
        # -----------------------------------------------
        separator("4. SUGGESTIONS CACHE -- Store, Retrieve, Expire")

        db.cache_suggestion(
            cache_key="manali_winter_tips",
            suggestion_data={
                "packing": ["Thermal wear", "Snow boots", "Sunscreen"],
                "budget_tips": "Book hotels 2 months in advance for 30% savings",
                "must_visit": ["Solang Valley", "Old Manali", "Rohtang Pass"],
            },
            category="travel_tips",
            ttl_hours=168,  # 1 week
        )
        db.cache_suggestion(
            cache_key="goa_budget_breakdown",
            suggestion_data={
                "accommodation": 8000,
                "food": 4000,
                "transport": 3000,
                "activities": 5000,
            },
            category="budget",
        )
        print("[OK] Cached 2 suggestions")

        # Retrieve (auto-increments hit count)
        tip = db.get_suggestion("manali_winter_tips")
        if tip:
            print(f"[TIP] Suggestion: {tip['cache_key']}")
            print(f"   Packing: {tip['suggestion_data']['packing']}")
            print(f"   Hit count: {tip['hit_count']}")

        # By category
        budgets = db.get_suggestions_by_category("budget")
        print(f"[STATS] Budget suggestions: {len(budgets)}")

        # -----------------------------------------------
        # 5. POPULAR DESTINATIONS
        # -----------------------------------------------
        separator("5. POPULAR DESTINATIONS -- Aggregate & Rank")

        # Refresh from live data (trip_history + user_searches)
        refreshed = db.refresh_popular_destinations()
        print(f"[REFRESH] Refreshed {refreshed} destination(s) from live data")

        # Read rankings
        popular = db.get_popular_destinations(limit=5)
        print("\n[RANK] Top Destinations:")
        for i, p in enumerate(popular, 1):
            print(f"   {i}. {p['destination']} -- "
                  f"score={p['trending_score']}, "
                  f"searches={p['search_count']}, "
                  f"trips={p['trip_count']}, "
                  f"avg Rs.{p['avg_predicted_cost']:,.0f}")

        # Manual upsert
        db.upsert_popular_destination(
            destination="Ladakh",
            search_count=200,
            trip_count=85,
            avg_predicted_cost=35000.0,
            avg_duration_days=8.0,
            most_common_season="Summer",
            trending_score=155.0,
        )
        print("\n[OK] Manually added Ladakh to popular destinations")

        # -----------------------------------------------
        # 6. MAINTENANCE UTILITIES
        # -----------------------------------------------
        separator("6. MAINTENANCE UTILITIES")

        # Table stats
        table_stats = db.get_table_stats()
        print("[STATS] Table Row Counts:")
        for table, count in table_stats.items():
            print(f"   * {table}: {count}")

        # Purge all expired caches
        purge_results = db.purge_all_expired()
        print(f"\n[PURGE] Cache purge results: {purge_results}")

        # Export a table to CSV
        export_path = tmp_dir / "exported_trips.csv"
        exported = db.export_table_to_csv("trip_history", export_path)
        print(f"[EXPORT] Exported {exported} trip(s) to {export_path.name}")

        # Vacuum
        db.vacuum()
        print("[VACUUM] Database vacuumed")

    # -----------------------------------------------
    # Cleanup demo files
    # -----------------------------------------------
    separator("CLEANUP")
    for f in [db_path, export_path]:
        if f.exists():
            f.unlink()
            print(f"[DEL] Removed {f.name}")
    print("\n[DONE] All examples completed successfully!")


if __name__ == "__main__":
    main()
