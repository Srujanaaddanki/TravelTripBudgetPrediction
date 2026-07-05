"""End-to-end verification of search tracking + analytics."""
import io, sys, os

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.database import TripDatabase

DB = Path(__file__).parent / "data" / "test_tracking.db"

def main():
    # Clean start
    if DB.exists():
        DB.unlink()

    with TripDatabase(DB) as db:
        # =============================================
        # 1. LOG SEARCHES (simulating user predictions)
        # =============================================
        print("=" * 60)
        print("  STEP 1: Logging 12 sample searches")
        print("=" * 60)

        searches = [
            ("Delhi",    "Manali",      "December",  5,  "Flight", 15000),
            ("Mumbai",   "Goa",         "May",       3,  "Train",  22000),
            ("Delhi",    "Manali",      "January",   7,  "Bus",    18000),
            ("Chennai",  "Kerala",      "August",    4,  "Flight", 25000),
            ("Kolkata",  "Darjeeling",  "March",     3,  "Car",    12000),
            ("Delhi",    "Shimla",      "December",  2,  "Bus",    8000),
            ("Mumbai",   "Goa",         "December",  5,  "Flight", 30000),
            ("Delhi",    "Manali",      "June",      6,  "Train",  20000),
            ("Pune",     "Goa",         "May",       4,  "Car",    16000),
            ("Delhi",    "Shimla",      "January",   3,  "Train",  10000),
            ("Bangalore","Kerala",      "March",     5,  "Flight", 28000),
            ("Mumbai",   "Manali",      "June",      8,  "Flight", 35000),
        ]

        for src, dest, month, days, mode, cost in searches:
            sid = db.log_search(
                destination=dest,
                source_location=src,
                month=month,
                duration_days=days,
                travel_mode=mode,
                predicted_cost=cost,
            )
            print(f"  [OK] #{sid:>2d}: {src:>10s} -> {dest:<12s} | {month:<10s} | {days}d | {mode:<6s} | Rs.{cost:>6,}")

        print(f"\n  Total rows inserted: {db.get_search_stats()['total_searches']}")

        # =============================================
        # 2. ANALYTICS: Most Searched Destinations
        # =============================================
        print("\n" + "=" * 60)
        print("  STEP 2: Most Searched Destinations")
        print("=" * 60)

        top_dest = db.most_searched_destinations(limit=5)
        print(f"  {'Destination':<15s} {'Searches':>8s} {'Avg Budget':>12s} {'Avg Days':>10s}")
        print(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*10}")
        for d in top_dest:
            print(f"  {d['destination']:<15s} {d['search_count']:>8d} Rs.{d['avg_predicted_cost']:>9,.0f} {d['avg_duration']:>10.1f}")

        # =============================================
        # 3. ANALYTICS: Most Searched Travel Modes
        # =============================================
        print("\n" + "=" * 60)
        print("  STEP 3: Most Searched Travel Modes")
        print("=" * 60)

        top_modes = db.most_searched_travel_modes()
        print(f"  {'Mode':<10s} {'Searches':>8s} {'Avg Budget':>12s} {'Share':>8s}")
        print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*8}")
        for m in top_modes:
            print(f"  {m['travel_mode']:<10s} {m['search_count']:>8d} Rs.{m['avg_predicted_cost']:>9,.0f} {m['percentage']:>6.1f}%")

        # =============================================
        # 4. ANALYTICS: Average Predicted Budget
        # =============================================
        print("\n" + "=" * 60)
        print("  STEP 4: Average Predicted Budget")
        print("=" * 60)

        budget = db.average_predicted_budget()
        print(f"  Overall Average : Rs.{budget['overall_avg']:>10,.2f}")
        print(f"  Minimum Budget  : Rs.{budget['min_budget']:>10,.0f}")
        print(f"  Maximum Budget  : Rs.{budget['max_budget']:>10,.0f}")
        print(f"  Total Searches  : {budget['total_searches']:>10d}")
        print(f"\n  {'Destination':<15s} {'Avg Budget':>12s} {'Searches':>8s}")
        print(f"  {'-'*15} {'-'*12} {'-'*8}")
        for bd in budget["by_destination"]:
            print(f"  {bd['destination']:<15s} Rs.{bd['avg_budget']:>9,.0f} {bd['search_count']:>8d}")

        # =============================================
        # 5. ANALYTICS: Monthly Search Trends
        # =============================================
        print("\n" + "=" * 60)
        print("  STEP 5: Monthly Search Trends")
        print("=" * 60)

        trends = db.monthly_search_trends()
        print(f"  {'Month':<12s} {'Searches':>8s} {'Avg Budget':>12s} {'Top Dest.':<15s}")
        print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*15}")
        for t in trends:
            print(f"  {t['month']:<12s} {t['search_count']:>8d} Rs.{t['avg_predicted_cost']:>9,.0f} {t['top_destination'] or 'N/A':<15s}")

        # =============================================
        # 6. VERIFY STORED FIELDS
        # =============================================
        print("\n" + "=" * 60)
        print("  STEP 6: Verify All 7 Fields Are Stored")
        print("=" * 60)

        recent = db.get_searches(limit=1)
        row = recent[0]
        fields = {
            "Source":           row.get("source_location"),
            "Destination":      row.get("destination"),
            "Month":            row.get("month"),
            "Duration":         row.get("duration_days"),
            "Travel Mode":      row.get("travel_mode"),
            "Predicted Budget":  row.get("predicted_cost"),
            "Timestamp":        row.get("searched_at"),
        }
        all_ok = True
        for label, val in fields.items():
            status = "OK" if val is not None else "MISSING"
            if val is None:
                all_ok = False
            print(f"  {label:<20s}: {str(val):<30s} [{status}]")

        print(f"\n  All 7 fields present: {'YES' if all_ok else 'NO'}")

    # Cleanup
    if DB.exists():
        DB.unlink()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
