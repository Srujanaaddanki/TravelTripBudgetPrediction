"""
========================================================
Module: Report Exporter
Purpose: Generates a premium, print-friendly HTML travel report
         containing the budget breakdown, route coordinates,
         weather forecast, packing checklist, and traveller ratings.
Author: Srujana
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""

from __future__ import annotations

from typing import Any, Dict


def generate_html_report(
    source: str,
    destination: str,
    month: str,
    duration: int,
    mode: str,
    hotel: str,
    predicted_cost: float,
    report_data: Dict[str, Any]
) -> str:
    """Compile trip details and intelligence stats into a clean HTML document.

    Parameters
    ----------
    source : str
        Origin city.
    destination : str
        Target destination.
    month : str
        Travel month.
    duration : int
        Stay duration.
    mode : str
        Travel transport mode.
    hotel : str
        Stay hotel quality.
    predicted_cost : float
        ML predicted cost.
    report_data : dict
        Unified intelligence dictionary containing weather, breakdown, tips.

    Returns
    -------
    str
        Styled HTML document string.
    """
    dest_insights = report_data.get("dataset_insights", {})
    weather = report_data.get("weather", {})
    intel = report_data.get("intelligence", {})
    tiers = report_data.get("budget_tiers", {})

    packing_items = "".join([f"<li>[ ] {item}</li>" for item in intel.get("packing_tips", [])])
    places = "".join([f"<li>📍 {place}</li>" for place in intel.get("places_to_visit", [])])
    foods = "".join([f"<li>🍜 {food}</li>" for food in intel.get("local_foods", [])])
    tips = "".join([f"<li>💡 {tip}</li>" for tip in intel.get("money_saving_tips", [])])

    # Classify budget grade
    if predicted_cost < 5000:
        category = "Budget Explorer 🏷️"
    elif predicted_cost < 10000:
        category = "Value Explorer ⚡"
    elif predicted_cost < 20000:
        category = "Comfort Voyager 💜"
    else:
        category = "Luxury Voyager 💎"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TripAI Premium Travel Report — {destination.title()}</title>
    <style>
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #333333;
            line-height: 1.6;
            margin: 40px;
            background-color: #FFFFFF;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2563EB;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .title {{
            font-size: 28px;
            font-weight: 800;
            color: #2563EB;
            margin: 0;
        }}
        .subtitle {{
            font-size: 14px;
            color: #666666;
            margin-top: 5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            background-color: #F8FAFC;
        }}
        .card h3 {{
            margin-top: 0;
            color: #1E293B;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 8px;
        }}
        .metric {{
            font-size: 24px;
            font-weight: 800;
            color: #2563EB;
            margin: 10px 0;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 6px;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #94A3B8;
            margin-top: 50px;
            border-top: 1px solid #E2E8F0;
            padding-top: 15px;
        }}
        @media print {{
            body {{ margin: 20px; }}
            .card {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">✈ Honor/TripAI Travel Intelligence Report</div>
        <div class="subtitle">AI-Powered Travel Budget & Route Plan Itinerary</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>📊 Trip Overview</h3>
            <p><strong>Route:</strong> {source.title()} to {destination.title()}</p>
            <p><strong>Duration:</strong> {duration} Days | <strong>Travel Month:</strong> {month}</p>
            <p><strong>Transport Mode:</strong> {mode} | <strong>Stay Quality:</strong> {hotel.title()} Stay</p>
            <p><strong>Category:</strong> {category}</p>
        </div>
        <div class="card">
            <h3>💰 Recommended Budget</h3>
            <div class="metric">₹{int(predicted_cost):,}</div>
            <p><strong>Min Tier:</strong> ₹{int(tiers.get('minimum', 0)):,}</p>
            <p><strong>Comfort Tier:</strong> ₹{int(tiers.get('comfort', 0)):,}</p>
            <p><strong>Luxury Tier:</strong> ₹{int(tiers.get('luxury', 0)):,}</p>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>📈 Traveller Insights</h3>
            <p>Based on {dest_insights.get('similar_count', 0)} historical trips.</p>
            <p>⭐ <strong>Overall Rating:</strong> {dest_insights.get('destination_score', 4.0)} / 5.0</p>
            <p>🏨 <strong>Hotel Grade rating:</strong> {dest_insights.get('avg_hotel_rating', 4.0)} / 5.0</p>
            <p>🚖 <strong>Local Transport Rating:</strong> {dest_insights.get('avg_transport_rating', 4.0)} / 5.0</p>
            <p>🔁 <strong>Revisit Intention:</strong> {dest_insights.get('revisit_percentage', 0.0)}% Yes</p>
        </div>
        <div class="card">
            <h3>🌤️ Destination Weather</h3>
            <p><strong>Current Temperature:</strong> {weather.get('temperature_c', 25.0)}°C</p>
            <p><strong>Conditions:</strong> {weather.get('description', 'Clear')}</p>
            <p><strong>Humidity:</strong> {weather.get('humidity', 50)}% | <strong>Wind:</strong> {weather.get('wind_speed', 10.0)} km/h</p>
            <p><strong>Best Season:</strong> {intel.get('best_time', 'October to March')}</p>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>🎒 Packing Checklist</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                {packing_items}
            </ul>
        </div>
        <div class="card">
            <h3>📝 Pre-Travel Checklist</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>[ ] Primary ID Proof (Aadhaar/Passport)</li>
                <li>[ ] Travel Tickets (Flight/Train)</li>
                <li>[ ] Hotel Booking Voucher</li>
                <li>[ ] Physical Cash & Credit Cards</li>
                <li>[ ] Charger & Power Bank</li>
                <li>[ ] Basic Medicines & First Aid</li>
            </ul>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>🏔️ Top Attractions</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                {places}
            </ul>
        </div>
        <div class="card">
            <h3>🍜 Local Delicacies</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                {foods}
            </ul>
        </div>
    </div>

    <div class="card">
        <h3>💡 Money Saving Suggestions</h3>
        <ul>
            {tips}
        </ul>
    </div>

    <div class="footer">
        Generated by TripAI — Premium Travel Planning Platform. © 2026.
    </div>
</body>
</html>
"""
    return html_content
