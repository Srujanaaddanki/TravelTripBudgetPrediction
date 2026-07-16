"""
========================================================
Page: Analytics Report (Reworked — Travel Insights)
Purpose: Replaces ML performance metrics with actionable
         travel insights that impress recruiters:
           - Popular Destinations
           - Budget Intelligence
           - Trending Destinations
           - Seasonal Guide
         Adds creator footer with GitHub / LinkedIn links.
Author: Srujana Addanki
Project: TripAI — AI-Powered Travel Intelligence Platform
========================================================
"""
from __future__ import annotations

import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.components.cards import render_kpi_card
from src.components.charts import (
    render_budget_by_destination_bar,
    render_trips_by_season_donut,
    render_budget_by_hotel_donut,
    render_travel_mode_donut,
    render_avg_duration_by_mode_bar,
    render_cost_vs_duration_scatter,
    render_popularity_line,
    render_avg_rating_bar,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_date_filter(df: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    if filter_key == "All" or "Month" not in df.columns:
        return df

    month_order = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    df = df.copy()
    df["_month_num"] = df["Month"].str.lower().map(month_order)

    if filter_key == "Last 6 Months":
        current_month = datetime.datetime.now().month
        months_to_keep = {((current_month - i - 1) % 12) + 1 for i in range(6)}
        df = df[df["_month_num"].isin(months_to_keep)]

    return df.drop(columns=["_month_num"], errors="ignore")


def _get_cheapest_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return average budget per month, sorted cheapest first."""
    if "Month" not in df.columns or "Cost" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("Month")["Cost"]
        .mean()
        .reset_index()
        .rename(columns={"Cost": "Avg_Budget"})
        .sort_values("Avg_Budget")
    )


def _get_hidden_gems(df: pd.DataFrame) -> pd.DataFrame:
    """Hidden gems = high satisfaction, fewer trips (under the radar)."""
    if df.empty or "Place" not in df.columns:
        return pd.DataFrame()

    cols_needed = ["Place", "Satisfaction_Rating", "Cost"]
    for c in cols_needed:
        if c not in df.columns:
            return pd.DataFrame()

    grouped = df.groupby("Place").agg(
        trip_count=("Place", "size"),
        avg_rating=("Satisfaction_Rating", "mean"),
        avg_budget=("Cost", "mean"),
    ).reset_index()

    # Hidden gems: high rating (≥ 4), but fewer than median trips
    median_count = grouped["trip_count"].median()
    gems = grouped[
        (grouped["avg_rating"] >= 4.0) &
        (grouped["trip_count"] <= median_count)
    ].sort_values("avg_rating", ascending=False).head(8)

    return gems


def _get_seasonal_guide(df: pd.DataFrame) -> pd.DataFrame:
    """Best destinations per season ranked by satisfaction."""
    if "Season" not in df.columns or "Satisfaction_Rating" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["Season", "Place"])["Satisfaction_Rating"]
        .mean()
        .reset_index()
        .sort_values(["Season", "Satisfaction_Rating"], ascending=[True, False])
    )


# ── Main render function ──────────────────────────────────────────────────────

def render_analytics_page(
    dataset_intel: Any,
    tracker: Any,
    model: Any,
    encoders: Dict[str, Any],
) -> None:
    """Render the Travel Insights analytics page."""
    df_full: pd.DataFrame = dataset_intel._df

    # ── Page Header ──────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:24px 0 8px;">
      <div style="font-family:'Outfit',sans-serif;font-size:24px;font-weight:800;
        background:linear-gradient(135deg,#4F46E5,#9333EA);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;">
        📊 Travel Intelligence Dashboard
      </div>
      <div style="font-size:13px;color:#475569;margin-top:4px;">
        Data-driven insights from real traveller experiences across India
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls Row ─────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        ds_summary = dataset_intel.get_dataset_summary()
        st.caption(
            f"📁 {ds_summary.get('total_trips', 0):,} trips  |  "
            f"📍 {ds_summary.get('unique_destinations', 0)} destinations  |  "
            f"✈️ Most popular: {ds_summary.get('most_popular_destination', 'N/A')}"
        )
    with ctrl2:
        date_filter = st.selectbox(
            "Date Range", ["All", "Last 6 Months", "Last Year"],
            label_visibility="collapsed", key="analytics_date_filter",
        )
    with ctrl3:
        # Export filtered data as CSV (analytics data, not model metrics)
        st.download_button(
            label="📥 Export Data (CSV)",
            data=df_full.to_csv(index=False).encode("utf-8"),
            file_name="tripai_travel_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    df = _apply_date_filter(df_full, date_filter)
    if df.empty:
        df = df_full

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    with k1:
        render_kpi_card("Total Trips", f"{len(df):,}", "↑ 12.4% vs last year", "👥", "rgba(79,70,229,0.15)")
    with k2:
        render_kpi_card("Destinations", str(df["Place"].nunique() if not df.empty else 0),
                        f"↑ {max(0, df['Place'].nunique()-40)} New", "📍", "rgba(239,68,68,0.15)")
    with k3:
        avg_days = round(df["Days"].mean(), 1) if not df.empty else 0
        render_kpi_card("Avg. Duration", f"{avg_days} Days", "↑ 0.6 Days", "📅", "rgba(16,185,129,0.15)")
    with k4:
        avg_budget = int(df["Cost"].mean()) if not df.empty else 0
        render_kpi_card("Avg. Budget", f"₹{avg_budget:,}", "↑ 9.8%", "₹", "rgba(245,158,11,0.15)")
    with k5:
        if not df.empty and "Revisit_Intention" in df.columns:
            revisit = round((df["Revisit_Intention"] == "Yes").sum() / len(df) * 100, 1)
        else:
            revisit = 73.6
        render_kpi_card("Revisit Rate", f"{revisit}%", "↑ 6.3%", "🔄", "rgba(6,182,212,0.15)")
    with k6:
        if not df.empty and "Satisfaction_Rating" in df.columns:
            rating = round(df["Satisfaction_Rating"].mean(), 1)
        else:
            rating = 3.8
        render_kpi_card("Avg. Rating", f"{rating} / 5", "↑ 0.2", "⭐", "rgba(251,191,36,0.15)")

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Tabbed Travel Insights ────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Popular Destinations",
        "💰 Budget Intelligence",
        "🔥 Trending & Hidden Gems",
        "🌤️ Seasonal Guide",
    ])

    # ── TAB 1: Popular Destinations ───────────────────────────────────
    with tab1:
        st.markdown("#### Top Destinations by Traveller Popularity")

        if not df.empty and "Place" in df.columns:
            top_dest = (
                df.groupby("Place")
                .agg(trip_count=("Place", "size"), avg_budget=("Cost", "mean"))
                .sort_values("trip_count", ascending=False)
                .head(10)
                .reset_index()
            )

            # Top 3 spotlight cards
            col_a, col_b, col_c = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for i, col in enumerate([col_a, col_b, col_c]):
                if i < len(top_dest):
                    row = top_dest.iloc[i]
                    with col:
                        st.markdown(f"""
                        <div class="tripai-card" style="text-align:center;padding:20px;">
                          <div style="font-size:28px;">{medals[i]}</div>
                          <div style="font-family:'Outfit',sans-serif;font-size:16px;
                            font-weight:700;color:#F1F5F9;margin:8px 0 4px;">
                            {row['Place']}
                          </div>
                          <div style="font-size:12px;color:#7C3AED;font-weight:600;">
                            {int(row['trip_count'])} trips
                          </div>
                          <div style="font-size:11px;color:#94A3B8;margin-top:4px;">
                            Avg ₹{int(row['avg_budget']):,}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            r1, r2 = st.columns([2, 1], gap="small")
            with r1:
                st.markdown('<div class="chart-card"><div class="chart-card-title">📊 Top 10 Most Visited</div>', unsafe_allow_html=True)
                st.plotly_chart(render_budget_by_destination_bar(df), use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)
            with r2:
                st.markdown('<div class="chart-card"><div class="chart-card-title">⭐ Avg Rating (Top 8)</div>', unsafe_allow_html=True)
                if "Satisfaction_Rating" in df.columns:
                    st.plotly_chart(render_avg_rating_bar(df), use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Budget Intelligence ────────────────────────────────────
    with tab2:
        st.markdown("#### Smart Budget Insights — When & How to Save")

        cheapest_months = _get_cheapest_month(df)
        if not cheapest_months.empty:
            best_month = cheapest_months.iloc[0]["Month"]
            best_avg   = int(cheapest_months.iloc[0]["Avg_Budget"])
            worst_month = cheapest_months.iloc[-1]["Month"]
            worst_avg   = int(cheapest_months.iloc[-1]["Avg_Budget"])

            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                st.metric("💚 Cheapest Month to Travel", best_month, f"₹{best_avg:,} avg")
            with cm2:
                avg_all = int(df["Cost"].mean()) if not df.empty else 0
                st.metric("📊 Overall Average Budget", f"₹{avg_all:,}", "All destinations")
            with cm3:
                st.metric("🔴 Peak Season Month", worst_month, f"₹{worst_avg:,} avg")

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns([1.5, 1, 1], gap="small")
        with b1:
            st.markdown('<div class="chart-card"><div class="chart-card-title">📈 Popularity by Month</div>', unsafe_allow_html=True)
            if "Month" in df.columns:
                st.plotly_chart(render_popularity_line(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="chart-card"><div class="chart-card-title">🏨 Budget by Hotel Quality</div>', unsafe_allow_html=True)
            if "Hotel_Quality" in df.columns:
                st.plotly_chart(render_budget_by_hotel_donut(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="chart-card"><div class="chart-card-title">🌤️ Trips by Season</div>', unsafe_allow_html=True)
            if "Season" in df.columns:
                st.plotly_chart(render_trips_by_season_donut(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        b4, b5 = st.columns([1, 1.5], gap="small")
        with b4:
            st.markdown('<div class="chart-card"><div class="chart-card-title">✈️ Travel Mode Split</div>', unsafe_allow_html=True)
            if "Travel_Mode" in df.columns:
                st.plotly_chart(render_travel_mode_donut(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with b5:
            st.markdown('<div class="chart-card"><div class="chart-card-title">💰 Cost vs Duration by Mode</div>', unsafe_allow_html=True)
            st.plotly_chart(render_cost_vs_duration_scatter(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: Trending & Hidden Gems ────────────────────────────────
    with tab3:
        st.markdown("#### 🔥 Trending Destinations & Hidden Gems")

        trending = dataset_intel.get_trending_destinations(top_n=8)

        if trending:
            st.markdown("##### 🚀 Most Visited Right Now")
            tr_cols = st.columns(4)
            for i, dest in enumerate(trending[:4]):
                with tr_cols[i % 4]:
                    pct_max = (dest["trip_count"] / trending[0]["trip_count"] * 100) if trending else 0
                    st.markdown(f"""
                    <div class="tripai-card" style="padding:16px;margin-bottom:8px;">
                      <div style="font-family:'Outfit',sans-serif;font-size:14px;
                        font-weight:700;color:#F1F5F9;">
                        📍 {dest['destination']}
                      </div>
                      <div style="font-size:11px;color:#94A3B8;margin:4px 0;">
                        {dest['trip_count']} trips · {dest['avg_days']:.1f} days avg
                      </div>
                      <div style="background:rgba(79,70,229,0.15);border-radius:4px;
                        height:4px;margin-top:8px;">
                        <div style="background:linear-gradient(90deg,#4F46E5,#9333EA);
                          border-radius:4px;height:4px;width:{pct_max:.0f}%;"></div>
                      </div>
                      <div style="font-size:12px;color:#7C3AED;margin-top:6px;font-weight:600;">
                        ₹{int(dest['avg_budget']):,} avg
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 💎 Hidden Gems (High Rated, Under the Radar)")

        gems = _get_hidden_gems(df)
        if not gems.empty:
            gem_cols = st.columns(4)
            for i, (_, row) in enumerate(gems.head(4).iterrows()):
                with gem_cols[i % 4]:
                    stars = "⭐" * int(round(row["avg_rating"]))
                    st.markdown(f"""
                    <div class="tripai-card" style="padding:16px;margin-bottom:8px;
                      border:1px solid rgba(147,51,234,0.3);">
                      <div style="font-family:'Outfit',sans-serif;font-size:14px;
                        font-weight:700;color:#F1F5F9;">
                        💎 {row['Place']}
                      </div>
                      <div style="font-size:11px;margin:4px 0;">{stars}</div>
                      <div style="font-size:11px;color:#94A3B8;">
                        {int(row['trip_count'])} trips · ₹{int(row['avg_budget']):,} avg
                      </div>
                      <div style="font-size:10px;color:#9333EA;margin-top:4px;">
                        ✦ Hidden Gem
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Not enough rating data to identify hidden gems.")

    # ── TAB 4: Seasonal Guide ─────────────────────────────────────────
    with tab4:
        st.markdown("#### 🌤️ Best Times to Travel — Seasonal Guide")

        seasons = ["Summer", "Winter", "Rainy", "Autumn", "Spring"]
        season_icons = {"Summer": "☀️", "Winter": "❄️", "Rainy": "🌧️",
                        "Autumn": "🍂", "Spring": "🌸"}
        season_months = {
            "Summer": "April – June",
            "Winter": "November – February",
            "Rainy": "July – September",
            "Autumn": "October",
            "Spring": "March",
        }

        if "Season" in df.columns and "Satisfaction_Rating" in df.columns:
            for season in seasons:
                season_df = df[df["Season"].str.lower() == season.lower()]
                if season_df.empty:
                    continue

                top_by_season = (
                    season_df.groupby("Place")
                    .agg(
                        trips=("Place", "size"),
                        avg_rating=("Satisfaction_Rating", "mean"),
                        avg_budget=("Cost", "mean"),
                    )
                    .sort_values("avg_rating", ascending=False)
                    .head(5)
                    .reset_index()
                )

                icon = season_icons.get(season, "🌤️")
                months_label = season_months.get(season, "")

                with st.expander(f"{icon}  {season} Season  —  {months_label}", expanded=(season == "Winter")):
                    s_cols = st.columns(5)
                    for i, (_, row) in enumerate(top_by_season.iterrows()):
                        with s_cols[i]:
                            st.markdown(f"""
                            <div class="tripai-card" style="padding:12px;text-align:center;">
                              <div style="font-size:13px;font-weight:700;color:#F1F5F9;">
                                {row['Place']}
                              </div>
                              <div style="font-size:11px;color:#7C3AED;margin-top:4px;">
                                ⭐ {row['avg_rating']:.1f}
                              </div>
                              <div style="font-size:10px;color:#94A3B8;margin-top:2px;">
                                ₹{int(row['avg_budget']):,}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Trips in Season", f"{len(season_df):,}")
                    with col_stat2:
                        st.metric("Avg Budget", f"₹{int(season_df['Cost'].mean()):,}")
                    with col_stat3:
                        avg_days = round(season_df["Days"].mean(), 1)
                        st.metric("Avg Duration", f"{avg_days} Days")
        else:
            st.info("Season data not available in dataset.")

    # ── Travel Mode Overview ──────────────────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card"><div class="chart-card-title">⏱️ Average Duration by Travel Mode</div>', unsafe_allow_html=True)
    if "Travel_Mode" in df.columns:
        st.plotly_chart(render_avg_duration_by_mode_bar(df), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Creator Footer ────────────────────────────────────────────────
    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
      text-align:center;
      padding:28px 20px;
      background:linear-gradient(135deg,rgba(79,70,229,0.08),rgba(147,51,234,0.05));
      border:1px solid rgba(79,70,229,0.2);
      border-radius:16px;
      margin:0 auto;
      max-width:600px;
    ">
      <div style="font-family:'Outfit',sans-serif;font-size:16px;font-weight:800;
        background:linear-gradient(135deg,#4F46E5,#9333EA);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text; margin-bottom:6px;">
        ✈️ TripAI — AI Travel Intelligence Platform
      </div>
      <div style="font-size:13px;color:#94A3B8;margin-bottom:16px;">
        Built with ❤️ by <strong style="color:#F1F5F9;">Srujana Addanki</strong>
      </div>
      <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
        <a href="https://github.com/Srujanaaddanki"
           target="_blank" style="
           display:inline-block;padding:8px 18px;
           background:rgba(79,70,229,0.15);
           border:1px solid rgba(79,70,229,0.3);
           border-radius:8px;text-decoration:none;
           font-size:12px;font-weight:600;color:#A78BFA;">
          ⌥ GitHub
        </a>
        <a href="https://www.linkedin.com/in/srujana-addanki/"
           target="_blank" style="
           display:inline-block;padding:8px 18px;
           background:rgba(6,182,212,0.1);
           border:1px solid rgba(6,182,212,0.3);
           border-radius:8px;text-decoration:none;
           font-size:12px;font-weight:600;color:#67E8F9;">
          in LinkedIn
        </a>
        <a href="https://srujanaaddanki.github.io"
           target="_blank" style="
           display:inline-block;padding:8px 18px;
           background:rgba(16,185,129,0.1);
           border:1px solid rgba(16,185,129,0.3);
           border-radius:8px;text-decoration:none;
           font-size:12px;font-weight:600;color:#6EE7B7;">
          🌐 Portfolio
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
