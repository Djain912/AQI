"""Streamlit dashboard for Real-Time AQI Predictor (Mumbai-focused)."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from data.fetcher import calculate_aqi_from_pm25, fetch_all_pollutants
from data.historical_store import HistoricalStore
from data.station_store import StationHistoricalStore
from modules.bayesian_network import AQIBayesianNetwork
from modules.fuzzy_logic import AQIFuzzySystem
from modules.neural_network import AQINeuralNetwork

CITY = "Mumbai"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]

CATEGORY_COLORS = {
    "Good": "#00B050",
    "Satisfactory": "#92D050",
    "Moderate": "#FFFF00",
    "Poor": "#FF9900",
    "Very Poor": "#FF0000",
    "Severe": "#800000",
}


def category_from_aqi(aqi: float | None) -> str:
    if aqi is None or pd.isna(aqi):
        return "Moderate"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def pivot_from_long(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty or not {"station", "pollutant", "avg"}.issubset(set(long_df.columns)):
        return pd.DataFrame()

    pivot = long_df.pivot_table(
        index=["station", "city", "state"],
        columns="pollutant",
        values="avg",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None

    meta_cols = [c for c in ["station", "lat", "lon", "last_update"] if c in long_df.columns]
    if "station" in meta_cols:
        meta = long_df.sort_values("last_update", ascending=False).drop_duplicates(subset=["station"])[meta_cols]
        pivot = pivot.merge(meta, on="station", how="left")

    for p in POLLUTANTS:
        if p not in pivot.columns:
            pivot[p] = float("nan")

    pivot["aqi_value"] = pivot["PM2.5"].apply(calculate_aqi_from_pm25)
    pivot["aqi_category"] = pivot["aqi_value"].apply(category_from_aqi)
    return pivot


def summarize_city(city: str, pivot_df: pd.DataFrame) -> Dict:
    means = {
        p: (float(pivot_df[p].mean()) if p in pivot_df.columns and not pivot_df[p].dropna().empty else None)
        for p in POLLUTANTS
    }

    now = datetime.now()
    hour = now.hour
    is_peak = 1 if hour in {7, 8, 9, 17, 18, 19, 20} else 0
    aqi_value = calculate_aqi_from_pm25(means.get("PM2.5"))

    return {
        "timestamp": now.isoformat(),
        "city": city,
        **means,
        "aqi_value": aqi_value,
        "aqi_category": category_from_aqi(aqi_value),
        "hour": hour,
        "is_peak": is_peak,
        "station_count": int(len(pivot_df)),
    }


@st.cache_data(ttl=300)
def cached_fetch_city(city: str):
    # Single API pull + local pivot avoids duplicate network calls per refresh.
    long_df = fetch_all_pollutants(city)
    pivot_df = pivot_from_long(long_df)
    summary = summarize_city(city, pivot_df) if not pivot_df.empty else {}
    return long_df, pivot_df, summary


def summary_from_history(df: pd.DataFrame) -> Dict:
    latest = df.sort_values("timestamp", ascending=False).iloc[0].to_dict()
    return {
        "timestamp": pd.to_datetime(latest.get("timestamp")).isoformat(),
        "city": CITY,
        "PM2.5": latest.get("PM2.5"),
        "PM10": latest.get("PM10"),
        "NO2": latest.get("NO2"),
        "SO2": latest.get("SO2"),
        "CO": latest.get("CO"),
        "OZONE": latest.get("OZONE"),
        "NH3": latest.get("NH3"),
        "aqi_value": latest.get("aqi_value"),
        "aqi_category": latest.get("aqi_category"),
        "hour": int(latest.get("hour", datetime.now().hour)),
        "is_peak": int(latest.get("is_peak", 0)),
        "station_count": int(latest.get("station_count", 0)),
    }


def render_aqi_badge(label: str, value: float | None, category: str):
    color = CATEGORY_COLORS.get(category, "#CCCCCC")
    text = "N/A" if value is None else f"{value:.1f}"
    st.markdown(
        f"""
        <div style='padding:14px;border-radius:12px;background:{color};font-weight:700;text-align:center;'>
            <div style='font-size:14px'>{label}</div>
            <div style='font-size:28px'>{text}</div>
            <div style='font-size:13px'>{category}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Real-Time AQI Predictor — India", layout="wide", page_icon="🌫️")

    for key, default in {
        "live_long_df": pd.DataFrame(),
        "live_pivot_df": pd.DataFrame(),
        "current_summary": {},
        "summary_source": "none",
        "prev_summary": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    store = HistoricalStore()  # city-level summary history
    station_store = StationHistoricalStore()  # station-level training history
    bn_model = AQIBayesianNetwork()
    fuzzy_model = AQIFuzzySystem()
    nn_model = AQINeuralNetwork()

    st.sidebar.header("CPCB Live Data — data.gov.in")
    st.sidebar.markdown("**City: Mumbai (fixed)**")

    do_fetch = st.sidebar.button("Fetch Live Data (Mumbai)")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("History Collection")
    batch_rounds = st.sidebar.number_input("Rounds", min_value=1, max_value=300, value=10, step=1)
    batch_interval = st.sidebar.number_input("Interval (sec)", min_value=0.0, max_value=10.0, value=0.5, step=0.5)
    collect_batches = st.sidebar.button("Collect API Batches")

    st.sidebar.subheader("Neural Training")
    min_rows = st.sidebar.number_input("Min rows to train", min_value=5, max_value=500, value=200, step=5)
    train_nn = st.sidebar.button("Train / Refresh Neural Model")

    history = store.load_history()
    mumbai_history = history[history["city"] == CITY] if not history.empty else pd.DataFrame()

    station_history = station_store.load_history()
    mumbai_station_history = (
        station_history[station_history["city"] == CITY] if not station_history.empty else pd.DataFrame()
    )

    if collect_batches:
        with st.spinner("Collecting station-level API snapshots for Mumbai..."):
            saved = station_store.collect_batches(
                CITY,
                rounds=int(batch_rounds),
                interval_seconds=float(batch_interval),
            )
            st.sidebar.success(f"Saved {saved} new station rows.")
            station_history = station_store.load_history()
            mumbai_station_history = (
                station_history[station_history["city"] == CITY] if not station_history.empty else pd.DataFrame()
            )

    if train_nn:
        with st.spinner("Preparing station-level historical data and training neural model..."):
            X, y = station_store.get_training_data(balance=True)
            train_info = nn_model.train(X, y, min_rows=int(min_rows))
            if train_info.get("fallback"):
                st.sidebar.warning(
                    f"NN fallback active: {train_info.get('reason')} (rows={train_info.get('rows')})."
                )
            else:
                backend = train_info.get("backend", "tensorflow")
                st.sidebar.success(
                    f"NN trained ({backend}). Test accuracy: {train_info.get('test_accuracy', 0.0):.2%}"
                )

    should_fetch = do_fetch or not st.session_state["current_summary"]
    if should_fetch:
        with st.spinner("Fetching live data from CPCB..."):
            try:
                long_df, pivot_df, summary = cached_fetch_city(CITY)
                if summary:
                    store.save_reading(summary)
                    st.session_state["live_long_df"] = long_df
                    st.session_state["live_pivot_df"] = pivot_df
                    st.session_state["current_summary"] = summary
                    st.session_state["summary_source"] = "live"
                    history = store.load_history()
                    mumbai_history = history[history["city"] == CITY] if not history.empty else pd.DataFrame()

                    station_history = station_store.load_history()
                    mumbai_station_history = (
                        station_history[station_history["city"] == CITY] if not station_history.empty else pd.DataFrame()
                    )
                else:
                    st.error("No data returned from API for Mumbai.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"API fetch failed: {exc}")

    summary = st.session_state["current_summary"]
    long_df = st.session_state["live_long_df"]

    if not summary and not mumbai_history.empty:
        summary = summary_from_history(mumbai_history)
        st.session_state["current_summary"] = summary
        st.session_state["summary_source"] = "history"
        st.warning("Showing latest saved Mumbai reading from history (live API unavailable).")

    st.sidebar.write(f"Mumbai city-summary rows: {len(mumbai_history)}")
    st.sidebar.write(f"Mumbai station-training rows: {len(mumbai_station_history)}")

    has_tf_or_sklearn_model = Path("models/nn_model.h5").exists() or Path("models/nn_model_sklearn.pkl").exists()
    if not has_tf_or_sklearn_model and len(mumbai_station_history) < int(min_rows):
        st.sidebar.info("NN uses fallback until enough station history is collected and training is run.")

    if summary:
        st.sidebar.write(f"Last updated: {summary['timestamp']}")
        st.sidebar.write(f"Stations reporting: {summary['station_count']}")

    if not long_df.empty:
        st.sidebar.subheader("Raw Station Records")
        cols = [c for c in ["station", "pollutant", "avg", "city", "state"] if c in long_df.columns]
        st.sidebar.dataframe(long_df[cols].head(30), use_container_width=True)

    st.title("Real-Time AQI Predictor for Indian Urban Zones")
    st.caption("Mumbai-only live pipeline. Source: CPCB data.gov.in API")

    source_label = st.session_state.get("summary_source", "none")
    status_left, status_right = st.columns(2)
    with status_left:
        st.info(f"Data source: {source_label}")
    with status_right:
        st.info(f"Station training rows: {len(mumbai_station_history)}")

    if summary and summary.get("station_count", 0) < 3:
        st.warning("Fewer than 3 stations are reporting for this city.")

    if summary:
        prev = st.session_state.get("prev_summary", {}).get(CITY, {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pm25 = summary.get("PM2.5")
            delta_pm25 = None
            if prev and prev.get("PM2.5") is not None and pm25 is not None:
                delta_pm25 = pm25 - prev["PM2.5"]
            st.metric(
                "PM2.5 (µg/m³)",
                f"{pm25:.1f}" if pm25 is not None else "N/A",
                f"{delta_pm25:+.1f}" if delta_pm25 is not None else None,
            )
        with c2:
            no2 = summary.get("NO2")
            st.metric("NO2 (µg/m³)", f"{no2:.1f}" if no2 is not None else "N/A")
        with c3:
            so2 = summary.get("SO2")
            st.metric("SO2 (µg/m³)", f"{so2:.1f}" if so2 is not None else "N/A")
        with c4:
            render_aqi_badge("AQI Score", summary.get("aqi_value"), summary.get("aqi_category", "Moderate"))

        c5, c6, c7 = st.columns(3)
        with c5:
            pm10 = summary.get("PM10")
            st.metric("PM10", f"{pm10:.1f}" if pm10 is not None else "N/A")
        with c6:
            co = summary.get("CO")
            st.metric("CO (mg/m³)", f"{co:.2f}" if co is not None else "N/A")
        with c7:
            ozone = summary.get("OZONE")
            st.metric("OZONE", f"{ozone:.1f}" if ozone is not None else "N/A")

        st.subheader("AI Model Predictions")
        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("### Bayesian Network (PGM)")
            try:
                bn_out = bn_model.predict(summary)
                dist_df = pd.DataFrame(
                    {
                        "Category": list(bn_out["distribution"].keys()),
                        "Probability": list(bn_out["distribution"].values()),
                    }
                ).set_index("Category")
                st.bar_chart(dist_df)
                st.markdown(f"**Predicted:** {bn_out['predicted']} ({bn_out['confidence']:.1%})")
                st.caption(bn_out["explanation"])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Bayesian model error: {exc}")

        with p2:
            st.markdown("### Fuzzy Logic (Soft Computing)")
            try:
                fuzzy_out = fuzzy_model.predict(
                    pm25_val=float(summary.get("PM2.5") or 0.0),
                    no2_val=float(summary.get("NO2") or 0.0),
                    so2_val=float(summary.get("SO2") or 0.0),
                )
                st.metric("AQI Risk Score", fuzzy_out["aqi_score"])
                st.markdown(f"**Category:** {fuzzy_out['category']}")
                top_mf = max(fuzzy_out["pm25_membership"], key=fuzzy_out["pm25_membership"].get)
                st.caption(f"PM2.5 membership: {top_mf} ({fuzzy_out['pm25_membership'][top_mf]:.2f})")
                st.bar_chart(pd.DataFrame.from_dict(fuzzy_out["pm25_membership"], orient="index", columns=["degree"]))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Fuzzy model error: {exc}")

        with p3:
            st.markdown("### Neural Network (Soft Computing)")
            try:
                nn_out = nn_model.predict_live(summary)
                st.markdown(f"**Predicted:** {nn_out['predicted_category']} ({nn_out['confidence']:.1%})")
                st.caption(f"Inference backend: {nn_out.get('mode', 'unknown')}")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Category": list(nn_out["all_probabilities"].keys()),
                            "Probability": list(nn_out["all_probabilities"].values()),
                        }
                    ),
                    use_container_width=True,
                )
                st.caption(f"Model trained on {len(mumbai_station_history)} Mumbai station readings")
                if nn_out.get("mode") == "fallback":
                    st.info("Neural output is fallback mode. Collect more history and train the model.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Neural model error: {exc}")

        st.session_state["prev_summary"] = st.session_state.get("prev_summary", {})
        st.session_state["prev_summary"][CITY] = summary

    st.subheader("Mumbai Historical Snapshot")
    if not mumbai_history.empty:
        trend_cols = [
            c
            for c in ["timestamp", "PM2.5", "NO2", "SO2", "aqi_value", "aqi_category", "station_count"]
            if c in mumbai_history.columns
        ]
        hist_view = mumbai_history.sort_values("timestamp", ascending=False)
        st.dataframe(hist_view[trend_cols].head(30), use_container_width=True)

        # Simple trend view for PM2.5 and AQI history.
        plot_df = hist_view[["timestamp", "PM2.5", "aqi_value"]].dropna().copy()
        if not plot_df.empty:
            plot_df = plot_df.sort_values("timestamp")
            plot_df = plot_df.set_index("timestamp")
            st.line_chart(plot_df)
    else:
        st.info("No historical Mumbai readings yet. Click Fetch Live Data or Collect API Batches.")

    st.subheader("Monitoring Stations Map")
    if not long_df.empty and {"lat", "lon"}.issubset(set(long_df.columns)):
        map_df = long_df[["lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"}).dropna()
        if not map_df.empty:
            st.map(map_df)

    if auto_refresh:
        st.caption("Auto refresh is ON. Next refresh in 60 seconds.")
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
