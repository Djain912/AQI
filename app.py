"""Streamlit dashboard for Real-Time AQI Predictor (Mumbai-focused)."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import streamlit as st

from data.fetcher import calculate_aqi_from_pm25, fetch_all_pollutants
from data.historical_store import HistoricalStore
from data.station_store import StationHistoricalStore
from modules.bayesian_network import AQIBayesianNetwork
from modules.fuzzy_logic import AQIFuzzySystem
from modules.markov_model import AQIMarkovModel
from modules.neural_network import AQINeuralNetwork
from modules.time_series_forecaster import AQITimeSeriesForecaster

CITY = "Mumbai"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]

CATEGORY_COLORS = {
    "Good": "#1E8E3E",
    "Satisfactory": "#8BC34A",
    "Moderate": "#FB8C00",
    "Poor": "#E53935",
    "Very Poor": "#7F0000",
    "Severe": "#7F0000",
}

CARD_TEXT_COLORS = {
    "Good": "#FFFFFF",
    "Satisfactory": "#111111",
    "Moderate": "#FFFFFF",
    "Poor": "#FFFFFF",
    "Very Poor": "#FFFFFF",
    "Severe": "#FFFFFF",
    "missing": "#FFFFFF",
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
    pm25_mean = means.get("PM2.5")
    pm10_mean = means.get("PM10")
    if pm25_mean is not None and not pd.isna(pm25_mean):
        aqi_value = calculate_aqi_from_pm25(pm25_mean)
    elif pm10_mean is not None and not pd.isna(pm10_mean):
        aqi_value = float(pm10_mean) * 1.5
    else:
        aqi_value = None

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
    if value is None or pd.isna(value):
        color = "#6C757D"
        text_color = CARD_TEXT_COLORS["missing"]
        value_text = "AQI: Not Available"
        subtitle = "Data missing"
    else:
        color = CATEGORY_COLORS.get(category, "#6C757D")
        text_color = CARD_TEXT_COLORS.get(category, "#FFFFFF")
        value_text = f"{value:.1f}"
        subtitle = category

    st.markdown(
        f"""
        <div style='padding:14px;border-radius:12px;background:{color};font-weight:700;text-align:center;color:{text_color};'>
            <div style='font-size:14px'>{label}</div>
            <div style='font-size:26px'>{value_text}</div>
            <div style='font-size:13px'>{subtitle}</div>
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
    markov_model = AQIMarkovModel(smoothing=1e-6, debug=False)
    nn_model = AQINeuralNetwork()
    ts_model = AQITimeSeriesForecaster(lookback_hours=48, forecast_hours=24)

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
    train_ts = st.sidebar.button("Train Future Forecaster (24h)")

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
                st.sidebar.info(
                    f"NN fallback active: {train_info.get('reason')} (rows={train_info.get('rows')})."
                )
            else:
                backend = train_info.get("backend", "tensorflow")
                st.sidebar.success(
                    f"NN trained ({backend}). Test accuracy: {train_info.get('test_accuracy', 0.0):.2%}"
                )

    if train_ts:
        with st.spinner("Compiling past 48h windows and training 24h Deep Forecaster..."):
            ts_res = ts_model.train(CITY)
            if ts_res.get("status") == "success":
                st.sidebar.success(f"Forecaster trained on {ts_res['trained_samples']} samples! MAE: {ts_res['mae']:.1f} AQI")
            else:
                st.sidebar.error(f"Error: {ts_res.get('message', 'Unknown Error')}")

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
        st.info("Showing latest saved Mumbai reading from history (live API unavailable).")

    st.sidebar.write(f"Mumbai city-summary rows: {len(mumbai_history)}")
    st.sidebar.write(f"Mumbai station-training rows: {len(mumbai_station_history)}")

    if not mumbai_history.empty and {"timestamp", "aqi_category"}.issubset(set(mumbai_history.columns)):
        markov_model.fit(mumbai_history[["timestamp", "aqi_category"]])

    has_tf_or_sklearn_model = Path("models/nn_model.h5").exists() or Path("models/nn_model_sklearn.pkl").exists()
    if not has_tf_or_sklearn_model and len(mumbai_station_history) < int(min_rows):
        st.sidebar.info("NN uses fallback until enough station history is collected and training is run.")

    if summary:
        st.sidebar.write(f"Last updated: {summary['timestamp']}")
        st.sidebar.write(f"Stations reporting: {summary['station_count']}")

    if summary:
        pm25_val = summary.get("PM2.5")
        pm10_val = summary.get("PM10")
        aqi_val = summary.get("aqi_value")

        if aqi_val is None or pd.isna(aqi_val):
            if pm25_val is not None and not pd.isna(pm25_val):
                aqi_val = calculate_aqi_from_pm25(float(pm25_val))
            elif pm10_val is not None and not pd.isna(pm10_val):
                aqi_val = float(pm10_val) * 1.5
            elif not mumbai_history.empty and "aqi_value" in mumbai_history.columns:
                prev_aqi = pd.to_numeric(mumbai_history["aqi_value"], errors="coerce").dropna()
                if not prev_aqi.empty:
                    aqi_val = float(prev_aqi.iloc[-1])

        if aqi_val is None or pd.isna(aqi_val):
            aqi_val = 100.0

        summary["aqi_value"] = float(aqi_val)
        summary["aqi_category"] = category_from_aqi(float(aqi_val))
        st.session_state["current_summary"] = summary

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
        st.info("Fewer than 3 stations are reporting for this city.")

    if summary:
        prev = st.session_state.get("prev_summary", {}).get(CITY, {})
        pm25_missing = summary.get("PM2.5") is None or pd.isna(summary.get("PM2.5"))
        if pm25_missing:
            st.info("PM2.5 missing - predictions may be less reliable")

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

        quality_features = ["PM2.5", "NO2", "SO2", "PM10", "CO", "OZONE"]
        non_null_count = sum(1 for f in quality_features if summary.get(f) is not None and not pd.isna(summary.get(f)))
        quality_score = (non_null_count / len(quality_features)) * 100
        if quality_score >= 80:
            quality_label = "High"
        elif quality_score >= 50:
            quality_label = "Medium"
        else:
            quality_label = "Low"

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            pm10 = summary.get("PM10")
            st.metric("PM10", f"{pm10:.1f}" if pm10 is not None else "N/A")
        with c6:
            co = summary.get("CO")
            st.metric("CO (mg/m³)", f"{co:.2f}" if co is not None else "N/A")
        with c7:
            ozone = summary.get("OZONE")
            st.metric("OZONE", f"{ozone:.1f}" if ozone is not None else "N/A")
        with c8:
            st.metric("Data Quality", f"{quality_score:.0f}% ({quality_label})")

        st.subheader("AI Model Predictions")
        st.markdown("---")

        # Compute all model outputs first so we can present a unified final decision.
        bn_out = None
        bn_pred = None
        bn_error = None
        bn_categories = ["Good", "Satisfactory", "Moderate", "Poor", "VeryPoor", "Severe"]

        def _bn_style(cat: str) -> str:
            return "VeryPoor" if cat == "Very Poor" else cat

        def _display_style(cat: str | None) -> str | None:
            if cat is None:
                return None
            return "Very Poor" if cat == "VeryPoor" else cat

        try:
            bn_out = bn_model.predict(summary)
            bn_pred = bn_out.get("predicted")
        except Exception as exc:  # noqa: BLE001
            bn_error = str(exc)

        if bn_out is None:
            mapped = _bn_style(str(summary.get("aqi_category", "Moderate")))
            base = {c: 0.6 / (len(bn_categories) - 1) for c in bn_categories}
            if mapped in base:
                base[mapped] = 0.4
            else:
                base = {c: 1.0 / len(bn_categories) for c in bn_categories}
                mapped = "Moderate"
            bn_out = {
                "distribution": base,
                "predicted": mapped,
                "confidence": float(base[mapped]),
                "explanation": "Fallback posterior used because Bayesian inference is unavailable for current inputs.",
            }
            bn_pred = mapped

        current_cat = summary.get("aqi_category")
        current_cat_str = str(current_cat) if current_cat is not None else ""
        markov_plot = None
        markov_next_dist = None
        markov_error = None
        try:
            markov_plot = markov_model.get_plot_data(current_cat_str)
            markov_next_dist = markov_model.predict_next(current_cat_str)
        except Exception as exc:  # noqa: BLE001
            markov_error = str(exc)

        if markov_next_dist is None:
            cats = list(markov_model.categories)
            mapped = str(summary.get("aqi_category", "Moderate"))
            base = {c: 0.6 / (len(cats) - 1) for c in cats}
            if mapped in base:
                base[mapped] = 0.4
            else:
                base = {c: 1.0 / len(cats) for c in cats}
                mapped = "Moderate"
            markov_next_dist = base
            markov_plot = pd.DataFrame({"Category": cats, "Probability": [base[c] for c in cats]})

        fuzzy_out = None
        fuzzy_error = None
        fuzzy_missing = any(
            summary.get(k) is None or pd.isna(summary.get(k))
            for k in ["PM2.5", "NO2", "SO2"]
        )
        fuzzy_used_estimate = False

        def _estimate_feature_value(feature: str, fallback: float) -> float:
            val = summary.get(feature)
            if val is not None and not pd.isna(val):
                return float(val)

            if not mumbai_history.empty and feature in mumbai_history.columns:
                series = pd.to_numeric(mumbai_history[feature], errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[-1])

            return float(fallback)

        fuzzy_inputs = {
            "PM2.5": _estimate_feature_value("PM2.5", 80.0),
            "NO2": _estimate_feature_value("NO2", 60.0),
            "SO2": _estimate_feature_value("SO2", 20.0),
        }
        if fuzzy_missing:
            fuzzy_used_estimate = True

        try:
            fuzzy_out = fuzzy_model.predict(
                pm25_val=float(fuzzy_inputs["PM2.5"]),
                no2_val=float(fuzzy_inputs["NO2"]),
                so2_val=float(fuzzy_inputs["SO2"]),
            )
        except Exception as exc:  # noqa: BLE001
            fuzzy_error = str(exc)

        nn_out = None
        nn_error = None
        missing_nn_features = [
            f for f in nn_model.FEATURE_ORDER if summary.get(f) is None or pd.isna(summary.get(f))
        ]
        nn_missing = len(missing_nn_features) > 0
        if not nn_missing:
            try:
                nn_out = nn_model.predict_live(summary)
            except Exception as exc:  # noqa: BLE001
                nn_error = str(exc)

        # Build a realistic fallback from temporal + probabilistic models instead of uniform bars.
        nn_fallback_categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        markov_vec = np.array([float(markov_next_dist.get(c, 0.0)) for c in nn_fallback_categories], dtype=float)

        bn_dist = bn_out.get("distribution", {}) if bn_out else {}
        bn_vec = np.array(
            [
                float(bn_dist.get("Good", 0.0)),
                float(bn_dist.get("Satisfactory", 0.0)),
                float(bn_dist.get("Moderate", 0.0)),
                float(bn_dist.get("Poor", 0.0)),
                float(bn_dist.get("VeryPoor", bn_dist.get("Very Poor", 0.0))),
                float(bn_dist.get("Severe", 0.0)),
            ],
            dtype=float,
        )

        if markov_vec.sum() <= 0:
            markov_vec = np.ones(len(nn_fallback_categories), dtype=float)
        if bn_vec.sum() <= 0:
            bn_vec = np.ones(len(nn_fallback_categories), dtype=float)

        markov_vec = markov_vec / markov_vec.sum()
        bn_vec = bn_vec / bn_vec.sum()
        blended_vec = (0.6 * markov_vec) + (0.4 * bn_vec)
        blended_vec = blended_vec / blended_vec.sum()
        nn_fallback_probs = {
            cat: float(blended_vec[i]) for i, cat in enumerate(nn_fallback_categories)
        }

        markov_pred = None
        if markov_next_dist:
            markov_pred = max(markov_next_dist, key=markov_next_dist.get)

        final_pred = None
        final_source = None
        nn_available = nn_out is not None and not nn_missing and nn_error is None

        if nn_available:
            final_pred = nn_out.get("predicted_category")
            final_source = "Neural"
        else:
            m_vote = _display_style(markov_pred)
            b_vote = _display_style(bn_pred)
            if m_vote and b_vote and m_vote == b_vote:
                final_pred = m_vote
                final_source = "Markov + Bayesian majority"
            elif m_vote is not None:
                final_pred = m_vote
                final_source = "Markov fallback"
            elif b_vote is not None:
                final_pred = b_vote
                final_source = "Bayesian fallback"

        model_votes = [p for p in [_display_style(bn_pred), _display_style(markov_pred), nn_out.get("predicted_category") if nn_out else None] if p]
        if final_pred:
            agreement_count = sum(1 for v in model_votes if v == final_pred)
            st.info(f"Final AQI Prediction: {final_pred} (source: {final_source})")
            st.caption(f"Based on model agreement: {agreement_count}/{len(model_votes)} models")
        else:
            st.info("Final AQI Prediction: Not Available")
            st.caption("Based on model agreement")

        st.subheader("Probabilistic Graphical Models (PGM)")
        p1, p2 = st.columns(2)

        with p1:
            st.markdown("### 🧠 Bayesian Network (PGM)")
            st.caption("Uses probabilistic dependencies between pollution factors.")
            dist_df = pd.DataFrame(
                {
                    "Category": list(bn_out["distribution"].keys()),
                    "Probability": list(bn_out["distribution"].values()),
                }
            ).set_index("Category")
            st.bar_chart(dist_df, height=240, use_container_width=True)
            st.markdown(f"**Predicted:** {_display_style(bn_out['predicted'])} ({bn_out['confidence']:.1%})")
            if bn_error:
                st.caption(f"Bayesian fallback active: {bn_error}")
            else:
                st.caption(bn_out["explanation"])

        with p2:
            st.markdown("### ⏳ Markov Chain (Temporal PGM)")
            st.caption("Models AQI transitions over time using historical trends.")
            st.bar_chart(markov_plot.set_index("Category"), height=240, use_container_width=True)
            next_cat = max(markov_next_dist, key=markov_next_dist.get)
            next_prob = float(markov_next_dist[next_cat])
            st.markdown(f"**Predicted:** {next_cat} ({next_prob:.0%})")
            step_forecast = markov_model.predict_n_steps(current_cat_str, steps=3) or []
            if step_forecast:
                top3 = [max(dist, key=dist.get) for dist in step_forecast]
                st.caption(f"Next 3 steps: {', '.join(top3)}")
            elif markov_error:
                st.caption(f"Markov fallback active: {markov_error}")
            else:
                st.caption("Markov fallback distribution used from current AQI category.")
            st.caption("Transition matrix available internally for analysis")

        st.markdown("---")
        st.subheader("Soft Computing Models")
        p3, p4 = st.columns(2)

        with p3:
            st.markdown("### 🌫️ Fuzzy Logic")
            st.caption("Handles uncertainty using rule-based linguistic logic.")
            if fuzzy_out is not None:
                st.metric("AQI Risk Score", fuzzy_out["aqi_score"])
                st.markdown(f"**Category:** {fuzzy_out['category']}")
                top_mf = max(fuzzy_out["pm25_membership"], key=fuzzy_out["pm25_membership"].get)
                st.caption(f"PM2.5 membership: {top_mf} ({fuzzy_out['pm25_membership'][top_mf]:.2f})")
                fuzzy_chart = pd.DataFrame.from_dict(
                    fuzzy_out["pm25_membership"], orient="index", columns=["degree"]
                )
                st.bar_chart(fuzzy_chart, height=240, use_container_width=True)
                if fuzzy_used_estimate:
                    st.caption(
                        "Fuzzy inputs estimated from recent history due to missing live PM2.5/NO2/SO2."
                    )
            elif fuzzy_error:
                approx_df = pd.DataFrame(
                    {
                        "Category": ["Low", "Medium", "High"],
                        "degree": [0.2, 0.6, 0.2],
                    }
                ).set_index("Category")
                st.bar_chart(approx_df, height=240, use_container_width=True)
                st.markdown("**Predicted:** Moderate (approx)")
                st.info("Fuzzy model used approximate fallback due to computation error")
                st.caption(f"Fuzzy fallback active: {fuzzy_error}")
            else:
                approx_df = pd.DataFrame(
                    {
                        "Category": ["Low", "Medium", "High"],
                        "degree": [0.2, 0.6, 0.2],
                    }
                ).set_index("Category")
                st.bar_chart(approx_df, height=240, use_container_width=True)
                st.markdown("**Predicted:** Moderate (approx)")
                st.info("Fuzzy model fallback output shown")
                st.caption("Approximate fuzzy profile displayed to keep panel interpretable.")

        with p4:
            st.markdown("### 🤖 Neural Network")
            st.caption("Learns AQI patterns from historical data using ML.")
            if nn_missing:
                nn_df = pd.DataFrame(
                    {
                        "Category": list(nn_fallback_probs.keys()),
                        "Probability": list(nn_fallback_probs.values()),
                    }
                )
                st.bar_chart(nn_df.set_index("Category"), height=240, use_container_width=True)
                st.dataframe(nn_df, use_container_width=True, height=240)
                nn_fallback_pred = max(nn_fallback_probs, key=nn_fallback_probs.get)
                st.markdown(f"**Predicted:** {nn_fallback_pred} (fallback)")
                st.info("Neural model fallback output (estimated from Markov + Bayesian)")
                if missing_nn_features:
                    st.caption(f"Missing fields: {', '.join(missing_nn_features)}")
            elif nn_error:
                nn_df = pd.DataFrame(
                    {
                        "Category": list(nn_fallback_probs.keys()),
                        "Probability": list(nn_fallback_probs.values()),
                    }
                )
                st.bar_chart(nn_df.set_index("Category"), height=240, use_container_width=True)
                st.dataframe(nn_df, use_container_width=True, height=240)
                nn_fallback_pred = max(nn_fallback_probs, key=nn_fallback_probs.get)
                st.markdown(f"**Predicted:** {nn_fallback_pred} (fallback)")
                st.info("Neural model fallback output (estimated from Markov + Bayesian)")
                st.caption(f"Neural fallback reason: {nn_error}")
            elif nn_out is not None:
                st.markdown(f"**Predicted:** {nn_out['predicted_category']} ({nn_out['confidence']:.1%})")
                st.caption(f"Inference backend: {nn_out.get('mode', 'unknown')}")
                nn_df = pd.DataFrame(
                    {
                        "Category": list(nn_out["all_probabilities"].keys()),
                        "Probability": list(nn_out["all_probabilities"].values()),
                    }
                )
                st.bar_chart(nn_df.set_index("Category"), height=240, use_container_width=True)
                st.dataframe(
                    nn_df,
                    use_container_width=True,
                    height=240,
                )
                st.caption(f"Model trained on {len(mumbai_station_history)} Mumbai station readings")
                if nn_out.get("mode") == "fallback":
                    st.caption("Neural output is in fallback mode until model confidence improves.")
            else:
                nn_df = pd.DataFrame(
                    {
                        "Category": list(nn_fallback_probs.keys()),
                        "Probability": list(nn_fallback_probs.values()),
                    }
                )
                st.bar_chart(nn_df.set_index("Category"), height=240, use_container_width=True)
                st.dataframe(nn_df, use_container_width=True, height=240)
                nn_fallback_pred = max(nn_fallback_probs, key=nn_fallback_probs.get)
                st.markdown(f"**Predicted:** {nn_fallback_pred} (fallback)")
                st.info("Neural model fallback output (estimated from Markov + Bayesian)")
                st.caption("Blended fallback distribution shown to keep panel realistic.")

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

    st.subheader("Deep Time-Series Future Forecast (Next 24 Hours)")
    ts_forecast = ts_model.predict_future_24h(CITY)
    if "error" in ts_forecast and ts_forecast["error"]:
        if "trained" in ts_forecast["error"].lower():
            st.info("The 24-hour Deep Forecaster hasn't been trained yet! Click 'Train Future Forecaster (24h)' in the sidebar.")
        else:
            st.info(f"Forecaster: {ts_forecast['error']}")
    else:
        clipped_forecast = [max(0.0, float(v)) for v in ts_forecast["forecast_aqi"]]
        forecast_df = pd.DataFrame({
            "Time": ts_forecast["future_labels"],
            "Forecasted AQI": clipped_forecast
        }).set_index("Time")
        st.caption("Auto-Regressive Multi-Output Model dynamically forecasting AQI score based on recent 48-hour sequence.")
        st.line_chart(forecast_df, color=["#FF4B4B"])

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
