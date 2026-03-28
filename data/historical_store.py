"""
Persistent historical AQI storage and training-data preparation.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.fetcher import calculate_aqi_from_pm25, fetch_pivoted


class HistoricalStore:
    """Store city-level AQI snapshots and prepare ML-ready training data."""

    HISTORY_COLUMNS = [
        "timestamp",
        "city",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "OZONE",
        "NH3",
        "aqi_value",
        "aqi_category",
        "hour",
        "is_peak",
        "station_count",
    ]

    FEATURE_COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "hour", "is_peak"]

    CATEGORY_MAP = {
        "Good": 0,
        "Satisfactory": 1,
        "Moderate": 2,
        "Poor": 3,
        "Very Poor": 4,
        "Severe": 5,
    }

    def __init__(self, history_path: str = "data/historical/aqi_history.csv") -> None:
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _category_from_aqi(aqi: float | None) -> str:
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

    def _summary_from_station_data(self, city: str) -> Dict:
        """Convert station-level pivoted frame into one city-level reading."""
        frame = fetch_pivoted(city)
        if frame.empty:
            return {}

        pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]
        means = {
            col: (float(frame[col].mean()) if col in frame.columns and not frame[col].dropna().empty else None)
            for col in pollutant_cols
        }

        ts_value = datetime.now()
        if "last_update" in frame.columns and not frame["last_update"].dropna().empty:
            ts_value = pd.to_datetime(frame["last_update"]).max().to_pydatetime()

        hour = int(ts_value.hour)
        is_peak = 1 if hour in {7, 8, 9, 17, 18, 19, 20} else 0
        aqi_value = calculate_aqi_from_pm25(means.get("PM2.5"))
        aqi_category = self._category_from_aqi(aqi_value)

        return {
            "timestamp": ts_value.isoformat(),
            "city": city,
            **means,
            "aqi_value": aqi_value,
            "aqi_category": aqi_category,
            "hour": hour,
            "is_peak": is_peak,
            "station_count": int(len(frame)),
        }

    def save_reading(self, city_data: Dict) -> None:
        """Append one city-level reading to CSV, creating file if required."""
        if not city_data:
            return

        row = {col: city_data.get(col) for col in self.HISTORY_COLUMNS}

        if row["timestamp"] is None:
            now = datetime.now()
            row["timestamp"] = now.isoformat()
            row["hour"] = int(now.hour)
            row["is_peak"] = 1 if row["hour"] in {7, 8, 9, 17, 18, 19, 20} else 0

        if row["aqi_value"] is None:
            row["aqi_value"] = calculate_aqi_from_pm25(row.get("PM2.5"))
        if not row["aqi_category"]:
            row["aqi_category"] = self._category_from_aqi(row["aqi_value"])

        row_df = pd.DataFrame([row], columns=self.HISTORY_COLUMNS)

        write_header = not self.history_path.exists()
        row_df.to_csv(self.history_path, mode="a", index=False, header=write_header)

    def load_history(self) -> pd.DataFrame:
        """Load historical CSV into a cleaned DataFrame."""
        if not self.history_path.exists():
            return pd.DataFrame(columns=self.HISTORY_COLUMNS)

        df = pd.read_csv(self.history_path)
        for col in self.HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.HISTORY_COLUMNS]

        # Preserve raw timestamp text for robust dedup across mixed datetime formats.
        df["_timestamp_raw"] = df["timestamp"].astype(str).str.strip()

        # Parse with mixed format support to avoid coercing valid ISO strings to NaT.
        ts = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
        still_bad = ts.isna() & df["_timestamp_raw"].ne("")
        if still_bad.any():
            ts.loc[still_bad] = pd.to_datetime(
                df.loc[still_bad, "_timestamp_raw"],
                errors="coerce",
                format="ISO8601",
            )
        df["timestamp"] = ts

        numeric_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3", "aqi_value", "hour", "is_peak", "station_count"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.drop_duplicates(subset=["city", "_timestamp_raw"], keep="last").reset_index(drop=True)
        df = df.drop(columns=["_timestamp_raw"])
        return df

    def collect_and_store(self, cities: List[str]) -> None:
        """Fetch city summaries and append them into history CSV."""
        saved = 0
        for city in cities:
            summary = self._summary_from_station_data(city)
            if not summary:
                continue
            self.save_reading(summary)
            saved += 1

        print(f"Saved {saved} city readings at {datetime.now().isoformat()}")

    def collect_api_batches(self, city: str, rounds: int = 30, interval_seconds: float = 2.0) -> int:
        """Collect repeated API snapshots for one city to accelerate history creation."""
        import time

        saved = 0
        for _ in range(max(0, int(rounds))):
            summary = self._summary_from_station_data(city)
            if summary:
                # Ensure unique row timestamps even in tight loops.
                summary["timestamp"] = datetime.now().isoformat()
                self.save_reading(summary)
                saved += 1
            if interval_seconds > 0:
                time.sleep(float(interval_seconds))

        return saved

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build scaled training arrays and persist scaler to models/scaler.pkl."""
        df = self.load_history()
        if df.empty:
            print("Training data shape: (0, 0), Classes: []")
            return np.empty((0, len(self.FEATURE_COLUMNS))), np.empty((0,), dtype=int)

        work = df.copy()
        work = work.dropna(subset=["PM2.5"])
        if work.empty:
            print("Training data shape: (0, 0), Classes: []")
            return np.empty((0, len(self.FEATURE_COLUMNS))), np.empty((0,), dtype=int)

        for col in self.FEATURE_COLUMNS:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            median_val = work[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            work[col] = work[col].fillna(median_val)

        if "aqi_category" not in work.columns:
            work["aqi_category"] = work["aqi_value"].apply(self._category_from_aqi)

        y = work["aqi_category"].map(self.CATEGORY_MAP).fillna(2).astype(int).to_numpy()
        X_raw = work[self.FEATURE_COLUMNS].to_numpy(dtype=float)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)
        joblib.dump(scaler, "models/scaler.pkl")

        unique_classes = sorted(set(y.tolist()))
        print(f"Training data shape: {X_scaled.shape}, Classes: {unique_classes}")
        return X_scaled, y
