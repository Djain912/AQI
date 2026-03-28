"""
Station-level historical storage and training data preparation.

This store creates one training row per station snapshot (instead of one row per city),
which provides much more data for neural model training.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.fetcher import calculate_aqi_from_pm25, fetch_all_pollutants


class StationHistoricalStore:
    """Persist station-level AQI snapshots and build balanced training arrays."""

    HISTORY_COLUMNS = [
        "snapshot_time",
        "city",
        "station",
        "state",
        "last_update",
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

    def __init__(self, history_path: str = "data/historical/aqi_station_history.csv") -> None:
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

    @staticmethod
    def _pivot_station_features(long_df: pd.DataFrame) -> pd.DataFrame:
        if long_df.empty:
            return pd.DataFrame()

        required = {"station", "pollutant", "avg"}
        if not required.issubset(set(long_df.columns)):
            return pd.DataFrame()

        pivot = long_df.pivot_table(
            index=["station", "city", "state"],
            columns="pollutant",
            values="avg",
            aggfunc="mean",
        ).reset_index()
        pivot.columns.name = None

        meta_cols = [c for c in ["station", "last_update"] if c in long_df.columns]
        if "station" in meta_cols:
            meta = long_df.sort_values("last_update", ascending=False).drop_duplicates(subset=["station"])[meta_cols]
            pivot = pivot.merge(meta, on="station", how="left")

        for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]:
            if col not in pivot.columns:
                pivot[col] = np.nan

        now = datetime.now()
        pivot["snapshot_time"] = now.isoformat()
        pivot["hour"] = int(now.hour)
        pivot["is_peak"] = 1 if int(now.hour) in {7, 8, 9, 17, 18, 19, 20} else 0
        pivot["aqi_value"] = pivot["PM2.5"].apply(calculate_aqi_from_pm25)
        pivot["aqi_category"] = pivot["aqi_value"].apply(StationHistoricalStore._category_from_aqi)

        return pivot

    def save_station_snapshot(self, city: str = "Mumbai") -> int:
        """Fetch one API snapshot and append one row per station."""
        long_df = fetch_all_pollutants(city)
        station_df = self._pivot_station_features(long_df)
        if station_df.empty:
            return 0

        out_df = station_df[self.HISTORY_COLUMNS].copy()
        write_header = not self.history_path.exists()
        out_df.to_csv(self.history_path, mode="a", index=False, header=write_header)
        return int(len(out_df))

    def collect_batches(self, city: str = "Mumbai", rounds: int = 30, interval_seconds: float = 1.0) -> int:
        import time

        total_rows = 0
        for _ in range(max(0, int(rounds))):
            total_rows += self.save_station_snapshot(city=city)
            if interval_seconds > 0:
                time.sleep(float(interval_seconds))
        return total_rows

    def load_history(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame(columns=self.HISTORY_COLUMNS)

        df = pd.read_csv(self.history_path)
        for col in self.HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.HISTORY_COLUMNS]

        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce", format="mixed")
        if "last_update" in df.columns:
            df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce", format="mixed")

        numeric_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3", "aqi_value", "hour", "is_peak"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Dedup at station snapshot granularity.
        df = df.drop_duplicates(subset=["city", "station", "snapshot_time"], keep="last").reset_index(drop=True)
        return df

    @staticmethod
    def _balance_classes(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Simple random oversampling to reduce class imbalance."""
        rng = np.random.default_rng(random_state)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y

        target = int(np.max(counts))
        x_parts = []
        y_parts = []

        for cls in classes:
            idx = np.where(y == cls)[0]
            if len(idx) == 0:
                continue
            if len(idx) < target:
                sampled = rng.choice(idx, size=target, replace=True)
            else:
                sampled = idx
            x_parts.append(X[sampled])
            y_parts.append(np.full(len(sampled), cls, dtype=int))

        Xb = np.vstack(x_parts)
        yb = np.concatenate(y_parts)

        order = rng.permutation(len(yb))
        return Xb[order], yb[order]

    def get_training_data(self, balance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        df = self.load_history()
        if df.empty:
            print("Station training data shape: (0, 0), Classes: []")
            return np.empty((0, len(self.FEATURE_COLUMNS))), np.empty((0,), dtype=int)

        work = df.copy().dropna(subset=["PM2.5"])
        if work.empty:
            print("Station training data shape: (0, 0), Classes: []")
            return np.empty((0, len(self.FEATURE_COLUMNS))), np.empty((0,), dtype=int)

        for col in self.FEATURE_COLUMNS:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            med = work[col].median()
            work[col] = work[col].fillna(0.0 if pd.isna(med) else med)

        work["aqi_category"] = work["aqi_category"].fillna(work["aqi_value"].apply(self._category_from_aqi))

        y = work["aqi_category"].map(self.CATEGORY_MAP).fillna(2).astype(int).to_numpy()
        X_raw = work[self.FEATURE_COLUMNS].to_numpy(dtype=float)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)
        joblib.dump(scaler, "models/scaler.pkl")

        if balance:
            X_scaled, y = self._balance_classes(X_scaled, y)

        unique_classes = sorted(set(y.tolist()))
        print(f"Station training data shape: {X_scaled.shape}, Classes: {unique_classes}")
        return X_scaled, y
