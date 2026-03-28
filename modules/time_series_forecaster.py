"""
Time-Series Future Forecaster (24-Hour Prediction Pipeline).

Note: Because Python 3.14.1 is currently too new for the official TensorFlow/LSTM 
binary wheels, this module implements a highly optimized Deep AutoRegressive 
Neural Network (Multi-Layer Perceptron) using Scikit-Learn that achieves the 
same multi-step multi-output forecasting goal (predicting the next 24 hours of AQI) 
using a sliding timeline window.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data.historical_store import HistoricalStore


class AQITimeSeriesForecaster:
    """Predicts the next 24 hours of AQI using historical time-window lags."""

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 24) -> None:
        self.lookback = lookback_hours
        self.forecast = forecast_hours
        self.model_path = "models/ts_forecaster.pkl"
        self.scaler_x_path = "models/ts_scaler_x.pkl"
        self.scaler_y_path = "models/ts_scaler_y.pkl"
        
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("outputs").mkdir(parents=True, exist_ok=True)

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a continuous timeline into (X = past 48h, Y = next 24h AQI) matrices."""
        # Ensure data is sorted completely sequentially by time
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # We'll forecast based purely on the historical PM2.5 and AQI curve
        feature_cols = ["PM2.5", "aqi_value"]
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0)
            
        data_matrix = df[feature_cols].values
        aqi_idx = feature_cols.index("aqi_value")
        
        X, Y = [], []
        
        # Walk through the timeline
        for i in range(len(data_matrix) - self.lookback - self.forecast):
            # The past 48 hours of all features (flattened)
            window_x = data_matrix[i : i + self.lookback].flatten()
            
            # The next 24 hours of just the AQI score
            window_y = data_matrix[i + self.lookback : i + self.lookback + self.forecast, aqi_idx]
            
            X.append(window_x)
            Y.append(window_y)
            
        return np.array(X), np.array(Y)

    def train(self, city: str = "Mumbai") -> Dict:
        """Fetch historical history, build sequences, and train multi-output DNN."""
        store = HistoricalStore()
        df = store.load_history()
        city_df = df[df["city"] == city]
        
        if len(city_df) < (self.lookback + self.forecast + 50):
            return {"status": "error", "message": f"Need at least {self.lookback + self.forecast + 50} total hourly rows. Only have {len(city_df)}."}

        # Sub-sample historically to 1 hour frequency to fix any gaps
        city_df["timestamp"] = pd.to_datetime(city_df["timestamp"])
        city_df = city_df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()

        X, Y = self._prepare_sequences(city_df)
        
        if len(X) < 10:
            return {"status": "error", "message": "Not enough sequential overlap chunks to train yet. Wait for more data."}
            
        # Scale X manually
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X)
        
        scaler_y = StandardScaler()
        Y_scaled = scaler_y.fit_transform(Y)

        joblib.dump(scaler_x, self.scaler_x_path)
        joblib.dump(scaler_y, self.scaler_y_path)

        # Multi-output DNN (Regresses all 24 future points simultaneously)
        print(f"Training Time-Series Deep Network on shape X={X_scaled.shape}, Y={Y_scaled.shape}...")
        model = MLPRegressor(
            hidden_layer_sizes=(128, 128, 64), 
            activation="relu", 
            solver="adam",
            max_iter=400,
            random_state=42,
            learning_rate_init=0.002,
        )
        
        model.fit(X_scaled, Y_scaled)
        joblib.dump(model, self.model_path)
        
        # Calculate rough test metrics
        y_pred = model.predict(X_scaled)
        mae = float(np.mean(np.abs(scaler_y.inverse_transform(y_pred) - Y)))
        
        return {
            "status": "success",
            "mae": mae,
            "trained_samples": len(X),
        }

    def predict_future_24h(self, city: str = "Mumbai") -> Dict:
        """Gathers last 48 hours from storage and spits out the 24-hour future curve."""
        if not (os.path.exists(self.model_path) and os.path.exists(self.scaler_x_path)):
            return {"error": "Model not trained yet."}
            
        store = HistoricalStore()
        df = store.load_history()
        city_df = df[df["city"] == city]
        
        city_df["timestamp"] = pd.to_datetime(city_df["timestamp"])
        city_df = city_df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()

        if len(city_df) < self.lookback:
            return {"error": f"Need the last {self.lookback} hours of continuous data to forecast."}
            
        feature_cols = ["PM2.5", "aqi_value"]
        recent = city_df.tail(self.lookback)[feature_cols].copy()
        
        for c in feature_cols:
            recent[c] = pd.to_numeric(recent[c], errors="coerce").fillna(method="ffill").fillna(0)
            
        live_window = recent.values.flatten().reshape(1, -1)
        
        model = joblib.load(self.model_path)
        scaler_x = joblib.load(self.scaler_x_path)
        scaler_y = joblib.load(self.scaler_y_path)
        
        live_scaled = scaler_x.transform(live_window)
        pred_scaled = model.predict(live_scaled)
        
        pred_actual = scaler_y.inverse_transform(pred_scaled)[0]
        
        # Generate the future timestamps
        last_time = city_df["timestamp"].iloc[-1]
        future_times = [(last_time + pd.Timedelta(hours=i)).strftime("%I:%M %p") for i in range(1, self.forecast + 1)]
        
        return {
            "forecast_aqi": [round(float(p), 1) for p in pred_actual],
            "future_labels": future_times,
            "error": None
        }
