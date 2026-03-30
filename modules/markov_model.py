"""
First-order Markov Chain model for AQI category transitions.

This module models temporal AQI dynamics by learning transition probabilities
between consecutive AQI categories from historical timestamped records.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class AQIMarkovModel:
    """Lightweight first-order Markov model for AQI category forecasting."""

    DEFAULT_CATEGORIES = [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe",
    ]

    def __init__(self, smoothing: float = 1e-6, debug: bool = False) -> None:
        self.smoothing = float(max(smoothing, 0.0))
        self.debug = bool(debug)

        self.categories: List[str] = list(self.DEFAULT_CATEGORIES)
        self.category_to_index: Dict[str, int] = {cat: idx for idx, cat in enumerate(self.categories)}
        self.index_to_category: Dict[int, str] = {idx: cat for cat, idx in self.category_to_index.items()}
        self.transition_matrix: np.ndarray | None = None
        self._is_fitted = False

    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[AQIMarkovModel] {message}")

    def _uniform_distribution(self) -> Dict[str, float]:
        n = len(self.categories)
        if n == 0:
            return {}
        p = 1.0 / n
        return {cat: p for cat in self.categories}

    def _normalize_rows(self, mat: np.ndarray) -> np.ndarray:
        """Normalize rows into probabilities with optional smoothing for zero rows."""
        if mat.size == 0:
            return mat

        row_sums = mat.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.flatten() == 0)
        if np.any(zero_rows):
            eps = self.smoothing if self.smoothing > 0 else 1e-6
            mat[zero_rows, :] += eps
            row_sums = mat.sum(axis=1, keepdims=True)

        # Final guard in case rows still sum to zero for any reason.
        row_sums[row_sums == 0] = 1.0
        return mat / row_sums

    def fit(self, dataframe: pd.DataFrame) -> None:
        """
        Build transition matrix from timestamped AQI category history.

        Expected columns:
          - timestamp
          - aqi_category
        """
        if dataframe is None or dataframe.empty:
            self.transition_matrix = None
            self._is_fitted = False
            self._log("No data provided. Model not fitted.")
            return

        required_cols = {"timestamp", "aqi_category"}
        if not required_cols.issubset(dataframe.columns):
            self.transition_matrix = None
            self._is_fitted = False
            self._log(f"Missing required columns: {required_cols - set(dataframe.columns)}")
            return

        work = dataframe[["timestamp", "aqi_category"]].copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", format="mixed")
        work = work.dropna(subset=["timestamp", "aqi_category"])

        # Keep only supported categories to avoid corrupt transitions.
        work = work[work["aqi_category"].isin(self.categories)]
        work = work.sort_values("timestamp").reset_index(drop=True)

        if len(work) < 2:
            self.transition_matrix = None
            self._is_fitted = False
            self._log("Insufficient sequential records (need at least 2 rows).")
            return

        n = len(self.categories)
        counts = np.zeros((n, n), dtype=float)

        cat_series = work["aqi_category"].tolist()
        for i in range(len(cat_series) - 1):
            src = self.category_to_index[cat_series[i]]
            dst = self.category_to_index[cat_series[i + 1]]
            counts[src, dst] += 1.0

        self.transition_matrix = self._normalize_rows(counts)
        self._is_fitted = True
        self._log(f"Model fitted with {len(work)} rows and {len(cat_series) - 1} transitions.")

    def predict_next(self, current_category: str) -> Dict[str, float] | None:
        """Return probability distribution for next AQI category."""
        if not self._is_fitted or self.transition_matrix is None:
            self._log("Model is not fitted. Returning None.")
            return None

        if current_category not in self.category_to_index:
            self._log(f"Unseen category '{current_category}'. Returning uniform distribution.")
            return self._uniform_distribution()

        row = self.transition_matrix[self.category_to_index[current_category]]
        return {cat: float(row[idx]) for idx, cat in enumerate(self.categories)}

    def predict_n_steps(self, current_category: str, steps: int = 24) -> List[Dict[str, float]] | None:
        """Propagate Markov state for N steps and return per-step distributions."""
        if not self._is_fitted or self.transition_matrix is None:
            self._log("Model is not fitted. Returning None.")
            return None

        steps = int(max(0, steps))
        if steps == 0:
            return []

        n = len(self.categories)
        state = np.zeros(n, dtype=float)

        if current_category in self.category_to_index:
            state[self.category_to_index[current_category]] = 1.0
        else:
            self._log(f"Unseen category '{current_category}' for multi-step prediction. Using uniform state.")
            state[:] = 1.0 / n

        results: List[Dict[str, float]] = []
        for _ in range(steps):
            state = state @ self.transition_matrix
            results.append({cat: float(state[idx]) for idx, cat in enumerate(self.categories)})

        return results

    def get_most_likely_next(self, current_category: str) -> str | None:
        """Return most probable next category for a given current category."""
        dist = self.predict_next(current_category)
        if not dist:
            return None
        return max(dist, key=dist.get)

    def get_plot_data(self, current_category: str) -> pd.DataFrame | None:
        """Return bar-chart ready DataFrame with Category and Probability columns."""
        dist = self.predict_next(current_category)
        if dist is None:
            return None

        return pd.DataFrame(
            {
                "Category": self.categories,
                "Probability": [dist[cat] for cat in self.categories],
            }
        )
