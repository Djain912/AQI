"""
Fuzzy Logic module for AQI risk scoring.

Implements Mamdani inference with centroid defuzzification using scikit-fuzzy
membership utilities, without the skfuzzy.control dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz


@dataclass(frozen=True)
class CPCBColors:
    mapping: Dict[str, str] = None

    def __post_init__(self) -> None:
        if self.mapping is None:
            object.__setattr__(
                self,
                "mapping",
                {
                    "Good": "#00B050",
                    "Satisfactory": "#92D050",
                    "Moderate": "#FFFF00",
                    "Poor": "#FF9900",
                    "Very Poor": "#FF0000",
                    "Severe": "#800000",
                },
            )


class AQIFuzzySystem:
    """Fuzzy system for AQI risk prediction."""

    def __init__(self) -> None:
        self.colors = CPCBColors().mapping

        self.pm25_u: np.ndarray | None = None
        self.no2_u: np.ndarray | None = None
        self.so2_u: np.ndarray | None = None
        self.aqi_u: np.ndarray | None = None

        self.pm25_mf: Dict[str, np.ndarray] = {}
        self.no2_mf: Dict[str, np.ndarray] = {}
        self.so2_mf: Dict[str, np.ndarray] = {}
        self.aqi_mf: Dict[str, np.ndarray] = {}

        self.rules: List[Tuple[List[Tuple[str, str]], str]] = []

    def build_system(self) -> None:
        """Create universes, membership functions, and all 12 fuzzy rules."""
        self.pm25_u = np.arange(0, 501, 1)
        self.no2_u = np.arange(0, 201, 1)
        self.so2_u = np.arange(0, 81, 1)
        self.aqi_u = np.arange(0, 501, 1)

        self.pm25_mf = {
            "good": fuzz.trapmf(self.pm25_u, [0, 0, 20, 40]),
            "satisf": fuzz.trimf(self.pm25_u, [30, 55, 80]),
            "moderate": fuzz.trimf(self.pm25_u, [60, 90, 130]),
            "poor": fuzz.trimf(self.pm25_u, [100, 160, 220]),
            "verypoor": fuzz.trimf(self.pm25_u, [180, 260, 340]),
            "severe": fuzz.trapmf(self.pm25_u, [300, 380, 500, 500]),
        }

        self.no2_mf = {
            "low": fuzz.trapmf(self.no2_u, [0, 0, 25, 50]),
            "moderate": fuzz.trimf(self.no2_u, [40, 80, 120]),
            "high": fuzz.trapmf(self.no2_u, [100, 150, 200, 200]),
        }

        self.so2_mf = {
            "low": fuzz.trapmf(self.so2_u, [0, 0, 20, 40]),
            "moderate": fuzz.trimf(self.so2_u, [30, 50, 70]),
            "high": fuzz.trapmf(self.so2_u, [60, 75, 80, 80]),
        }

        self.aqi_mf = {
            "good": fuzz.trapmf(self.aqi_u, [0, 0, 30, 55]),
            "satisf": fuzz.trimf(self.aqi_u, [45, 75, 105]),
            "moderate": fuzz.trimf(self.aqi_u, [90, 145, 205]),
            "poor": fuzz.trimf(self.aqi_u, [180, 245, 305]),
            "verypoor": fuzz.trimf(self.aqi_u, [280, 350, 415]),
            "severe": fuzz.trapmf(self.aqi_u, [390, 450, 500, 500]),
        }

        self.rules = [
            ([("pm25", "good"), ("no2", "low")], "good"),
            ([("pm25", "good"), ("no2", "moderate")], "satisf"),
            ([("pm25", "satisf"), ("no2", "low")], "satisf"),
            ([("pm25", "satisf"), ("no2", "moderate")], "moderate"),
            ([("pm25", "moderate"), ("no2", "low")], "moderate"),
            ([("pm25", "moderate"), ("no2", "moderate")], "poor"),
            ([("pm25", "moderate"), ("no2", "high")], "poor"),
            ([("pm25", "poor"), ("no2", "high")], "verypoor"),
            ([("pm25", "verypoor"), ("no2", "moderate")], "verypoor"),
            ([("pm25", "verypoor"), ("no2", "high")], "severe"),
            ([("pm25", "severe"), ("no2", "high")], "severe"),
            ([("so2", "high"), ("pm25", "moderate")], "poor"),
        ]

    @staticmethod
    def _category_from_score(aqi_score: float) -> str:
        if aqi_score <= 50:
            return "Good"
        if aqi_score <= 100:
            return "Satisfactory"
        if aqi_score <= 200:
            return "Moderate"
        if aqi_score <= 300:
            return "Poor"
        if aqi_score <= 400:
            return "Very Poor"
        return "Severe"

    def _deg(self, var_name: str, mf_name: str, value: float) -> float:
        if var_name == "pm25":
            return float(fuzz.interp_membership(self.pm25_u, self.pm25_mf[mf_name], value))
        if var_name == "no2":
            return float(fuzz.interp_membership(self.no2_u, self.no2_mf[mf_name], value))
        return float(fuzz.interp_membership(self.so2_u, self.so2_mf[mf_name], value))

    def predict(self, pm25_val: float, no2_val: float, so2_val: float) -> Dict:
        """Predict AQI risk score and category using Mamdani inference."""
        if self.aqi_u is None:
            self.build_system()

        pm25_val = float(np.clip(pm25_val, 0, 500))
        no2_val = float(np.clip(no2_val, 0, 200))
        so2_val = float(np.clip(so2_val, 0, 80))

        values = {"pm25": pm25_val, "no2": no2_val, "so2": so2_val}
        aggregated = np.zeros_like(self.aqi_u, dtype=float)

        for antecedents, consequent in self.rules:
            strengths = [self._deg(var, mf, values[var]) for var, mf in antecedents]
            firing = min(strengths) if strengths else 0.0
            if firing > 0:
                clipped = np.fmin(firing, self.aqi_mf[consequent])
                aggregated = np.fmax(aggregated, clipped)

        if float(np.max(aggregated)) <= 0.0:
            aqi_score = max(pm25_val * 1.8, 0.0)
        else:
            aqi_score = float(fuzz.defuzz(self.aqi_u, aggregated, "centroid"))

        category = self._category_from_score(aqi_score)

        pm25_membership = {
            name: float(fuzz.interp_membership(self.pm25_u, mf, pm25_val))
            for name, mf in self.pm25_mf.items()
        }

        return {
            "aqi_score": round(aqi_score, 2),
            "category": category,
            "pm25_membership": pm25_membership,
            "color_hex": self.colors[category],
        }

    def plot_membership_functions(self, save_path: str = "outputs/fuzzy_mf.png") -> str:
        """Plot and save membership functions for report/presentation usage."""
        if self.aqi_u is None:
            self.build_system()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 14))

        for name, mf in self.pm25_mf.items():
            axes[0].plot(self.pm25_u, mf, label=name)
        axes[0].set_title("PM2.5 Membership Functions")
        axes[0].legend(loc="upper right")

        for name, mf in self.no2_mf.items():
            axes[1].plot(self.no2_u, mf, label=name)
        axes[1].set_title("NO2 Membership Functions")
        axes[1].legend(loc="upper right")

        for name, mf in self.so2_mf.items():
            axes[2].plot(self.so2_u, mf, label=name)
        axes[2].set_title("SO2 Membership Functions")
        axes[2].legend(loc="upper right")

        for name, mf in self.aqi_mf.items():
            axes[3].plot(self.aqi_u, mf, label=name)
        axes[3].set_title("AQI Risk Output Membership Functions")
        axes[3].legend(loc="upper right")

        for ax in axes:
            ax.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        return save_path
