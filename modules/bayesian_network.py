"""
Bayesian Network module for AQI category prediction.

Implements a compact domain-knowledge model using pgmpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.models import BayesianNetwork

    PGMPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    TabularCPD = None  # type: ignore[assignment]
    VariableElimination = None  # type: ignore[assignment]
    BayesianNetwork = object  # type: ignore[assignment]
    PGMPY_AVAILABLE = False


@dataclass(frozen=True)
class BNStates:
    time_of_day: Tuple[str, ...] = ("OffPeak", "Peak")
    traffic_proxy: Tuple[str, ...] = ("Low", "High")
    pm25_level: Tuple[str, ...] = ("Good", "Moderate", "High", "VeryHigh")
    nox_level: Tuple[str, ...] = ("Low", "Moderate", "High")
    wind_proxy: Tuple[str, ...] = ("High", "Low")
    aqi_category: Tuple[str, ...] = (
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "VeryPoor",
        "Severe",
    )


class AQIBayesianNetwork:
    """AQI Bayesian Network wrapper for build, discretize and predict."""

    def __init__(self) -> None:
        self.states = BNStates()
        self.model: BayesianNetwork | None = None
        self._supports_pgmpy = PGMPY_AVAILABLE

    @staticmethod
    def _tabular_from_columns(
        variable: str,
        variable_states: Tuple[str, ...],
        evidence: List[str],
        evidence_states: List[Tuple[str, ...]],
        distributions: Dict[Tuple[str, ...], List[float]],
    ) -> TabularCPD:
        """Create a TabularCPD by specifying distributions per parent-state column."""
        combos = list(product(*evidence_states)) if evidence_states else [tuple()]

        matrix = [[] for _ in variable_states]
        for combo in combos:
            probs = distributions[combo]
            for row_i, p in enumerate(probs):
                matrix[row_i].append(float(p))

        evidence_card = [len(s) for s in evidence_states]
        state_names = {variable: list(variable_states)}
        for ev_name, ev_states in zip(evidence, evidence_states):
            state_names[ev_name] = list(ev_states)

        return TabularCPD(
            variable=variable,
            variable_card=len(variable_states),
            values=matrix,
            evidence=evidence if evidence else None,
            evidence_card=evidence_card if evidence else None,
            state_names=state_names,
        )

    def _build_pm25_cpd(self) -> TabularCPD:
        dists = {
            ("OffPeak", "Low"): [0.60, 0.30, 0.08, 0.02],
            ("OffPeak", "High"): [0.20, 0.45, 0.30, 0.05],
            ("Peak", "Low"): [0.10, 0.50, 0.30, 0.10],
            ("Peak", "High"): [0.05, 0.10, 0.50, 0.35],
        }
        return self._tabular_from_columns(
            variable="PM25Level",
            variable_states=self.states.pm25_level,
            evidence=["TimeOfDay", "TrafficProxy"],
            evidence_states=[self.states.time_of_day, self.states.traffic_proxy],
            distributions=dists,
        )

    def _build_nox_cpd(self) -> TabularCPD:
        dists = {
            ("Good",): [0.70, 0.25, 0.05],
            ("Moderate",): [0.30, 0.50, 0.20],
            ("High",): [0.10, 0.40, 0.50],
            ("VeryHigh",): [0.05, 0.25, 0.70],
        }
        return self._tabular_from_columns(
            variable="NOxLevel",
            variable_states=self.states.nox_level,
            evidence=["PM25Level"],
            evidence_states=[self.states.pm25_level],
            distributions=dists,
        )

    def _build_aqi_cpd(self) -> TabularCPD:
        pm25_idx = {s: i for i, s in enumerate(self.states.pm25_level)}
        nox_idx = {s: i for i, s in enumerate(self.states.nox_level)}
        wind_penalty = {"High": 0.0, "Low": 0.5}

        def dist_from_score(score: float) -> List[float]:
            centers = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
            sigma = 0.9
            w = np.exp(-((centers - score) ** 2) / (2 * sigma**2))
            w /= w.sum()
            return w.round(4).tolist()

        dists: Dict[Tuple[str, str, str], List[float]] = {}
        for pm, nx, wd in product(
            self.states.pm25_level,
            self.states.nox_level,
            self.states.wind_proxy,
        ):
            score = pm25_idx[pm] + 0.7 * nox_idx[nx] + wind_penalty[wd]
            dists[(pm, nx, wd)] = dist_from_score(score)

        dists[("VeryHigh", "High", "Low")] = [0.0, 0.0, 0.01, 0.04, 0.25, 0.70]
        dists[("VeryHigh", "High", "High")] = [0.0, 0.0, 0.03, 0.07, 0.45, 0.45]
        dists[("High", "High", "Low")] = [0.0, 0.0, 0.05, 0.35, 0.50, 0.10]
        dists[("High", "High", "High")] = [0.0, 0.03, 0.12, 0.45, 0.35, 0.05]
        dists[("Good", "Low", "High")] = [0.70, 0.25, 0.04, 0.01, 0.0, 0.0]
        dists[("Good", "Low", "Low")] = [0.55, 0.35, 0.08, 0.02, 0.0, 0.0]

        return self._tabular_from_columns(
            variable="AQICategory",
            variable_states=self.states.aqi_category,
            evidence=["PM25Level", "NOxLevel", "WindProxy"],
            evidence_states=[
                self.states.pm25_level,
                self.states.nox_level,
                self.states.wind_proxy,
            ],
            distributions=dists,
        )

    def build_network(self) -> BayesianNetwork:
        """Create and validate the Bayesian Network with hardcoded CPTs."""
        if not self._supports_pgmpy:
            self.model = None
            return self.model  # type: ignore[return-value]

        model = BayesianNetwork(
            [
                ("TimeOfDay", "PM25Level"),
                ("TrafficProxy", "PM25Level"),
                ("PM25Level", "NOxLevel"),
                ("PM25Level", "AQICategory"),
                ("NOxLevel", "AQICategory"),
                ("WindProxy", "AQICategory"),
            ]
        )

        cpd_time = TabularCPD(
            variable="TimeOfDay",
            variable_card=2,
            values=[[0.65], [0.35]],
            state_names={"TimeOfDay": list(self.states.time_of_day)},
        )
        cpd_traffic = TabularCPD(
            variable="TrafficProxy",
            variable_card=2,
            values=[[0.55], [0.45]],
            state_names={"TrafficProxy": list(self.states.traffic_proxy)},
        )
        cpd_wind = TabularCPD(
            variable="WindProxy",
            variable_card=2,
            values=[[0.60], [0.40]],
            state_names={"WindProxy": list(self.states.wind_proxy)},
        )

        cpd_pm25 = self._build_pm25_cpd()
        cpd_nox = self._build_nox_cpd()
        cpd_aqi = self._build_aqi_cpd()

        model.add_cpds(cpd_time, cpd_traffic, cpd_wind, cpd_pm25, cpd_nox, cpd_aqi)
        model.check_model()

        self.model = model
        return model

    def _fallback_distribution(self, evidence_states: Dict[str, str]) -> Dict[str, float]:
        """Rule-based posterior approximation when pgmpy is unavailable."""
        labels = self.states.aqi_category
        pm_score = {
            "Good": 0,
            "Moderate": 2,
            "High": 3,
            "VeryHigh": 4,
        }.get(evidence_states.get("PM25Level", "Moderate"), 2)
        nox_score = {
            "Low": 0,
            "Moderate": 1,
            "High": 2,
        }.get(evidence_states.get("NOxLevel", "Moderate"), 1)
        peak_bonus = {
            "OffPeak": 0,
            "Peak": 1,
        }.get(evidence_states.get("TimeOfDay", "OffPeak"), 0)

        severity = min(5.0, pm_score + 0.8 * nox_score + 0.4 * peak_bonus)
        centers = np.arange(6, dtype=float)
        sigma = 1.0
        w = np.exp(-((centers - severity) ** 2) / (2 * sigma**2))
        w /= w.sum()

        return {label: float(prob) for label, prob in zip(labels, w)}

    @staticmethod
    def _is_peak_hour(hour: int) -> bool:
        return hour in {7, 8, 9, 17, 18, 19, 20}

    def discretize(self, live_data: Dict) -> Dict[str, str]:
        """Discretize continuous live values into BN states."""
        pm25 = float(live_data.get("PM2.5", 0.0) or 0.0)
        no2 = float(live_data.get("NO2", 0.0) or 0.0)

        if pm25 < 30:
            pm25_level = "Good"
        elif pm25 < 60:
            pm25_level = "Moderate"
        elif pm25 <= 90:
            pm25_level = "High"
        else:
            pm25_level = "VeryHigh"

        if no2 < 40:
            nox_level = "Low"
        elif no2 <= 80:
            nox_level = "Moderate"
        else:
            nox_level = "High"

        hour = int(live_data.get("hour", 12) or 12)
        is_peak = int(live_data.get("is_peak", 1 if self._is_peak_hour(hour) else 0))
        time_of_day = "Peak" if self._is_peak_hour(hour) else "OffPeak"
        traffic_proxy = "High" if is_peak == 1 else "Low"

        station_count = int(live_data.get("station_count", 0) or 0)
        wind_proxy = "Low" if station_count <= 5 else "High"

        return {
            "PM25Level": pm25_level,
            "NOxLevel": nox_level,
            "TimeOfDay": time_of_day,
            "TrafficProxy": traffic_proxy,
            "WindProxy": wind_proxy,
        }

    def predict(self, live_data: Dict) -> Dict:
        """Predict AQI category distribution from discretized live data."""
        if self.model is None:
            self.build_network()

        evidence_states = self.discretize(live_data)
        if self._supports_pgmpy and self.model is not None:
            inference = VariableElimination(self.model)
            query_evidence = {
                "PM25Level": evidence_states["PM25Level"],
                "NOxLevel": evidence_states["NOxLevel"],
                "TimeOfDay": evidence_states["TimeOfDay"],
            }
            result = inference.query(["AQICategory"], evidence=query_evidence, show_progress=False)
            probs = result.values
            labels = self.states.aqi_category
            distribution = {label: float(prob) for label, prob in zip(labels, probs)}
        else:
            distribution = self._fallback_distribution(evidence_states)
            labels = self.states.aqi_category
            probs = np.array([distribution[label] for label in labels], dtype=float)

        predicted_idx = int(np.argmax(probs))
        predicted = labels[predicted_idx]
        confidence = float(probs[predicted_idx])

        explanation = (
            f"{evidence_states['PM25Level']} PM2.5 with {evidence_states['NOxLevel']} NO2 "
            f"during {evidence_states['TimeOfDay']} hours gives highest probability for {predicted} AQI "
            f"({confidence:.2%})."
        )

        return {
            "distribution": distribution,
            "predicted": predicted,
            "confidence": confidence,
            "explanation": explanation,
            "discrete_inputs": evidence_states,
        }
