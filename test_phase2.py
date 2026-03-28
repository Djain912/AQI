"""
Phase 2 test script for:
1) Bayesian Network (pgmpy)
2) Fuzzy Logic (scikit-fuzzy)

Run:
    python test_phase2.py
"""

from modules.bayesian_network import AQIBayesianNetwork
from modules.fuzzy_logic import AQIFuzzySystem


def run_bayesian_test() -> None:
    print("\n=== Bayesian Network Test ===")
    bn = AQIBayesianNetwork()
    bn.build_network()

    test_input = {
        "PM2.5": 150,
        "NO2": 90,
        "hour": 8,
        "is_peak": 1,
        "station_count": 4,
    }

    result = bn.predict(test_input)
    dist = result["distribution"]

    severe_mass = dist.get("Poor", 0.0) + dist.get("VeryPoor", 0.0) + dist.get("Severe", 0.0)

    print("Input:", test_input)
    print("Predicted:", result["predicted"])
    print("Confidence:", f"{result['confidence']:.3f}")
    print("Poor+VeryPoor+Severe:", f"{severe_mass:.3f}")
    print("Explanation:", result["explanation"])


def run_fuzzy_test() -> None:
    print("\n=== Fuzzy Logic Test ===")
    fuzzy = AQIFuzzySystem()
    fuzzy.build_system()

    result = fuzzy.predict(pm25_val=220, no2_val=140, so2_val=55)

    print("Input: pm25=220, no2=140, so2=55")
    print("AQI Score:", result["aqi_score"])
    print("Category:", result["category"])
    print("Color:", result["color_hex"])

    max_mf = max(result["pm25_membership"], key=result["pm25_membership"].get)
    print("Top PM2.5 Membership:", max_mf, f"({result['pm25_membership'][max_mf]:.3f})")

    out_path = fuzzy.plot_membership_functions()
    print("Membership plot saved at:", out_path)


if __name__ == "__main__":
    run_bayesian_test()
    run_fuzzy_test()
