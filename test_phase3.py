"""
Phase 3 prep test script:
- Historical storage pipeline
- Neural network fallback prediction

Run:
    python test_phase3.py
"""

from datetime import datetime, timedelta

from data.historical_store import HistoricalStore
from modules.neural_network import AQINeuralNetwork


def seed_history(store: HistoricalStore) -> None:
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    samples = [
        {"city": "Delhi", "PM2.5": 180, "PM10": 260, "NO2": 95, "SO2": 35, "CO": 2.1, "OZONE": 60, "NH3": 40},
        {"city": "Mumbai", "PM2.5": 75, "PM10": 140, "NO2": 55, "SO2": 18, "CO": 1.2, "OZONE": 45, "NH3": 20},
        {"city": "Pune", "PM2.5": 42, "PM10": 90, "NO2": 38, "SO2": 12, "CO": 0.9, "OZONE": 35, "NH3": 15},
        {"city": "Chennai", "PM2.5": 22, "PM10": 60, "NO2": 25, "SO2": 9, "CO": 0.6, "OZONE": 25, "NH3": 10},
    ]

    for i, s in enumerate(samples):
        ts = base_time - timedelta(hours=i)
        hour = ts.hour
        store.save_reading(
            {
                "timestamp": ts.isoformat(),
                "city": s["city"],
                "PM2.5": s["PM2.5"],
                "PM10": s["PM10"],
                "NO2": s["NO2"],
                "SO2": s["SO2"],
                "CO": s["CO"],
                "OZONE": s["OZONE"],
                "NH3": s["NH3"],
                "hour": hour,
                "is_peak": 1 if hour in {7, 8, 9, 17, 18, 19, 20} else 0,
                "station_count": 8,
            }
        )


def run_historical_test() -> None:
    print("\n=== Historical Store Test ===")
    store = HistoricalStore()

    seed_history(store)
    df = store.load_history()
    print("History shape:", df.shape)
    print(df[["timestamp", "city", "PM2.5", "aqi_category"]].head().to_string(index=False))

    X, y = store.get_training_data()
    print("X shape:", X.shape)
    print("y shape:", y.shape)


def run_nn_fallback_test() -> None:
    print("\n=== Neural Network Fallback Test ===")
    nn = AQINeuralNetwork()

    live = {
        "PM2.5": 145,
        "PM10": 220,
        "NO2": 80,
        "SO2": 22,
        "CO": 1.8,
        "OZONE": 55,
        "hour": 8,
        "is_peak": 1,
    }

    out = nn.predict_live(live)
    print("Mode:", out["mode"])
    print("Predicted:", out["predicted_category"])
    print("Confidence:", out["confidence"])
    print("Probabilities:", out["all_probabilities"])


if __name__ == "__main__":
    run_historical_test()
    run_nn_fallback_test()
