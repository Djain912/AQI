"""
Collect Mumbai API snapshots in bulk and train neural model from history.

Usage examples:
  python scripts/collect_and_train_mumbai.py --rounds 60 --interval 2
  python scripts/collect_and_train_mumbai.py --rounds 120 --interval 1 --skip-train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.historical_store import HistoricalStore
from data.station_store import StationHistoricalStore
from modules.neural_network import AQINeuralNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk collect Mumbai AQI API snapshots and train NN")
    parser.add_argument("--rounds", type=int, default=60, help="Number of API collection rounds")
    parser.add_argument("--interval", type=float, default=2.0, help="Sleep seconds between rounds")
    parser.add_argument("--skip-train", action="store_true", help="Only collect history, do not train")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="Minimum rows required before training starts (default 200)",
    )
    parser.add_argument(
        "--dataset",
        choices=["station", "city"],
        default="station",
        help="Training dataset mode: station-level (recommended) or city-level",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing for station-level training",
    )
    args = parser.parse_args()

    city = "Mumbai"

    if args.dataset == "station":
        store = StationHistoricalStore()
        print(
            f"Collecting STATION snapshots for {city}: rounds={args.rounds}, interval={args.interval}s"
        )
        saved = store.collect_batches(city=city, rounds=args.rounds, interval_seconds=args.interval)
        print(f"Saved {saved} new station rows for {city}.")

        history = store.load_history()
        mumbai_rows = int((history["city"] == city).sum()) if not history.empty else 0
        print(f"Current Mumbai station history rows: {mumbai_rows}")

        if args.skip_train:
            print("Skipped training as requested.")
            return

        X, y = store.get_training_data(balance=not args.no_balance)
    else:
        store = HistoricalStore()
        print(f"Collecting CITY snapshots for {city}: rounds={args.rounds}, interval={args.interval}s")
        saved = store.collect_api_batches(city=city, rounds=args.rounds, interval_seconds=args.interval)
        print(f"Saved {saved} new city rows for {city}.")

        history = store.load_history()
        mumbai_rows = int((history["city"] == city).sum()) if not history.empty else 0
        print(f"Current Mumbai city history rows: {mumbai_rows}")

        if args.skip_train:
            print("Skipped training as requested.")
            return

        X, y = store.get_training_data()

    nn = AQINeuralNetwork()
    result = nn.train(X, y, min_rows=args.min_rows)

    if result.get("fallback"):
        print(
            "NN fallback active: "
            f"reason={result.get('reason')}, rows={result.get('rows')}. "
            "Collect more data and run again."
        )
    else:
        print(
            f"NN training complete. Backend={result.get('backend', 'tensorflow')}, "
            f"Test accuracy={result.get('test_accuracy', 0.0):.2%}"
        )


if __name__ == "__main__":
    main()
