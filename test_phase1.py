"""
test_phase1.py
==============
Phase 1 test script — demonstrates AQI fetcher output for Mumbai.

Run:
    python test_phase1.py               # uses demo data (no API key needed)
    python test_phase1.py --live        # tries real API (requires .env)
"""

import sys
import argparse
import pandas as pd

# ── Pretty display helpers ───────────────────────────────────────────────────
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.2f}".format)

DIVIDER = "=" * 72


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — AQI calculation accuracy (no API needed)
# ═══════════════════════════════════════════════════════════════════════════════

def test_aqi_calculation() -> None:
    from data.fetcher import calculate_aqi_from_pm25, _aqi_category

    section("TEST 1 · AQI Calculation (PM2.5 → AQI breakpoints)")

    test_cases = [
        (0.0,   "Good"),
        (8.5,   "Good"),
        (12.0,  "Good"),
        (25.0,  "Satisfactory"),
        (45.0,  "Moderate"),
        (75.0,  "Poor"),
        (175.0, "Very Poor"),
        (275.0, "Severe"),
        (400.0, "Hazardous"),
        (float("nan"), "N/A"),
    ]

    print(f"\n{'PM2.5 (µg/m³)':>16} {'AQI':>8} {'Category':<15} {'Status'}")
    print("-" * 55)

    all_pass = True
    for pm25, expected_cat in test_cases:
        aqi  = calculate_aqi_from_pm25(pm25)
        cat  = _aqi_category(pm25)
        ok   = cat == expected_cat
        all_pass &= ok
        pm_str  = f"{pm25:.1f}" if not (isinstance(pm25, float) and pm25 != pm25) else "NaN"
        aqi_str = f"{aqi:.1f}" if aqi is not None else "None"
        status  = "✓ PASS" if ok else f"✗ FAIL (expected {expected_cat})"
        print(f"{pm_str:>16} {aqi_str:>8} {cat:<15} {status}")

    print(f"\n{'All tests passed ✓' if all_pass else 'SOME TESTS FAILED ✗'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Demo data (offline, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

def test_demo_data(city: str = "Mumbai") -> pd.DataFrame:
    from data.fetcher import load_demo_data

    section(f"TEST 2 · Demo / Offline Data — {city}")

    df = load_demo_data(city)
    print(f"\nShape: {df.shape}   (rows × cols)")
    print(f"Columns: {list(df.columns)}\n")

    # ── Station summary ───────────────────────────────────────────────────────
    display_cols = ["station", "PM2.5", "PM10", "NO2", "AQI", "AQI_Category"]
    print("── Monitoring Stations ──────────────────────────────────────")
    print(df[display_cols].to_string(index=False))

    # ── Descriptive stats ─────────────────────────────────────────────────────
    section(f"TEST 2b · Descriptive Statistics — {city}")
    numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "OZONE", "NH3", "AQI"]
    print(df[numeric_cols].describe().round(2).to_string())

    # ── AQI category distribution ─────────────────────────────────────────────
    section("TEST 2c · AQI Category Distribution")
    cat_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe", "Hazardous"]
    counts = df["AQI_Category"].value_counts().reindex(cat_order).dropna().astype(int)
    print()
    for cat, n in counts.items():
        bar = "█" * n
        print(f"  {cat:<15} {bar} ({n})")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Live API fetch (requires API key in .env)
# ═══════════════════════════════════════════════════════════════════════════════

def test_live_api(city: str = "Mumbai") -> None:
    section(f"TEST 3 · Live API Fetch — {city}")

    try:
        from data.fetcher import fetch_city_pollutant, fetch_pivoted

        # 3a: Single pollutant (quick test)
        print(f"\n[3a] fetch_city_pollutant('{city}', 'PM2.5')")
        df_single = fetch_city_pollutant(city, "PM2.5")
        if df_single.empty:
            print("  ⚠  No data returned — check city name or API key.")
        else:
            print(f"  Rows: {len(df_single)}")
            print(df_single[["station", "pollutant", "avg", "min_val", "max_val", "last_update"]].head(5).to_string(index=False))

        # 3b: Pivoted (all pollutants)
        print(f"\n[3b] fetch_pivoted('{city}')")
        df_wide = fetch_pivoted(city)
        if df_wide.empty:
            print("  ⚠  No data returned.")
        else:
            print(f"  Shape: {df_wide.shape}")
            cols = ["station", "PM2.5", "PM10", "NO2", "AQI", "AQI_Category"]
            print(df_wide[[c for c in cols if c in df_wide.columns]].to_string(index=False))

    except EnvironmentError as e:
        print(f"\n  ✗ API Key Error:\n  {e}")
    except Exception as e:  # noqa: BLE001
        print(f"\n  ✗ Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 — AQI Fetcher Tests")
    parser.add_argument("--live", action="store_true", help="Run live API test (requires .env)")
    parser.add_argument("--city", default="Mumbai", help="City name for live test")
    args = parser.parse_args()

    print("\n🌫️  AQI Data Fetcher — Phase 1 Test Suite")
    print(f"   City: {args.city} | Mode: {'LIVE API' if args.live else 'DEMO (offline)'}")

    test_aqi_calculation()
    demo_df = test_demo_data(args.city)

    if args.live:
        test_live_api(args.city)
    else:
        print(f"\n\n💡 Tip: Run with --live to test the real data.gov.in API.")
        print("   Make sure DATA_GOV_API_KEY is set in .env first.\n")

    print(f"\n{DIVIDER}")
    print("  Phase 1 complete. ✓")
    print(DIVIDER)
