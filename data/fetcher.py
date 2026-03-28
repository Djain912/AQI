"""
data/fetcher.py
================
Real-time AQI data fetcher from data.gov.in CPCB API.

Functions:
    fetch_city_pollutant()  - Fetch a single pollutant for a city
    fetch_all_pollutants()  - Fetch all pollutants for a city
    fetch_pivoted()         - Wide-format DataFrame (one row per station)
    calculate_aqi_from_pm25() - Standard EPA breakpoint AQI calculation

Usage:
    from data.fetcher import fetch_pivoted, calculate_aqi_from_pm25
    df = fetch_pivoted("Mumbai")
    df["AQI_calc"] = df["PM2.5"].apply(calculate_aqi_from_pm25)

API Source: https://data.gov.in (CPCB Real-Time Air Quality)
"""

import os
import time
import logging
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL    = "https://api.data.gov.in/resource/{resource_id}"
#  Primary CPCB real-time AQI dataset (data.gov.in resource ID)
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
REQUEST_TIMEOUT = 15

DEFAULT_API_KEY = (
    os.getenv("DATA_GOV_KEY")
    or os.getenv("DATA_GOV_API_KEY")
    or "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
)

DEFAULT_CITY = "Mumbai"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]

# ── EPA PM2.5 → AQI Breakpoints ──────────────────────────────────────────────
# Source: US EPA AQI Technical Assistance Document (2018 rev.)
PM25_BREAKPOINTS = [
    # (C_lo, C_hi, I_lo, I_hi, category)
    (0.0,   12.0,   0,   50,  "Good"),
    (12.1,  35.4,  51,  100,  "Satisfactory"),
    (35.5,  55.4, 101,  150,  "Moderate"),
    (55.5, 150.4, 151,  200,  "Poor"),
    (150.5, 250.4, 201, 300,  "Very Poor"),
    (250.5, 350.4, 301, 400,  "Severe"),
    (350.5, 500.4, 401, 500,  "Hazardous"),
]

# ── Helper utilities ─────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Return API key from env, raise clearly if missing."""
    key = os.getenv("DATA_GOV_KEY") or os.getenv("DATA_GOV_API_KEY") or DEFAULT_API_KEY
    if not key or key == "your_api_key_here":
        raise EnvironmentError(
            "DATA_GOV_API_KEY not set.\n"
            "1. Register at https://data.gov.in\n"
            "2. Copy .env.example → .env\n"
            "3. Paste your key in .env"
        )
    return key


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, replacing non-numeric / sentinel text with NaN."""
    # CPCB API sometimes returns 'NA', '-', 'N/A', empty string
    return pd.to_numeric(series.replace({"NA": None, "N/A": None, "-": None, "": None}),
                         errors="coerce")


def _fetch_city_all(city: str) -> list[dict]:
    """
    Fetch all pollutant records for one city in one API call (city filter only).

    Strategy: filter only by city — avoids the timeout bug caused by
    URL-encoding 'PM2.5' (dot in filter value). All pollutant splitting
    is done in-memory with pandas after this single fetch.

    Returns
    -------
    list[dict]  Raw records from API (all pollutants for the city).
    """
    url      = BASE_URL.format(resource_id=RESOURCE_ID)
    all_recs = []
    offset   = 0
    limit    = 500   # max per page to minimise round-trips

    while True:
        params = {
            "api-key":        _get_api_key(),
            "format":         "json",
            "limit":          limit,
            "offset":         offset,
            "filters[city]": city,
        }
        try:
            logger.info("  API call: offset=%d (%ss timeout)...", offset, REQUEST_TIMEOUT)
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("Timeout at offset=%d after %ss. Returning %d records collected so far.",
                         offset, REQUEST_TIMEOUT, len(all_recs))
            break
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP %s: %s", exc.response.status_code, exc.response.text[:300])
            break
        except requests.exceptions.RequestException as exc:
            logger.error("Network error: %s", exc)
            break

        data = resp.json()
        if "records" not in data:
            logger.warning("Unexpected response shape: %s", list(data.keys()))
            break

        batch = data["records"]
        if not batch:
            break

        all_recs.extend(batch)
        total = int(data.get("total", len(all_recs)))
        logger.info("  Got %d records (total available: %d, collected: %d)",
                    len(batch), total, len(all_recs))

        if len(all_recs) >= total:
            break

        offset += limit
        time.sleep(0.2)

    return all_recs


# ── Core clean-up ────────────────────────────────────────────────────────────

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data-cleaning steps to a raw API response DataFrame.

    Steps:
      1. Rename columns to consistent snake_case names
      2. Parse Last_Update as datetime
      3. Coerce numeric pollutant columns
      4. Drop duplicate rows
      5. Reset index
    """
    if df.empty:
        return df

    # 1. Column name normalisation
    rename_map = {
        "country":      "country",
        "station":      "station",
        "state":        "state",
        "city":         "city",
        "pollutant_id": "pollutant",
        "avg_value":    "avg",        # actual API field name
        "min_value":    "min_val",    # actual API field name
        "max_value":    "max_val",    # actual API field name
        "last_update":  "last_update",
        "latitude":     "lat",
        "longitude":    "lon",
    }
    # Lowercase all incoming column names first
    df.columns = [c.lower().strip() for c in df.columns]
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # 2. Datetime parsing
    if "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], dayfirst=True, errors="coerce")

    # 3. Numeric coercion
    for col in ["avg", "min_val", "max_val", "lat", "lon"]:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])

    # 4. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 5. Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_city_pollutant(city: str = DEFAULT_CITY, pollutant: str = "PM2.5") -> pd.DataFrame:
    """
    Fetch one pollutant for a city.

    Uses the bulk-fetch strategy: downloads all city data in one API call,
    then filters to the requested pollutant in-memory.

    Parameters
    ----------
    city      : str  City name (default 'Mumbai').
    pollutant : str  Pollutant ID string (default 'PM2.5')

    Returns
    -------
    pd.DataFrame  Columns: station | city | state | pollutant | avg |
                           min_val | max_val | last_update | lat | lon

    Example
    -------
    >>> df = fetch_city_pollutant(pollutant="PM2.5")
    >>> print(df.head())
    """
    logger.info("Fetching %s data for %s (single bulk call)...", pollutant, city)
    records = _fetch_city_all(city)

    if not records:
        logger.warning("No records returned from API.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = _clean_dataframe(df)

    # Filter to requested pollutant in-memory (no URL-encoding issues)
    if "pollutant" in df.columns and pollutant:
        df = df[df["pollutant"] == pollutant].reset_index(drop=True)

    if df.empty:
        logger.warning("No records for pollutant=%s in %s data.", pollutant, city)
    else:
        logger.info("Done — %d records for %s in %s.", len(df), pollutant, city)
    return df


def fetch_all_pollutants(city: str = DEFAULT_CITY) -> pd.DataFrame:
    """
    Fetch all pollutants for one city in one API call, return long-format DataFrame.

    This is the recommended entry point — makes exactly ONE API request.

    Returns
    -------
    pd.DataFrame  Long-format: one row per (station, pollutant)

    Example
    -------
    >>> df = fetch_all_pollutants()
    >>> print(df["pollutant"].value_counts())
    """
    logger.info("Fetching ALL pollutants for %s (one bulk API call)...", city)
    records = _fetch_city_all(city)

    if not records:
        logger.warning("No data retrieved from API.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = _clean_dataframe(df)

    # Keep only recognised pollutants
    if "pollutant" in df.columns:
        df = df[df["pollutant"].isin(POLLUTANTS)].reset_index(drop=True)

    logger.info("All pollutants fetched — %d rows, %d unique pollutants.",
                len(df), df["pollutant"].nunique() if "pollutant" in df.columns else 0)
    return df


def fetch_pivoted(city: str) -> pd.DataFrame:
    """
    Fetch data for all pollutants and return a WIDE (pivoted) DataFrame,
    one row per monitoring station.

    Columns: station, city, state, lat, lon, last_update,
             PM2.5, PM10, SO2, NO2, CO, OZONE, NH3, Pb,
             AQI (calculated from PM2.5)

    Parameters
    ----------
    city : str  City name (e.g. 'Mumbai')

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> df = fetch_pivoted("Mumbai")
    >>> print(df[["station", "PM2.5", "AQI"]].to_string())
    """
    long_df = fetch_all_pollutants(city)

    if long_df.empty:
        logger.warning("Cannot pivot — no data available.")
        return pd.DataFrame()

    # Check required columns exist
    required = {"station", "pollutant", "avg"}
    if not required.issubset(long_df.columns):
        logger.error("Missing columns for pivot: %s", required - set(long_df.columns))
        return long_df  # return long-format as fallback

    # ── Pivot: rows=station, cols=pollutant, values=avg ──────────────────────
    pivot = long_df.pivot_table(
        index=["station", "city", "state"],
        columns="pollutant",
        values="avg",
        aggfunc="mean",   # average across multiple readings if any
    ).reset_index()

    pivot.columns.name = None   # remove column-axis name artefact

    # ── Attach latest lat/lon/last_update per station ────────────────────────
    meta = (
        long_df.sort_values("last_update", ascending=False)
               .drop_duplicates(subset=["station"])
               [["station", "lat", "lon", "last_update"]]
    )
    pivot = pivot.merge(meta, on="station", how="left")

    # ── Ensure all standard pollutant columns exist (NaN if missing) ─────────
    for poll in POLLUTANTS:
        if poll not in pivot.columns:
            pivot[poll] = float("nan")

    # ── Calculate AQI from PM2.5 ─────────────────────────────────────────────
    pivot["AQI"] = pivot["PM2.5"].apply(calculate_aqi_from_pm25)
    pivot["AQI_Category"] = pivot["PM2.5"].apply(_aqi_category)

    # ── Nice column order ─────────────────────────────────────────────────────
    base_cols  = ["station", "city", "state", "lat", "lon", "last_update"]
    poll_cols  = [p for p in POLLUTANTS if p in pivot.columns]
    aqi_cols   = ["AQI", "AQI_Category"]
    pivot = pivot.reindex(columns=base_cols + poll_cols + aqi_cols)

    logger.info("Pivoted DataFrame shape: %s", pivot.shape)
    return pivot


# ═══════════════════════════════════════════════════════════════════════════════
# AQI CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_aqi_from_pm25(pm25: float) -> Optional[float]:
    """
    Calculate AQI from PM2.5 concentration using EPA linear interpolation.

    Formula:  AQI = ((I_hi - I_lo) / (C_hi - C_lo)) * (C - C_lo) + I_lo

    Parameters
    ----------
    pm25 : float  PM2.5 concentration in µg/m³ (24-hour average)

    Returns
    -------
    float or None  AQI value (0–500), or None if input is NaN / out of range

    Example
    -------
    >>> calculate_aqi_from_pm25(45.0)
    126.0
    >>> calculate_aqi_from_pm25(float('nan'))
    None
    """
    if pm25 is None or (isinstance(pm25, float) and pd.isna(pm25)):
        return None

    pm25 = round(float(pm25), 1)   # CPCB truncates to 1 decimal

    for c_lo, c_hi, i_lo, i_hi, _ in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
            return round(aqi, 1)

    # Concentration above highest breakpoint → cap at 500
    if pm25 > 500.4:
        return 500.0

    return None   # below 0 or otherwise invalid


def _aqi_category(pm25: float) -> str:
    """Return human-readable AQI category string for a PM2.5 value."""
    if pm25 is None or (isinstance(pm25, float) and pd.isna(pm25)):
        return "N/A"

    pm25 = round(float(pm25), 1)
    for c_lo, c_hi, _, _, category in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return category

    return "Hazardous" if pm25 > 500.4 else "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
# OFFLINE DEMO — For testing without an API key
# ═══════════════════════════════════════════════════════════════════════════════

def load_demo_data(city: str = "Mumbai") -> pd.DataFrame:
    """
    Return a realistic synthetic pivoted DataFrame for testing/demo purposes
    when no API key is available.

    Parameters
    ----------
    city : str  Used to label the demo rows.

    Returns
    -------
    pd.DataFrame  Same schema as fetch_pivoted()
    """
    import numpy as np
    rng = np.random.default_rng(42)

    stations = [
        "Bandra Kurla Complex",
        "Sion",
        "Worli",
        "Chembur",
        "Mazagaon (MPCB)",
        "Colaba",
        "Borivali East",
        "Andheri East",
    ]

    pm25_vals  = rng.uniform(20, 180, len(stations)).round(1)
    pm10_vals  = (pm25_vals * rng.uniform(1.5, 2.5, len(stations))).round(1)
    so2_vals   = rng.uniform(5, 40, len(stations)).round(1)
    no2_vals   = rng.uniform(15, 90, len(stations)).round(1)
    co_vals    = rng.uniform(0.5, 4.0, len(stations)).round(2)
    ozone_vals = rng.uniform(10, 80, len(stations)).round(1)
    nh3_vals   = rng.uniform(5, 30, len(stations)).round(1)

    lats = rng.uniform(18.90, 19.25, len(stations)).round(5)
    lons = rng.uniform(72.78, 72.98, len(stations)).round(5)

    df = pd.DataFrame({
        "station":    stations,
        "city":       city,
        "state":      "Maharashtra",
        "lat":        lats,
        "lon":        lons,
        "last_update": pd.Timestamp.now().floor("h"),
        "PM2.5":      pm25_vals,
        "PM10":       pm10_vals,
        "SO2":        so2_vals,
        "NO2":        no2_vals,
        "CO":         co_vals,
        "OZONE":      ozone_vals,
        "NH3":        nh3_vals,
        "Pb":         rng.uniform(0.1, 1.0, len(stations)).round(2),
    })

    df["AQI"]          = df["PM2.5"].apply(calculate_aqi_from_pm25)
    df["AQI_Category"] = df["PM2.5"].apply(_aqi_category)

    return df
