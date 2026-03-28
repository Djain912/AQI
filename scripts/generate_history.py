"""
Generate 2 months of synthetic historical CPCB data for Mumbai to train the Neural Network.
The real-time data.gov.in API only provides current snapshots, so this script generates
backdated realistic hourly data to immediately allow NN training.
"""

import os
from datetime import datetime, timedelta
import random
import pandas as pd
from data.station_store import StationHistoricalStore

def generate_historical_data(days=60):
    print(f"Generating {days} days of historical data for Mumbai...")
    
    store = StationHistoricalStore()
    
    stations = [
        "Bandra Kurla Complex",
        "Sion",
        "Worli",
        "Chembur",
        "Mazagaon",
        "Colaba",
        "Borivali East",
        "Andheri East",
    ]
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    curr_time = start_time
    history_records = []
    
    print("Simulating realistic pollutant patterns by hour...")
    while curr_time <= end_time:
        hour = curr_time.hour
        # Higher pollution during rush hours (7-10 AM, 5-9 PM)
        is_peak = 1 if hour in {7, 8, 9, 10, 17, 18, 19, 20} else 0
        
        # Base multipliers
        peak_mult = 1.5 if is_peak else 1.0
        night_mult = 0.7 if hour < 6 else 1.0
        
        for station in stations:
            # Base ranges for Mumbai
            pm25 = random.uniform(30, 90) * peak_mult * night_mult
            pm10 = pm25 * random.uniform(1.2, 2.0)
            no2 = random.uniform(20, 60) * peak_mult
            so2 = random.uniform(10, 30)
            co = random.uniform(0.5, 2.0) * peak_mult
            ozone = random.uniform(20, 80) if 10 <= hour <= 16 else random.uniform(10, 30)
            nh3 = random.uniform(5, 25)
            
            # Add a chance for a "Poor/Severe" spike to teach the NN those classes
            spike = random.random()
            if spike > 0.98: # 2% chance of severe
                pm25 *= 4.0
            elif spike > 0.90: # 8% chance of poor/very poor
                pm25 *= 2.5
                
            record = {
                "timestamp": curr_time.isoformat(),
                "city": "Mumbai",
                "station": station,
                "lat": round(random.uniform(18.9, 19.3), 4),
                "lon": round(random.uniform(72.8, 73.0), 4),
                "PM2.5": round(pm25, 2),
                "PM10": round(pm10, 2),
                "NO2": round(no2, 2),
                "SO2": round(so2, 2),
                "CO": round(co, 2),
                "OZONE": round(ozone, 2),
                "NH3": round(nh3, 2),
                "hour": hour,
                "is_peak": is_peak
            }
            history_records.append(record)
            
        curr_time += timedelta(hours=1)
        
    # Convert to DataFrame and save via the app's standard storage system
    df = pd.DataFrame(history_records)
    
    # Needs aqi_value and aqi_category for training
    from data.fetcher import calculate_aqi_from_pm25
    from app import category_from_aqi
    
    df["aqi_value"] = df["PM2.5"].apply(calculate_aqi_from_pm25)
    df["aqi_category"] = df["aqi_value"].apply(category_from_aqi)
    
    # Save the huge dataset
    history_file = store.history_path
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    df.to_csv(history_file, index=False)
    print(f"✅ Successfully generated and saved {len(df)} historical records to {history_file}!")
    
if __name__ == "__main__":
    generate_historical_data(60)
