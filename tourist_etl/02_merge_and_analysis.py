"""
02_merge_and_analysis.py
------------------------
Merges tables safely using dtype alignment and auto key detection.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/database/2025-11-06")
MERGED_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/analysis/merged")
MERGED_PATH.mkdir(parents=True, exist_ok=True)

# ---------- Load Cleaned Tables ----------
def load(name):
    return pd.read_parquet(DATA_PATH / f"{name}_cleaned.parquet")

print("📂 Loading cleaned data...")
tables = {}
for file in DATA_PATH.glob("*_cleaned.parquet"):
    tables[file.stem.replace("_cleaned", "")] = pd.read_parquet(file)
    print(f"✅ {file.stem}")

# ---------- Helper: Safe merge ----------
def safe_merge(left, right, keys):
    for key in keys:
        if key in left.columns and key in right.columns:
            print(f"🔗 Merging on {key}")
            # Align data types
            if left[key].dtype != right[key].dtype:
                left[key] = left[key].astype(str)
                right[key] = right[key].astype(str)
            return left.merge(right, on=key, how="left")
    print("⚠️ No key found, skipping merge.")
    return left.copy()

# ---------- Perform Merges ----------
if "bookings" in tables and "destinations" in tables:
    merged_bookings = safe_merge(tables["bookings"], tables["destinations"],
                                 ["destination_id", "destinationID", "destination"])
    merged_bookings.to_parquet(MERGED_PATH / "bookings_destinations.parquet")
    print("✅ Saved bookings_destinations.parquet")

if "reviews" in tables and "hotels" in tables:
    merged_reviews = safe_merge(tables["reviews"], tables["hotels"],
                                ["hotel_id", "hotelID", "hotel"])
    merged_reviews.to_parquet(MERGED_PATH / "reviews_hotels.parquet")
    print("✅ Saved reviews_hotels.parquet")

if "payments" in tables and "tours" in tables:
    merged_tours = safe_merge(tables["payments"], tables["tours"],
                              ["tour_id", "tourID", "tour"])
    merged_tours.to_parquet(MERGED_PATH / "payments_tours.parquet")
    print("✅ Saved payments_tours.parquet")

print("\n🎯 Merging completed successfully!")
