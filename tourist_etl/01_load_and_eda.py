"""
01_load_and_eda.py
------------------
Loads Parquet files and performs basic EDA.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/database/2025-11-06")
EDA_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/analysis/eda")
EDA_PATH.mkdir(parents=True, exist_ok=True)

print("📦 Loading data...")

tables = {}
for file in DATA_PATH.glob("*.parquet"):
    df = pd.read_parquet(file)
    tables[file.stem] = df
    print(f"✅ Loaded {file.stem}: {df.shape}")

# ---------- Basic Cleaning ----------
def clean(df):
    df = df.drop_duplicates().copy()
    for c in df.select_dtypes("object"):
        df[c] = df[c].astype(str).str.strip().replace("nan", "Unknown").fillna("Unknown")
    for c in df.select_dtypes("number"):
        df[c] = df[c].fillna(df[c].median())
    return df

for k, v in tables.items():
    tables[k] = clean(v)

sns.set(style="whitegrid")

# ---------- Simple EDA ----------
if "bookings" in tables:
    print("\n🔍 Booking summary:")
    print(tables["bookings"].describe(include="all").T)

if "payments" in tables:
    plt.figure(figsize=(6, 4))
    sns.histplot(tables["payments"]["amount"], bins=30, kde=True)
    plt.title("Distribution of Payment Amounts")
    plt.tight_layout()
    plt.savefig(EDA_PATH / "payment_distribution.png")
    plt.close()
    print("📊 Saved → payment_distribution.png")

if "reviews" in tables:
    plt.figure(figsize=(6, 4))
    sns.histplot(tables["reviews"]["rating"], bins=5)
    plt.title("Hotel Review Ratings")
    plt.tight_layout()
    plt.savefig(EDA_PATH / "ratings_distribution.png")
    plt.close()
    print("📊 Saved → ratings_distribution.png")

# ---------- Save cleaned versions for next steps ----------
for name, df in tables.items():
    df.to_parquet(DATA_PATH / f"{name}_cleaned.parquet", index=False)

print("\n🎯 EDA completed. Cleaned files saved for next step.")
