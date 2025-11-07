"""
TouristApp Data Cleaning, Analysis & Prediction (Auto Key Version)
------------------------------------------------------------------
Automatically detects foreign keys (e.g., destination_id, destinationID)
and performs EDA + prediction without schema errors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- CONFIG ----------
DATA_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/database/2025-11-06")
OUTPUT_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/analysis")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("📦 Loading Parquet files...")

# ---------- LOAD ALL TABLES ----------
tables = {}
for file in DATA_PATH.glob("*.parquet"):
    name = file.stem
    df = pd.read_parquet(file)
    tables[name] = df
    print(f"✅ Loaded {name}: {df.shape}")

# ---------- CLEANING FUNCTION ----------
def clean_dataframe(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes("object"):
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", "Unknown").fillna("Unknown")
    for col in df.select_dtypes("number"):
        df[col] = df[col].fillna(df[col].median())
    return df

for name, df in tables.items():
    tables[name] = clean_dataframe(df)
    print(f"🧹 Cleaned {name}: {tables[name].shape}")

# ---------- HELPER FUNCTION ----------
def safe_merge(left, right, possible_keys):
    """Merge two DataFrames using the first common key found."""
    for key in possible_keys:
        if key in left.columns and key in right.columns:
            print(f"🔗 Merging on key: {key}")
            return left.merge(right, on=key, how="left")
    print("⚠️ No matching key found, skipping merge.")
    return left.copy()

sns.set(style="whitegrid")

# ---------- EDA 1: Top 10 Destinations ----------
if "bookings" in tables and "destinations" in tables:
    bookings = tables["bookings"]
    dests = tables["destinations"]
    merged = safe_merge(bookings, dests, ["destination_id", "destinationID", "destination"])
    
    if "destination_name" in merged.columns:
        top_dest = merged["destination_name"].value_counts().head(10)
    elif "name" in merged.columns:
        top_dest = merged["name"].value_counts().head(10)
    else:
        print("⚠️ Destination name column not found.")
        top_dest = pd.Series()

    if not top_dest.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=top_dest.values, y=top_dest.index, palette="viridis")
        plt.title("Top 10 Destinations by Bookings")
        plt.xlabel("Number of Bookings")
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / "top_destinations.png")
        plt.close()
        print("📊 Saved → top_destinations.png")

# ---------- EDA 2: Top 10 Tours by Revenue ----------
if "payments" in tables and "tours" in tables:
    payments = tables["payments"]
    tours = tables["tours"]
    merged = safe_merge(payments, tours, ["tour_id", "tourID", "tour"])
    name_col = "tour_name" if "tour_name" in merged.columns else "name" if "name" in merged.columns else None

    if name_col:
        revenue = merged.groupby(name_col)["amount"].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=revenue.values, y=revenue.index, palette="magma")
        plt.title("Top 10 Tours by Revenue")
        plt.xlabel("Revenue (₹)")
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / "top_tours_revenue.png")
        plt.close()
        print("📊 Saved → top_tours_revenue.png")
    else:
        print("⚠️ Could not find tour name column, skipping chart.")

# ---------- EDA 3: Top 10 Hotels by Rating ----------
if "reviews" in tables and "hotels" in tables:
    reviews = tables["reviews"]
    hotels = tables["hotels"]
    merged = safe_merge(reviews, hotels, ["hotel_id", "hotelID", "hotel"])
    name_col = "hotel_name" if "hotel_name" in merged.columns else "name" if "name" in merged.columns else None

    if name_col and "rating" in merged.columns:
        avg_rating = merged.groupby(name_col)["rating"].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=avg_rating.values, y=avg_rating.index, palette="coolwarm")
        plt.title("Top 10 Hotels by Average Rating")
        plt.xlabel("Average Rating")
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / "top_hotels_ratings.png")
        plt.close()
        print("📊 Saved → top_hotels_ratings.png")
    else:
        print("⚠️ Missing columns for hotel ratings analysis.")

# ---------- MACHINE LEARNING MODEL ----------
print("\n🤖 Building Payment Prediction Model...")

if "bookings" in tables and "payments" in tables:
    bookings = tables["bookings"]
    payments = tables["payments"]
    data = safe_merge(bookings, payments, ["booking_id", "bookingID", "booking"])
    
    if "amount" in data.columns:
        numeric_cols = data.select_dtypes("number").columns.tolist()
        if len(numeric_cols) >= 2:
            X = data[numeric_cols].drop(columns=["amount"], errors="ignore")
            y = data["amount"].fillna(data["amount"].median())
            
            X = X.fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"📈 Model R² Score: {r2:.3f}")
            print(f"💰 Mean Absolute Error: ₹{mae:.2f}")
            
            with open(OUTPUT_PATH / "model_report.txt", "w") as f:
                f.write(f"R² Score: {r2:.3f}\n")
                f.write(f"Mean Absolute Error: ₹{mae:.2f}\n")
                f.write(f"Trained on {len(X_train)} samples\n")
            print("🧾 Saved → model_report.txt")
        else:
            print("⚠️ Not enough numeric data for prediction.")
    else:
        print("⚠️ No 'amount' column found in payments for prediction.")
else:
    print("⚠️ Missing bookings or payments table.")

# ---------- SUMMARY ----------
with open(OUTPUT_PATH / "analysis_log.txt", "a") as f:
    f.write(f"{datetime.now()} - Completed cleaning, EDA, and prediction.\n")

print("\n🎯 Analysis and Prediction completed successfully!")
print(f"📂 Results saved to: {OUTPUT_PATH}")

