"""
04_feature_engineering_and_forecasting.py
Memory-optimized script for feature engineering, model training, and forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------- Paths ----------
BASE_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/analysis")
INPUT_FILE = BASE_PATH / "merged" / "payments_tours.parquet"
OUTPUT_PATH = BASE_PATH / "forecast"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ---------- Load Data ----------
print("📂 Loading merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"✅ Loaded: {df.shape}")

# ---------- Data Cleaning ----------
print("🧹 Cleaning data...")
df = df.drop_duplicates()
df = df.fillna(0)

# Detect numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ---------- Feature Engineering ----------
print("🧠 Creating new features...")
for col in numeric_cols:
    df[f"{col}_log"] = np.log1p(df[col])
    df[f"{col}_zscore"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

# ---------- Encode Categorical Columns ----------
from sklearn.preprocessing import LabelEncoder

print("🔤 Encoding categorical columns...")
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

for col in tqdm(cat_cols, desc="Encoding"):
    try:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    except Exception:
        df[col] = df[col].astype("category").cat.codes

# ---------- Convert Datetime Columns ----------
print("⏱️ Converting datetime columns...")
date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[col] = df[col].view("int64") // 10**9  # convert to Unix timestamp (seconds)



# Add total feature count as complexity indicator
df["feature_count"] = df[numeric_cols].gt(0).sum(axis=1)

# ---------- Target Variable ----------
target_col = "amount" if "amount" in df.columns else numeric_cols[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# ---------- Split Data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Train Model (Memory Safe) ----------
sample_size = min(50000, len(X_train))
X_sample = X_train.sample(sample_size, random_state=42)
y_sample = y_train.loc[X_sample.index]

print(f"🤖 Training Random Forest on {sample_size:,} samples...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_sample, y_sample)

# ---------- Evaluate Model ----------
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📈 R² Score (RandomForest): {r2:.3f}")
print(f"💰 Mean Absolute Error: ₹{mae:.2f}")

# ---------- Save Model Report ----------
with open(OUTPUT_PATH / "rf_model_report.txt", "w") as f:
    f.write(f"R² Score: {r2:.3f}\n")
    f.write(f"MAE: ₹{mae:.2f}\n")
    f.write(f"Samples Used: {sample_size}\n")
    f.write(f"Train Shape: {X_train.shape}\n")

print("🧾 Model performance saved successfully.")

# ---------- Feature Importance ----------
print("📊 Saving feature importance chart...")
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_features.plot(kind='barh')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "feature_importance.png")
plt.close()

print("✅ Feature importance chart saved!")

# ---------- Optional Prophet Forecast (if date exists) ----------
date_cols = [c for c in df.columns if "date" in c.lower()]
if date_cols:
    try:
        from prophet import Prophet
        print(f"📆 Running forecast using column: {date_cols[0]}")
        time_df = df[[date_cols[0], target_col]].copy()
        time_df.columns = ["ds", "y"]
        time_df["ds"] = pd.to_datetime(time_df["ds"], errors="coerce")
        time_df = time_df.dropna()

        model = Prophet()
        model.fit(time_df)

        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        plt.figure(figsize=(10, 5))
        model.plot(forecast)
        plt.title("90-Day Revenue Forecast (Prophet)")
        plt.savefig(OUTPUT_PATH / "future_revenue_forecast.png")
        plt.close()

        print("📊 Prophet forecast saved.")
    except Exception as e:
        print(f"⚠️ Forecast skipped due to: {e}")

print("🎯 Done! All outputs saved successfully.")

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ---------- Ensure output folder exists ----------
OUTPUT_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/database")
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"💾 Output directory confirmed → {OUTPUT_PATH}")

# ---------- Feature Importance Plot ----------
print("📊 Generating Feature Importance plot...")

try:
    importances = rf.feature_importances_
    features = X_sample.columns
    indices = np.argsort(importances)[::-1]
    sorted_features = features[indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[:20][::-1], sorted_importances[:20][::-1])
    plt.title("Top 20 Feature Importances (Random Forest)", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    fi_path = OUTPUT_PATH / "feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"✅ Saved → {fi_path}")

except Exception as e:
    print(f"⚠️ Could not generate feature importance plot: {e}")

# ---------- Actual vs Predicted Scatter Plot ----------
print("📈 Generating Actual vs Predicted plot...")

try:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
    plt.xlabel("Actual Amount")
    plt.ylabel("Predicted Amount")
    plt.title("Actual vs Predicted Values", fontsize=14)
    plt.tight_layout()

    avp_path = OUTPUT_PATH / "actual_vs_predicted.png"
    plt.savefig(avp_path, dpi=300)
    plt.close()
    print(f"✅ Saved → {avp_path}")

except Exception as e:
    print(f"⚠️ Could not save Actual vs Predicted plot: {e}")

# ---------- Model Performance Report ----------
print("🧾 Saving Model Report...")

try:
    report_path = OUTPUT_PATH / "rf_model_report.txt"
    with open(report_path, "w") as f:
        f.write("Random Forest Model Performance Report\n")
        f.write("======================================\n")
        f.write(f"R² Score: {r2:.3f}\n")
        f.write(f"Mean Absolute Error: ₹{mae:.2f}\n")
        f.write(f"Samples Used: {len(X_sample)}\n")
    print(f"✅ Report saved → {report_path}")
except Exception as e:
    print(f"⚠️ Could not save report: {e}")

# ---------- Save Predictions ----------
print("💡 Saving Predictions sample...")

try:
    pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    pred_path = OUTPUT_PATH / "predictions_sample.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"✅ Predictions saved → {pred_path}")
except Exception as e:
    print(f"⚠️ Could not save predictions: {e}")

