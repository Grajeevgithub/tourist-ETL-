"""
03_prediction_model.py
----------------------
Predicts payment amounts using merged data.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

MERGED_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/analysis/merged")
OUTPUT_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/analysis/model")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("📊 Loading merged data...")
file = MERGED_PATH / "payments_tours.parquet"
data = pd.read_parquet(file)
print(f"✅ Loaded {file.name}: {data.shape}")

# ---------- Prepare Data ----------
if "amount" not in data.columns:
    raise ValueError("❌ 'amount' column missing in dataset.")

X = data.select_dtypes("number").drop(columns=["amount"], errors="ignore")
y = data["amount"].fillna(data["amount"].median())
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Train Model ----------
print("🤖 Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📈 R² Score: {r2:.3f}")
print(f"💰 Mean Absolute Error: ₹{mae:.2f}")

# ---------- Save Results ----------
with open(OUTPUT_PATH / "model_report.txt", "w") as f:
    f.write(f"R² Score: {r2:.3f}\n")
    f.write(f"MAE: ₹{mae:.2f}\n")
    f.write(f"Training Samples: {len(X_train)}\n")

print("🧾 Model report saved to model_report.txt")
