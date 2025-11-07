import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------- Paths ----------
DATA_PATH = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/database/2025-11-06")
OUTPUT_PATH = DATA_PATH
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("📂 Loading route data...")

# ---------- Load Datasets ----------
routes = pd.read_parquet(DATA_PATH / "routes_cleaned.parquet")
origins = pd.read_parquet(DATA_PATH / "origins_cleaned.parquet")
destinations = pd.read_parquet(DATA_PATH / "destinations_cleaned.parquet")

print(f"✅ Loaded: routes({routes.shape}), origins({origins.shape}), destinations({destinations.shape})")

# ---------- Normalize Column Names ----------
routes.columns = routes.columns.str.lower()
origins.columns = origins.columns.str.lower()
destinations.columns = destinations.columns.str.lower()

# ---------- Merge origin and destination names ----------
print("🔗 Merging origin and destination names...")

routes = routes.merge(
    origins[['origin_id', 'origin_name']],
    on='origin_id',
    how='left'
)

routes = routes.merge(
    destinations[['destination_id', 'name']],
    on='destination_id',
    how='left'
).rename(columns={'name': 'destination_name'})

print("✅ Merged successfully — routes now have origin and destination names!")

# ---------- Route Counts by Transport ----------
transport_col = 'transport_type'
transport_counts = routes[transport_col].value_counts().reset_index()
transport_counts.columns = ['Transport_Mode', 'Count']

print("\n🚦 Route counts by transport mode:")
print(transport_counts)

plt.figure(figsize=(8, 5))
sns.barplot(x='Transport_Mode', y='Count', data=transport_counts)
plt.title("Most Common Transport Modes", fontsize=14)
plt.xlabel("Transport Mode")
plt.ylabel("Number of Routes")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "transport_mode_counts.png", dpi=300)
plt.close()
print(f"✅ Saved → transport_mode_counts.png")

# ---------- Top Routes per Mode ----------
print("\n🔝 Finding top routes per mode...")

top_routes = (
    routes.groupby([transport_col, 'origin_name', 'destination_name'])
    .size()
    .reset_index(name='Count')
    .sort_values(['Count'], ascending=False)
)
top_routes.to_csv(OUTPUT_PATH / "top_routes_summary.csv", index=False)
print(f"✅ Saved → top_routes_summary.csv")

# ---------- Visualize Each Mode ----------
for mode in ['Bus', 'Plane', 'Train']:
    mode_routes = top_routes[top_routes[transport_col].str.contains(mode, case=False, na=False)]
    if not mode_routes.empty:
        print(f"\n📍 Top 5 {mode} Routes:")
        print(mode_routes.head())

        plt.figure(figsize=(9, 6))
        sns.barplot(
            x='Count',
            y=mode_routes.head(10).apply(lambda r: f"{r['origin_name']} → {r['destination_name']}", axis=1),
            data=mode_routes.head(10),
            palette='viridis'
        )
        plt.title(f"Top 10 {mode} Routes", fontsize=14)
        plt.xlabel("Number of Trips")
        plt.ylabel("Route")
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / f"top_{mode.lower()}_routes.png", dpi=300)
        plt.close()
        print(f"✅ Saved → top_{mode.lower()}_routes.png")
    else:
        print(f"⚠️ No routes found for {mode}")

print("\n🎯 Route Analysis Complete! Check your output folder:")
print(OUTPUT_PATH)

