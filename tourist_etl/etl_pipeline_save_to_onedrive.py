"""
ETL Pipeline: Export all MySQL tables from `touristapp` into Parquet files
and save them inside your OneDrive folder mounted in WSL.
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm  # ✅ import tqdm for progress bar

# ---------- CONFIG ----------
HOST = "192.168.0.101"
USER = "rajeev"
PASSWORD = "ityug123"
DATABASE = "touristapp"

# 📂 Create dated folder in OneDrive automatically
base_path = Path("/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp/database")
date_folder = datetime.now().strftime("%Y-%m-%d")
TARGET_PATH = base_path / date_folder
TARGET_PATH.mkdir(parents=True, exist_ok=True)

print("🔌 Connecting to MySQL...")

# 🔗 Create SQLAlchemy connection
engine = create_engine(
    f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/{DATABASE}",
    connect_args={"autocommit": True},
    pool_pre_ping=True
)

# 🧠 Connect and fetch tables
with engine.connect() as conn:
    tables = pd.read_sql(text("SHOW TABLES;"), conn).iloc[:, 0].tolist()
    print(f"📦 Found {len(tables)} tables: {tables}")

    # ✅ >>> Replace your old “for table in tables:” loop with this:
    for table in tqdm(tables, desc="Exporting Tables"):
        print(f"\n⬇️ Exporting table: {table}")
        df = pd.read_sql(text(f"SELECT * FROM {table}"), conn)
        file_path = TARGET_PATH / f"{table}.parquet"
        df.to_parquet(file_path, index=False)
        print(f"✅ Saved → {file_path}")

# 🧾 Log the ETL run
log_file = base_path / "etl_log.txt"
with open(log_file, "a") as f:
    f.write(f"{datetime.now()} - Exported {len(tables)} tables to {TARGET_PATH}\n")

print("\n🎯 ETL completed successfully! Files saved to OneDrive.")

