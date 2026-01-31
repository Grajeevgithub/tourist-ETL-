









# 🌍 Tourist Data ETL & Analysis Projec

This project presents a complete **Data Engineering and Analysis pipeline** for tourism data using **Python, MySQL, and Power BI**.  
It automates data extraction, cleaning, transformation, and visualization — delivering deep insights into bookings, transport modes, and yearly travel trends.

---

## 🎯 Objective

To build an end-to-end pipeline that:
- Extracts data from a **MySQL database**
- Cleans, transforms, and stores it efficiently in **Parquet format**
- Analyzes tourism metrics like routes, bookings, and transport usage
- Predicts future trends using **machine learning models**

---

## 🧰 Tools & Skills Used

- **Python** (Pandas, SQLAlchemy, Scikit-learn, Matplotlib)
- **MySQL** (Data Source)
- **Power BI** (Dashboard & KPIs)
- **ETL Automation** (via Ubuntu / WSL)
- **Machine Learning** (Random Forest for forecasting)

---

## ✅ Key Features

- ⚙️ Automated ETL pipeline (MySQL → Parquet → Cleaned datasets)  
- 🧹 Data cleaning, EDA, and merging of multi-table relationships  
- 🧠 Predictive modeling using Random Forest  
- 🚍 Route and transport trend analysis (Bus, Train, Flight)  
- 📆 Year-wise route forecasting with visual trend charts  
- 📈 Exported reports, graphs, and model metrics  

---

## 🔍 Project Insights

- Bus routes have the **highest frequency**, followed closely by trains.  
- Average travel **duration and cost vary significantly** by mode of transport.  
- **Revenue and route activity** show clear seasonal and yearly growth trends.  
- Predictive models provide insight into **future travel demand patterns**.

---

## 📊 Generated Outputs

| Output Type | File Example |
|--------------|--------------|
| Cleaned Data | `*_cleaned.parquet` |
| Model Report | `rf_model_report.txt` |
| Feature Importance | `feature_importance.png` |
| Forecast Visualization | `actual_vs_predicted.png` |
| Route Trends | `route_trends_by_year.png` |

---

## 📂 Project Folder Structure

touristapp/
│
├── etl_pipeline_save_to_onedrive.py # Extract & save MySQL data to OneDrive
├── 01_load_and_eda.py # Load and explore cleaned data
├── 02_merge_and_analysis.py # Merge and analyze datasets
├── 03_prediction_model.py # Train and evaluate model
├── 04_feature_engineering_and_forecasting.py # Forecasting & visualization
├── 05_route_analysis.py # Transport route insights
├── 06_route_trend_analysis.py # Year-wise trend analysis
├── analysis_and_prediction.py # Combined predictive analysis
└── *.parquet / *.png / *.txt # Generated data and output files

---


---

## 🖥️ How to Run

```bash
# Step 1 — Open Ubuntu terminal
cd "/mnt/c/Users/DELL/OneDrive/Desktop/data stimulate/touristapp"

# Step 2 — Run the ETL and analysis scripts
python3 etl_pipeline_save_to_onedrive.py
python3 01_load_and_eda.py
python3 02_merge_and_analysis.py
python3 03_prediction_model.py
python3 04_feature_engineering_and_forecasting.py
python3 05_route_analysis.py
python3 06_route_trend_analysis.py
