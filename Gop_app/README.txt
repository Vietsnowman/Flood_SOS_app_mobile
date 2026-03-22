README này có:

Project overview

Architecture

Folder structure

Installation

Data pipeline

Model description

Running instructions

Output explanation

.gitignore recommendation

Future improvements

🌊 Flood Prediction & Emergency Response System

A machine learning–based flood prediction and emergency response system designed for commune-level flood risk analysis in Nghệ An Province, Vietnam.

The system integrates:

🌦 Real-time weather data (Open-Meteo API)

🤖 Machine learning flood prediction models

📊 Impact estimation on population and infrastructure

🚨 Emergency response planning (shelters & SOS signals)

The system provides near real-time flood risk predictions and decision support tools for disaster management.

🧠 System Overview

The pipeline performs:

1️⃣ Collect weather data
2️⃣ Generate predictive features
3️⃣ Run trained ML models
4️⃣ Estimate flood probability and affected area
5️⃣ Calculate population impact
6️⃣ Generate response outputs

🏗 System Architecture
Weather API (Open-Meteo)
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Models
 (Flood Probability / Flood Area / Impact Ratio)
        │
        ▼
Flood Prediction Engine
        │
        ▼
Impact Assessment
        │
        ▼
Emergency Response Planning
        │
        ▼
Realtime Outputs + Dashboard
📂 Project Structure
Gop_app
│
├── main_unified.py                # Main pipeline runner
│
├── predict.py                     # Flood prediction logic
├── realtime.py                    # Realtime processing
├── open_meteo.py                  # Weather API integration
├── feature_engineering.py         # Feature generation
├── db_utils.py                    # Database utilities
├── report_utils.py                # Reporting utilities
├── scheduler_cache.py             # Forecast caching
│
├── models/                        # Trained ML models
│   ├── model_area_no_leak.pkl
│   ├── model_prob_no_leak.pkl
│   └── model_ratio_no_leak.pkl
│
├── data/                          # Input datasets
│   ├── DanSo_Xa.csv               # Population data
│   ├── shelters.csv               # Shelter locations
│   ├── resources_default.json     # Emergency resources
│   └── weather_flood datasets
│
├── cache/                         # Cached merged datasets
│
├── realtime_outputs/              # Realtime prediction results
│
├── db/                            # SQLite databases
│
├── src/                           # 🆕 Core package structure cho import sạch
│   ├── api/                       # API modules re-exports (priority.py, routing.py)
│   ├── dashboard/                 # Streamlit dashboards re-exports
│   └── data_processing/           # Data pipelines re-exports
│
├── tests/                         # 🆕 Pytest test suite
│   ├── test_priority_api.py       # API unit tests (FastAPI TestClient)
│   └── test_routing_api.py
│
├── training scripts
│   ├── 0_merge_data.py
│   ├── 1_train_models_no_leak.py
│   └── train_flood_model_final.py
│
├── app6.py                        # Dashboard / visualization
└── app_SOS_shelters7.py           # Emergency interface
⚙️ System Requirements

Recommended environment:

Python 3.10+
RAM ≥ 8GB (recommended for large datasets)

Operating systems supported:

Windows

Linux

macOS

🧪 Installation
1️⃣ Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd Gop_app
2️⃣ Create virtual environment
python -m venv venv

Activate environment:

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
3️⃣ Install dependencies

Create requirements.txt:

pandas
numpy
scikit-learn
joblib
requests
sqlalchemy
geopandas
networkx
shapely
pyarrow

Install:

pip install -r requirements.txt
📊 Dataset

The system uses multiple datasets:

Population
data/DanSo_Xa.csv

Contains population statistics per commune.

Shelter locations
data/shelters.csv

Emergency shelters for evacuation planning.

Weather + flood training data
NgheAn_weather32full_flood_merge_by_commune_time_*.csv

These files are large datasets (>300MB) and should not be uploaded to GitHub.

🤖 Machine Learning Models

Three models are used in the system:

Model	Purpose
model_prob_no_leak.pkl	Predict flood probability
model_area_no_leak.pkl	Predict flooded area
model_ratio_no_leak.pkl	Estimate affected population ratio

Models are trained using historical weather + flood data.

🚀 Running the System

The main program is:

main_unified.py

Run:

python main_unified.py

The program will:

1️⃣ Fetch weather forecast data
2️⃣ Generate prediction features
3️⃣ Run trained ML models
4️⃣ Estimate flood probability and impact
5️⃣ Generate response outputs

📤 Output Files

Results are generated in:

realtime_outputs/

Example outputs:

File	Description
flood_point_probability_rt.csv	Flood probability per location
flood_extent_mask_rt.csv	Flooded area mask
impact_click_zone_rt.csv	Population impact zones
zone_points_rt.csv	Commune data points
meta_rt.json	Metadata
sos_signals.csv	SOS emergency signals
osm_graph.graphml	Road network graph
🧠 Model Training (Optional)

To retrain models:

Step 1
python 0_merge_data.py
Step 2
python 1_train_models_no_leak.py

or

python train_flood_model_final.py

New models will be saved to:

models/
🖥 Dashboard

Two dashboard interfaces are available:

app6.py
app_SOS_shelters7.py

Run example:

python app6.py

This launches the visual flood monitoring dashboard.

🧪 Unit Testing (MỚI)
Hệ thống AI đã được tích hợp bộ test tự động sử dụng `pytest` kết hợp với `httpx` (FastAPI TestClient).
Để chạy toàn bộ test suit, từ folder Gop_app chạy lệnh:

uv run pytest tests/ -v

Test files bao gồm các scenarios biên đổi:
- `tests/test_priority_api.py`: Check logic AI chấm điểm (hỗ trợ graceful fallback).
- `tests/test_routing_api.py`: Check định dạng routing và logic ngập.

🗂 Recommended .gitignore
cache/
realtime_outputs/
db/

__pycache__/
*.pyc

*.db
*.parquet
*.pkl

large_datasets/

Large datasets should be stored outside GitHub or using Git LFS.

🔬 Future Improvements

Planned improvements:

Deep learning flood models

Real-time IoT rainfall sensors

Satellite flood detection

Interactive GIS dashboard

Mobile alert system

Cloud deployment

👨‍💻 Authors: Tran Quoc Viet

Flood Prediction & Disaster Response System
Developed for early flood warning and emergency planning.

FLOOD_UNIFIED/
├─ main_unified.py                # ✅ App gộp (entrypoint duy nhất)
│
├─ app_SOS_shelters7.py           # ✅ App realtime gốc (giữ nguyên)
├─ app6.py                        # ✅ App forecast/DB gốc (giữ nguyên)
│
├─ realtime.py                    # (nếu muốn chạy tạo outputs realtime)
├─ train_flood_model_final.py     # (nếu muốn train lại point model)
│
├─ 0_merge_data.py                # (merge dữ liệu xã)
├─ 1_train_models_no_leak.py      # (train 3 model xã)
├─ scheduler_cache.py             # (build cache dự báo xã)
├─ predict.py / open_meteo.py / feature_engineering.py / db_utils.py / report_utils.py
│
├─ models/                        # (cho app forecast)
│  ├─ model_prob_no_leak.pkl
│  ├─ model_area_no_leak.pkl
│  └─ model_ratio_no_leak.pkl
│
├─ db/
│  └─ app.db                      # (app6 tự tạo nếu chưa có)
│
├─ cache/                         # (app6)
│  ├─ merged.csv
│  └─ forecasts/
│      └─ YYYY-MM-DD_commune_forecasts_7d.parquet (khuyến nghị)
│
├─ data/                          # (app6)
│  ├─ DanSo_Xa.csv
│  ├─ shelters.csv
│  ├─ resources_default.json
│  └─ NgheAn_weather32full_flood_merge_by_commune_time_20xx_NO_NaN.csv ...
│
└─ realtime_outputs/              # (app SOS)
   ├─ flood_point_probability_rt.csv
   ├─ flood_extent_mask_rt.csv
   ├─ meta_rt.json
   ├─ openmeteo_cache.json
   ├─ osm_graph.graphml
   ├─ shelters.csv
   └─ sos_signals.csv
