# 🌊 FloodSOS — Hệ Thống Cứu Hộ Lũ Lụt Thông Minh Nghệ An

> Hệ thống cứu hộ lũ lụt toàn diện kết hợp **AI dự báo ngập** (Python/Streamlit/FastAPI) + **Ứng dụng SOS Mobile** (Flutter + Node.js + MongoDB).

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logo=polars&logoColor=white)](https://pola.rs)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flutter.dev)
[![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com)

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Tính Năng AI Core](#-tính-năng-ai-core)
- [Hướng Dẫn Cài Đặt & Chạy](#-hướng-dẫn-cài-đặt--chạy)
- [Pipeline Dữ Liệu & Training](#-pipeline-dữ-liệu--training)
- [API Endpoints](#-api-endpoints)
- [Cấu Hình Dữ Liệu](#-cấu-hình-dữ-liệu)

---

## 🎯 Giới Thiệu

**FloodSOS** là hệ thống cứu hộ toàn diện hỗ trợ người dân vùng **Nghệ An** trong tình huống lũ lụt khẩn cấp. Khi người dân bấm nút SOS trên điện thoại, hệ thống AI lập tức phân tích vị trí, đánh giá mức ngập, và trả về **chỉ dẫn cứu hộ cụ thể** trong vài giây.

### 3 thành phần chính:
| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| **🐍 AI Core** | Python 3.11.9 + FastAPI + Streamlit | Dự báo ngập, định tuyến cứu hộ, điều phối |
| **📱 Mobile App** | Flutter (Calm Crisis UI) | Gửi SOS, hiển thị chỉ dẫn weather/routing |
| **🔧 SOS Backend** | Node.js + MongoDB | Nhận SOS, lưu trữ, proxy sang AI |

---

## 🏗️ Kiến Trúc Hệ Thống

```
Mobile App (Flutter)
    │  POST /api/sos/voice  →  Gửi SOS (vị trí + audio)
    │  POST /api/sos/route  →  Nhận chỉ dẫn tuyến đường
    ▼
Node.js Server (:3002)
    ├── MongoDB  →  Lưu SOS, archive lịch sử
    ├── /api/sos/voice  →  calls priority_api  ←── priority_api.py (:8765)
    └── /api/sos/route  →  calls routing_api   ←── routing_api.py  (:8766)
                                                        │
                                          ┌─────────────┴──────────────┐
                                          │    flood_prob tại vị trí   │
                                          │    (từ RT CSV hoặc model)  │
                                          │                            │
                                    Ngập thấp (<50%)         Ngập cao (≥50%)
                                    Self-Evacuation          Rescue Dispatch
                                    → route đến shelter      → route từ trạm
                                    → chỉ dẫn xe/đi bộ      → per-segment plan
                                                                (xuồng/xe tải)

Streamlit Dashboard
    ├── app6.py               →  Giám sát + điều phối + phân bổ nguồn lực
    └── app_SOS_shelters7.py  →  SOS map + routing + shelter management
```

---

## 📁 Cấu Trúc Dự Án

```
Flood_SOS_app_mobile/
│
├── Gop_app/                     # 🐍 AI Core (Python 3.11.9, uv)
│   ├── data/                    # Dữ liệu đầu vào
│   │   ├── DanSo_Xa.csv             # Dân số theo xã
│   │   ├── shelters.csv             # Điểm trú ẩn
│   │   ├── resources_default.json   # Mặc định nguồn lực cứu hộ
│   │   └── NgheAn_weather32full_flood_merge_*.csv  # Thời tiết 2023-2025
│   ├── dataset_mét_độ(2020).csv # Dataset pixel-level (57 MB, 2020)
│   │
│   ├── 0_merge_data.py          # [Polars] Gộp 3 CSV thời tiết → cache/merged.csv
│   ├── 1_train_models_no_leak.py # Train LightGBM commune-level (forecast 7 ngày)
│   ├── 2_build_base_points.py   # [Polars] Tạo base_points_static.csv từ dataset pixel
│   ├── train_flood_model_final.py # Train model pixel-level (realtime)
│   │
│   ├── realtime.py              # Chạy dự báo realtime → realtime_outputs/
│   ├── scheduler_cache.py       # Cập nhật cache dự báo 7 ngày hàng ngày
│   │
│   ├── priority_api.py          # FastAPI :8765 — Urgency Scoring (MLP)
│   ├── routing_api.py           # FastAPI :8766 — Flood-Aware SOS Routing (MỚI)
│   ├── predict.py               # Commune-level forecast engine
│   ├── predict_sos_priority.py  # MLP inference cho SOS urgency
│   ├── route_calculator_osm.py  # OSM routing wrapper (osmnx)
│   │
│   ├── app6.py                  # Streamlit: Dashboard điều phối + forecast
│   ├── app_SOS_shelters7.py     # Streamlit: SOS map + shelter routing
│   ├── db_utils.py              # SQLite helpers
│   └── pyproject.toml           # uv dependencies (Python 3.11.9)
│
├── FloodSOS-Complete/
│   ├── frontend-flutter/        # 📱 Flutter Mobile App (Calm Crisis UI)
│   │   ├── lib/
│   │   │   ├── config/theme_config.dart           # Ocean/Teal palette
│   │   │   ├── widgets/glass_widgets.dart         # Glassmorphism widgets
│   │   │   ├── screens/...                        # UI screens
│   │   │   ├── services/api_service.dart          # HTTP client (sử dụng Dio + dotenv)
│   │   │   └── providers/...                      # State management
│   │   ├── .env                                   # Cấu hình BACKEND_URL, API_KEY
│   │   └── pubspec.yaml
│   │
│   └── Sos-backend/             # 🔧 Node.js Backend (MVC Architecture)
│       ├── config/              # Kết nối Database
│       ├── controllers/         # Xử lý Logic (Auth, Chat, SOS)
│       ├── models/              # Schema MongoDB
│       ├── routes/              # Express Router API
│       ├── services/            # Logic nghiệp vụ gọi AI Core, xử lý CSV
│       ├── tests/               # 🆕 Jest Unit test files (100% coverage routes)
│       ├── server.js            # Entry point thu gọn
│       └── .env                 # Biến môi trường hệ thống (Cổng, Database, API Keys, Credentials)
│
├── docker-compose.yml           # 🐳 Khởi chạy toàn bộ hệ thống bằng 1 lệnh
└── README.md
```

---

## 🤖 Tính Năng AI Core

### 1. Dự báo ngập 7 ngày (Commune-level)
- Model: **LightGBM** (Group K-Fold, no data leakage)
- Đầu vào: thời tiết Open-Meteo + đặc trưng địa hình xã
- Đầu ra: `p_flood`, `flood_area_m2`, `flood_ratio`, `priority_score`

### 2. Dự báo ngập realtime (Pixel-level)
- Model: **Calibrated LightGBM** trên dataset 2020 (UTM grid)
- Đầu vào: đặc trưng tĩnh địa hình (Slope/TWI/DEM) + mưa rolling window
- Đầu ra: `flood_prob` tại từng điểm lưới → `realtime_outputs/flood_point_probability_rt.csv`

### 3. SOS Urgency Scoring — priority_api.py (port 8765)
- Model: **MLP** (PyTorch) → xác suất cần xử lý trong ≤15 phút
- Fallback graceful nếu model chưa train

### 4. 🆕 Flood-Aware SOS Routing — routing_api.py (port 8766)
**Khi SOS được gửi:**
- `flood_prob < 50%` → **Self-Evacuation**: Route người dân đến shelter gần nhất
- `flood_prob ≥ 50%` → **Rescue Dispatch**: Route đội cứu hộ từ trạm → điểm SOS, phân tích từng chặng:

| Mức ngập chặng | Phương án |
|---|---|
| 🔴 `heavy` (≥65%) | Xuồng máy / áo phao bắt buộc |
| 🟠 `moderate` (40–65%) | Xe tải cao hoặc thuyền nhỏ |
| 🟡 `low` (20–40%) | Xe bán tải, đi chậm cẩn thận |
| 🟢 `none` (<20%) | Xe ô tô thông thường |

---

### 5. 🌊 Giao Diện Calm Crisis (Mobile)
Mobile app được thiết kế theo phong cách **Calm Crisis**:
- **Glassmorphism Design:** Các thẻ thông tin, form đăng nhập, box chat đều tựa như kính mờ trên nền đại dương sâu.
- **Ocean & Teal Palette:** Sử dụng gradient biển sâu (`#0D1B2A` → `#1F3A4B`) và accent màu lục lam (`#00BCD4`) tạo cảm giác bình tĩnh nhưng dứt khoát.
- **Interactive SOS Button:** Nút bấm SOS dạng xung (pulse animation) mô phỏng sóng lan tỏa, giúp người dùng dễ dàng thu hút sự chú ý.
- **Bottom Navigation Bar:** Hỗ trợ điều hướng đa năng (SOS / Bản đồ Cứu hộ / Hotline AI / Cảnh báo Thời tiết).

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### Yêu cầu
- **Python 3.11.9** (qua [pyenv](https://github.com/pyenv/pyenv) hoặc tải trực tiếp)
- **uv** — `pip install uv`
- **Node.js** ≥18 + npm
- **Flutter** ≥3.19 (stable)
- **MongoDB** đang chạy local (:27017)

---

### Bước 1 — Cài môi trường Python (Gop_app)

```bash
cd Gop_app
uv sync          # cài tất cả dependency vào .venv (Python 3.11.9)
```

### Bước 2 — Cài Node.js backend

```bash
cd FloodSOS-Complete/Sos-backend
npm install
```

###**Cấu hình biến môi trường:**
Tạo file `.env` (copy từ mẫu có sẵn) và điền:
```env
PORT=3002
MONGO_URI=mongodb://127.0.0.1:27017/floodsos
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
JWT_SECRET=your-secret
OPENWEATHER_KEY=your-api-key
PRIORITY_API_URL=http://127.0.0.1:8765/predict
ROUTING_API_URL=http://127.0.0.1:8766/route
```

### 3. Cài Đặt Frontend

```bash
cd ../frontend-flutter
flutter pub get
```

**Cấu hình Flutter:**
Tạo file `.env` tại thư mục root của Flutter chứa:
```env
BACKEND_URL=http://127.0.0.1:3002
OWM_API_KEY=your_openweather_key
```

### 4. Kiểm Thử Hệ Thống (Unit Tests)

**Backend Node.js (Jest & Supertest):**
```bash
cd sos-backend
npm test
```

**AI Core Python (Pytest & TestClient):**
```bash
cd Gop_app
uv run pytest tests/ -v
```

---

## 🎮 Khởi Chạy Ứng Dụng

Hệ thống được thiết kế để chạy tự động bằng Docker hoặc chạy độc lập từng service.

### Cách 1: Chạy Siêu Tốc Bằng Docker (Khuyên Dùng) 🐳
Đảm bảo bạn đã cài Docker Desktop & máy ảo Python đã train ra file models/ .joblib.
Từ thư mục root của dự án:
```bash
docker compose up -d --build
```
*Lệnh này sẽ tự động khởi tạo MongoDB, Node.js Backend, 2 Python FastAPI và 2 màn hình Streamlit.*

### Cách 2: Chạy Thủ Công Từng Dev Server

**Khởi động AI Core (Terminal 1 & 2):**
```bash
cd Gop_app
uv run uvicorn src.api.priority:app --port 8765 --reload
uv run uvicorn src.api.routing:app --port 8766 --reload
```

**Khởi động Node.js Backend Server (Terminal 3):**
```bash
cd FloodSOS-Complete/Sos-backend
npm run dev
```

**Kết quả mong đợi:**
```
🚀 SERVER ĐANG CHẠY TẠI: http://0.0.0.0:3002       
📡 API Gửi SOS: POST http://localhost:3002/api/sos/voice
✅ MongoDB Connected thành công!
```

---

## 📊 Pipeline Dữ Liệu & Training

### Pipeline Commune-level (dự báo 7 ngày)

```bash
cd Gop_app

# Bước 1: Gộp dữ liệu thời tiết 3 năm (dùng Polars — nhanh ~4x)
uv run python 0_merge_data.py
# → cache/merged.csv (280,320 rows × 98 cols)

# Bước 2: Train LightGBM (không data leakage, Group K-Fold)
uv run python 1_train_models_no_leak.py
# → models/model_prob_no_leak.pkl
# → models/model_area_no_leak.pkl
# → models/model_ratio_no_leak.pkl

# Bước 3: Cache dự báo 7 ngày (cập nhật hàng ngày)
uv run python scheduler_cache.py
# → cache/forecasts/YYYY-MM-DD_commune_forecasts.parquet
```

### Pipeline Pixel-level (dự báo realtime)

```bash
cd Gop_app

# Bước 0: Tạo base points từ dataset pixel (dùng Polars)
uv run python 2_build_base_points.py
# → base_points_static.csv (1,000 điểm lưới × 14 cột)

# Bước 1: Train model pixel-level (Calibrated LightGBM)
uv run python train_flood_model_final.py
# → calib_model.joblib
# → feature_list.joblib
# → model_meta.json

# Bước 2: Chạy dự báo realtime (lấy mưa từ Open-Meteo API)
uv run python realtime.py
# → realtime_outputs/flood_point_probability_rt.csv  ← routing_api đọc file này
# → realtime_outputs/flood_extent_mask_rt.csv
```

### Sơ đồ luồng dữ liệu

```
data/NgheAn_weather32full_*.csv (2023-2025)
    ↓ 0_merge_data.py (Polars lazy scan)
    → cache/merged.csv
    ↓ 1_train_models_no_leak.py
    → models/*.pkl                   ← predict.py (forecast)

dataset_mét_độ(2020).csv (57 MB)
    ↓ 2_build_base_points.py (Polars)
    → base_points_static.csv
    ↓ train_flood_model_final.py
    → calib_model.joblib + feature_list.joblib
    ↓ realtime.py (mỗi 30 phút)
    → realtime_outputs/flood_point_probability_rt.csv  ← routing_api.py

data/shelters.csv            ← routing_api.py (tìm shelter gần nhất)
data/resources_default.json  ← app6.py sidebar (mặc định nguồn lực)
data/DanSo_Xa.csv            ← scheduler_cache.py + app6.py (dân số)
```

---

## 🔌 API Endpoints

### Node.js Backend (port 3002)

| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/api/auth/login` | Đăng nhập admin |
| GET | `/api/sos` | Lấy danh sách SOS đang mở |
| POST | `/api/sos/voice` | Gửi SOS (multipart audio + location) |
| PUT | `/api/sos/:id/resolve` | Đánh dấu đã cứu |
| DELETE | `/api/sos/:id` | Xóa SOS |
| POST | **`/api/sos/route`** | 🆕 Phân tích tuyến đường flood-aware |
| POST | `/api/chat` | Chatbot hỗ trợ khẩn cấp |

### Priority API (port 8765)

| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/predict` | Urgency score → `{ urgency_prob, is_urgent }` |
| GET | `/health` | Trạng thái model |

### Routing API (port 8766)

| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/route` | Phân tích tuyến đường → `{ flood_level, mode, segments, route, ... }` |
| GET | `/health` | Trạng thái + kiểm tra file CSV/shelter |

**Ví dụ request `/route`:**
```json
{ "lat": 19.34, "lon": 105.71 }
```

**Response flood thấp:**
```json
{
  "flood_level": "low",
  "flood_prob": 0.28,
  "mode": "self_evacuation",
  "shelter": { "name": "UBND xã X", "lat": 19.35, "lon": 105.72, "distance_km": 1.2 },
  "segments": [
    { "index": 0, "flood_level": "low", "plan": "🟡 Xe máy, đi chậm", "distance_km": 1.2 }
  ],
  "route": [[19.34, 105.71], [19.345, 105.715], ...]
}
```

**Response flood cao:**
```json
{
  "flood_level": "high",
  "flood_prob": 0.82,
  "mode": "rescue_dispatch",
  "rescue_base": { "lat": 19.30, "lon": 105.68, "source": "base_json" },
  "sos_target": [19.34, 105.71],
  "segments": [
    { "index": 0, "flood_level": "heavy", "plan": "🔴 Xuồng máy / áo phao bắt buộc" },
    { "index": 1, "flood_level": "low",   "plan": "🟡 Xe bán tải, đi chậm" }
  ],
  "summary": "Tuyến cứu hộ gồm 2 chặng: 1 ngập nặng (xuồng), 1 ngập nhẹ (xe bán tải)."
}
```

---

## ⚙️ Cấu Hình Dữ Liệu

### Trạm cứu hộ (rescue base)
Tạo file `Gop_app/realtime_outputs/rescue_base.json`:
```json
{ "lat": 19.25, "lon": 105.65 }
```
Nếu không có, routing_api fallback dùng shelter đầu tiên trong `data/shelters.csv`.

### Ngưỡng nguồn lực mặc định
Chỉnh file `Gop_app/data/resources_default.json`:
```json
{
  "teams": 10, "boats": 15, "trucks": 8,
  "food_packs": 5000, "water_liters": 10000, "medical_kits": 600
}
```
Dashboard Streamlit tự đọc file này khi khởi động.

### Điểm trú ẩn
File `Gop_app/data/shelters.csv` cần có tối thiểu các cột:
```
name, lat, lon, capacity, commune_gid3
```

---

## 🛠️ Công Cụ & Môi Trường

| Tool | Phiên bản | Mục đích |
|---|---|---|
| Python | 3.11.9 (strict) | AI Core |
| uv | latest | Package manager (thay pip) |
| Polars | 1.39.x | Fast CSV I/O cho file lớn |
| Pandas | 2.2.x | Streamlit + sklearn integration |
| LightGBM | 4.4+ | Mô hình ML chính |
| FastAPI + uvicorn | 0.111+ | AI microservice APIs |
| osmnx + networkx | 1.9+ | Routing thực tế theo đường bộ |
| Streamlit | 1.35+ | Dashboard điều phối |
| Flutter | 3.19+ | Mobile App |
| Node.js | 18+ | Backend |
| MongoDB | 7+ | Lưu trữ SOS |

---

## 🔍 Kiểm Tra Nhanh

```bash
# Kiểm tra lint toàn bộ Python
cd Flood_SOS_app_mobile
uv run ruff check .

# Kiểm tra routing API
curl -X POST http://localhost:8766/route \
  -H "Content-Type: application/json" \
  -d '{"lat": 19.34, "lon": 105.71}'

# Kiểm tra priority API  
curl http://localhost:8765/health

# Kiểm tra Node.js backend
curl http://localhost:3002/api/sos
```
