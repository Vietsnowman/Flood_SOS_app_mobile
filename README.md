# 🌊 FloodSOS — Hệ Thống Cứu Hộ Lũ Lụt Thông Minh Nghệ An

> Hệ thống cứu hộ lũ lụt toàn diện kết hợp **AI dự báo ngập lụt** (Python/Streamlit) + **Ứng dụng SOS Mobile** (Flutter + Node.js + MongoDB).

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flutter.dev)
[![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com)

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Tính Năng](#-tính-năng)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Cài Đặt & Khởi Chạy](#-cài-đặt--khởi-chạy)
- [API Endpoints](#-api-endpoints)
- [Xử Lý Lỗi](#-xử-lý-lỗi)

---

## 🎯 Giới Thiệu

**FloodSOS** là hệ thống cứu hộ toàn diện được thiết kế để hỗ trợ người dân vùng **Nghệ An** trong tình huống lũ lụt khẩn cấp. Hệ thống gồm **3 thành phần chính**:

1. **🐍 AI Prediction Dashboard** (Python/Streamlit) — Dự báo ngập 7 ngày, bản đồ realtime, điều phối nguồn lực
2. **📱 Mobile SOS App** (Flutter) — Ứng dụng gửi SOS, GPS, ghi âm, chatbot cho người dân
3. **🔧 SOS Backend** (Node.js/MongoDB) — API xử lý SOS, auth, chatbot

---

## ✨ Tính Năng

### 🐍 AI Prediction (Streamlit)
| Module | Chức năng |
|--------|-----------|
| **Realtime SOS + Shelters** | Bản đồ điểm ngập realtime, SOS routing, shelter, priority scoring |
| **Forecast + Điều phối** | Dự báo 7 ngày theo xã, quản lý sự cố, hotline, phân bổ nguồn lực |

### 📱 Mobile App (Flutter)
- ✅ Gửi SOS với GPS tự động + ghi âm giọng nói
- ✅ Chatbot hỗ trợ khẩn cấp (113, 114, 115)
- ✅ Bản đồ cứu hộ realtime với markers SOS
- ✅ Theo dõi thời tiết (OpenWeatherMap)
- ✅ Admin Dashboard — quản lý SOS trên bản đồ

### 🔗 Tích hợp 2 chiều (Bidirectional SOS Integration)
- 📌 **Từ Mobile tới Python**: Node.js tự động quét và tính toán (Haversine) các trạm cứu hộ gần nhất từ file dữ liệu hệ thống Python, đồng thời tự động chèn điểm SOS mới vào file CSV của Streamlit.
- 📌 **Từ Python tới Mobile**: Chỉ sau vài giây, điểm SOS xuất hiện trên màn hình Live Map Streamlit tĩnh, đồng thời người gọi SOS trên điện thoại nhận ngay thông tin tên và khoảng cách đến Trạm cứu hộ gần họ nhất trên màn hình máy.

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────┐
│                    NGƯỜI DÙNG                            │
│  📱 Flutter App (SOS, GPS, Ghi âm, Chat, Bản đồ)       │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (port 3002) - Trả về Trạm cứu hộ gần nhất
┌──────────────────────▼──────────────────────────────────┐
│              NODE.JS BACKEND                             │
│  Express API: SOS, Auth, Chatbot   ──── MongoDB         │
│         │                                                │
│         │ Đọc file shelters.csv                          │
│         │ Ghi đè vào sos_signals.csv                     │
└─────────▼────────────────────────────────────────────────┘
          │ (Auto-sync File System)
┌─────────▼───────────────────────────────────────────────┐
│           PYTHON ML BACKEND (Streamlit)                  │
│  Realtime Dashboard (port 8501)                          │
│  ├── Dự báo ngập 7 ngày (ML models)                     │
│  ├── Bản đồ điểm ngập / SOS / Shelters                  │
│  ├── Điều phối nguồn lực (SQLite DB)                     │
│  └── Open-Meteo API → Feature Engineering → Predict      │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Dự Án

```
Gop_app/
│
├── Gop_app/                        # 🐍 Python ML Backend
│   ├── main_unified.py             # ✅ Entrypoint gộp 2 app Streamlit
│   ├── app_SOS_shelters7.py        # Realtime SOS + bản đồ (point-based)
│   ├── app6.py                     # Forecast 7 ngày + DB + điều phối (commune-based)
│   ├── predict.py                  # Engine dự báo 7 ngày
│   ├── realtime.py                 # Job tạo dữ liệu realtime cho bản đồ
│   ├── open_meteo.py               # Gọi API Open-Meteo
│   ├── feature_engineering.py      # Tạo features cho ML
│   ├── db_utils.py                 # SQLite DB (sự cố, điều phối, hotline)
│   ├── report_utils.py             # Xuất báo cáo CSV/PDF
│   ├── scheduler_cache.py          # Cache dự báo hằng ngày
│   ├── 0_merge_data.py             # Merge dữ liệu nhiều năm
│   ├── 1_train_models_no_leak.py   # Train ML models (no leak)
│   ├── models/                     # ML models (.pkl)
│   ├── cache/                      # merged.csv + forecast parquet
│   ├── data/                       # Dân số, shelters, weather data
│   ├── db/                         # SQLite database
│   └── realtime_outputs/           # Output cho app realtime
│
├── Floodsos/                       # 📱 Mobile App System
│   ├── Sos-backend/                # 🔧 Node.js Express API
│   │   ├── server.js               # SOS, Auth, Chatbot (port 3002)
│   │   ├── .env                    # Config (JWT, API keys)
│   │   └── package.json            # Dependencies
│   └── frontend-flutter/           # 📱 Flutter App
│       ├── lib/
│       │   ├── main.dart           # Entry point
│       │   ├── config/             # App config, theme
│       │   ├── screens/            # 15 màn hình UI
│       │   ├── services/           # API, Socket, Weather, GPS
│       │   ├── providers/          # State management
│       │   ├── models/             # Data models
│       │   └── widgets/            # Reusable widgets
│       └── pubspec.yaml            # Flutter dependencies
│
├── train_flood_model_final.py      # Train point-based model (dataset 2020)
└── 0_merge_data.py                 # Merge dữ liệu xã nhiều năm
```

---

## 💻 Yêu Cầu Hệ Thống

| Công cụ | Phiên bản | Mục đích | Link tải |
|---------|-----------|----------|----------|
| **Python** | ≥ 3.9 | ML Backend + Streamlit | [python.org](https://python.org) |
| **Node.js** | ≥ 16 | SOS Backend API | [nodejs.org](https://nodejs.org/) |
| **MongoDB** | ≥ 4.x | Lưu dữ liệu SOS | [mongodb.com](https://www.mongodb.com/try/download/community) |
| **Flutter SDK** | Latest stable | Mobile App | [flutter.dev](https://docs.flutter.dev/install/archive) |
| **Java JDK** | ≥ 17 | Build Android APK | [oracle.com/java](https://www.oracle.com/java/technologies/downloads/) |
| **CMake** | ≥ 3.28 | Build Windows Desktop | [cmake.org](https://cmake.org/download/) |

---

## 🚀 Cài Đặt & Khởi Chạy

> **Thứ tự theo `run.txt`** — Mở lần lượt từng terminal, chạy đúng thứ tự bước 1 → 6.

### Bước 0: Clone & Cài dependencies (lần đầu)

```bash
git clone https://github.com/Vietsnowman/Gop_app.git
```

**Python dependencies:**
```bash
pip install streamlit pandas numpy folium streamlit-folium pyproj scikit-learn joblib ^
  alphashape osmnx networkx scikit-image streamlit-autorefresh streamlit-geolocation ^
  reportlab pyarrow
```

**Node.js dependencies:**
```bash
cd Gop_app/Floodsos/Sos-backend
npm install && npm install sqlite3 --save
```

---

### 🖥️ Bước 1 — Merge dữ liệu (lần đầu hoặc có data mới)

```bash
# Terminal 1 — root project (Gop_app/)
cd Gop_app
python 0_merge_data.py
```
> Tạo ra `cache/merged.csv` từ các file CSV gốc.

---

### ⏱️ Bước 2 — Cache dự báo 7 ngày (nên chạy sớm, lặp mỗi 3h)

```bash
# Terminal 2 — Gop_app/Gop_app/
cd Gop_app/Gop_app
python scheduler_cache.py
```
> Tạo file `cache/forecasts/<date>_commune_forecasts_7d.parquet`. Nếu chưa có cache, app vẫn chạy nhưng chậm hơn.

---

### 🌊 Bước 3 — Realtime map (bản đồ điểm ngập & SOS)

```bash
# Terminal 3 — Gop_app/Gop_app/
cd Gop_app/Gop_app
python realtime.py
```
> Tạo `realtime_outputs/flood_point_probability_rt.csv` cho bản đồ realtime.

---

### 🖥️ Bước 4 — Chạy Streamlit Dashboard ⭐

```bash
# Terminal 1 (tiếp theo Bước 1) — Gop_app/Gop_app/
cd Gop_app/Gop_app
python -m streamlit run main_unified.py
```
> → Dashboard mở tại **http://localhost:8501**  
> Gồm 2 module: **Realtime SOS Map** (`app_SOS_shelters7.py`) + **Forecast + Điều phối** (`app6.py`)

---

### 🔧 Bước 5 — Chạy Node.js Backend (SOS API)

```bash
# Terminal 4 — Gop_app/Floodsos/Sos-backend/
cd Gop_app/Floodsos/Sos-backend
npm start
```
> → API chạy tại **http://localhost:3002**  
> ⚠️ Cần MongoDB đang chạy: `net start MongoDB` (Windows)

---

### 📱 Bước 6 — Chạy Flutter App (Mobile/Desktop)

```bash
# Terminal 5 — Gop_app/Floodsos/frontend-flutter/
cd Gop_app/Floodsos/frontend-flutter
flutter pub get          # lần đầu

# Chạy Desktop (Windows)
flutter build windows

# Hoặc chạy nhanh trên Chrome
flutter run -d chrome

# Hoặc build APK Android
flutter build apk --release
```

---

### ⚡ Tóm tắt nhanh (tất cả đã cài sẵn)

| Terminal | Lệnh | Mục đích |
|----------|------|----------|
| **T1** | `python 0_merge_data.py` → `python -m streamlit run main_unified.py` | Dashboard chính |
| **T2** | `python scheduler_cache.py` | Cache dự báo 7 ngày |
| **T3** | `python realtime.py` | Dữ liệu realtime |
| **T4** | `npm start` (Sos-backend/) | API SOS Node.js |
| **T5** | `flutter build windows` (frontend-flutter/) | App mobile/desktop |

---



## ⚙️ Ports & Services

| Service | Port | URL | Ghi chú |
|---------|------|-----|---------|
| Streamlit Dashboard | 8501 | http://localhost:8501 | AI Prediction + Điều phối |
| Node.js SOS API | 3002 | http://localhost:3002 | SOS, Auth, Chatbot |
| MongoDB | 27017 | mongodb://127.0.0.1:27017 | Database cho SOS |

---

## 📡 API Endpoints (Node.js Backend)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/auth/login` | Đăng nhập admin (`admin / admin123`) |
| `POST` | `/api/sos/voice` | Gửi SOS (text + GPS + audio) |
| `GET` | `/api/sos` | Lấy danh sách tất cả SOS |
| `DELETE` | `/api/sos/:id` | Xóa SOS |
| `PUT` | `/api/sos/:id/resolve` | Xác nhận đã cứu hộ → xóa SOS |
| `POST` | `/api/chat` | Chatbot (hỏi 113, 114, 115, SOS) |

### Ví dụ gửi SOS (Dùng FormData vì có file audio)

Vì API gửi SOS sử dụng `multer` để nhận file ghi âm, bạn cần gửi dữ liệu dưới dạng `multipart/form-data` thay vì JSON thuần túy.

Ví dụ dùng Fetch API (JavaScript):
```javascript
const formData = new FormData();
formData.append('name', 'Nguyễn Văn A');
formData.append('phone', '0123456789');
formData.append('lat', '18.6733');
formData.append('lon', '105.6924');
formData.append('message', 'Cần cứu hộ gấp, nước dâng cao');
formData.append('water_level', 'Cao');
formData.append('people_count', '5');
// formData.append('audio', fileBlob, 'sos.aac'); // Tuỳ chọn

fetch('http://localhost:3002/api/sos/voice', {
    method: 'POST',
    body: formData
}).then(res => res.json()).then(console.log);
```

---

## 🔧 Xử Lý Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `MongoDB connection failed` | MongoDB chưa chạy | `net start MongoDB` hoặc cài MongoDB |
| `streamlit not found` | Không trong PATH | Dùng `python -m streamlit run ...` |
| `flutter not found` | Chưa cài Flutter SDK | Tải từ flutter.dev, thêm vào PATH |
| `CMake not found` | Thiếu CMake cho Windows build | Cài CMake, restart terminal |
| `Không tìm thấy merged.csv` | Chưa chạy merge data | Chạy `python 0_merge_data.py` (cần file data) |
| `API connection error` | Backend không chạy | Đảm bảo Node.js đang chạy port 3002 |

---

## 📊 Pipeline Dữ Liệu

```
Data CSVs (2023-2025) → 0_merge_data.py → cache/merged.csv
                                             ↓
                              1_train_models_no_leak.py → models/*.pkl
                                             ↓
                          scheduler_cache.py → cache/forecasts/*.parquet
                                             ↓
                                    app6.py (Streamlit Dashboard)

train_flood_model_final.py → calib_model.joblib + feature_list.joblib
                                             ↓
                              realtime.py → realtime_outputs/*.csv
                                             ↓
                               app_SOS_shelters7.py (Realtime Map)
```

---

## 👥 Tác Giả

- **Vietsnowman** — [GitHub](https://github.com/Vietsnowman)
- **Khanh Vu** — [GitHub](https://github.com/KhanhKH069)

---

## 🤝 Đóng Góp

1. Fork repository
2. Tạo branch: `git checkout -b feature/NewFeature`
3. Commit: `git commit -m 'Add NewFeature'`
4. Push: `git push origin feature/NewFeature`
5. Mở Pull Request
