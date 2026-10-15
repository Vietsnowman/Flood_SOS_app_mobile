# FloodSOS: Technical Analysis for International Scientific Research Paper

> **Hệ thống Cứu hộ Lũ lụt Thông minh tỉnh Nghệ An**  
> A Multi-layer AI-powered Flood Emergency Response System

---

## ABSTRACT (Tổng quan nghiên cứu)

**FloodSOS** là một hệ thống cứu hộ lũ lụt đa tầng tích hợp học máy (Machine Learning), định tuyến theo ngữ cảnh ngập lụt (Flood-Aware Routing), và ứng dụng di động chịu lỗi ngoại tuyến (Offline-Resilient Mobile App) phục vụ tỉnh Nghệ An, Việt Nam. Hệ thống kết hợp dữ liệu khí tượng từ Open-Meteo API, dữ liệu địa hình không gian (DEM, TWI, Slope, Flow Accumulation), và hành vi xã hội khẩn cấp để đưa ra quyết định cứu hộ theo thời gian thực. Hệ thống được triển khai theo kiến trúc microservices container hóa (Docker) với 6 container độc lập.

---

## I. SYSTEM ARCHITECTURE (Kiến trúc hệ thống)

### 1.1 Kiến trúc tổng thể (4-layer Stack)

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4 — Mobile Client (Flutter/Dart v3.0+)               │
│  • Calm Crisis UI    • Offline A* Routing                   │
│  • P2P Mesh UDP      • Hybrid Online/Offline Tracking       │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST/WebSocket
┌───────────────────────▼─────────────────────────────────────┐
│  LAYER 3 — API Gateway (Node.js v18+ / Express 4.x)         │
│  • REST API          • Socket.IO v4.8.3 (Real-time)         │
│  • JWT Auth          • MVC Architecture                     │
│  • MongoDB v7+       • CSV Backup Service                   │
└────────┬──────────────────────────────┬─────────────────────┘
         │ HTTP/8765                     │ HTTP/8766
┌────────▼────────────┐  ┌──────────────▼──────────────────┐
│  AI Priority API    │  │  AI Routing API                 │
│  (FastAPI/uvicorn)  │  │  (FastAPI/uvicorn)              │
│  Port: 8765         │  │  Port: 8766                     │
│  • MLP Urgency Score│  │  • Flood-Aware Routing          │
│  • Flood Zone Query │  │  • OSM Graph Navigation         │
└────────┬────────────┘  └──────────────┬──────────────────┘
         │                              │
┌────────▼──────────────────────────────▼──────────────────┐
│  LAYER 1 — AI Core (Python 3.11.9)                        │
│  • LightGBM Flood Forecasting  • Streamlit Dashboards     │
│  • Open-Meteo API Integration  • Realtime Pipeline        │
└───────────────────────────────────────────────────────────┘
```

### 1.2 Docker Microservices (6 Containers)

| Container | Port | Image | Vai trò |
|-----------|------|-------|---------|
| `floodsos-mongodb` | 27017 | mongo:7 | NoSQL database (SOS alerts, user data) |
| `floodsos-backend` | 3002 | node:18-alpine | API Gateway + Socket.IO |
| `floodsos-priority-api` | 8765 | Dockerfile.api (~500MB) | MLP Urgency Scoring |
| `floodsos-routing-api` | 8766 | Dockerfile.api (~500MB) | Flood-Aware Routing |
| `floodsos-streamlit-dash` | 8501 | Dockerfile (~4GB) | Management Dashboard |
| `floodsos-streamlit-sos` | 8502 | Dockerfile (~4GB) | SOS & Shelter Map |

**Internal Network:** `floodsos-net` (Docker bridge network)

---

## II. MACHINE LEARNING ALGORITHMS (Các thuật toán ML)

### 2.1 Mô hình Dự báo Ngập Cấp Xã — LightGBM (Commune-Level Flood Forecast)

**Mục đích:** Dự báo xác suất ngập lụt 7 ngày tới theo đơn vị hành chính cấp xã (commune-level).

#### Thuật toán: Light Gradient Boosted Machine (LightGBM)

| Loại mô hình | Tên | Đối tượng dự báo |
|---|---|---|
| `LGBMClassifier` | `model_prob_no_leak.pkl` | `p_flood` — Xác suất ngập [0,1] |
| `LGBMRegressor` | `model_area_no_leak.pkl` | `flood_area_m2` — Diện tích ngập (m²) |
| `LGBMRegressor` | `model_ratio_no_leak.pkl` | `flood_ratio` — Tỷ lệ % ngập xã |

#### Hyperparameters (Commune-level model)

```python
LGBMClassifier(
    n_estimators   = 1200,        # Số cây quyết định
    learning_rate  = 0.03,        # Tốc độ học
    num_leaves     = 64,          # Số lá tối đa mỗi cây
    subsample      = 0.8,         # Tỷ lệ sample dữ liệu mỗi vòng
    colsample_bytree = 0.8,       # Tỷ lệ features mỗi cây
    random_state   = 42
)

LGBMRegressor(
    n_estimators   = 2000,        # Số cây (nhiều hơn classification)
    learning_rate  = 0.03,
    num_leaves     = 64,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    random_state   = 42
)
```

#### Cross-Validation Strategy: Group K-Fold (Chống data leakage)

```python
GroupKFold(n_splits=5)
# groups = df["GID_3"].values  ← Nhóm theo mã xã (commune ID)
```

> [!IMPORTANT]
> **Anti-Leakage Design:** Sử dụng `GroupKFold` với nhóm theo `GID_3` (Mã xã GADM Level 3) để đảm bảo không một xã nào xuất hiện đồng thời trong tập train và validation. Đây là chuẩn phòng ngừa data leakage cho dữ liệu không gian-thời gian.

#### Target Engineering (Log-transform)

```python
y_flood = (flood_ratio > 0.001).astype(int)    # Binary: ngập/không ngập
y_area  = np.log1p(flood_area_m2)               # Log-transform diện tích
y_ratio = np.log1p(flood_ratio)                 # Log-transform tỷ lệ
# Inverse: np.expm1(prediction)
```

#### Feature Engineering — Commune-level

| Nhóm Feature | Các biến | Ý nghĩa |
|---|---|---|
| Lag Features | `tp_lag{1,2,3,7}` | Lượng mưa trễ 1-7 ngày trước |
| Rolling Sum | `tp_sum_{3,7,14}d` | Tổng mưa tích lũy 3/7/14 ngày |
| Rolling Max | `tp_max_{3,7,14}d` | Mưa cao nhất trong 3/7/14 ngày |
| Rain Days | `tp_rain_days_{3,7,14}d` | Số ngày mưa trong cửa sổ |
| Hourly Peak | `tp_max1h_max_{w}d` | Mưa giờ cao nhất trong cửa sổ |
| Rain Hours | `rain_hours_sum_{w}d` | Tổng số giờ mưa |
| Time Features | `month`, `dayofyear` | Chu kỳ mùa vụ |
| Static Geo | `GID_3`, commune polygon | Đặc trưng địa lý cố định |

---

### 2.2 Mô hình Dự báo Ngập Pixel-level — Calibrated LightGBM

**Mục đích:** Dự báo xác suất ngập theo lưới không gian (UTM grid) với độ phân giải cao, dùng cho định tuyến thời gian thực.

#### Dataset

```
Tên file   : dataset_mét_độ(2020).csv
Kích thước : ~57 MB
Độ phân giải: UTM Grid cells — cell_area ≈ 5.7 m²
Hệ tọa độ : EPSG:32648 (WGS 84 / UTM Zone 48N)
```

#### Hyperparameters (Pixel-level model)

```python
lgb.LGBMClassifier(
    n_estimators     = 800,
    learning_rate    = 0.03,
    num_leaves       = 31,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    scale_pos_weight = (1 - pos_rate) / (pos_rate + 1e-9),  # Xử lý class imbalance
    min_data_in_leaf = 300,           # Chống overfitting trên dataset lớn
    reg_lambda       = 5.0,           # L2 regularization
    reg_alpha        = 0.0            # L1 regularization
)
```

#### Probability Calibration — Isotonic Regression

```python
CalibratedClassifierCV(
    estimator = lgb_model,
    method    = "isotonic",   # Non-parametric monotone calibration
    cv        = 3             # 3-fold cross-validation
)
```

> [!TIP]
> **Isotonic Calibration** được dùng để điều chỉnh xác suất đầu ra của LightGBM sao cho `P(flood=1 | score=s) ≈ s`, cải thiện độ tin cậy trong quyết định định tuyến (routing threshold = 0.5 và 0.65).

#### Feature List — Pixel-level

| Nhóm | Feature | Nguồn |
|---|---|---|
| **Địa hình (Terrain)** | `Slope_UTM` — Độ dốc | DEM processing |
| | `TWI_UTM` — Topographic Wetness Index | Flow accumulation |
| | `DEMC_UTM` — Digital Elevation Model (corrected) | GIS |
| | `Flowlog_UT` — Log flow accumulation | Hydrology |
| | `RiverDist_` — Khoảng cách đến sông | GIS |
| | `RiverDC_UTM` — River density coefficient | GIS |
| | `FlowAccum` — Tích lũy dòng chảy | Hydrology |
| **Lượng mưa rolling** | `rain_{1,3,6,12,24}h_sum` | Open-Meteo |
| | `rain_{1,3,6,12,24}h_max` | Open-Meteo |
| **Lượng mưa lag** | `rain_lag_{1,3,6}h` | Open-Meteo |
| **Phơi lộ (Exposure)** | `population_density` | Population census |
| | `road_density` | OSM |
| **Thời gian** | `hour`, `day`, `month` | Temporal |

#### Train/Test Split Strategy — Spatial Block + Time Purge

```python
BLOCK_MULTIPLIER  = 30          # Kích thước block = 30 × cell_size
TEST_BLOCK_RATIO  = 0.30        # 30% blocks → test set
LEAD_HOURS        = 3           # Forecast horizon: 3 giờ tới
PURGE_HOURS       = LEAD_HOURS  # Purge gap giữa train/test

# Spatial blocks: df["block_id"] = floor(X/BLOCK_SIZE) + "_" + floor(Y/BLOCK_SIZE)
# Time cut: 70% timestamps → train, top 30% với purge gap → test
target = df.groupby(["X","Y"])["Is_Flood"].shift(-LEAD_HOURS)  # Target dịch 3h
```

#### Flood Extent Smoothing — Morphological Operations

```python
from scipy.ndimage import binary_opening, binary_closing

# Xử lý spatial noise trong mặt nạ ngập:
grid2 = binary_opening(grid)    # Loại bỏ các pixel đơn lẻ (salt noise)
grid3 = binary_closing(grid2)   # Lấp các lỗ nhỏ trong vùng ngập (pepper noise)
```

#### Evaluation Metrics

| Metric | Công thức | Mục đích |
|---|---|---|
| **ROC-AUC** | Area under ROC curve | Phân biệt ngập/không ngập |
| **PR-AUC** | Average Precision Score | Hiệu quả trên dữ liệu mất cân bằng |
| **Best Threshold (F1)** | `argmax(2*P*R/(P+R+ε))` | Tối ưu ngưỡng phân loại |

```python
prec, rec, thr = precision_recall_curve(y_test, pred_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_threshold = thr[np.argmax(f1)]
```

---

### 2.3 Mô hình Phân loại Độ Khẩn Cấp SOS — PyTorch MLP

**Mục đích:** Phân loại tín hiệu SOS nhận được có cần xử lý trong ≤15 phút hay không (binary classification: `urgent_15m`).

#### Kiến trúc Multilayer Perceptron (MLP)

```
Input (8 features)
    ↓
Linear(8 → 32) + ReLU
    ↓
Dropout(p=0.30)       ← Regularization chống overfitting
    ↓
Linear(32 → 16) + ReLU
    ↓
Linear(16 → 1) + Sigmoid
    ↓
Output: urgency_prob ∈ [0.0, 1.0]
```

#### Training Configuration — MLP

```python
# Model
SOSPriorityNet(n_features=8)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss Function
criterion = nn.BCELoss()          # Binary Cross-Entropy

# Scheduler — ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

# Training Settings
EPOCHS     = 150
BATCH_SIZE = 32
EARLY_STOP = 15    # Dừng nếu không cải thiện val_loss sau 15 epochs
TEST_SIZE  = 0.20  # Stratified 80/20 split

# Normalization
StandardScaler()   # Z-score normalization cho tất cả input features
```

#### Input Features — MLP (8 biến)

| Feature | Kiểu | Mô tả |
|---|---|---|
| `lat` | float | Vĩ độ điểm SOS |
| `lon` | float | Kinh độ điểm SOS |
| `flood_prob_near` | float [0,1] | Xác suất ngập tại điểm gần nhất |
| `people_count` | float | Số người cần cứu hộ |
| `status_enc` | int {0,1,2} | Trạng thái SOS (open=0, in_progress=1, closed=2) |
| `hour_of_day` | int [0,23] | Giờ trong ngày |
| `is_night` | binary | 1 nếu 22:00–05:59 |
| `is_weekend` | binary | 1 nếu Thứ 7/Chủ nhật |

#### Graceful Fallback (khi model chưa train)

```python
# Heuristic fallback thay thế Neural Network:
urgency_prob = min(1.0, flood_prob_near * 0.7 + min(people_count, 10) / 20.0)
```

---

## III. ROUTING ALGORITHMS (Thuật toán Định tuyến)

### 3.1 Flood-Aware SOS Routing (routing_api.py — Port 8766)

**Thuật toán cốt lõi:** Kết hợp Graph Shortest Path (Dijkstra qua NetworkX) + Flood Probability Lookup + Rule-based Segment Classification.

#### Logic Điều phối Hai Chế Độ

```
Nhận SOS (lat, lon)
    ↓
flood_prob = nearest-point lookup trong flood_point_probability_rt.csv
    ↓
if flood_prob < 0.50:                   # NGƯỠNG PHÂN LOẠI
    Mode: SELF-EVACUATION
    → Tìm shelter gần nhất (Haversine)
    → Tính đường bộ OSM → shelter
    → Chia chặng 0.8 km → gán cấp độ ngập từng chặng
else:
    Mode: RESCUE-DISPATCH
    → Lấy vị trí trạm cứu hộ (rescue_base.json)
    → Tính đường bộ OSM: trạm → điểm SOS
    → Chia chặng 0.8 km → gán phương tiện từng chặng
```

#### Phân cấp Ngập Lụt (4 Level System)

| Level | flood_prob | Phương tiện bắt buộc |
|---|---|---|
| `heavy` | ≥ 0.65 | Xuồng máy + áo phao bắt buộc |
| `moderate` | 0.40 – 0.65 | Xe tải cao hoặc thuyền nhỏ |
| `low` | 0.20 – 0.40 | Xe máy/xe bán tải, đi chậm |
| `none` | < 0.20 | Xe ô tô/xe máy bình thường |

#### Haversine Distance Formula

```python
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))
```

#### Nearest Shelter Lookup

```python
# Tìm shelter gần nhất bằng Haversine distance
df["_dist"] = df.apply(lambda r: _haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
nearest = df.loc[df["_dist"].idxmin()]
```

#### Nearest Flood Point Lookup

```python
# Nearest-neighbor trong flood probability grid (Euclidean approx)
rt_df["_dist"] = ((rt_df["lat"] - lat)**2 + (rt_df["lon"] - lon)**2)**0.5
nearest = rt_df.loc[rt_df["_dist"].idxmin()]
# Chỉ tin khi điểm gần trong vòng ~11 km (0.1 độ decimal)
if nearest["_dist"] > 0.1: return 0.3  # neutral fallback
```

### 3.2 OpenStreetMap Graph Routing (route_calculator_osm.py)

**Thuật toán:** Dijkstra Shortest Path (NetworkX) trên đồ thị đường bộ OSM.

```python
import osmnx as ox
import networkx as nx

# Tải graph đường bộ (cache trong memory)
G = ox.graph_from_point(
    (center_lat, center_lon),
    dist        = radius_m,       # 2000–8000m tùy khoảng cách
    network_type= 'drive',        # Chỉ đường ô tô
    retain_all  = False
)

# Tìm đường ngắn nhất bằng Dijkstra (weight = length in meters)
route_nodes = nx.shortest_path(G, orig_node, dest_node, weight='length')
```

#### Graph Parameters

```python
radius_m = max(2000, min(8000, int(dist_approx_m * 0.6)))  # Adaptive radius
center   = (lat1 + lat2) / 2, (lon1 + lon2) / 2           # Midpoint center
cache_key = (round(lat, 3), round(lon, 3), radius_m)       # In-memory cache
```

#### Fallback: Linear Interpolation

```python
# Khi OSM không available (offline / vùng thiếu dữ liệu):
n = max(10, int(haversine_km / SEGMENT_LEN_KM * 3))
waypoints = [(lat1 + (lat2-lat1)*i/n, lon1 + (lon2-lon1)*i/n) for i in range(n+1)]
```

### 3.3 Offline A* Routing (Flutter Mobile — On-device)

**Vị trí:** Flutter app (`offline_routing` service)

```
Thuật toán: A* (A-Star) Search
Dữ liệu:   Bản đồ và tham chiếu đường bộ lưu trong assets/data/
Kích hoạt: Khi không có kết nối Internet hoặc Server không phản hồi
Mục đích:  Chỉ đường đến nơi an toàn hoàn toàn trên thiết bị
```

---

## IV. FLOOD ZONE SCORING — TWI-Based Heuristic

**Vị trí:** `priority_api.py` — `/flood-zones` endpoint  
**Phạm vi:** 1,000 điểm lưới phủ Nghệ An (`base_points_static.csv`)

#### Công thức điểm ngập (Flood Probability Formula)

```
p_flood = TWI_UTM/15 × 0.5  
        + (1 - elevation_norm/20) × 0.3  
        + (1 - river_dist_norm/500) × 0.2
```

| Thành phần | Trọng số | Biến | Ý nghĩa |
|---|---|---|---|
| TWI Component | 0.50 | `TWI_UTM` | Chỉ số độ ẩm địa hình |
| Elevation Component | 0.30 | `DEMC_UTM` | Cao trình (DEM corrected) |
| River Proximity | 0.20 | `RiverDist_` | Khoảng cách đến sông |

#### Risk Level Classification

```
p_flood ≥ 0.70 → CRITICAL  (radius 600m trên bản đồ mobile)
p_flood ≥ 0.40 → HIGH      (radius 450m)
p_flood ≥ 0.20 → MEDIUM    (radius 350m)
p_flood < 0.20 → LOW       (radius 280m)
```

#### Impact Score Formula (Event-Level)

```python
# Đánh giá tác động sự kiện ngập:
impact_score = 100 * (0.35 * pop_exposed_pct    # 35% dân số bị ảnh hưởng
                    + 0.25 * road_exposed_pct    # 25% đường bộ bị ngập
                    + 0.15 * flooded_area_pct)   # 15% diện tích ngập
```

| Priority Level | impact_score | Hành động |
|---|---|---|
| Low | < 25 | Theo dõi |
| Medium | 25 – 50 | Cảnh báo |
| High | 50 – 75 | Điều phối |
| Emergency | ≥ 75 | Khẩn cấp tối cao |

#### Priority Score Formula (7-day Forecast)

```python
priority_score = 100 × (0.5 × p_flood + 0.3 × ap_norm + 0.2 × ratio_norm)
# ap_norm  = normalize(affected_population)
# ratio_norm = normalize(flood_ratio)
```

---

## V. REAL-TIME DATA PIPELINE

### 5.1 Open-Meteo API Integration

```
API: https://api.open-meteo.com/v1/forecast
Parameters:
  - latitude, longitude: Tâm vùng Nghệ An (tính trung bình base_points)
  - hourly: "precipitation"   (mm/hour)
  - timezone: "Asia/Bangkok"
  - start_date → end_date: today + 2 days

Cache policy: 30 phút (CACHE_MAX_AGE_MIN = 30)
Update: Mỗi lần realtime.py chạy, kiểm tra tuổi cache
```

### 5.2 Realtime Prediction Pipeline

```
Open-Meteo API (Forecast 12h)
    ↓
build_rain_features():
    rain_{1,3,6,12,24}h_sum  ← Rolling sum
    rain_{1,3,6,12,24}h_max  ← Rolling max  
    rain_lag_{1,3,6}h        ← Lag features
    ↓
Broadcast to 1,000 base_points_static.csv
    (X,Y UTM → lat/lon via pyproj EPSG:32648→4326)
    ↓
calib_model.predict_proba(X[features])[:, 1]
    ↓
flood_point_probability_rt.csv   ← routing_api.py reads
flood_extent_mask_rt.csv
meta_rt.json

Forecast horizon: FORECAST_HOURS_AHEAD = 12 giờ
Update interval: mỗi 30 phút (scheduler_cache.py)
EPSG: 32648 → 4326 (UTM → WGS84 lat/lon)
```

### 5.3 7-Day Commune Forecast Pipeline

```
Open-Meteo Daily API (7 ngày)
    ↓
Historical data combination:
  - precipitation_sum, precipitation_hours, precipitation_max_1h
  - Lag: 1,2,3,7 days
  - Rolling: 3,7,14 days
    ↓
predict_7days():
  → p_flood   = LGBMClassifier.predict_proba[:, 1]
  → flood_area = exp(LGBMRegressor.predict) - 1
  → flood_ratio = exp(LGBMRegressor.predict) - 1
    ↓
Dashboard Streamlit (app6.py, app_SOS_shelters7.py)
```

---

## VI. REAL-TIME COMMUNICATION ARCHITECTURE

### 6.1 Socket.IO (Online Mode)

**Thư viện:** `socket.io v4.8.3` (Node.js) + `socket_io_client v3.0.0` (Flutter)

| Event | Hướng | Mô tả |
|---|---|---|
| `nearby_sos` | Server → Client | Broadcast SOS trong bán kính 500m cho tình nguyện viên |
| `volunteer_update` | Bidirectional | Cập nhật trạng thái tình nguyện viên cứu hộ |
| `sos_resolved` | Server → Client | Thông báo SOS đã được giải quyết |
| `tracking_update` | Client → Server | Real-time position tracking |
| `tracking_sync_offline` | Client → Server | Đồng bộ dữ liệu tracking offline |

### 6.2 P2P Mesh Network (Offline Mode — UDP Multicast)

**Vị trí:** Flutter `nearby_rescue_screen.dart`

```
Công nghệ: UDP Multicast (không cần Internet/WiFi router)
Kịch bản: Hạ tầng mạng 4G/WiFi bị sập do lũ lụt
Chức năng:
  - Phát vị trí SOS giữa các thiết bị lân cận
  - Kết nối tình nguyện viên P2P trong bán kính ngắn
  - Không cần kết nối server trung tâm
```

### 6.3 Hybrid Tracking (Admin Dashboard)

```
Admin tracking_screen.dart nhận tín hiệu từ đồng thời:
  ├── Online:  Socket.IO tracking_update  (qua Internet)
  └── Offline: UDP Mesh radar signals      (qua mạng cục bộ)
→ Kết hợp để có bức tranh toàn cảnh không bị gián đoạn
```

---

## VII. DATA CHARACTERISTICS (Đặc điểm dữ liệu)

### 7.1 Training Datasets

| Dataset | Kích thước | Mô tả |
|---|---|---|
| `NgheAn_weather32full_flood_merge_..._2023.csv` | ~340 MB | Thời tiết 32 biến + ngập theo xã 2023 |
| `NgheAn_weather32full_flood_merge_..._2024.csv` | ~340 MB | Thời tiết 32 biến + ngập theo xã 2024 |
| `NgheAn_weather32full_flood_merge_..._2025.csv` | ~406 MB | Thời tiết 32 biến + ngập theo xã 2025 |
| `dataset_mét_độ(2020).csv` | ~57 MB | Pixel-level flood data 2020 (UTM grid) |
| Merged cache | 280,320 rows × 98 cols | Sau gộp bằng Polars lazy scan |

### 7.2 Data Processing: Polars vs Pandas

```python
# Polars (dùng cho files lớn — 4x nhanh hơn Pandas):
import polars as pl
df = pl.scan_csv("cache/merged.csv").collect()  # Lazy evaluation

# Pandas (dùng cho Streamlit + sklearn integration):
import pandas as pd
df = pd.read_csv(MERGED_FILE)
```

### 7.3 Base Points Grid

```
File: base_points_static.csv
Số điểm: 1,000 điểm lưới
Phạm vi: lat [18.77, 18.80], lon [105.58, 105.61] (vùng Nghệ An)
Cột: X, Y (UTM EPSG:32648), lat, lon, TWI_UTM, DEMC_UTM,
     RiverDist_, Slope_UTM, population_density, road_density
Cell area: ~5.7 m²
```

### 7.4 Shelter Database

```
File: data/shelters.csv
Cột bắt buộc: name, lat, lon, capacity, commune_gid3
Dùng cho: Nearest shelter lookup trong routing_api.py
```

### 7.5 Population Data

```
File: data/DanSo_Xa.csv
Nội dung: Dân số theo từng xã (commune-level)
Dùng cho: affected_population estimate, scheduler_cache.py, app6.py
```

---

## VIII. MOBILE APPLICATION TECHNOLOGY STACK

### 8.1 Flutter App (v3.0+ / Dart SDK ≥3.0.0)

#### Core Dependencies

| Package | Version | Vai trò |
|---|---|---|
| `dio` | ^5.4.1 | HTTP client thống nhất (thay http) |
| `socket_io_client` | ^3.0.0 | Real-time WebSocket |
| `flutter_map` | ^7.0.2 | OSM-based map widget |
| `geolocator` | ^13.0.2 | GPS geolocation |
| `latlong2` | ^0.9.1 | Geo coordinate operations |
| `provider` | ^6.1.2 | State management |
| `sqflite` | ^2.3.2 | Local SQLite (offline data) |
| `shared_preferences` | ^2.5.5 | Key-value offline storage |
| `firebase_core` | ^3.1.0 | Firebase integration |
| `firebase_messaging` | ^15.0.1 | Push notifications |
| `cloud_firestore` | ^5.0.1 | Firestore real-time DB |
| `fl_chart` | ^1.2.0 | Biểu đồ phân tích 24h |
| `record` | ^5.2.0 | Voice SOS recording |
| `audioplayers` | ^6.0.0 | Audio playback |
| `flutter_tts` | ^4.2.5 | Text-to-speech alerts |
| `lottie` | ^3.1.0 | Animated UI elements |
| `shimmer` | ^3.0.0 | Loading skeleton UI |
| `battery_plus` | ^7.0.0 | Battery monitoring |
| `device_info_plus` | ^12.3.0 | Device identification |
| `vibration` | ^3.1.8 | Haptic alerts |
| `flutter_dotenv` | ^5.1.0 | Environment configuration |

#### Screens (20 màn hình)

| Screen | Chức năng |
|---|---|
| `home_screen.dart` | Trang chủ: minimap, chỉ số ngập, nút SOS nhanh |
| `map_screen.dart` | Bản đồ tổng hợp flood zones + CircleLayer |
| `voice_sos_screen.dart` | SOS bằng giọng nói (Record package) |
| `panic_sos_screen.dart` | Nút SOS khẩn cấp một chạm |
| `sos_routing_result_screen.dart` | Hiển thị polyline tuyến đường cứu hộ |
| `alerts_list_screen.dart` | Danh sách SOS đang mở |
| `nearby_rescue_screen.dart` | P2P volunteer rescue (UDP Mesh) |
| `weather_screen.dart` | Dự báo thời tiết 24h (fl_chart) |
| `chat_screen.dart` | Chatbot hỗ trợ khẩn cấp |
| `admin_dashboard_screen.dart` | Dashboard giám sát toàn cảnh |
| `admin_map_screen.dart` | Bản đồ quản trị |
| `admin_tracking_screen.dart` | Hybrid Online+Offline tracking |
| `admin_users_screen.dart` | Quản lý người dùng |
| `admin_settings_screen.dart` | Cấu hình hệ thống |

#### GPS Strategy (Fault-tolerant Location)

```dart
// Windows/Desktop: Skip GPS → Dùng tọa độ Nghệ An preset
if (Platform.isWindows) { usePresetLocation(); }

// Mobile GPS ngoài bbox Nghệ An → fallback
bool inNgheAnBbox = lat >= 18.5 && lat <= 20.0 
                 && lon >= 104.0 && lon <= 106.5;
if (!inNgheAnBbox) { useRandomNgheAnLocation(); }

// 7 tọa độ Nghệ An preset (nghe_an_locations.dart)
// → Random 1 điểm khi khởi động
```

### 8.2 UI Design System — "Calm Crisis"

```
Theme: "Calm Crisis" — Tạo cảm giác bình tĩnh trong khẩn cấp
Primary Gradient: #0D1B2A → #1F3A4B  (Ocean deep)
Accent Color:     #00BCD4             (Teal/Cyan)
Style: Glassmorphism (backdrop-blur cards)
Animation: Lottie animated alerts, Shimmer loading
Typography: Custom sans-serif
```

### 8.3 Offline-First Architecture

```
Offline capabilities:
  1. Offline A* Routing   → assets/data/ (bundled map data)
  2. Cached Weather       → SharedPreferences (last fetch)
  3. Local SQLite         → sqflite (SOS history cache)
  4. P2P Mesh Network    → UDP Multicast (no Internet)
  5. Offline Chatbot     → OfflineChatbotService (local FAQ)
```

---

## IX. BACKEND TECHNOLOGY STACK

### 9.1 Node.js Backend (Port 3002)

```
Framework: Express.js v4.22.1
Pattern:   MVC Architecture
Auth:      JWT (jsonwebtoken v9.0.3) + bcryptjs v2.4.3
Database:  MongoDB v7+ (mongoose v9.1.2)
Real-time: Socket.IO v4.8.4
File Upload: Multer v2.0.2 (voice SOS audio)
Testing:   Jest v29.7.0 + Supertest v7.2.2
```

#### REST API Endpoints

| Method | Endpoint | Chức năng |
|---|---|---|
| POST | `/api/auth/login` | Đăng nhập (JWT) |
| GET | `/api/sos` | Lấy danh sách SOS |
| POST | `/api/sos/voice` | Gửi SOS (audio + location) |
| POST | `/api/sos/route` | Yêu cầu định tuyến flood-aware |
| PUT | `/api/sos/:id/resolve` | Đánh dấu đã cứu |
| DELETE | `/api/sos/:id` | Xóa SOS |
| POST | `/api/sos/:id/volunteer/accept` | Tình nguyện viên nhận SOS |
| GET | `/api/flood-zones` | Proxy → Python priority-api |
| POST | `/api/chat` | Chatbot FAQ |

#### MongoDB Schemas

```javascript
// SOSAlert:
{ lat, lon, description, urgency_prob, flood_prob_near,
  people_count, status, audio_url, created_at }

// SOSHistory: Archive của resolved alerts
// User: { username, password_hash, role, location }
```

#### Services Layer

```
aiService.js    → Gọi priority-api (urgency) + routing-api (route)
floodService.js → Tra cứu flood_prob từ base_points_static.csv
csvService.js   → Backup SOS data ra CSV file
```

### 9.2 AI API Services (Python FastAPI + uvicorn)

```
Framework: FastAPI v0.111+
ASGI Server: uvicorn[standard] v0.30+
Validation: Pydantic BaseModel (auto schema)
CORS: CORSMiddleware (allow_origins=["*"])
```

---

## X. DATA SCIENCE INFRASTRUCTURE

### 10.1 Python Environment

```
Interpreter: Python 3.11.9 (strict version)
Package Manager: uv (thay thế pip — faster)
Dependency file: pyproject.toml
```

### 10.2 Key Scientific Libraries

| Library | Version | Vai trò |
|---|---|---|
| `LightGBM` | ≥4.4.0 | Gradient Boosting ML models |
| `PyTorch` | ≥2.3.1 | Deep Learning (MLP) |
| `scikit-learn` | ≥1.5.1 | Calibration, metrics, preprocessing |
| `Polars` | ≥0.20.0 | Fast DataFrame (lazy evaluation) |
| `Pandas` | ≥2.2.2 | Data manipulation |
| `NumPy` | ≥1.26.4 | Numerical computing |
| `SciPy` | ≥1.13.1 | Morphological image operations |
| `osmnx` | ≥1.9.4 | OSM graph download + processing |
| `NetworkX` | ≥3.3 | Graph algorithms (Dijkstra) |
| `pyproj` | ≥3.6.1 | CRS transformation (UTM ↔ WGS84) |
| `Shapely` | ≥2.0.6 | Geometric operations |
| `alphashape` | ≥1.3.1 | Concave hull (flood extent polygon) |
| `folium` | ≥0.17.0 | Map rendering (Streamlit) |
| `Streamlit` | ≥1.35.0 | Interactive web dashboard |
| `joblib` | ≥1.4.2 | Model serialization |
| `fpdf2` | ≥2.7.9 | PDF report generation |
| `pymongo` | ≥4.8.0 | MongoDB Python driver |

---

## XI. INNOVATION HIGHLIGHTS (Điểm đổi mới cho bài báo)

### 11.1 Multi-Scale Flood Forecasting

```
Scale 1: Commune-level (Cấp xã)
  - Model: LightGBM (1,200 trees)
  - Input: 32 weather variables + lag/rolling features
  - Output: p_flood, flood_area_m2, flood_ratio
  - Horizon: 7 ngày

Scale 2: Pixel-level (Lưới UTM ~ 5.7 m²)
  - Model: Calibrated LightGBM (800 trees) + Isotonic calibration
  - Input: Terrain features (DEM, TWI, Slope, River Distance) + Rainfall
  - Output: flood_prob per grid point
  - Horizon: 12 giờ tiếp theo (thời gian thực)
```

### 11.2 Flood-Context-Aware Routing (Novel Contribution)

```
Không chỉ tìm đường ngắn nhất → tìm đường an toàn nhất trong lũ:
  • Mỗi chặng đường 0.8km được tra cứu flood_prob riêng
  • Tự động chuyển đổi phương tiện phù hợp theo mực ngập từng chặng
  • Hai chế độ hoạt động: dân chúng tự di tản vs. đội cứu hộ điều phối
  • Real-time: Dùng dữ liệu flood_point_probability_rt.csv (cập nhật 30 phút)
```

### 11.3 Offline Resilience Architecture (Critical for Disaster Zones)

```
Problem: Hạ tầng mạng thường sập khi lũ nghiêm trọng nhất
Solution: 3-layer offline fallback
  Layer 1: Cached data (SharedPreferences, SQLite)
  Layer 2: On-device A* routing (không cần server)
  Layer 3: P2P UDP Mesh (không cần Internet hoàn toàn)
```

### 11.4 SOS Urgency Scoring (AI-powered Triage)

```
Bài toán triage y tế áp dụng vào cứu hộ lũ lụt:
  • MLP 3-layer (8→32→16→1) với Dropout regularization
  • Input: Tọa độ + Mức ngập + Số người + Thời gian + Trạng thái
  • Output: urgency_prob → ưu tiên dispatcher điều phối đội cứu hộ
  • Fallback: Heuristic rule khi model chưa train
```

---

## XII. EVALUATION FRAMEWORK

### 12.1 ML Model Quality Metrics

| Model | Metric | Giá trị |
|---|---|---|
| Commune LightGBM | CV AUC | Ghi log khi train (`np.mean(aucs)`) |
| Commune Area Reg | CV MAE (log_area) | `np.mean(maes_area)` |
| Commune Ratio Reg | CV MAE (log_ratio) | `np.mean(maes_ratio)` |
| Pixel LightGBM (calib) | ROC-AUC | Sau spatial block split |
| Pixel LightGBM (calib) | PR-AUC | Đo trên imbalanced test set |
| MLP Priority | Val Accuracy | `accuracy_score(val_true, val_pred)` |
| MLP Priority | Val AUC-ROC | `roc_auc_score(val_true, val_probs)` |

### 12.2 Anti-Leakage Validation Strategy

| Strategy | Áp dụng cho | Mục đích |
|---|---|---|
| `GroupKFold(5)` | Commune-level | Không để xã xuất hiện cross-fold |
| Spatial Block Split | Pixel-level | Tách vùng địa lý train/test |
| Temporal Purge Gap | Pixel-level | Loại overlap thời gian giữa train/test |
| Leak Column Removal | Both | Loại tường minh các cột là target leakage |

### 12.3 Testing Framework

```
Python AI:    pytest + FastAPI TestClient
Node.js:      Jest v29.7.0 + Supertest v7.2.2
             Coverage: 100% routes
Flutter:      flutter_test SDK
```

---

## XIII. DEPLOYMENT & INFRASTRUCTURE

### 13.1 Docker Configuration

```yaml
# docker-compose.yml tóm tắt:
Services: 6
Network: floodsos-net (bridge)
Volumes: mongodb-data (persistent)

Image sizes:
  Dockerfile (Full):       ~4 GB (torch, CUDA, Streamlit)
  Dockerfile.api (Light):  ~500 MB (API only, no torch initially)
```

### 13.2 System Requirements

| Component | Requirement |
|---|---|
| Python | 3.11.9 (strict) |
| Node.js | ≥18.0.0 |
| Flutter | ≥3.19 (stable) |
| MongoDB | v7+ |
| Docker | Engine 24+ |
| RAM (AI Server) | ≥8 GB (training), ≥4 GB (inference) |

### 13.3 Port Configuration

```
3002  — Node.js Backend (REST + Socket.IO)
8765  — Priority API (FastAPI — MLP urgency scoring)
8766  — Routing API (FastAPI — Flood-aware routing)
8501  — Streamlit Dashboard (Dispatch coordination)
8502  — Streamlit SOS Map (Shelter + SOS tracking)
27017 — MongoDB
```

---

## XIV. SUGGESTED PAPER SECTIONS (Gợi ý cấu trúc bài báo)

### Abstract Keywords
`Flood prediction`, `LightGBM`, `Multilayer Perceptron`, `Flood-aware routing`, `OpenStreetMap`, `Disaster management`, `Mobile emergency response`, `Offline resilience`, `P2P mesh network`, `Real-time GIS`, `Nghệ An Vietnam`

### Proposed Paper Structure

1. **Introduction** — Flood disaster context in Nghệ An; motivation for integrated AI system
2. **Related Work** — Flood prediction ML, disaster routing, emergency mobile apps
3. **System Architecture** — 4-layer microservices, Docker deployment
4. **Flood Prediction Models** — LightGBM commune-level (7-day) + Pixel-level (real-time)
5. **SOS Urgency Scoring** — MLP architecture, features, training methodology
6. **Flood-Aware Routing Algorithm** — Dual-mode routing, segment classification
7. **Mobile Application** — Offline resilience, P2P mesh, calm crisis UI
8. **Experiments & Evaluation** — Dataset description, anti-leakage validation, metrics
9. **Discussion** — Innovation comparison, limitations, future work
10. **Conclusion**

### Potential Target Journals
- *International Journal of Disaster Risk Reduction* (Elsevier)
- *Natural Hazards and Earth System Sciences* (EGU/Copernicus)
- *IEEE Transactions on Geoscience and Remote Sensing*
- *Computers, Environment and Urban Systems* (Elsevier)
- *International Journal of Applied Earth Observation and Geoinformation*

---

## XV. SUMMARY TABLE — All Algorithms & Technologies

| # | Algorithm/Technology | Category | Parameters | Purpose |
|---|---|---|---|---|
| 1 | **LightGBM Classifier** | ML - Boosting | n=1200, lr=0.03, leaves=64 | Commune flood probability |
| 2 | **LightGBM Regressor (×2)** | ML - Boosting | n=2000, lr=0.03, leaves=64 | Flood area + ratio prediction |
| 3 | **LightGBM Classifier** | ML - Boosting | n=800, lr=0.03, leaves=31, λ=5.0 | Pixel-level flood prob |
| 4 | **Isotonic Calibration** | Calibration | cv=3, method=isotonic | Probability calibration |
| 5 | **PyTorch MLP** | Deep Learning | 8→32→16→1, Dropout=0.3 | SOS urgency scoring |
| 6 | **BCELoss + Adam** | Optimization | lr=1e-3, ReduceLROnPlateau | MLP training |
| 7 | **Group K-Fold CV** | Validation | k=5, group=GID_3 | Anti-leakage cross-validation |
| 8 | **Spatial Block Split** | Validation | block=30×cell, test=30% | Spatial train/test separation |
| 9 | **Temporal Purge** | Validation | purge=LEAD_HOURS=3h | Time gap between train/test |
| 10 | **Dijkstra Shortest Path** | Graph Algorithm | weight=length (m) | OSM road routing |
| 11 | **Haversine Formula** | Geospatial | R=6371 km | Distance between coordinates |
| 12 | **Nearest-Neighbor Lookup** | Spatial Index | Euclidean approx < 0.1° | Flood prob at SOS point |
| 13 | **A* Search** | Graph Algorithm | On-device assets | Offline routing (Flutter) |
| 14 | **Binary Morphology** | Image Processing | open→close | Flood extent smoothing |
| 15 | **TWI Heuristic** | Geospatial | w=(0.5, 0.3, 0.2) | Static flood zone scoring |
| 16 | **Rolling Window Features** | Feature Engineering | w=[1,3,6,12,24]h | Temporal rainfall aggregation |
| 17 | **Log-Transform Targets** | Feature Engineering | log1p / expm1 | Handling skewed distributions |
| 18 | **StandardScaler** | Preprocessing | Z-score | MLP input normalization |
| 19 | **CRS Transform (pyproj)** | Geospatial | EPSG:32648→4326 | UTM → WGS84 conversion |
| 20 | **Socket.IO** | Real-time | v4.8.3, event-driven | P2P rescue coordination |
| 21 | **UDP Multicast (P2P Mesh)** | Networking | LAN, no Internet | Offline SOS broadcasting |
| 22 | **Polars Lazy Scan** | Data Processing | LazyFrame | Fast large CSV processing |
| 23 | **F1-based Threshold** | Model Selection | argmax(F1 on PR curve) | Optimal classification cutoff |
| 24 | **Impact Score Formula** | Decision Support | w=(0.35, 0.25, 0.15) | Event-level severity scoring |
| 25 | **Priority Score Formula** | Decision Support | w=(0.5, 0.3, 0.2) | 7-day forecast triage |

---

*Document generated by Antigravity AI — FloodSOS Project Analysis for Scientific Paper*  
*Date: 2026-04-16 | Conversation: 6f0b72a3-1686-44f6-887c-6c88c18b49d3*
