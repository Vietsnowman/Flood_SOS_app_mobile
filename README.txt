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

main_unified.py → chạy app_SOS_shelters7.py hoặc app6.py
app6.py → dùng db_utils.py, đọc cache parquet do scheduler_cache.py tạo, có thể dùng report_utils.py
scheduler_cache.py → gọi predict.py
predict.py → gọi open_meteo.py + feature_engineering.py + load model do 1_train_models_no_leak.py train
app_SOS_shelters7.py → đọc output do realtime.py sinh + routing OSM (osmnx)
realtime.py → dùng model do train_flood_model_final.py train + gọi Open-Meteo hourly


1) File “chạy app” (entrypoint)
main_unified.py
Vai trò: “App gộp” (Streamlit) để chọn chạy 1 trong 2 app gốc bằng sidebar.
Cơ chế: dùng runpy.run_path() để chạy app_SOS_shelters7.py hoặc app6.py.
Điểm đặc biệt: “monkeypatch” st.set_page_config để tránh lỗi set_page_config gọi nhiều lần khi chạy app con.

app_SOS_shelters7.py
Vai trò: Streamlit Realtime dạng point-based (điểm lưới) + SOS + Shelter + Routing (OSM).
Đọc dữ liệu chính: trong realtime_outputs/ (ví dụ flood_point_probability_rt.csv, flood_extent_mask_rt.csv, sos_signals.csv, osm_graph.graphml, …).
Chức năng nổi bật:
Dashboard cứu hộ theo thời gian thực
Hiển thị điểm rủi ro, vùng ngập (mask)
Nhận/đọc tín hiệu SOS, chấm điểm ưu tiên SOS
Click bản đồ để đặt “căn cứ”, tính đường đi ngắn nhất (routing) trên dữ liệu OSM (osmnx/networkx)
Có phần map dành cho người dân (“bản đồ an toàn”)

app6.py
Vai trò: Streamlit dạng commune-based (theo xã) + forecast 7 ngày + DB + điều phối nguồn lực.
Đọc dữ liệu chính:
cache/forecasts/*.parquet (dự báo đã cache)
cache/merged.csv (dữ liệu lịch sử đã merge)
data/DanSo_Xa.csv (dân số)
DB: db/app.db (trạng thái sự cố, phân bổ, hotline requests, tasks…)
Chức năng nổi bật:
Xem bảng ưu tiên theo xã (priority)
Quản lý “incident status” theo xã
Nhập & lưu yêu cầu cứu hộ (hotline) vào DB
Tạo & cập nhật task điều phối
Quản lý điểm trú ẩn (còn chỗ, nhu cầu, log chuyển người)
Xuất báo cáo (CSV/PDF) (thường gọi report_utils.py)

2) Pipeline dự báo theo xã (forecast 7 ngày)
open_meteo.py
Vai trò: gọi API Open-Meteo để lấy weather forecast.
Output: trả về DataFrame có:
daily: nhiệt độ max/min, precipitation_sum, precipitation_hours, wind…
hourly aggregated: mean temp/humidity/cloud/wind + precipitation_max_1h
Được gọi bởi: predict.py (và gián tiếp bởi scheduler_cache.py, app6.py).

feature_engineering.py
Vai trò: tạo feature cho mô hình theo xã:
Lag mưa: tp_lag1/2/3/7
Rolling sum/max: tp_sum_3d/7d/14d, tp_max_...
Số ngày mưa trong cửa sổ: tp_rain_days_...
max 1h rolling: tp_max1h_max_...
tổng giờ mưa rolling: rain_hours_sum_...
thêm time features (month/dayofyear)
attach static features (đính kèm biến tĩnh của xã vào forecast)
Được gọi bởi: predict.py.

predict.py
Vai trò: “engine dự báo 7 ngày” theo xã.
Làm gì:
load 3 model “no leak” trong models/ (prob/area/ratio)
gọi Open-Meteo lấy forecast
ghép lịch sử mưa + forecast để tạo lag/rolling
dự đoán:
p_flood
flood_area_m2
flood_ratio
nếu có population → tính affected_population
tính priority_score từ tổ hợp các yếu tố (prob + affected_pop + ratio)
Được gọi bởi: scheduler_cache.py (và có thể bởi app khi cần dự báo on-demand).

scheduler_cache.py
Vai trò: chạy batch mỗi ngày để cache dự báo 7 ngày cho tất cả xã.
Input: cache/merged.csv + data/DanSo_Xa.csv
Output: cache/forecasts/{YYYY-MM-DD}_commune_forecasts.parquet
Mục đích: app6 mở ra “nhanh” (không phải dự báo on-the-fly cho từng xã).

3) Chuẩn bị dữ liệu & train model theo xã (no-leak)
0_merge_data.py
Vai trò: merge dữ liệu nhiều năm (2023/2024/2025) → 1 file dùng chung.
Input: data/NgheAn_weather32full_flood_merge_by_commune_time_202*_NO_NaN.csv
Output: cache/merged.csv
Ý nghĩa: tạo dataset lịch sử chuẩn để train và chạy cache.
1_train_models_no_leak.py
Vai trò: train các model “no leak” cho pipeline theo xã.
Train ra 3 model:
models/model_prob_no_leak.pkl (classifier dự đoán có ngập hay không)
models/model_area_no_leak.pkl (regressor dự đoán log(area))
models/model_ratio_no_leak.pkl (regressor dự đoán log(ratio))
Chú ý “no leak”: có danh sách LEAK_COLS để loại các cột có nguy cơ “lộ nhãn/đích”.
Cách CV: GroupKFold theo GID_3 để tránh rò rỉ theo địa lý.

4) Realtime point-based + SOS (mô hình theo điểm)
train_flood_model_final.py
Vai trò: script train mô hình theo điểm (dataset 2020) để phục vụ realtime.
Output quan trọng (thường):
model đã calibrate: calib_model.joblib
danh sách feature: feature_list.joblib
meta/cấu hình: model_meta.json
(có thể) mask/extent file dạng csv
Ý nghĩa: đây là “nguồn gốc” tạo model + feature cho realtime.py.

realtime.py
Vai trò: job tạo output realtime để app_SOS_shelters7.py hiển thị.
Input:
calib_model.joblib, feature_list.joblib, model_meta.json
base_points_static.csv (lưới điểm chuẩn)
gọi Open-Meteo hourly precipitation + cache openmeteo_cache.json
Output (đổ vào realtime_outputs/):
flood_point_probability_rt.csv (xác suất ngập theo điểm)
flood_extent_mask_rt.csv (mask vùng ngập)
impact_click_zone_rt.csv, zone_points_rt.csv (vùng tác động khi click)
meta_rt.json (metadata runtime)
Ý nghĩa: app realtime chỉ “vẽ và điều phối”; còn realtime.py là “máy tính dữ liệu realtime”.

5) Database + báo cáo (support)
db_utils.py
Vai trò: toàn bộ hàm làm việc với SQLite DB (mặc định trỏ tới db/app.db).
Những nhóm bảng/chức năng hay có:
incidents: trạng thái sự cố theo gid3
allocations: phân bổ nguồn lực theo gid3
requests: hotline/đầu vào yêu cầu cứu hộ
tasks: nhiệm vụ điều phối (tạo từ request hoặc tạo tay)
có thêm các hàm get/upsert/list/update cho từng nhóm
Được gọi bởi: chủ yếu app6.py (để lưu/đọc trạng thái, điều phối, hotline).

report_utils.py
Vai trò: xuất báo cáo:
export_csv(df, out_path)
export_pdf(top_df, summary_dict, out_path) (PDF bảng top + summary)
Được gọi bởi: thường app6.py khi bấm “Xuất PDF/CSV”.