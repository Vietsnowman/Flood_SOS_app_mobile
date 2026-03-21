"""
predict_sos_priority.py
========================
Inference helper — load model + scaler một lần, expose hàm predict_priority().

Dùng từ Streamlit hoặc bất kỳ Python script nào:
  from predict_sos_priority import predict_priority
  prob = predict_priority(lat=19.34, lon=105.71,
                          flood_prob_near=0.72, people_count=4,
                          status="open")
  # prob là float [0, 1] — xác suất SOS được xử lý trong ≤ 15 phút (urgent)

Chú ý: script này KHÔNG train model. Hãy chạy train_sos_priority.py trước.
"""

import os
import functools
import numpy as np
import joblib
import torch
import torch.nn as nn

# ─── CONFIG ───────────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(__file__)
_MODEL_PATH  = os.path.join(_BASE_DIR, "models", "sos_priority_model.pt")
_SCALER_PATH = os.path.join(_BASE_DIR, "models", "sos_priority_scaler.joblib")

_STATUS_ENC = {
    "open":        0,
    "warning":     0,
    "in_progress": 1,
    "closed":      2,
    "false_alarm": 2,
}

_DEFAULT_THRESHOLD = 0.5
# ──────────────────────────────────────────────────────────────────────────────


class _SOSPriorityNet(nn.Module):
    """Phải khớp với kiến trúc đã dùng khi train."""

    def __init__(self, n_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@functools.lru_cache(maxsize=1)
def _load_artifacts():
    """Load model + scaler một lần duy nhất (cached)."""
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Model chưa được train. Hãy chạy train_sos_priority.py trước.\n"
            f"(Không tìm thấy: {_MODEL_PATH})"
        )
    if not os.path.exists(_SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler không tồn tại. Hãy chạy train_sos_priority.py trước.\n"
            f"(Không tìm thấy: {_SCALER_PATH})"
        )

    checkpoint  = torch.load(_MODEL_PATH, map_location="cpu")
    n_features  = checkpoint.get("n_features", 6)
    model       = _SOSPriorityNet(n_features=n_features)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scaler    = joblib.load(_SCALER_PATH)
    threshold = checkpoint.get("threshold", _DEFAULT_THRESHOLD)

    return model, scaler, threshold


def predict_priority(
    lat: float,
    lon: float,
    flood_prob_near: float,
    people_count: float = 1.0,
    status: str = "open",
    hour_of_day: int | None = None,  # 0-23; None → giờ hiện tại
    is_night: int | None = None,     # 1 nếu 22:00-05:59
    is_weekend: int | None = None,   # 1 nếu Thứ 7/CN
) -> float:
    """
    Trả về xác suất SOS là khẩn cấp (urgent_15m=1).

    Parameters
    ----------
    lat             : vĩ độ SOS
    lon             : kinh độ SOS
    flood_prob_near : xác suất ngập tại điểm gần nhất [0, 1]
    people_count    : số người trong SOS (mặc định 1)
    status          : "open" | "in_progress" | "closed" | ...
    hour_of_day     : giờ trong ngày (0-23); mặc định giờ hiện tại
    is_night        : 1 nếu 22:00-05:59; mặc định tự tính
    is_weekend      : 1 nếu Thứ 7/CN; mặc định tự tính

    Returns
    -------
    float in [0.0, 1.0]
    """
    model, scaler, _ = _load_artifacts()

    import datetime as _dt
    _now       = _dt.datetime.now()
    _hour      = hour_of_day if hour_of_day is not None else _now.hour
    _is_night  = is_night    if is_night    is not None else (1 if _hour >= 22 or _hour < 6 else 0)
    _is_wkend  = is_weekend  if is_weekend  is not None else (1 if _now.weekday() >= 5 else 0)

    status_enc = _STATUS_ENC.get((status or "open").strip().lower(), 0)

    features = np.array([[
        float(lat),
        float(lon),
        float(flood_prob_near) if flood_prob_near is not None else 0.0,
        float(people_count) if people_count is not None else 1.0,
        float(status_enc),
        float(_hour),
        float(_is_night),
        float(_is_wkend),
    ]], dtype=np.float32)

    features_scaled = scaler.transform(features).astype(np.float32)
    x_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = model(x_tensor).item()

    return float(prob)


def is_urgent(
    lat: float,
    lon: float,
    flood_prob_near: float,
    people_count: float = 1.0,
    status: str = "open",
    hour_of_day: int | None = None,
    is_night_flag: int | None = None,
    is_weekend_flag: int | None = None,
    threshold: float | None = None,
) -> bool:
    """Convenience wrapper — trả về True nếu SOS được dự đoán là khẩn cấp."""
    _, _, default_thr = _load_artifacts()
    thr  = threshold if threshold is not None else default_thr
    prob = predict_priority(lat, lon, flood_prob_near, people_count, status,
                            hour_of_day, is_night_flag, is_weekend_flag)
    return prob >= thr


# ── Demo khi chạy trực tiếp ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Demo: predict_sos_priority ===\n")

    test_cases = [
        dict(lat=19.34, lon=105.71, flood_prob_near=0.80, people_count=5,
             status="open",        label="[ngập cao, open]"),
        dict(lat=19.34, lon=105.71, flood_prob_near=0.05, people_count=1,
             status="in_progress", label="[ngập thấp, in_progress]"),
        dict(lat=19.33, lon=105.70, flood_prob_near=0.50, people_count=3,
             status="open",        label="[ngập trung bình, open]"),
    ]

    for tc in test_cases:
        label = tc.pop("label")
        try:
            prob = predict_priority(**tc)
            urgency = "🚨 URGENT" if prob >= 0.5 else "✅ Normal"
            print(f"{label:40s}  prob={prob:.4f}  {urgency}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            break
