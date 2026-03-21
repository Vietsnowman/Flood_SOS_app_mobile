"""
priority_api.py
===============
Lightweight FastAPI microservice wrapping predict_sos_priority.py.
Node.js server.js gọi endpoint này sau khi nhận SOS để lấy urgency score.

Chạy (với uv):
    uv run uvicorn priority_api:app --host 0.0.0.0 --port 8765 --reload

Endpoint:
    POST /predict
    Body:  { "lat": 19.34, "lon": 105.71, "flood_prob_near": 0.72,
              "people_count": 3, "status": "open" }
    Returns: { "urgency_prob": 0.87, "is_urgent": true, "model_loaded": true }

    GET  /health  →  { "status": "ok", "model_loaded": bool }
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import predict module từ cùng thư mục.
try:
    from predict_sos_priority import predict_priority, _load_artifacts
    _MODEL_AVAILABLE = True
except Exception as _e:
    logging.warning(f"[priority_api] Không load được model: {_e}")
    _MODEL_AVAILABLE = False

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SOS Priority API",
    description="Predict urgency score cho SOS alert dựa trên flood_prob và thông tin SOS.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class SOSInput(BaseModel):
    lat: float            = Field(..., description="Vĩ độ SOS")
    lon: float            = Field(..., description="Kinh độ SOS")
    flood_prob_near: float = Field(0.0, ge=0.0, le=1.0,
                                   description="Xác suất ngập tại điểm gần nhất [0,1]")
    people_count: Optional[float] = Field(1.0, ge=0,
                                          description="Số người cần cứu")
    status: Optional[str] = Field("open",
                                  description="Trạng thái SOS: open|in_progress|closed")
    # Temporal features — tự động tính từ giờ hiện tại nếu không cung cấp.
    hour_of_day: Optional[int]  = Field(None, ge=0, le=23)
    is_night:    Optional[int]  = Field(None, ge=0, le=1)
    is_weekend:  Optional[int]  = Field(None, ge=0, le=1)


class PriorityResult(BaseModel):
    urgency_prob: float   # [0, 1]
    is_urgent: bool       # True nếu prob >= 0.5
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    loaded = _MODEL_AVAILABLE
    if loaded:
        try:
            _load_artifacts()
        except Exception:
            loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PriorityResult)
def predict(body: SOSInput):
    """
    Trả về urgency_prob: xác suất SOS cần xử lý trong ≤ 15 phút.
    Nếu model chưa train, trả về fallback dựa trên flood_prob_near.
    """
    if not _MODEL_AVAILABLE:
        # Fallback: nếu không có model, ước lượng từ flood_prob và people_count
        fallback_prob = min(1.0, body.flood_prob_near * 0.7
                            + min(body.people_count or 1, 10) / 20.0)
        return PriorityResult(
            urgency_prob=round(fallback_prob, 4),
            is_urgent=fallback_prob >= 0.5,
            model_loaded=False,
        )

    try:
        prob = predict_priority(
            lat=body.lat,
            lon=body.lon,
            flood_prob_near=body.flood_prob_near,
            people_count=body.people_count or 1.0,
            status=body.status or "open",
            hour_of_day=body.hour_of_day,
            is_night=body.is_night,
            is_weekend=body.is_weekend,
        )
        return PriorityResult(
            urgency_prob=round(float(prob), 4),
            is_urgent=bool(prob >= 0.5),
            model_loaded=True,
        )
    except FileNotFoundError:
        # Model file missing — fallback graceful
        fallback_prob = min(1.0, body.flood_prob_near * 0.7
                            + min(body.people_count or 1, 10) / 20.0)
        return PriorityResult(
            urgency_prob=round(fallback_prob, 4),
            is_urgent=fallback_prob >= 0.5,
            model_loaded=False,
        )


# ── Entrypoint standalone ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("priority_api:app", host="0.0.0.0", port=8765, reload=True)
