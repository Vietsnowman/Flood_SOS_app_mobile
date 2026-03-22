# tests/test_priority_api.py — pytest tests for priority_api.py FastAPI endpoints
"""
Chạy:
    cd Gop_app
    uv run pytest tests/ -v
"""
# pytest imported for future fixture use
from fastapi.testclient import TestClient

# Import từ root (uvicorn module)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from priority_api import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_has_status_ok(self):
        res = client.get("/health")
        assert res.json()["status"] == "ok"

    def test_health_has_model_loaded_field(self):
        res = client.get("/health")
        assert "model_loaded" in res.json()


class TestPredictEndpoint:
    def test_predict_with_valid_input(self):
        res = client.post("/predict", json={
            "lat": 19.34,
            "lon": 105.71,
            "flood_prob_near": 0.7,
            "people_count": 3,
            "status": "open"
        })
        assert res.status_code == 200
        data = res.json()
        assert "urgency_prob" in data
        assert "is_urgent" in data
        assert "model_loaded" in data
        assert 0.0 <= data["urgency_prob"] <= 1.0
        assert isinstance(data["is_urgent"], bool)

    def test_predict_with_low_flood_prob(self):
        res = client.post("/predict", json={
            "lat": 19.34,
            "lon": 105.71,
            "flood_prob_near": 0.05,
            "people_count": 1,
        })
        assert res.status_code == 200

    def test_predict_with_high_flood_prob(self):
        res = client.post("/predict", json={
            "lat": 19.34,
            "lon": 105.71,
            "flood_prob_near": 0.95,
            "people_count": 10,
        })
        assert res.status_code == 200
        # High flood + many people → urgency_prob should be higher than zero
        assert res.json()["urgency_prob"] > 0.0

    def test_predict_missing_lat_returns_422(self):
        res = client.post("/predict", json={"lon": 105.71, "flood_prob_near": 0.5})
        assert res.status_code == 422

    def test_predict_flood_prob_out_of_range_returns_422(self):
        res = client.post("/predict", json={
            "lat": 19.34,
            "lon": 105.71,
            "flood_prob_near": 1.5,  # > 1.0, violates ge=0.0, le=1.0 constraint
        })
        assert res.status_code == 422
