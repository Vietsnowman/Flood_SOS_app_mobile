# tests/test_routing_api.py — pytest tests for routing_api.py FastAPI endpoints
"""
Chạy:
    cd Gop_app
    uv run pytest tests/ -v
"""
# pytest imported for future fixture use
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from routing_api import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_has_expected_fields(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "rt_csv_exists" in data
        assert "shelters_exists" in data


class TestRouteEndpoint:
    """Tests /route POST endpoint."""

    def test_route_with_valid_coords(self):
        res = client.post("/route", json={"lat": 19.34, "lon": 105.71})
        assert res.status_code == 200
        data = res.json()
        assert "flood_level" in data
        assert "flood_prob" in data
        assert "mode" in data
        assert "route" in data
        assert "segments" in data
        assert "total_distance_km" in data
        assert "summary" in data

    def test_route_flood_level_is_low_or_high(self):
        res = client.post("/route", json={"lat": 19.34, "lon": 105.71})
        assert res.json()["flood_level"] in ("low", "high")

    def test_route_mode_matches_flood_level(self):
        res = client.post("/route", json={"lat": 19.34, "lon": 105.71}).json()
        if res["flood_level"] == "low":
            assert res["mode"] == "self_evacuation"
        else:
            assert res["mode"] == "rescue_dispatch"

    def test_route_missing_lat_returns_422(self):
        res = client.post("/route", json={"lon": 105.71})
        assert res.status_code == 422

    def test_route_missing_lon_returns_422(self):
        res = client.post("/route", json={"lat": 19.34})
        assert res.status_code == 422

    def test_route_returns_list_of_waypoints(self):
        res = client.post("/route", json={"lat": 19.34, "lon": 105.71})
        route = res.json()["route"]
        assert isinstance(route, list)
        if len(route) > 0:
            assert len(route[0]) == 2  # [lat, lon]
