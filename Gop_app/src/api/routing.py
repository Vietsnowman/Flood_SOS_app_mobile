# src/api/routing.py — Re-export from root routing_api.py
# Uvicorn còn chạy: uv run uvicorn routing_api:app (từ root)
# Module này cho phép import sạch trong code khác hoặc test:
#   from src.api.routing import app
from routing_api import app  # noqa: F401
