# src/api/priority.py — Re-export from root priority_api.py
# Uvicorn còn chạy: uv run uvicorn priority_api:app (từ root)
# Module này cho phép import sạch trong code khác hoặc test:
#   from src.api.priority import app
from priority_api import app  # noqa: F401
