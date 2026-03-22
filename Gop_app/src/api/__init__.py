# src/api/__init__.py — Re-exports FastAPI app instances from root-level files.
# Giúp import sạch hơn: from src.api.priority import app
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
