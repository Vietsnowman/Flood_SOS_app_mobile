# Dockerfile for Python AI Core (Gop_app)
# Used for: priority_api, routing_api, streamlit dashboards
FROM python:3.11.9-slim

WORKDIR /app

# Install uv (fast package manager)
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install all Python dependencies into the system environment
RUN uv sync --no-dev

# Copy application source
COPY . .

# Default command is overridden per-service in docker-compose.yml
CMD ["uv", "run", "uvicorn", "priority_api:app", "--host", "0.0.0.0", "--port", "8765"]