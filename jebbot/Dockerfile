FROM python:3.11-slim

# Install stockfish
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Create venv and install CPU-only torch + other deps
RUN uv venv && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install flask-cors && \
    uv sync --no-dev

# Copy application code and model
COPY jebbot/ ./jebbot/
COPY data/models/best_model.pt ./data/models/best_model.pt

CMD ["uv", "run", "python", "-m", "jebbot.play.server"]
