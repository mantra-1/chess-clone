# JebBot Chess Server - Railway Deployment
FROM python:3.11-slim

# Install system dependencies including Stockfish
RUN apt-get update && apt-get install -y \
    stockfish \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && uv sync --frozen

# Copy source code
COPY jebbot/ ./jebbot/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Expose port (Railway will set PORT env var)
EXPOSE 8080

# Run the server
CMD ["uv", "run", "python", "-m", "jebbot.play.server"]
