FROM python:3.11-slim

# Install stockfish binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish && \
    rm -rf /var/lib/apt/lists/* && \
    which stockfish && stockfish --help || true

WORKDIR /app

# Install CPU-only torch first, then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask flask-cors numpy python-chess stockfish requests

# Copy application code and model
COPY jebbot/ ./jebbot/
COPY data/models/best_model.pt ./data/models/best_model.pt

ENV PYTHONPATH=/app
ENV STOCKFISH_PATH=/usr/games/stockfish

EXPOSE 8767

CMD ["python", "-m", "jebbot.play.server"]
