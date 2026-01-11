# Chess Style Clone

A neural network that learns to play chess in your personal style.

Train a model on your Chess.com games to create a bot that makes moves the way you would.

## How it works

1. Downloads your games from Chess.com
2. Extracts positions and the moves you played
3. Uses Stockfish to generate candidate moves
4. Trains a neural network to pick moves that match your style
5. Plays games using your trained style model

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Stockfish](https://stockfishchess.org/download/) chess engine

## Install

```bash
# Clone the repo
git clone <repo-url>
cd jebbot

# Install dependencies
uv sync

# Install Stockfish
# Mac: brew install stockfish
# Ubuntu: sudo apt install stockfish
```

## Usage

### 1. Download your games

```bash
uv run python scripts/fetch_games.py YOUR_CHESS_COM_USERNAME
```

### 2. Build training dataset

```bash
# Basic (fast)
uv run python scripts/build_dataset.py

# With Stockfish analysis (recommended, slower)
uv run python scripts/build_dataset.py --stockfish
```

### 3. Train the model

```bash
uv run python scripts/train.py
```

Training takes 10-30 minutes on a Mac M1/M2/M3.

### 4. Play against your bot

```bash
uv run python scripts/play_jebbot.py
```

Opens a browser where you can play against your trained model.

## Configuration

Edit `jebbot/data/parse.py` to change `DEFAULT_USERNAME` to your Chess.com username.

## Project structure

```
jebbot/
├── jebbot/
│   ├── data/          # Data loading and encoding
│   ├── model/         # Neural network architecture
│   ├── play/          # Game server and engine
│   ├── training/      # Training utilities
│   └── visualization/ # Training visualizer
├── scripts/           # CLI scripts
└── data/              # Downloaded games and trained models (gitignored)
```

## License

MIT
