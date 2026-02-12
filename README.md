# JebBot

A neural network trained to play chess in a specific player's style. It downloads a player's game history from Chess.com, trains a model to predict their moves, then combines that model with Stockfish to play games that match the player's style.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

You'll also need [Stockfish](https://stockfishchess.org/download/) installed on your system.

## Scripts

All scripts are run from the project root with `uv run python scripts/<script>.py`.

### Data pipeline

| Script | What it does |
|--------|-------------|
| `fetch_games.py` | Downloads a player's games from Chess.com. Takes an optional username arg (defaults to `jebhead`). |
| `build_dataset.py` | Parses downloaded games into training positions. Pass `--stockfish` to include engine analysis. |
| `rebuild_dataset_with_stockfish.py` | Re-processes an existing dataset to add Stockfish evaluations. |
| `explore_games.py` | Prints stats about downloaded games (openings, win rates, etc). |

### Training

| Script | What it does |
|--------|-------------|
| `train.py` | Trains the StyleSelector model. Options: `--epochs`, `--lr`, `--batch-size`, `--visualize`. |
| `test_model.py` | Evaluates a trained model on the test set. |
| `train_test.py` | Unit tests for training code. |
| `start_visualizer.py` | Starts a live training dashboard in the browser. |
| `view_interesting.py` | Browse interesting/unusual model predictions. |

### Playing

| Script | What it does |
|--------|-------------|
| `play_jebbot.py` | Starts a web server at `localhost:8767` where you can play against JebBot. |

## Forking this for your own player

1. **Fetch your games:** `uv run python scripts/fetch_games.py YOUR_CHESSCOM_USERNAME`
2. **Build dataset:** `uv run python scripts/build_dataset.py`
3. **Train:** `uv run python scripts/train.py`
4. **Update the opening book** in `jebbot/play/openings.py` to match your own opening repertoire.
5. **Play:** `uv run python scripts/play_jebbot.py`

## How it works

The model (`StyleSelector`) is a small CNN (~500K params) that takes a board position + candidate move and outputs the probability that the player would choose that move. During play, Stockfish generates the top candidate moves and the model picks the one most matching the player's style.
