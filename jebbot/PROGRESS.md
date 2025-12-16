# JebBot Progress

## Phase 1: Data Pipeline ✅ COMPLETE

**Completed December 15, 2025**

### What we built:
- Chess.com API client to download games
- PGN parser to extract positions
- Position encoder (FEN → 12x8x8 tensor)
- Move encoder (UCI → index 0-4095)
- PyTorch Dataset class

### Stats:
- 7,956 games downloaded (2016-2025)
- 219,663 training positions extracted
- 31 positions per game average
- 48MB training file

### Key files:
- `jebbot/data/download.py` - Chess.com API client
- `jebbot/data/parse.py` - PGN to positions
- `jebbot/data/encode.py` - Tensor encoding + Dataset class
- `data/processed/training_positions.json` - Training data

### My chess profile (from the data):
- 97% blitz player
- 51% win rate as White, 48% as Black
- Top openings: Modern Defense, Danish Gambit, Philidor Defense
- Low draw rate (4%) - I play for decisive results

## Phase 2: Model Architecture (NEXT)
- Design StyleSelector neural network
- Integrate with Stockfish for candidate moves
- ~500K parameter model

## Phase 3: Training (TODO)
- Cloud GPU training ($20-40)
- Track loss curves, accuracy metrics
- Target: >50% top-1 move prediction accuracy

## Phase 4: Evaluation (TODO)
- Blind test: can friends tell bot from real games?
- Play against it myself

## Phase 5: Deployment (TODO)
- FastAPI backend on Railway
- Web frontend with chessboard.js

---

## Video Ideas
- [ ] Live board visualization during training (show predicted vs actual move)
- [ ] Training loss curves timelapse
- [ ] Blind test reveal with friends
