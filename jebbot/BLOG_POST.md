# Building JebBot: A Chess AI That Plays Like Me

*Everyone's obsessed with living forever. I just want my friends to lose to a bot playing the Danish Gambit poorly after I'm gone.*

---

## The Idea

There's something deeply appealing about the idea of a digital clone. Not the sci-fi version where a copy of your consciousness wakes up confused in a server somewhere, but something simpler: a bot that captures how you do one specific thing.

For me, that thing is playing chess badly.

I've played thousands of games on chess.com since 2016. Blitz mostly, always going for the most fun line rather than the most correct one. I love gambits, hate draws, and have developed what can only be described as "a style" over the years.

The question that led to this project: could I train a neural network to recognize that style? Not to play good chess—there's Stockfish for that—but to play *my* chess?

## Three Goals

This wasn't just about the end result. I had three things I wanted to accomplish:

**1. Build a digital chess-playing clone of myself.** Something that captures my patterns, my opening preferences, my tendency to play moves that are slightly worse but more interesting.

**2. Use AI coding tools for a real project.** I've been using Claude, Cursor, and Claude Code for little things, but I wanted to push them on something substantial. Something with actual ML training, data pipelines, and debugging weird tensor errors at 2am.

**3. Actually understand how neural networks work.** Not just conceptually—I've watched the 3Blue1Brown videos like everyone else—but viscerally. I wanted to write the forward pass, debug the loss function, watch the gradients flow. The whole thing.

## The Data Pipeline

### Downloading 8,000 Games

Chess.com has a nice API. Point it at a username, get a list of monthly game archives. Point it at an archive, get PGN files. I wrote a simple client that pulls down everything and stores it as JSON.

Final count: **7,956 games** spanning from 2016 to 2025. That's about a decade of me clicking pieces around a board.

### Extracting Positions

Not every game is useful. I filtered out:
- Abandoned games
- Timeouts in clearly won positions
- The opening moves (first 5) since those are just memorized theory

What remained: **219,663 positions** where I made an actual decision. About 31 meaningful moves per game on average.

From my games I learned some things about myself:
- 97% blitz player (no patience for long thinks)
- 51% win rate as White, 48% as Black
- Low draw rate at 4% (I play for decisive results)
- Top openings: Modern Defense, Danish Gambit, Philidor Defense

### Encoding Positions

Neural networks eat tensors. A chess position becomes a 12x8x8 array:
- 12 channels (one for each piece type × 2 colors)
- 8×8 board (obviously)
- Each channel is binary: 1 where that piece sits, 0 elsewhere

Moves become indices 0-4095. There are 64 squares a piece can move from, 64 squares it can move to. 64×64 = 4096. Simple.

## The Model Architecture

Here's the key insight that makes this work: **I'm not teaching the model to play chess. I'm teaching it to recognize my moves.**

The model is a binary classifier. Given a position and a candidate move, it outputs a probability: "Is this something Jeb would play?"

```
INPUT:
├── Position: 12×8×8 tensor (one channel per piece type)
└── Move: index 0-4095 → embedded to 64-dim vector

ARCHITECTURE:
├── 3 convolutional layers (12→64→128→128 channels)
├── Flatten to 8,192 features
├── Concatenate with 64-dim move embedding
├── 3 fully connected layers (8,256→256→64→1)
└── Sigmoid output (probability 0-1)

OUTPUT: "Is this a Jeb move?" (0.0 to 1.0)
```

Total parameters: **~540,000**. Tiny by modern standards. GPT-4 has over a trillion. My model could fit on a floppy disk if anyone still had those.

## The Training Trick: Negative Examples

Here's where I got stuck for a while. My first training run hit 100% accuracy instantly. Something was wrong.

The problem: I was only showing the model moves I actually played. It learned that *any* move passed to it was probably a Jeb move, because that's all it ever saw. The model wasn't learning my style; it was learning to always say "yes."

The fix: **negative examples**. For every position, I generate two training examples:
1. **Positive**: The move I actually played (target = 1.0)
2. **Negative**: A move I didn't play (target = 0.0)

But not random moves—that would be too easy. I use Stockfish set to 1500 Elo (roughly my level) to find the top 6 "reasonable" moves, then pick one I didn't play. This creates the perfect contrast: "Here's a good move, but it's not what Jeb chose."

Now the model has to learn what makes my moves *mine*, not just what makes them legal.

## The Training Loop

With the data pipeline solid, training was straightforward:

- **80/10/10 split** for train/validation/test
- **Binary Cross-Entropy loss** (standard for classification)
- **Adam optimizer** with learning rate 1e-4
- **Early stopping** with 10 epochs patience
- **Dropout (0.3)** to prevent overfitting

I trained on Apple Silicon using MPS acceleration. Had to add some memory management—clearing the cache every 1000 batches—but it worked well enough.

### The Visualization That Made It Click

One of my favorite parts of this project was building a real-time training visualizer. Every 100 batches, the trainer sends 6 random positions to a local web server that displays:
- The board position
- The move being evaluated
- The model's confidence
- Whether it was actually my move

Watching the model learn in real-time was mesmerizing. Early on, it would confidently say "that's a Jeb move!" about Stockfish's top choice. By epoch 20, it started recognizing my weird preferences. The confidence bars would swing—high on my aggressive queen moves, low on the boring defensive retreats I'd never play.

Green border for correct predictions. Red for wrong. The flash animations made it feel alive.

## Results

After training, the model achieves **~60.5% accuracy** distinguishing my moves from reasonable alternatives.

Is that good? Consider:
- Random guessing would get 50% (it's binary classification with balanced classes)
- 100% would mean it perfectly predicts every move I make, which is impossible because humans are inconsistent
- 60% means it's captured *something* about my style

The interesting examples tell the story:

**High-Confidence Correct (>90%)**: The model recognizes my signature moves. Aggressive queen sorties. The Danish Gambit continuation. Pushing pawns when I should be developing.

**High-Confidence Wrong (>85%)**: Moves the model thinks I'd love, but I actually didn't play. Often these are *exactly* the kind of move I'd play in a different position. The model has learned my patterns but overgeneralizes.

**Missed Jeb Moves (<15%)**: Moves I played that the model thought were too boring for me. Usually defensive moves I made reluctantly.

## Playing Against JebBot

The trained model doesn't play chess by itself—it scores moves. To actually play, I built an engine that combines:

### 1. Opening Book (First ~5 Moves)
Opening play is mostly memorized. Instead of letting the neural network struggle with theory, I built a weighted random book based on my actual opening preferences:

- As White: Always 1.e4, then probabilistic responses (Danish Gambit 30%, Italian 30%, etc.)
- As Black: Against 1.e4, 45% Caro-Kann, 45% Sicilian, 10% Scandinavian

### 2. Endgame Detection
When either side has ≤12 material points (roughly a rook or less), JebBot hands off to pure Stockfish. Endgames are tactical—my "style" doesn't help when it's K+R vs K. Let the engine calculate.

### 3. Style-Quality Balancing
For everything in between, the engine uses a clever selection algorithm:

```
1. Get top 5 moves from Stockfish (ranked by quality)
2. Score each with the neural network (0-1 confidence)
3. Find the "safe pick": first Stockfish move with >50% Jeb score
4. If another move scores 15%+ higher, play that instead
5. Fallback: if nothing is >50%, play highest Jeb score anyway
```

This balances two competing goals:
- Don't make moves so bad they're obviously wrong
- Let personality emerge when the choices are close

The result feels remarkably like playing against... me. It favors the same openings. It makes the same slightly-dubious piece sacrifices. It has the same blind spots.

## The Tech Stack

- **python-chess**: Board representation and move validation
- **stockfish**: Candidate move generation and endgame play
- **PyTorch**: Neural network training
- **wandb** (optional): Training logging and visualization
- **Flask**: Simple API server for the game UI
- **chessboard.js**: Interactive chess board in the browser

Everything runs locally. No cloud needed for inference.

## What I Learned

### About Neural Networks
They're not magic. They're function approximators that find patterns in data. The tricky part is giving them data that captures what you actually care about. Negative examples matter more than I expected. Balanced datasets matter more than I expected.

### About AI Coding Tools
Claude Code wrote probably 70% of this codebase. Not by magic—I had to be very specific about what I wanted, catch its mistakes, and iterate. But it was genuinely useful in a way that surprised me. The debugging help alone was worth it. When my MPS memory kept exploding, Claude suggested the cache-clearing pattern that fixed it.

### About My Chess
I play far more gambits than I realized. My opening repertoire is narrower than I thought. And apparently I have a tell: the model learned that I almost never retreat my queen in the first 15 moves. Even when I should.

## What's Next

JebBot is functional but not deployed. The next steps would be:
- Host the API somewhere (Railway, Fly.io)
- Build a nicer frontend
- Maybe add personality to the move explanations ("Ah yes, the classic Jeb attack")

But honestly? The project accomplished what I wanted. I have a digital chess clone. I understand neural networks much better. And I pushed AI coding tools to build something real.

*Now I just need to figure out who to bequeath my chess bot to in my will.*

---

## Technical Appendix

### Model Size Breakdown
| Component | Parameters |
|-----------|-----------|
| Conv1 (12→64) | 6,976 |
| Conv2 (64→128) | 73,856 |
| Conv3 (128→128) | 147,584 |
| Move Embedding | 262,144 |
| FC1 (8256→256) | 2,113,792 |
| FC2 (256→64) | 16,448 |
| FC3 (64→1) | 65 |
| **Total** | **~2.6M** |

### Data Statistics
- Total games: 7,956
- Total positions: 219,663
- Dataset size (with negatives): 439,326 examples
- Training file: 48MB
- Trained model: 31MB

### Training Configuration
- Epochs: 50 (with early stopping)
- Batch size: 64
- Learning rate: 1e-4
- Weight decay: 0.0001
- Dropout: 0.3
- Early stopping patience: 10 epochs

### Repository Structure
```
jebbot/
├── jebbot/
│   ├── data/          # Download, parse, encode
│   ├── model/         # StyleSelector network
│   ├── play/          # Engine, openings, server
│   ├── training/      # Training utilities
│   └── visualization/ # Real-time training display
├── scripts/           # CLI entry points
└── data/
    ├── raw/           # Downloaded games
    ├── processed/     # Training positions
    └── models/        # Trained checkpoints
```

---

*Inspired by Peter Whidden's "AI Learns Pokemon" video, but using supervised learning instead of reinforcement learning. Much simpler, much faster, and perfectly suited for behavioral cloning.*
