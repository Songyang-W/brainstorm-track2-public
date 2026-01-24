# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BrainStorm 2026 Track 2 challenge: Build a real-time visualization tool to guide neurosurgeons in placing a brain-computer interface (BCI) array over the optimal region of the motor cortex. The application processes live neural data from a 1024-channel micro-ECoG array (32×32 grid) and identifies velocity-tuned neural regions.

## Commands

```bash
# Setup
make install                                    # Install dependencies + git hooks
uv run python -m scripts.download hard          # Download dataset (super_easy/easy/medium/hard)

# Development (run in separate terminals)
uv run brainstorm-stream --from-file data/hard/ # Terminal 1: Stream neural data
uv run brainstorm-serve                         # Terminal 2: Serve web app at localhost:8000

# Code quality
make format                                     # Format with ruff
make lint                                       # Lint with ruff --fix
make type-check                                 # mypy on scripts/
make test                                       # pytest
make check-all                                  # All of the above
```

## Architecture

```
Data Flow:
[Parquet Files] → [stream_data.py ws://localhost:8765] → [Your App (browser/backend)]

Key Scripts (DO NOT modify stream_data.py protocol):
├── scripts/stream_data.py    # WebSocket server, streams at 500Hz in batches of 10
├── scripts/serve.py          # Static file server for web apps
├── scripts/download.py       # HuggingFace dataset downloader
└── scripts/control_client.py # Arrow key controls for live evaluation

Example App (intentionally minimal - replace/extend):
└── example_app/
    ├── index.html
    ├── app.js               # WebSocket client, renders 32×32 heatmap
    └── style.css
```

## Data Format

**Neural Data**: 1024 channels (32×32 grid), 500 Hz sampling rate
- WebSocket sends JSON batches: `{type: "sample_batch", neural_data: [[...], ...], start_time_s: float}`
- Channels indexed 0-1023, coordinates are 1-indexed (1,1) to (32,32)

**Ground Truth** (development only, not available in live eval):
- `vx`, `vy`: cursor velocity
- `vx_pos_center_row/col`, `vx_neg_center_row/col`, etc.: tuned region positions

**Difficulty Levels** (see `data/*/README.md` for details):

| Dataset | Noise | Bad Channels | Array Drift |
|---------|-------|--------------|-------------|
| `super_easy` | None | None | Static at (16,16) |
| `easy` | Low white | None | Drifts during recording |
| `medium` | White + 60Hz | None | Static at (16,16) |
| `hard` | High white + 60Hz | 4 dead, 4 artifact | 4 phases (see below) |

**Hard dataset phases** (used for final evaluation):
- Easy (20%): starts at (8,8), static
- Hard (30%): drifts to (26,26)
- Recovery (30%): returns to (16,16)
- Stable (20%): remains at (16,16)

## Signal Processing Guidance

The neural signals encode cursor velocity in specific frequency bands. Recommended pipeline:

1. **Bandpass Filter** (70-150 Hz high-gamma often works well for motor signals)
2. **Power/Envelope Extraction** (squared signal or Hilbert transform)
3. **Temporal Smoothing** (EMA or moving average to reduce flicker)
4. **Reshape to 32×32 Grid**
5. **Spatial Smoothing** (optional Gaussian blur)

Four velocity-tuned regions exist: Vx+, Vx-, Vy+, Vy- (respond to rightward, leftward, upward, downward movement respectively). The goal is identifying stable **areas** of tuned activity, not individual transient spikes.

## Design Constraints

- **Operating Room Environment**: High contrast, readable from 6 feet, zero training required
- **Latency Budget**: <50ms total (network + processing + rendering)
- **Target Frame Rate**: 30-60 FPS for smooth visualization
- **WebSocket Throughput**: ~2 MB/s (50 messages/second × 40KB each)

## What You Can Modify

- `example_app/` - Replace entirely with your solution
- Add custom backend (Python/Node) between data stream and web app
- Use any signal processing or visualization libraries

## What You Cannot Modify

- Data streaming protocol (WebSocket message format)
- How data is transmitted during live evaluation
