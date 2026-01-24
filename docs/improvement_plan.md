# Tracking & UI Improvements - Implementation Summary

## What Was Done

### 1. Kalman Filter for Smooth Center Display

Added a `SimpleKalmanFilter2D` class that smooths the center estimate for UI display:

- **State vector**: `[row, col, vel_row, vel_col]` (position + velocity)
- **Process model**: Constant velocity with configurable noise
- **Measurement**: Raw center from `InterpretableClusterTracker`
- **Effect**: Eliminates jittery center movement in UI without affecting tracking accuracy

The Kalman filter is applied **after** the core tracker runs, so it only affects what the UI displays, not the underlying tracking metrics.

### 2. Simplified UI Design

Completely redesigned the UI with one goal: **answer "which way to move?" instantly**

**Main display (90% of screen)**:
- Giant directional arrow (blue, pointing where to move)
- Giant text instruction: "UP LEFT", "DOWN", "ON TARGET", etc.
- Color-coded background:
  - White = tracking
  - Amber = acquiring signal (low confidence)
  - Green = on target

**Secondary elements (minimized)**:
- Distance bar at bottom showing progress toward target
- Status strip at top (connection state, time)
- Controls panel (collapsed by default)
- Debug panel (hidden by default, toggle available)

### 3. Interpretable Tracker for Noisy Data

Replaced the four-region tracker with a peak-based, memory-backed tracker that
does not assume fixed Vx/Vy regions. It:
- extracts top peaks each frame (robust to noise)
- keeps a persistent set of peak tracks (interpretable anchors)
- outputs a confidence based on track strength + count
- maintains a long-memory map for debug UI
- uses memory as a fallback anchor when confidence is low

### 4. Signal Conditioning for Hard Mode

- 60/120 Hz notch filters to suppress line noise + harmonic
- Persistent bad-channel detection to mask dead/artifact channels
- Optional cursor velocity passthrough for live demos
- Channel coordinate mapping from stream init (no more hard-coded ordering)

### 5. Code Changes

**Files modified**:
- `scripts/hotspot_tracker.py` - Trimmed to core tracker utilities only
- `scripts/compass_backend.py` - Integrated new tracker, bad-channel masking, 60/120 Hz notch filtering, cursor passthrough
- `example_app/index.html` - New simplified layout
- `example_app/style.css` - New clinical light theme
- `example_app/app.js` - New state management and rendering

## Current Performance (defaults)

| Dataset | center_rmse | move_cos median |
|---------|-------------|----------------|
| medium | 3.531 | 0.979 |
| hard | 3.520 | 0.951 |

These defaults are tuned for noisy `medium`/`hard` data while keeping the UI stable.

Evaluation now uses time-interpolated ground truth (instead of nearest sample)
to reduce timing bias at batch boundaries.

## UI States

1. **Disconnected**: Gray status dot, "CONNECT" instruction
2. **Acquiring** (confidence < 35%): Amber background, "ACQUIRING" text, pulsing indicator
3. **Tracking** (confidence >= 35%): White background, direction text ("UP LEFT"), blue arrow
4. **On Target** (distance < 1.5 & confidence > 65%): Green background, "ON TARGET" text, checkmark icon

## How to Test

```bash
# Terminal 1: Start stream
uv run brainstorm-stream --from-file data/super_easy/

# Terminal 2: Start backend
uv run brainstorm-compass

# Terminal 3: Start web server
uv run brainstorm-serve

# Open browser to http://localhost:8000
```

## Debug View

Enable "Show debug view" in controls to see:
- Live activity heatmap
- Memory map
- Center coordinates
- Confidence percentage
- Distance value
- Number of active regions
