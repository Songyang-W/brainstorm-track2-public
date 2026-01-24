# Tracking & UI Improvements - Implementation Summary

## What Was Done

### 1. Kalman Filter for Smooth Center Display

Added a `SimpleKalmanFilter2D` class that smooths the center estimate for UI display:

- **State vector**: `[row, col, vel_row, vel_col]` (position + velocity)
- **Process model**: Constant velocity with configurable noise
- **Measurement**: Raw center from `InterpretableClusterTracker`
- **Effect**: Removes jitter while still allowing abrupt user-driven moves (process noise is set high)

The Kalman filter runs **after** the core tracker, so it does not change which anchors are tracked—only the reported center.

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

Replaced the four-region tracker with a peak-based tracker that
does not assume fixed Vx/Vy regions. It:
- extracts top peaks each frame (robust to noise)
- keeps a persistent set of peak tracks (interpretable anchors)
- outputs a confidence based on track strength + count
- uses a weighted anchor fallback when confidence is low

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

## Current Performance (120-second evaluation)

| Dataset | center_rmse | center_mae | move_cos median |
|---------|-------------|------------|-----------------|
| super_easy | 3.625 | 3.222 | 0.978 |
| medium | 3.464 | 3.081 | 0.979 |
| hard | 4.449 | 3.974 | 0.949 |

Defaults: `ema_tau_s=0.20`, `spatial_sigma=1.2`, `drift_factor=0.1`,
`kalman process_noise=0.2`, `measurement_noise=2.5`

Evaluation uses time-interpolated ground truth (instead of nearest sample)
to reduce timing bias at batch boundaries.

### Key Algorithmic Improvement: Conservative Drift Correction

The tracker no longer propagates global drift to all tracks. Instead:
- Matched tracks are updated directly via EMA
- Unmatched tracks receive only 10% of the estimated drift
- This prevents cumulative error from aggressive drift propagation
- Error no longer accumulates over time (confirmed via time-segment analysis)

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
- Center coordinates
- Confidence percentage
- Distance value
- Number of active regions
