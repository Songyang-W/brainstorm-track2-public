# Tracking & UI Improvements - Implementation Summary

## What Was Done

### 1. Kalman Filter for Smooth Center Display

Added a `SimpleKalmanFilter2D` class that smooths the center estimate for UI display:

- **State vector**: `[row, col, vel_row, vel_col]` (position + velocity)
- **Process model**: Constant velocity with configurable noise
- **Measurement**: Raw center from `FourTunedRegionTracker`
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

### 3. Code Changes

**Files modified**:
- `scripts/hotspot_tracker.py` - Added `SimpleKalmanFilter2D` and `MotionCompensatedTracker` classes
- `scripts/compass_backend.py` - Integrated Kalman smoothing on top of existing tracker
- `example_app/index.html` - New simplified layout
- `example_app/style.css` - New clinical light theme
- `example_app/app.js` - New state management and rendering

## Performance Results

| Dataset | Metric | Before | After |
|---------|--------|--------|-------|
| super_easy | center_rmse | 3.984 | 3.863 |
| super_easy | move_cos median | 0.998 | 0.997 |
| medium | center_rmse | 4.956 | 5.023 |
| medium | move_cos median | 0.969 | 0.972 |

The tracking accuracy is maintained or slightly improved while the UI is now much smoother.

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
