# Simplification Summary (Before/After)

## What Was Removed (and Why)

- **Memory heatmap + memory map tracking**
  - Removed because it was only used for debug visualization; it did not improve
    center tracking accuracy.
- **Suppressed/extra heatmaps in the UI**
  - Removed because the UI only needs the live activity map for debugging.
- **`spots` and `spots_mem` payload fields**
  - Removed to reduce payload size and simplify the frontend; no effect on UI.
- **Multiple notch filters list structure**
  - Replaced with a single stacked SOS filter for 60/120 Hz to keep the filter
    bank compact and easy to reason about.

## What Stayed (and Why)

- **Kalman center smoothing**
  - Keeps UI guidance stable while still allowing abrupt user-driven moves.
- **Peak-based tracker with persistent tracks**
  - Avoids fixed Vx/Vy assumptions and stays robust in hard data.
- **Bad-channel detection**
  - Needed to prevent dead/artifact channels from dominating the map.

## Evaluation (120s)

| Dataset | center_rmse | center_mae | move_cos median |
|---------|-------------|------------|-----------------|
| super_easy | 3.625 | 3.222 | 0.978 |
| medium | 3.464 | 3.081 | 0.979 |
| hard | 4.449 | 3.974 | 0.949 |
