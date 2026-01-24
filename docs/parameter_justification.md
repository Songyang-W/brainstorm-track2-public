# Parameter Justification (Compass)

This document explains why each remaining tunable parameter exists and why its
default value is reasonable for the Track 2 evaluation data.

## Signal Conditioning (scripts/compass_backend.py)

- `notch_freqs = [60, 120] Hz`
  - 60 Hz line noise and its harmonic contaminate ECoG; 120 Hz lies inside the
    high-gamma band and must be suppressed explicitly.
- `notch_Q = 30`
  - Narrow notches remove line peaks while preserving neighboring signal power.
- `bandpass = 70-150 Hz`, `order = 4`
  - High-gamma carries movement-related power; 4th order is a stable causal
    filter with sufficient roll-off without heavy ringing.
- `ema_tau_s = 0.20`
  - ~200 ms temporal smoothing reduces flicker but keeps updates responsive for
    surgical guidance.
- `spatial_sigma = 1.2`
  - Mild spatial smoothing merges neighboring electrodes without blurring out
    localized hotspots.
- `z_cap = 8.0`
  - Limits rare outliers so single-channel spikes cannot dominate the map.

## Bad-Channel Detection (scripts/compass_backend.py)

- `bad_z_hi = 6.0`, `bad_z_lo = -3.0`
  - Robust z-thresholds catch extreme variance (artifacts) and near-flat lines
    (dead/saturated) without over-masking.
- `bad_decay = 0.97`, `bad_gain = 0.03`, `bad_thresh = 0.6`
  - Leaky integrator requires persistence (~0.5–1 s) before masking, preventing
    transient glitches from being treated as dead channels.

## Peak Tracking (scripts/hotspot_tracker.py)

- `peak_n = 6`
  - Captures multiple active regions without overfitting to noise.
- `peak_suppress_radius = 7`, `peak_com_radius = 2`
  - Enforces spatial separation and refines peak position using a local center
    of mass.
- `peak_min_abs = 0.15`
  - Discards weak peaks after normalization to avoid noisy anchors.
- `max_tracks = 10`
  - Enough persistent anchors for hard data while keeping tracking simple.
- `ema_alpha = 0.4`
  - Moderate update rate for track positions (smooth but responsive).
- `max_match_dist = 7.5`
  - Allows for drift across the array without mismatching unrelated peaks.
- `strength_gain = 0.3`, `strength_decay = 0.98`
  - Builds confidence over ~1–2 s and decays stale tracks gradually.
- `max_age = 240`
  - Limits track lifetime to avoid long-lived ghosts.
- `drift_factor = 0.1`
  - Conservative drift correction prevents global drift from accumulating.
- `conf_strength_weight = 0.7`, `conf_count_weight = 0.3`
  - Confidence prioritizes strong anchors while still rewarding multiple peaks.
- `age_tau_updates = 120`
  - Downweights old tracks over a few seconds to keep anchors current.

## Center Smoothing (scripts/hotspot_tracker.py + scripts/compass_backend.py)

- `process_noise = 0.2`, `measurement_noise = 2.5`
  - High process noise avoids assuming a fixed direction while still damping
    jitter; measurement noise provides modest smoothing.
- `confidence scaling`
  - Measurement noise is divided by confidence so low-confidence measurements
    influence the center less.
- `anchor_fallback_conf = 0.35`
  - Below this threshold, a weighted anchor centroid stabilizes the center.
- `conf_alpha = 0.3`
  - EMA on confidence smooths UI state transitions without delaying feedback.

## UI/Server Defaults (scripts/compass_backend.py)

- `ui_hz = 15`
  - Keeps the UI responsive while keeping CPU and websocket bandwidth stable.
- `heatmaps = true`
  - Debug-friendly for local development; can be disabled to reduce payload.
