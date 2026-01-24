# Technical Approach: BrainStorm Track 2 - Neural Hotspot Tracker

## The Problem

Surgeons placing brain-computer interface arrays need rapid feedback about where motor cortex activity is strongest. In the operating room they cannot use a keyboard, so the system must reduce 1024 channels of neural data streaming at 500 Hz into an at-a-glance cue: move left, move right, or hold position. Conventional research views (raw traces, spectra) are too dense for this setting.

---

## Signal Processing Pipeline

### Why High-Gamma (70-150 Hz)

With ECoG we cannot resolve individual spikes, but high-gamma power (70–150 Hz) is a strong proxy for local population firing and is relatively spatially localized. Other bands were less spatially specific for this task, so we focus on high-gamma.

### The Filter Chain

**1. Bandpass (IIR Butterworth)**

Fourth-order Butterworth, 70–150 Hz. A stateful IIR filter keeps latency low and supports continuous streaming:
- Stable behavior after initialization
- Consistent phase across the stream
- Low compute cost per channel

The JavaScript implementation uses pre-computed biquad coefficients and maintains zi state vectors for all 1024 channels. The Python implementation uses scipy's sosfilt for numerical stability.

**2. Power Computation**

Compute mean power over each batch window by squaring the filtered signal and averaging. A Hilbert envelope is possible but adds complexity for limited benefit here.

**3. Temporal Smoothing (EMA)**

EMA with α = 0.2 reduces flicker while preserving movement onset visibility (≈100 ms).

**4. Spatial Smoothing (Gaussian)**

Gaussian blur with σ = 1.5 pixels on the 32×32 grid encourages local spatial coherence and suppresses isolated artifacts.

**5. Percentile Normalization**

Clip at the 95th percentile and rescale to [0,1] so outliers (artifacts, bad channels) do not dominate the display.

---

## The UNet Denoiser

### Why an Autoencoder

The signal processing pipeline is effective, but the hard dataset includes 60 Hz line noise leakage, intermittent channel spikes, and subtle hotspots near the noise floor.

A small U-Net autoencoder learns a prior over plausible hotspot geometry from clean training data. It maps a noisy 32×32 grid to a denoised 32×32 confidence map; skip connections preserve detail while the bottleneck enforces structure.

### Architecture

```
Input (1, 32, 32)
   │
   ├─► Conv 1→16→16 + ReLU ────────────────────────────┐
   │              │                                     │
   ▼              │                                     │
MaxPool 2x2       │                                     │
   │              │                                     │
   ├─► Conv 16→32→32 + ReLU ───────────┐               │
   │              │                     │               │
   ▼              │                     │               │
MaxPool 2x2       │                     │               │
   │              │                     │               │
   ▼              │                     │               │
Conv 32→64 (bottleneck)                 │               │
   │                                    │               │
   ▼                                    │               │
ConvTranspose 64→32 ◄──── concat ◄──────┘               │
   │                                                    │
   ▼                                                    │
ConvTranspose 32→16 ◄──── concat ◄──────────────────────┘
   │
   ▼
Conv 16→1 + Sigmoid
   │
   ▼
Output (1, 32, 32) confidence map
```

Two pooling layers yield an 8×8 bottleneck. The model is ~274 KB and runs in <5 ms on CPU.

### What It Learns

The U-Net is trained on pairs of noisy grids and clean targets, and learns common structure:
- Hotspots are spatially coherent blobs
- Single-pixel spikes are typically noise
- Smooth falloff is more reliable than sharp edges

---

## Cluster Tracking: The Hard Problem

The electrode array can drift during surgery. Hotspots may shift in grid coordinates even when stable in brain coordinates, so tracking must separate true changes from sensor motion.

### Anchored Peaks

Each detected peak gets an AnchoredPeak object that tracks:
- **Position**: smoothed (row, col) on the grid
- **Age**: how many frames it has been alive
- **Activation count**: how often it has been detected
- **Confidence**: running average of intensity at this location

We increase inertia with age: early detections are allowed to move (and die) quickly, while persistent peaks become stable. This is implemented with age-based smoothing weights:

```javascript
const ageWeight = Math.min(this.age / 30, 0.95);
this.row = ageWeight * this.row + (1 - ageWeight) * observedRow;
```

Young peaks (age < 30) are responsive; older peaks are stable.

### Distance Penalties

Observations far from a peak are unlikely to be the same hotspot. We down-weight matches using a distance penalty:

```javascript
const dist = Math.sqrt((dr * dr) + (dc * dc));
const penalty = Math.exp(-dist / 5);  // decay with distance
observation.weight *= penalty;
```

This prevents distant outliers from pulling the centroid.

### Cluster Entities

Multiple peaks that consistently appear together form a ClusterEntity. The cluster maintains:
- A set of all peaks (active and inactive)
- A weighted centroid computed from all peaks
- A smoothed centroid for extra stability
- Structure stability metric (how much the peak configuration changes)

The critical behavior: **clusters remember structure even when peaks turn off**. If you have 4 peaks in a diamond pattern, and 2 go quiet temporarily, the centroid stays in the middle of the diamond—not the midpoint of the 2 remaining peaks. This prevents jitter.

---

## Global Mapping: Persistent Memory

### The Coordinate Problem

As the array moves, the 32×32 grid covers a different cortical patch. The system needs a short-term memory of where hotspots were relative to the current array position.

GlobalMap maintains hotspots in brain-relative coordinates. When we see a cluster at grid position (12, 8), we store it at global position (grid_offset_x + 12, grid_offset_y + 8). As the array drifts, we update the offset, so the global positions stay fixed relative to the brain.

### Structure Preservation

GlobalClusters store all peaks, not just currently active ones. If a hotspot goes quiet briefly, we mark it inactive but retain its position so re-activation matches the same global cluster.

This matters because the four tuning regions (Vx+, Vx-, Vy+, Vy-) can activate at different times; persistent tracking avoids dropping identities between activations.

---

## Velocity Decoding

### Center of Mass

The velocity decoder computes an intensity-weighted center of mass; displacement from center sets direction and magnitude.

```python
raw_vx = (avg_col - 15.5) / 15.5  # normalize to [-1, 1]
raw_vy = -(avg_row - 15.5) / 15.5  # invert Y (row 0 is top)
```

### Intensity-Squared Weighting

We square the intensity before computing COM to emphasize strong hotspots and suppress diffuse background.

### Exponentiation for Small Displacements

COM displacements can be small, so we apply a sign-preserving power function to expand the dynamic range:

```python
exaggeration = 0.6
vx = sign(raw_vx) * abs(raw_vx) ** exaggeration
```

This makes small displacements more visible without clipping large ones.

---

## Interface Design

### Operating Room Constraints

The surgeon is:
- Standing, hands occupied, cannot touch controls
- Looking at the patient, glances at screen peripherally
- 6+ feet from the display
- Under time pressure, cannot interpret complex visualizations

### Design Decisions

**Heatmap, not numbers**: Color encodes intensity with minimal annotation.

**Large panel layout**: Four quadrants, each at least 300×300 pixels, for visibility at distance.

**Minimal text**: Frame count, connection status, and current velocity.

**Audio feedback**: Click rate increases near hotspots; a steady tone indicates on-target.

**Flexible layouts**: Seven presets (tabs, splits, grid, main+side) with one-click switching.

### Panel Types

- **Live heatmap**: Current denoised activity, updated 30 FPS
- **Global map**: Persistent brain-relative hotspot positions
- **Trajectory**: Cursor path based on decoded velocity
- **Velocity arrow**: Current movement direction and magnitude
- **Metrics**: Correlation to ground truth (for development)

---

## Performance

### Latency Budget

- Network: <10 ms (local WebSocket)
- Processing: <20 ms (filter + denoise + track)
- Rendering: <16 ms (one frame at 60 FPS)
- Total: <50 ms

In practice, the main limiter is typically the browser's `requestAnimationFrame` cadence rather than processing.

### Throughput

- 500 Hz sampling
- 10-sample batches (50 messages/sec)
- ~40 KB per message
- ~2 MB/sec sustained

This throughput fits comfortably within a local WebSocket stream without compression.

### Statefulness

IIR filters, EMA smoothers, and trackers are stateful:
- Stable behavior after initialization
- Consistent output over time
- Requires in-order processing (expected for real-time streaming)

---

## What Makes This Work

1. **Right frequency band**: High-gamma is the correct physiological target
2. **Stateful filtering**: Smooth output, no transients
3. **Learned denoising**: U-Net removes artifacts without destroying structure
4. **Age-based anchoring**: Old peaks do not jump around
5. **Structure preservation**: Clusters maintain shape when peaks go quiet
6. **Global coordinates**: Hotspots stay fixed relative to brain, not array
7. **Audio feedback**: Surgeon can work without looking

Combining classical signal processing, learned denoising, and motion-aware tracking yields stable, interpretable hotspot localization in real time, presented in a surgeon-appropriate interface.
