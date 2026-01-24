# BrainStorm Track 2 - Implementation Plan

## The Compass: Global Brain Map with Array Guidance System

### Executive Summary

This document outlines the complete overhaul of `example_app` to create an OR-ready neurosurgical guidance system for Kat, our Clinical Operator persona. The system will:

1. **Build a persistent global brain map** that accumulates activity over time
2. **Track cluster movements** to infer array position changes
3. **Provide directional guidance** showing the surgeon where to move for optimal placement
4. **Display a clear "Found It" signal** when the array is well-positioned

---

## User Persona: Kat (Clinical Operator)

**Key Requirements from User Persona:**
- PhD Neuroscience, 200+ surgeries experience
- Needs to interpret data from **6 feet away** in a crowded OR
- Expects **raw data fidelity** over aesthetics
- Requires **zero-training interface** - instantly interpretable
- Must provide **precise verbal directions** to surgeon (e.g., "Shift 3mm medial")
- Critical alerts must be **impossible to miss**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────┘

   WebSocket Server (ws://192.168.1.152:8765/stream)
                     │
                     ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      INCOMING MESSAGE TYPES                             │
   │  • init: channels_coords, grid_size, fs, batch_size                     │
   │  • sample_batch: neural_data[10×1024], cursor_data[{vx, vy}×10]        │
   └─────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    SIGNAL PROCESSING PIPELINE                           │
   │                                                                          │
   │  1. Bandpass Filter (70-150 Hz high-gamma)                              │
   │  2. Power Computation (squared amplitude)                                │
   │  3. Temporal Smoothing (EMA, alpha=0.2)                                 │
   │  4. Reshape to 32×32 grid                                               │
   │  5. Spatial smoothing (Gaussian, sigma=1.5)                             │
   └─────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    GLOBAL BRAIN MAP SYSTEM                              │
   │                                                                          │
   │  • Persistent activity accumulation (decaying weighted average)         │
   │  • Cluster detection (connected component analysis)                      │
   │  • Cluster tracking (frame-to-frame correspondence)                      │
   │  • Array movement inference (cluster motion → inverse array motion)     │
   │  • Global coordinate system maintenance                                  │
   └─────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      VISUALIZATION OUTPUTS                              │
   │                                                                          │
   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
   │  │ GLOBAL MAP   │  │ CURRENT VIEW │  │  GUIDANCE    │                  │
   │  │              │  │              │  │              │                  │
   │  │ • Full brain │  │ • Live 32×32 │  │ • Direction  │                  │
   │  │   coverage   │  │   heatmap    │  │   arrow      │                  │
   │  │ • Best       │  │ • Current    │  │ • Distance   │                  │
   │  │   clusters   │  │   hotspots   │  │   indicator  │                  │
   │  │ • Array      │  │ • Cursor     │  │ • "Found It" │                  │
   │  │   position   │  │   velocity   │  │   signal     │                  │
   │  └──────────────┘  └──────────────┘  └──────────────┘                  │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Algorithm: Global Brain Map with Array Tracking

### 1. Cluster Detection

```javascript
// For each frame, detect clusters in the current 32×32 grid
function detectClusters(grid, threshold = 0.3) {
    // 1. Threshold the grid to binary
    // 2. Find connected components
    // 3. For each component:
    //    - Calculate centroid (intensity-weighted)
    //    - Calculate total intensity
    //    - Calculate area (number of cells)
    //    - Calculate bounding box
    return clusters; // [{centroid, intensity, area, bbox}, ...]
}
```

### 2. Cluster Tracking (Frame-to-Frame)

```javascript
// Track clusters across frames to understand movement
function trackClusters(prevClusters, currClusters) {
    // Hungarian algorithm for optimal assignment
    // Cost = distance between centroids + intensity difference
    // Track: which previous cluster matches which current cluster
    // Detect: new clusters, disappeared clusters
    return {
        matches: [{prev, curr, displacement}],
        newClusters: [...],
        lostClusters: [...]
    };
}
```

### 3. Array Movement Inference

**Key Insight**: If a cluster moves LEFT on the array view, the array is moving RIGHT over the brain.

```javascript
function inferArrayMovement(clusterTracking) {
    // Weighted average of cluster displacements
    // Weight by cluster intensity (brighter = more reliable)
    const avgDisplacement = weightedAverageDisplacement(clusterTracking.matches);

    // Invert to get array movement
    return {
        arrayDx: -avgDisplacement.dx,
        arrayDy: -avgDisplacement.dy
    };
}
```

### 4. Global Map Accumulation

```javascript
class GlobalBrainMap {
    constructor(size = 128) {
        this.map = new Float32Array(size * size); // Extended canvas
        this.arrayPosition = {x: 64, y: 64}; // Current array position on global map
        this.clusters = []; // Tracked best clusters in global coordinates
    }

    update(localGrid, inferredMovement, cursorVelocity) {
        // 1. Update array position
        this.arrayPosition.x += inferredMovement.arrayDx;
        this.arrayPosition.y += inferredMovement.arrayDy;

        // 2. Place current local grid onto global map
        this.placeLocalGrid(localGrid);

        // 3. Apply decay to old values (persistence with fade)
        this.applyDecay(0.995);

        // 4. Identify and track best clusters globally
        this.updateGlobalClusters();
    }
}
```

---

## UI Design: OR-Ready Interface

### Layout (Designed for 6-Foot Viewing)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BRAINSTORM COMPASS                                    ● CONNECTED  12 FPS  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────┐    ┌───────────────────────────────┐    │
│  │                               │    │                               │    │
│  │                               │    │                               │    │
│  │       GLOBAL BRAIN MAP        │    │      CURRENT ARRAY VIEW       │    │
│  │                               │    │                               │    │
│  │    (128×128 accumulated)      │    │       (32×32 live)            │    │
│  │                               │    │                               │    │
│  │    [Array position shown]     │    │    [Hotspots + velocity]      │    │
│  │    [Best clusters marked]     │    │                               │    │
│  │                               │    │                               │    │
│  └───────────────────────────────┘    └───────────────────────────────┘    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                                                                       │ │
│  │                        GUIDANCE PANEL                                 │ │
│  │                                                                       │ │
│  │    ┌─────────────┐      ┌─────────────────────────────────────────┐  │ │
│  │    │             │      │  MOVE:  ← 3.2mm LEFT                    │  │ │
│  │    │   ◄════     │      │         ↑ 1.5mm SUPERIOR                │  │ │
│  │    │             │      │                                          │  │ │
│  │    │  DIRECTION  │      │  CONFIDENCE: ████████░░ 78%             │  │ │
│  │    │   ARROW     │      │  DISTANCE TO TARGET: 3.5mm              │  │ │
│  │    │             │      │                                          │  │ │
│  │    └─────────────┘      └─────────────────────────────────────────┘  │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  ▲ VX: +42.3  VY: -18.7  │  STATUS: SEARCHING  │  BEST @ 3.2mm NW   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Color Scheme (High Contrast for OR)

```css
--bg-dark: #000000;           /* Pure black background */
--grid-bg: #0a0a12;           /* Slightly lighter for grid areas */
--activity-low: #1a1a3a;      /* Dim activity */
--activity-mid: #4a2060;      /* Medium activity */
--activity-high: #ff6b35;     /* High activity - ORANGE (visible) */
--activity-peak: #ffff00;     /* Peak activity - YELLOW (max contrast) */
--guidance-arrow: #00ff88;    /* Green guidance arrow */
--found-it: #00ff00;          /* Bright green "FOUND IT" */
--warning: #ff4444;           /* Red warnings */
--text-primary: #ffffff;      /* White text */
--text-secondary: #888899;    /* Gray secondary text */
```

### "Found It" Signal

When the array is optimally positioned (>80% of best cluster within view with high intensity):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                        ████  FOUND IT  ████                                │
│                                                                             │
│                    OPTIMAL POSITION REACHED                                 │
│                                                                             │
│                        [Pulsing green glow]                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Files

### File Structure

```
example_app/
├── index.html           # Main HTML with OR-ready layout
├── style.css            # High-contrast OR styling
├── app.js               # Main application logic
├── signal-processor.js  # DSP: bandpass filter, power computation
├── global-map.js        # Global brain map accumulation & tracking
├── cluster-tracker.js   # Cluster detection and frame-to-frame tracking
├── guidance-system.js   # Direction calculation and "Found It" logic
└── visualizer.js        # Canvas rendering for all panels
```

---

## Data Stream Integration

### Connecting to Test Server

```javascript
// Test time connection
const WS_URL = 'ws://192.168.1.152:8765/stream';

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'sample_batch') {
        // Extract cursor velocity if present
        if (data.cursor_data) {
            const avgVx = data.cursor_data.reduce((s, d) => s + d.vx, 0) / data.cursor_data.length;
            const avgVy = data.cursor_data.reduce((s, d) => s + d.vy, 0) / data.cursor_data.length;
            updateCursorVelocity(avgVx, avgVy);
        }

        // Process neural data
        processNeuralBatch(data.neural_data);
    }
};
```

### Arrow Key Control Integration

The array movement is controlled via `brainstorm-control`. Our UI will:
1. **Display inferred array position** from cluster tracking
2. **Show direction to best clusters** regardless of control method
3. **React in real-time** to position changes

---

## Signal Processing (JavaScript Implementation)

### Bandpass Filter (Butterworth, 70-150 Hz)

```javascript
class BandpassFilter {
    constructor(fs = 500, lowcut = 70, highcut = 150, order = 4) {
        // Pre-computed Butterworth coefficients
        this.coeffs = this.computeCoefficients(fs, lowcut, highcut, order);
        this.state = new Float32Array(order * 2 * 1024); // Filter state per channel
    }

    process(samples) {
        // Apply IIR filter with state preservation
        return this.applyFilter(samples);
    }
}
```

### Power & Smoothing

```javascript
class PowerEstimator {
    constructor(alpha = 0.2) {
        this.alpha = alpha;
        this.smoothedPower = null;
    }

    process(filteredSamples) {
        // Square for power
        const power = filteredSamples.map(s => s * s);

        // Average across batch
        const avgPower = this.averageAcrossBatch(power);

        // EMA smoothing
        if (!this.smoothedPower) {
            this.smoothedPower = avgPower;
        } else {
            for (let i = 0; i < avgPower.length; i++) {
                this.smoothedPower[i] = this.alpha * avgPower[i] +
                                        (1 - this.alpha) * this.smoothedPower[i];
            }
        }

        return this.smoothedPower;
    }
}
```

---

## Success Metrics

### Technical
- [ ] <50ms total latency (target: <30ms)
- [ ] 30+ FPS rendering
- [ ] Accurate cluster detection on hard dataset
- [ ] Reliable array movement inference

### UX (Kat's Requirements)
- [ ] Readable from 6 feet away
- [ ] Zero-training interpretability
- [ ] Clear directional guidance
- [ ] Unambiguous "Found It" signal
- [ ] High contrast visibility under OR lights

---

## Next Steps

1. **Implement signal processing pipeline** (signal-processor.js)
2. **Build cluster detection system** (cluster-tracker.js)
3. **Create global map accumulation** (global-map.js)
4. **Develop guidance algorithm** (guidance-system.js)
5. **Design and implement OR-ready UI** (index.html, style.css)
6. **Integrate visualization** (visualizer.js, app.js)
7. **Test with hard dataset**
8. **Connect to test server for final evaluation**
