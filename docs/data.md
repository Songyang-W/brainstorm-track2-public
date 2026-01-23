# Data

This page describes the **datasets**, **file formats**, and **signal content**.

For the WebSocket streaming protocol, see [Data Stream](data_stream.md). For signal processing guidance, see [Getting Started](getting_started.md).

## Datasets

Four difficulty levels on [HuggingFace](https://huggingface.co/datasets/PrecisionNeuroscience/BrainStorm2026-track2):

| Difficulty | Signal Quality | Noise | Artifacts | Use Case |
|------------|---------------|-------|-----------|----------|
| **super_easy** | Crystal-clear | None | None | Understanding the signal |
| **easy** | Clean | Low | None | Initial development |
| **medium** | Moderate | Medium | None | Robustness testing |
| **hard** | Challenging | High | Multiple | **Final testing & live evaluation** |

### Dataset Parameters

| Difficulty | Spots/Region | Noise | Background | Bad Channels |
|------------|--------------|-------|------------|--------------|
| **super_easy** | 1 (centered) | None | None | None |
| **easy** | 3 (scattered) | Low white noise | Light | None |
| **medium** | 5 (scattered) | White + 60Hz line | Moderate | None |
| **hard** | 15 (scattered) | White + 60Hz line | Moderate | 4 dead, 4 artifact |

Each difficulty folder includes a `README.md` with exact generation parameters.

> **Note**: All difficulty levels use **identical array movement patterns**, allowing direct comparison of how noise affects your algorithm.

### Array Movement (Drift)

The simulated array drifts across the cortex in a corner cycle pattern:

| Position | Grid Coordinates [row, col] |
|----------|-----------------------------|
| Center | [16, 16] |
| Top-right | [26, 26] |
| Bottom-right | [26, 6] |
| Bottom-left | [6, 6] |
| Top-left | [6, 26] |

- Each transition takes **5 seconds**
- Cycle: center ↔ corner ↔ center ↔ next corner...
- **Drift starts at t=0** — tuned regions move from the first sample

### Downloading

```bash
uv run python -m scripts.download hard
```

```python
from scripts.download import download_track2_data, load_track2_data

download_track2_data("hard")
data, ground_truth = load_track2_data("hard")
```

## File Formats

### `track2_data.parquet`

Neural signal data:

| Property | Description |
|----------|-------------|
| **Index** | `time_s` — timestamp in seconds |
| **Columns** | Channel numbers `0` to `1023` |
| **Shape** | `(n_samples, 1024)` |
| **dtype** | `float32` |

```python
import pandas as pd

data = pd.read_parquet("data/hard/track2_data.parquet")
print(data.shape)       # (17500, 1024) for 35s at 500Hz
print(data.index.name)  # 'time_s'

sample = data.loc[1.0]  # Sample at t=1.0s
channel_42 = data[42]   # All samples from channel 42
```

### `ground_truth.parquet`

Cursor kinematics and tuned region positions (development only):

| Column | Description | Units |
|--------|-------------|-------|
| `time_s` | Timestamp | seconds |
| `vx`, `vy` | Cursor velocity | arbitrary |
| `vx_pos_center_row/col` | X+ region position | grid (1-32) |
| `vx_neg_center_row/col` | X- region position | grid (1-32) |
| `vy_pos_center_row/col` | Y+ region position | grid (1-32) |
| `vy_neg_center_row/col` | Y- region position | grid (1-32) |

> **Note**: Ground truth is for development only. During live evaluation, you will NOT have access to these positions.

### Channel Coordinates

Channels are arranged in a 32×32 grid. Coordinates are `[row, col]` pairs (1-indexed):

```
Channel 0:    (1, 1)   — top-left
Channel 31:   (1, 32)  — top-right  
Channel 992:  (32, 1)  — bottom-left
Channel 1023: (32, 32) — bottom-right
```

## Signal Content

### Velocity-Tuned Regions

The data simulates a cursor control task with four velocity-tuned neural regions:

| Region | Responds To | Description |
|--------|-------------|-------------|
| **Vx+** | +X velocity | Active when cursor moves right |
| **Vx-** | -X velocity | Active when cursor moves left |
| **Vy+** | +Y velocity | Active when cursor moves up |
| **Vy-** | -Y velocity | Active when cursor moves down |

Each region consists of multiple "spots" with Gaussian spatial profiles. When the cursor moves in a direction, channels near the corresponding region show increased activity.

### Understanding the Signal

The neural activity pattern depends on **both**:
1. **Cursor direction** — Which tuned region is active
2. **Array position** — Where the tuned regions appear on the grid

This means the pattern constantly changes. Your solution should identify **stable areas** of tuned activity, not chase individual transient spikes.

## Physiological Background

### Motor Cortex Encoding

Motor cortex neurons are "tuned" to movement aspects:
- **Directional tuning** — Regions respond preferentially to specific movement directions
- **Velocity encoding** — Activity correlates with movement speed
- **Spatial organization** — Nearby electrodes have similar tuning

### ECoG Signal Characteristics

Electrocorticography records electrical potentials from the cortical surface:

- **Broadband activity** — Both oscillatory and non-rhythmic components
- **Frequency bands carry different information**:
  - **< 30 Hz** — Larger-scale network dynamics, movement preparation
  - **> 50 Hz** — Often correlates with local neural activity, movement execution
- **Movement-related changes** — Systematic power changes during movement

### Key Frequency Bands

| Band | Frequency | Often Associated With |
|------|-----------|----------------------|
| Theta/Alpha | 4-12 Hz | Movement planning, attention |
| Beta | 12-30 Hz | Movement preparation/suppression |
| Low Gamma | 30-70 Hz | Local cortical processing |
| **High Gamma** | 70-150 Hz | **Motor execution, local firing** |

> **Tip**: High-gamma (70-150 Hz) power is frequently used in BCI research as it correlates well with movement intent. This is often a good starting point for feature extraction.

### Relevant Literature

For deeper understanding:
- **Schalk & Leuthardt (2011)** — "Brain-Computer Interfaces Using Electrocorticographic Signals" (review)
- **Crone et al. (1998)** — Functional mapping with ECoG spectral analysis
- **Miller et al. (2007, 2009)** — Broadband/high-frequency activity in human motor cortex
- **Wolpaw & Wolpaw (2012)** — "Brain-Computer Interfaces: Principles and Practice"

> You don't need to read these to complete the challenge, but they provide useful context.

## Noise and Artifacts

Depending on difficulty level:

| Type | Description | How to Identify |
|------|-------------|-----------------|
| **White Noise** | Gaussian noise on all channels | Random fluctuations |
| **Line Noise** | 60 Hz interference | Periodic oscillation |
| **Dead Channels** | Flat signal | Zero or constant values |
| **Saturated Channels** | Stuck at rail voltage | Constant extreme values |
| **Artifact Channels** | Excessive noise | High variance, non-physiological |

Your application should detect and handle problematic channels automatically.

## Data Usage

### Local Development

```bash
# Start with super_easy to understand the signal
uv run brainstorm-stream --from-file data/super_easy/

# Develop with hard (matches live evaluation)
uv run brainstorm-stream --from-file data/hard/
```

### Live Evaluation

During evaluation, data streams from our server with the same format but no ground truth access. See [Submissions](submissions.md) for the live evaluation workflow.
