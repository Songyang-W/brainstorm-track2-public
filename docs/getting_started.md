# Getting Started

This guide walks you through developing your Track 2 solution.

## Prerequisites

Complete the [Installation](installation.md) steps first.

## Step 1: Understand the Architecture

You have flexibility in how you build your solution:

| Component | Description | Modify? |
|-----------|-------------|---------|
| `stream_data.py` | Streams data from files | ❌ No |
| `serve.py` | Static file server | ✅ Optional |
| `example_app/` | Basic visualization | ✅ Yes |
| Your backend | Custom processing (optional) | ✅ Yes |

**Two approaches:**

1. **Direct connection** — Web app connects directly to the data stream and processes in JavaScript (like the example app)
2. **Custom backend** — Build middleware in Python/Node that processes data before sending to your web app

See [Data Stream](data_stream.md) for architecture details.

## Step 2: Run the Example

**Terminal 1:**
```bash
uv run brainstorm-stream --from-file data/hard/
```

**Terminal 2:**
```bash
uv run brainstorm-serve
```

Open http://localhost:8000. The example app is intentionally basic — **your solution should be much better!**

## Step 3: Understand the Data

### Neural Signals

1024 channels arranged in a 32×32 grid, sampled at 500 Hz:

```python
import pandas as pd

data = pd.read_parquet("data/hard/track2_data.parquet")
print(data.shape)  # (n_samples, 1024)
```

### Ground Truth (Development Only)

For development, you have cursor kinematics and tuned region positions:

```python
gt = pd.read_parquet("data/hard/ground_truth.parquet")

# Cursor velocity
vx, vy = gt['vx'], gt['vy']

# Tuned region positions (row, col in 1-32 range)
vx_pos_row = gt['vx_pos_center_row']
vx_pos_col = gt['vx_pos_center_col']
```

> **Important**: During live evaluation, you will NOT have access to ground truth. Your algorithm must identify hotspots from neural signals alone.

### Signal Content

The data simulates cursor movement with velocity-tuned neural regions:

| Region | Responds To |
|--------|-------------|
| Vx+ | Rightward movement (+X velocity) |
| Vx- | Leftward movement (-X velocity) |
| Vy+ | Upward movement (+Y velocity) |
| Vy- | Downward movement (-Y velocity) |

The neural activity pattern changes based on cursor direction — when the cursor moves right, the Vx+ region activates; when it moves left, Vx- activates. Your solution should identify **areas** of tuned activity, not individual transient spikes.

See [Data](data.md) for detailed format and signal content documentation.

## Step 4: Signal Processing

This is where the challenge lies. The neural signals encode cursor velocity information that you need to extract in real-time.

### Key Concepts for Real-Time Processing

**1. Bandpass Filtering**

ECoG signals contain information across different frequency bands. Motor-related activity is often strongest in specific bands:

```python
from scipy.signal import butter, sosfiltfilt
import numpy as np

def bandpass_filter(data, lowcut, highcut, fs=500, order=4):
    """Apply bandpass filter to extract frequency band of interest."""
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=0)

# Example: Extract high-gamma band (often motor-related)
filtered = bandpass_filter(neural_data, lowcut=70, highcut=150, fs=500)
```

Common frequency bands to explore:
- **Theta/Alpha (4-12 Hz)**: Slower rhythms, movement planning
- **Beta (12-30 Hz)**: Movement preparation/suppression
- **Low Gamma (30-70 Hz)**: Local processing
- **High Gamma (70-150 Hz)**: Often correlates with local neural firing, motor execution

> **Tip**: High-gamma power is frequently used in BCI research as it correlates well with movement intent. Start there!

**2. Power Estimation**

After filtering, compute the signal power (amplitude envelope):

```python
def compute_power(filtered_data, window_ms=100, fs=500):
    """Compute smoothed power envelope."""
    # Square the signal
    power = filtered_data ** 2
    
    # Smooth with moving average
    window_samples = int(window_ms * fs / 1000)
    kernel = np.ones(window_samples) / window_samples
    smoothed = np.convolve(power, kernel, mode='same')
    
    return smoothed
```

Or use the Hilbert transform for the analytic amplitude:

```python
from scipy.signal import hilbert

def compute_envelope(filtered_data):
    """Compute amplitude envelope using Hilbert transform."""
    analytic = hilbert(filtered_data, axis=0)
    return np.abs(analytic)
```

**3. Temporal Smoothing**

Real-time visualizations need smoothing to avoid flickering:

```python
class ExponentialSmoother:
    """Exponential moving average for real-time smoothing."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Higher = faster response, more noise
        self.state = None
    
    def update(self, new_value):
        if self.state is None:
            self.state = new_value
        else:
            self.state = self.alpha * new_value + (1 - self.alpha) * self.state
        return self.state
```

**4. Spatial Analysis**

The tuned regions are spatially localized. Consider:
- Spatial smoothing (Gaussian blur over the 32×32 grid)
- Clustering to identify coherent regions
- Computing regional statistics

```python
from scipy.ndimage import gaussian_filter

def spatial_smooth(grid_data, sigma=1.5):
    """Apply Gaussian smoothing to 32x32 grid."""
    return gaussian_filter(grid_data.reshape(32, 32), sigma=sigma)
```

### Real-Time Processing Pipeline

A typical pipeline for real-time processing:

```
Raw Data (500 Hz, 1024 channels)
    │
    ▼
Bandpass Filter (e.g., 70-150 Hz)
    │
    ▼
Power/Envelope Extraction
    │
    ▼
Temporal Smoothing (EMA or sliding window)
    │
    ▼
Reshape to 32×32 Grid
    │
    ▼
Spatial Smoothing (optional)
    │
    ▼
Visualization
```

### JavaScript Implementation

If processing in the browser:

```javascript
// Simple moving average for temporal smoothing
class MovingAverage {
    constructor(windowSize) {
        this.windowSize = windowSize;
        this.buffer = [];
    }
    
    update(values) {
        this.buffer.push(values);
        if (this.buffer.length > this.windowSize) {
            this.buffer.shift();
        }
        
        // Average across buffer
        const result = new Float32Array(values.length);
        for (let i = 0; i < values.length; i++) {
            let sum = 0;
            for (const frame of this.buffer) {
                sum += frame[i];
            }
            result[i] = sum / this.buffer.length;
        }
        return result;
    }
}

// Compute power (squared values)
function computePower(samples) {
    return samples.map(v => v * v);
}
```

For more sophisticated filtering in JavaScript, consider Web Audio API or libraries like `dsp.js`.

### Exploration Strategy

1. **Start with `super_easy` data** — Signals are crystal clear, helps you understand what you're looking for
2. **Use ground truth** — Correlate neural activity with cursor velocity to understand the signal
3. **Visualize raw vs filtered** — See how filtering affects the signal
4. **Test on `hard` data** — Ensure your approach is robust to noise

## Step 5: Build Your Visualization

### Where to Add Your Code

**Option A: Process in Browser (simplest)**

Modify `example_app/app.js`:

```javascript
function processBatch(neuralData, startTimeS) {
    // Your signal processing here
    const features = extractFeatures(neuralData);
    renderVisualization(features);
}
```

**Option B: Custom Backend (more control)**

Build a Python/Node backend that processes data:

```python
import asyncio
import websockets

async def process_and_serve():
    async with websockets.connect('ws://localhost:8765') as stream:
        async for message in stream:
            data = json.loads(message)
            if data['type'] == 'sample_batch':
                # NumPy/SciPy signal processing
                processed = your_processing_function(data['neural_data'])
                await send_to_frontend(processed)
```

### Visualization Design Principles

Remember the operating room environment:

- **Large, clear indicators** — Readable from 6 feet
- **High contrast colors** — Visible under bright OR lights
- **Directional guidance** — Where should the array move?
- **Confidence indication** — How certain is the detection?
- **"Found it" signal** — Clear indication when positioned correctly
- **Minimize cognitive load** — No small controls or complex menus

## Step 6: Test with Hard Difficulty

Your solution must work well with the `hard` dataset — this matches live evaluation:

```bash
uv run brainstorm-stream --from-file data/hard/
```

Verify that your solution:
- Identifies hotspots despite noise
- Handles bad channels gracefully
- Maintains real-time performance
- Provides useful guidance

## Step 7: Prepare for Live Evaluation

See [Submissions](submissions.md) for the complete live evaluation workflow, including:
- Connecting to the live server
- Using `brainstorm-control` for interactive array movement
- Recording your submission video

## Common Pitfalls

| Pitfall | Why It's Bad | Solution |
|---------|--------------|----------|
| Skipping filtering | Raw signals are noisy | Apply bandpass filtering |
| No temporal smoothing | Visualization flickers | Use EMA or moving average |
| Over-engineering | Complex = slow & fragile | Start simple |
| Ignoring latency | Delayed feedback frustrates | Profile and optimize |
| Tiny indicators | Can't see from 6 feet | Make everything bigger |
| Low contrast | Hard to see in bright OR | Use high-contrast colors |

## Next Steps

- [Data](data.md) — Detailed format and signal content
- [Data Stream](data_stream.md) — WebSocket protocol reference
- [User Persona](user_persona.md) — Understanding your target user
- [Submissions](submissions.md) — Live evaluation and submission process
