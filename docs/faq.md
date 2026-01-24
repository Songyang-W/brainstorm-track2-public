# Frequently Asked Questions

## Data

### Where do I get the data?

Download from [HuggingFace](https://huggingface.co/datasets/PrecisionNeuroscience/BrainStorm2026-track2):

```bash
uv run python -m scripts.download hard
```

See [Installation](installation.md) for setup and [Data](data.md) for format details.

### What's the difference between difficulty levels?

| Difficulty | Noise | Artifacts | Use Case |
|------------|-------|-----------|----------|
| **super_easy** | None | None | Understanding the signal |
| **easy** | Low | None | Initial development |
| **medium** | Medium | Some bad channels | Robustness testing |
| **hard** | High | Multiple artifacts | **Live evaluation** |

### Which difficulty do I need to win?

Your video must be recorded during **live evaluation** (which uses hard difficulty). Only live-mode submissions are eligible for prizes.

### What's in the ground truth file?

Cursor kinematics (`vx`, `vy`) and tuned region positions. This is for **development only** — not available during live evaluation.

## What Can I Modify?

✅ **You CAN**:
- Build a custom backend/middleware
- Create your own web app from scratch
- Use any signal processing approach
- Use any visualization framework
- Add any dependencies
- Downsample, buffer, or transform the data

❌ **You CANNOT**:
- Modify the streaming protocol or message format
- Change how data is transmitted during evaluation

## Technical

### Can I use a different programming language?

**Yes!** Any language that can:
- Connect to a WebSocket server
- Parse JSON messages
- Render a web-based visualization

Examples: Python (Streamlit, Dash), JavaScript/TypeScript, Rust, Go, C++

### Can I build a custom backend?

**Yes!** See [Data Stream](data_stream.md) for architecture options. A backend lets you use Python libraries (NumPy, SciPy, MNE-Python) for signal processing.

### Can I use external libraries?

**Absolutely!** Use any libraries that help:
- **Signal Processing**: NumPy, SciPy, scikit-learn, PyTorch, MNE-Python
- **Visualization**: Plotly, D3.js, Three.js, WebGL
- **Web Frameworks**: React, Vue, Svelte, Streamlit, Dash, FastAPI

### Do I need to process all 1024 channels in real-time?

Not necessarily. Options:
- Process all channels if performance allows
- Spatially downsample (average over patches)
- Use adaptive sampling (focus on regions of interest)

The key is **near real-time visualization updates**.

### How should I handle bad channels?

Detect and handle them (exclude, interpolate, or mark visually). Don't let bad channels obscure true hotspots. This is part of the challenge!

### Can I use machine learning?

**Yes**, though the challenge emphasizes **real-time signal processing** and **visualization design** over ML performance.

## Evaluation

### How will submissions be judged?

- **40% User Experience** — Can a surgeon use it effectively in the OR?
- **40% Technical Execution** — Accurate hotspot detection with low latency
- **20% Innovation** — Novel approach, compelling presentation

See [Overview](overview.md) for detailed criteria.

### Can I work with a team?

**Yes!** Teams of any size. The challenge benefits from diverse skills in signal processing, web development, and UX design.

### What should my video include?

Your 3-5 minute video should:
1. Show your app during **live evaluation** (not downloaded data)
2. Demonstrate interactive array control with arrow keys
3. "Play the game" — find and track tuned regions
4. Explain your visualizations and design rationale
5. Highlight key technical features

See [Submissions](submissions.md) for full requirements.

## Live Evaluation

### What happens on Day 2?

1. **Dev Server (early)** — Practice connection with simplified data
2. **Final Server (later)** — Record your submission video with hard-difficulty data

See [Submissions](submissions.md) for the complete workflow.

### What's the dev server for?

> ⚠️ **Connection practice only** — The dev server streams simplified "toy" data. Do NOT use it to develop or tune your visualization.

### Where do I submit?

Upload your video to YouTube (Public or Unlisted), then update `SUBMISSION.YAML` with the link and push to `main`. See [Submissions](submissions.md) for details.
