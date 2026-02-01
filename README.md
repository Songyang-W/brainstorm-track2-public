[![BrainStorm 2026 - Track 2 Demo](https://img.youtube.com/vi/a3brRuCG1hk/0.jpg)](https://www.youtube.com/watch?v=a3brRuCG1hk)

# BrainStorm 2026 - Track 2: The "Compass" Challenge

Build a real-time visualization tool to guide neurosurgeons in placing a brain-computer interface array over the optimal region of the motor cortex.

## 🎯 The Challenge

Design and build a web application that:
- Processes a live stream of neural data from a 1024-channel micro-ECoG array
- Identifies **areas** of functionally tuned neural activity (not just individual transient hotspots)
- Visualizes tuned regions relative to the array position
- Provides clear, intuitive visual guidance for array placement optimization
- Works in the high-pressure environment of an operating room

> **Key insight**: Neural activity over tuned regions is not uniform — it depends on cursor movement direction. A good solution identifies coherent **areas of interest** rather than chasing individual activation spikes.

> **The Analogy**: Think of it like processing a live video feed from a shaky, moving camera. We have to filter out the 'snow' (noise) to find specific objects that light up, track them as the camera moves, and point an arrow telling the surgeon exactly where to center the frame.
### ⚙️ Backend Engineering Challenges
* **Latency vs. Complexity**: We theoretically know how to filter noise, but we cannot use heavy, slow algorithms. The system must provide **hard real-time feedback** to the surgeon, meaning signal processing must be extremely optimized.
* **Spontaneous Activity**: The brain is never silent. Background regions fire randomly even without specific intent. Your system must distinguish true functional "lighting up" from this constant biological background noise.
* **Scientific Uncertainty**: This is an open research field—there is no algorithm in the world that can 100% perfectly decode these signals. Your solution must be robust enough to work despite this inherent ambiguity.
* **Blind Navigation**: You need to calculate a guidance vector (telling the doctor where to move) without knowing their current hand trajectory or velocity. You are guiding a moving target with no external motion tracking.


## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[Overview](docs/overview.md)** | Challenge description, requirements, judging criteria |
| **[Installation](docs/installation.md)** | Setup instructions |
| **[Getting Started](docs/getting_started.md)** | Development workflow and signal processing hints |
| **[Data](docs/data.md)** | Dataset formats, signal content, and processing guidance |
| **[Data Stream](docs/data_stream.md)** | WebSocket protocol reference |
| **[User Persona](docs/user_persona.md)** | Understanding your target user |
| **[Submissions](docs/submissions.md)** | Live evaluation and how to submit |
| **[FAQ](docs/faq.md)** | Common questions and rules |

## 🚀 Quick Start

```bash
# 1. Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup
make install

# 3. Download data (start with super_easy, develop with hard)
uv run python -m scripts.download super_easy
uv run python -m scripts.download hard

# 4. Stream data (Terminal 1)
uv run brainstorm-stream --from-file data/hard/

# 5. Run example app (Terminal 2)
uv run brainstorm-serve
# Open http://localhost:8000
```

The example app shows a basic heatmap. **Your solution should go far beyond this!**

See [Installation](docs/installation.md) for detailed setup and [Getting Started](docs/getting_started.md) for development guidance.

## 📊 The Data

Four difficulty levels on [HuggingFace](https://huggingface.co/datasets/PrecisionNeuroscience/BrainStorm2026-track2):

| Difficulty | Description | Use Case |
|------------|-------------|----------|
| `super_easy` | Crystal-clear signals | Understanding the signal |
| `easy` | Clean signals, minimal noise | Initial development |
| `medium` | Moderate noise | Testing robustness |
| `hard` | Challenging conditions | **Final testing & live evaluation** |

- **Array**: 1024 channels (32×32 grid)
- **Sampling Rate**: 500 Hz
- **Protocol**: WebSocket (JSON batches)

See [Data](docs/data.md) for detailed format and signal processing guidance.

## 🎨 What Makes a Great Solution?

### User Experience (40%)
- Instantly interpretable (< 1 second to understand)
- Readable from 6 feet away (high contrast, large indicators)
- Visualizes tuned **areas** relative to the array
- Provides directional guidance for movement optimization
- Unambiguous "found it" signal when positioned correctly

### Technical Execution (40%)
- Accurate identification of tuned regions
- Real-time performance (low latency, smooth updates)
- Robust to noise and bad channels
- Aggregates signal over time to identify stable areas

### Innovation (20%)
- Novel visualization approaches beyond simple heatmaps
- Creative signal processing
- Compelling video demonstration

## 🛠️ What You Can Modify

✅ **You CAN**:
- Modify or replace the example app completely
- Build a custom backend/middleware (Python, Node, etc.)
- Use any signal processing or visualization approach
- Add any dependencies or frameworks

❌ **You CANNOT**:
- Modify the data streaming protocol
- Change how data is transmitted during evaluation

## 📦 Key Files

```
brainstorm2026-track2/
├── scripts/
│   ├── download.py       # Download datasets from HuggingFace
│   ├── stream_data.py    # Stream data locally
│   ├── serve.py          # Static file server for example app
│   └── control_client.py # Send keyboard controls (live eval)
├── example_app/          # Minimal reference implementation
├── data/                 # Downloaded datasets (gitignored)
└── docs/                 # Full documentation
```

## 🎥 Deliverables

> **⚠️ Your submission is a VIDEO** — recorded during live evaluation and uploaded to YouTube.

1. **Video demo** (3-5 minutes) — Screen recording during live evaluation with voice narration
2. **SUBMISSION.YAML** — Updated with your YouTube link and pushed to `main`
3. **Your application** (code repository with documentation)

See [Submissions](docs/submissions.md) for detailed instructions.

## 🏥 Design for the Operating Room

- **User**: Clinical operator with neuroscience PhD (see [User Persona](docs/user_persona.md))
- **Environment**: Crowded, high-stress operating room
- **Viewing Distance**: Up to 6 feet from screen
- **Cognitive Load**: Must be immediately interpretable
- **Stakes**: This guides permanent array placement in a patient's brain

## 💡 Tips

- Start simple and iterate
- Test with streaming data early
- **Develop with the `hard` dataset** — this matches final evaluation
- Prioritize clarity over complexity
- Think like a surgeon, not a researcher
- Make it readable from 6 feet away
