# Installation

## Prerequisites

- **Python**: 3.10 or higher
- **OS**: macOS, Linux, or Windows
- **Package Manager**: [UV](https://github.com/astral-sh/uv) (recommended) or pip

## Step 1: Install UV

UV is a fast Python package manager:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Step 2: Clone and Install

```bash
git clone <track2-repo-url>
cd brainstorm2026-track2

# Install using Makefile
make install
```

This creates a virtual environment and installs all dependencies.

**Manual setup alternative:**

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

uv sync
```

## Step 3: Download Data

```bash
# Start with super_easy to understand the signal
uv run python -m scripts.download super_easy

# Develop with hard (matches live evaluation)
uv run python -m scripts.download hard
```

Or in Python:

```python
from scripts.download import download_track2_data, load_track2_data

download_track2_data("hard")
data, ground_truth = load_track2_data("hard")
```

Files are saved to `data/{difficulty}/`.

## Step 4: Verify Installation

**Terminal 1 — Start the data stream:**
```bash
uv run brainstorm-stream --from-file data/hard/
```

**Terminal 2 — Start the example app:**
```bash
uv run brainstorm-serve
```

Open **http://localhost:8000** in your browser. You should see a grid visualization updating in real-time.

## Available Commands

| Command | Description |
|---------|-------------|
| `brainstorm-stream` | Stream data from downloaded datasets |
| `brainstorm-serve` | Static file server for example web app |
| `brainstorm-control` | Send keyboard controls to remote server (live eval only) |

### Command Options

```bash
# Stream options
brainstorm-stream --from-file data/hard/           # Stream dataset
brainstorm-stream --from-file data/hard/ --no-loop # Stop at end
brainstorm-stream --from-file data/hard/ --port 8766

# Server options
brainstorm-serve                    # Default (port 8000)
brainstorm-serve --port 8001        # Custom port
```

## Troubleshooting

### Command not found

Activate the virtual environment or use `uv run`:

```bash
source .venv/bin/activate
# or
uv run brainstorm-stream --from-file data/hard/
```

### WebSocket connection refused

Make sure `brainstorm-stream` is running in another terminal.

### Data not found

Download the data first:

```bash
uv run python -m scripts.download hard
```

### Port already in use

Use different ports:

```bash
uv run brainstorm-stream --from-file data/hard/ --port 8766
uv run brainstorm-serve --port 8001
# Update your app's WebSocket URL to ws://localhost:8766
```

## Next Steps

Head to [Getting Started](getting_started.md) to begin developing your solution!
