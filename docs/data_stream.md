# Data Stream

Technical reference for the WebSocket streaming protocol.

For data content and formats, see [Data](data.md). For signal processing guidance, see [Getting Started](getting_started.md).

## Architecture

### Option A: Direct Connection (Example App)

Your web app connects directly to the data stream:

```
┌──────────────────┐                          ┌──────────────────┐
│  stream_data.py  │    WebSocket (500 Hz)    │    Web App       │
│  (Data Source)   │ ────────────────────────▶│   (Browser)      │
│  ws://:8765      │                          └──────────────────┘
└──────────────────┘                                   ▲
                                                       │ serves static files
                                               ┌──────────────────┐
                                               │    serve.py      │
                                               │  http://:8000    │
                                               └──────────────────┘
```

Benefits:
- Simple setup — no middleware
- Easy to deploy (just static files)
- For live evaluation, just change the WebSocket URL

### Option B: Custom Backend

Build middleware that processes data before sending to your web app:

```
┌──────────────────┐                          ┌──────────────────┐
│  Data Stream     │    WebSocket             │  Your Backend    │
│  ws://:8765      │ ────────────────────────▶│  (Python/Node)   │
└──────────────────┘                          └────────┬─────────┘
                                                       │
                                                       ▼ your protocol
                                              ┌──────────────────┐
                                              │    Web App       │
                                              │   (Browser)      │
                                              └──────────────────┘
```

Benefits:
- Use Python libraries (NumPy, SciPy, MNE-Python)
- Complex algorithms easier in Python
- Control output format and frame rate
- Run computation on a more powerful machine

## WebSocket Protocol

### Connection Endpoints

| Mode | Data Stream URL |
|------|-----------------|
| Local Dev | `ws://localhost:8765` |
| Live Eval | `ws://<server-ip>:8765/stream` |

### Message Types

#### 1. Initialization Message

Sent once when a client connects:

```json
{
  "type": "init",
  "channels_coords": [[1, 1], [1, 2], ..., [32, 32]],
  "grid_size": 32,
  "fs": 500.0,
  "batch_size": 10
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"init"` |
| `channels_coords` | array | 1024 coordinate pairs `[row, col]` (1-indexed) |
| `grid_size` | int | Grid dimension (32) |
| `fs` | float | Sampling frequency (500.0 Hz) |
| `batch_size` | int | Samples per message |

#### 2. Sample Batch Message

Sent continuously at high rate:

```json
{
  "type": "sample_batch",
  "neural_data": [[0.1, -0.2, ...], [0.15, -0.18, ...], ...],
  "start_time_s": 1.234,
  "sample_count": 10,
  "fs": 500.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"sample_batch"` |
| `neural_data` | array | List of samples, each with 1024 values |
| `start_time_s` | float | Timestamp of first sample in batch |
| `sample_count` | int | Number of samples in batch |
| `fs` | float | Sampling frequency |

### Data Rates

| Metric | Value |
|--------|-------|
| Sampling Rate | 500 Hz |
| Batch Size | 10 samples (default) |
| Message Rate | 50 messages/second |
| Payload Size | ~40 KB per message |
| Bandwidth | ~2 MB/s |

## Connecting Your Application

### JavaScript (Direct Connection)

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'init') {
        setupGrid(data.channels_coords, data.grid_size);
    } else if (data.type === 'sample_batch') {
        processBatch(data.neural_data, data.start_time_s);
    }
};
```

### Python Backend

```python
import asyncio
import json
import websockets

async def connect_to_stream():
    async with websockets.connect('ws://localhost:8765') as ws:
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'init':
                setup(data)
            elif data['type'] == 'sample_batch':
                processed = process_batch(data['neural_data'])
                await send_to_frontend(processed)
```

## Control Endpoint (Live Evaluation Only)

During live evaluation, you send keyboard controls via `brainstorm-control`:

```bash
uv run brainstorm-control --host <server-ip> --port 8765
```

The control client captures arrow keys and sends them to the server, which updates the array position. Your app receives the resulting neural data changes.

### Control Message Format

Messages sent to `/control`:

```json
{
  "type": "key",
  "key": "up",
  "pressed": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"key"` |
| `key` | string | `"up"`, `"down"`, `"left"`, `"right"` |
| `pressed` | boolean | `true` for key down, `false` for key up |

See [Submissions](submissions.md) for the complete live evaluation workflow.

## Performance Considerations

### Latency Budget

| Stage | Target |
|-------|--------|
| Network | < 10 ms |
| Processing | < 20 ms |
| Rendering | < 16 ms (60 FPS) |
| **Total** | < 50 ms |

### Optimization Tips

1. **Batch processing** — Process batches, not individual samples
2. **Vectorized operations** — Use NumPy for signal processing
3. **Async I/O** — Don't block on WebSocket operations
4. **Frame skipping** — Skip frames if processing falls behind
5. **Web Workers** — Offload heavy JS computation
