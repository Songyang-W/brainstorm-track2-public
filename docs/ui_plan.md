# OR UI Plan: "Compass HUD"

This plan targets the Clinical Operator persona in `docs/user_persona.md` and the OR constraints in `docs/overview.md`:
- glanceable from ~6 feet
- minimal cognitive load
- unambiguous "move this way" + "found it" signal
- robust to noisy, direction-dependent activations (`docs/data.md`)

## Core Mental Model

The tuned region appears as a cluster of direction-dependent sub-regions that only partially light up at any moment. The UI should:
1. Estimate the *stable cluster center* (the placement target) from streaming neural data.
2. Show a single dominant guidance vector: where to move the array to center the cluster.
3. Communicate certainty and signal quality.

## Screen Layout (Single Screen, No Tabs)

### A) Primary Guidance (Center Stage)
- **Big Compass Arrow**: points in the direction the surgeon should move the array.
- **Distance Ring**: radial meter showing how far the estimated center is from the array midpoint.
- **ON TARGET State**: when "close enough" and confidence is high, replace arrow with a large "ON TARGET" badge + steady green ring.

### B) Context (Secondary, Still Glanceable)
- **Live Heatmap (32x32)**: current high-gamma power map, with center marker and peak overlays.
- **Memory Heatmap**: persistence view that accumulates cluster parts over time (shows the stable shape, including parts that are currently off).

### C) Operator Status Strip (Top/Bottom Bar)
- Connection state (Connected / Buffering / Disconnected)
- Timestamp / latency
- Confidence meter (0-100) with clear thresholds
- Signal-quality flags:
  - "Too Noisy" / "Dead Channels" (if detected)
  - "Low Confidence" (if template score collapses)

## Interaction Model

No UI controls during active use beyond:
- Server URL field (setup only)
- "Connect" button
- "Reset" button (clears accumulation and re-centers tracking)
  - Implementation detail: sends `{"type":"reset"}` to the backend when connected.

Everything else is read-only and big.

## States & Visual Language

### 1) Searching
- Arrow becomes a subtle rotating sweep (like sonar) with text: "SEARCHING"
- Confidence bar muted / gray
- Heatmaps still visible but de-emphasized

Trigger: confidence < threshold for N frames OR no stable estimate.

### 2) Tracking
- Arrow is stable and decisive
- Ring shows distance with a single numeric readout (e.g. "3.2 mm eq")
- Confidence meter visible and rising/falling smoothly

### 3) Locked (Found It)
- Screen simplifies:
  - center shows "ON TARGET" + steady green ring
  - arrow disappears
- Optional small "Hold" timer indicating stability duration

Trigger: distance < `lock_radius` AND confidence > `lock_conf` for `lock_hold_s`.

### 4) Signal Fault
- Full-width amber/red banner:
  - "SIGNAL QUALITY LOW"
  - a single recommended action (e.g. "Check connections" / "Ignore channels 12, 801, ...")

## Guidance Vector Definition (Critical)

If the tuned region appears offset in the array frame, the surgeon should move the array **toward that offset in the real world**. Because the array frame moves with the array, guidance is the **opposite** of the observed offset:
- `delta = estimated_center - grid_center`
- `move = -delta` (direction to move the array)

The arrow displays `move`, not `delta`.

## Performance Targets

From `docs/data_stream.md`:
- ingest 500 Hz batches, but UI update at ~10-20 Hz (frame skipping allowed)
- processing should be causal and stable; prioritize low-latency over perfect offline filtering

## Color / Typography Direction

Industrial + clinical:
- Deep charcoal background, off-white type
- Surgical green for "go/locked"
- Amber for "uncertain/searching"
- Red for "fault"
- Typography: condensed display for the compass wordmark, highly legible sans for numbers

## What We Render (Data Contract)

Frontend expects `type: "compass_frame"` messages:
- `t_s`
- `center_row`, `center_col`
- `confidence` (0..1)
- `distance` (grid units)
- `move_row`, `move_col` (normalized -1..1)
- `spots` (list of `[row, col, weight]`, optional)
- `spots_mem` (list of `[row, col, weight]` in memory/template coords, optional)
- `heatmap` (32x32 float array, optional)
- `memory` (32x32 float array, optional)

## "OR-Ready" Details

- Default font sizes assume distance viewing.
- Avoid tiny legends; use a small number of labels, always in the same place.
- Use motion only to communicate state changes (lock acquisition, confidence collapse), never decorative.
