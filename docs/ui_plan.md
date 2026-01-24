# Array Placement Guide - UI Specification

This document describes the UI design for the intraoperative array placement guidance tool, targeting the Clinical Operator persona defined in `docs/user_persona.md`.

## Design Principles

1. **Light clinical theme** - Bright, neutral background suitable for OR lighting conditions
2. **6-foot readability** - Large typography, high contrast, simple shapes
3. **Instruction-first** - Primary job is telling the surgeon what to do NOW
4. **Zero-training** - Instantly understandable interface
5. **Stable** - No frantic animations; motion only for state transitions

## Visual Design

### Color Palette
- Background: Light gray (#f5f6f8)
- Surface: White (#ffffff)
- Primary accent: Blue (#2563eb) for guidance elements
- Success: Green (#059669) for on-target state
- Warning: Amber (#d97706) for acquiring/uncertain states
- Error: Red (#dc2626) for disconnected/fault states

### Typography
- Primary: Atkinson Hyperlegible (chosen for maximum legibility at distance)
- Monospace: IBM Plex Mono (for numeric values)
- Instruction text: 36-64px bold (responsive)
- Secondary labels: 11-13px uppercase

## Screen Layout

### Header (Status Bar)
- Logo and application title ("Array Placement")
- Connection status chip with indicator dot
- Time display

### Main Content Area

#### Primary: Instruction Panel (Full Width)
The dominant UI element - a large text panel showing one of:
- **CONNECT TO BEGIN** - Disconnected state
- **ACQUIRING SIGNAL** - Low confidence, building signal
- **MOVE [direction]** - Active guidance (e.g., "MOVE UP LEFT")
- **ON TARGET** - Array is centered on the tuned region

The panel background changes color to reinforce state:
- Default: White
- Acquiring: Light amber (#fef3c7)
- On Target: Light green (#d1fae5)

#### Secondary: Guidance Pad
A circular visualization showing the direction and magnitude of required movement:
- Central dot represents current position
- Arrow points in direction to move
- Target ring indicates the "on-target" zone
- Cardinal labels (UP/DOWN/LEFT/RIGHT) for orientation

When on-target:
- Arrow hides
- Target ring turns solid green

#### Metrics Row
Two metric cards below the guidance pad:
- **Confidence**: Bar + percentage showing tracking quality
- **Distance**: Numeric value in grid units

### Sidebar: Secondary Information

#### Tuned Regions Card
Four horizontal bars showing signal strength for each direction-tuned region:
- Vx+ (right-tuned)
- Vx- (left-tuned)
- Vy+ (up-tuned)
- Vy- (down-tuned)

#### Heatmap Card
One small heatmap:
- **Live Activity**: Current high-gamma power map with center marker

### Footer (Controls)
- Backend URL input field
- Connect/Disconnect button
- Reset button (clears tracking state)

## UI States

### 1. Disconnected
- Status chip: Red indicator, "DISCONNECTED"
- Instruction: "CONNECT TO BEGIN"
- All metrics show "--"
- Guidance arrow hidden

### 2. Acquiring
- Status chip: Amber indicator (pulsing), "CONNECTED"
- Instruction panel: Amber background, "ACQUIRING SIGNAL"
- Sub-instruction: "Move slowly while signal stabilizes"
- Confidence < 40%

### 3. Tracking
- Status chip: Green indicator, "CONNECTED"
- Instruction: Direction command (e.g., "MOVE UP RIGHT")
- Guidance arrow visible, pointing in direction to move
- Confidence bar shows current value

### 4. On Target
- Instruction panel: Green background, "ON TARGET"
- Sub-instruction: "Hold position - array is centered"
- Target ring turns solid green
- Guidance arrow hidden
- Triggers when: distance < 1.5 grid units AND confidence > 72%

## Data Contract

The frontend expects WebSocket messages from the backend.

### Init Message
```json
{
  "type": "init",
  "grid_size": 32,
  "fs": 500.0,
  "ui_hz": 15.0
}
```

### Compass Frame Message
```json
{
  "type": "compass_frame",
  "t_s": 12.34,
  "center_row": 15.5,
  "center_col": 14.2,
  "confidence": 0.85,
  "distance": 2.3,
  "move_row": 0.3,
  "move_col": -0.2,
  "regions": {
    "anchor_1": [15.5, 22.0, 0.9],
    "anchor_2": [15.5, 8.0, 0.85],
    "anchor_3": [8.0, 15.5, 0.7],
    "anchor_4": [22.0, 15.5, 0.6]
  },
  "heatmap": [[...], ...]
}
```

### Reset Command (Frontend to Backend)
```json
{
  "type": "reset"
}
```

## Guidance Vector Definition

The guidance arrow shows where to MOVE the array, not where the signal is:
- `delta = estimated_center - grid_center`
- `move = -delta` (opposite of offset)

This is because moving the array in direction X causes the signal to appear to move in direction -X on the grid.

## Performance

- UI updates at 10-20 Hz (configurable via backend)
- Heatmap is optional (can be disabled for bandwidth)
- All rendering is lightweight (no heavy frameworks)

## Mobile Support

The layout is responsive:
- Below 1024px: Single column layout
- Below 768px: Stacked panels, smaller instruction text
- Touch-friendly button sizes

## Files

- `example_app/index.html` - HTML structure
- `example_app/style.css` - Styles (light theme)
- `example_app/app.js` - WebSocket connection and rendering logic
