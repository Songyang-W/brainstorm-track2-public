# Modular UI & Improved Cluster Anchoring Plan

## Overview

This document outlines the plan for:
1. **Modular Layout System** - Flexible view arrangements (tabs, splits, grids)
2. **Improved Cluster Anchoring** - Better persistence of cluster positions when parts turn off

---

## Part 1: Modular Layout System

### Current State
- Three fixed tabs: Guide, Global Map, Debug
- Only one view visible at a time
- No flexibility for surgeon preferences

### Proposed Layout Options

```
┌─────────────────────────────────────────────────────────────┐
│                      LAYOUT SELECTOR                         │
├─────────────────────────────────────────────────────────────┤
│  [Tabs] [H-Split] [V-Split] [2x2 Grid] [1+2 Layout]        │
└─────────────────────────────────────────────────────────────┘
```

#### Layout Types:

1. **Tabs (current)** - One view at a time, full screen
   ```
   ┌─────────────────────┐
   │                     │
   │      [Active]       │
   │                     │
   └─────────────────────┘
   ```

2. **Horizontal Split** - Two views side by side
   ```
   ┌──────────┬──────────┐
   │          │          │
   │  [Left]  │  [Right] │
   │          │          │
   └──────────┴──────────┘
   ```

3. **Vertical Split** - Two views stacked
   ```
   ┌─────────────────────┐
   │        [Top]        │
   ├─────────────────────┤
   │       [Bottom]      │
   └─────────────────────┘
   ```

4. **2x2 Grid** - Four views in quadrants
   ```
   ┌──────────┬──────────┐
   │   [TL]   │   [TR]   │
   ├──────────┼──────────┤
   │   [BL]   │   [BR]   │
   └──────────┴──────────┘
   ```

5. **1+2 Layout** - Main view + two smaller views
   ```
   ┌──────────┬──────────┐
   │          │   [TR]   │
   │  [Main]  ├──────────┤
   │          │   [BR]   │
   └──────────┴──────────┘
   ```

6. **Triple Horizontal** - Three views side by side
   ```
   ┌───────┬───────┬───────┐
   │ [L]   │ [M]   │ [R]   │
   └───────┴───────┴───────┘
   ```

### Available View Panels

Each panel slot can contain one of:
- **Guide** - Direction arrow + instruction (primary surgeon view)
- **Global Map** - Accumulated activity map with trajectory
- **Live Heatmap** - Real-time 32x32 activity
- **Cluster Memory** - Tracked hotspots with decay visualization
- **Metrics** - Numerical debug data

### Implementation Approach

#### 1. Panel Component Architecture

```javascript
// Each view becomes a self-contained panel
class Panel {
    constructor(type, container) {
        this.type = type;           // 'guide', 'global-map', 'live', 'memory', 'metrics'
        this.container = container;
        this.canvas = null;         // For canvas-based views
        this.ctx = null;
    }

    render(data) { /* type-specific rendering */ }
    resize(width, height) { /* handle container resize */ }
    destroy() { /* cleanup */ }
}
```

#### 2. Layout Manager

```javascript
class LayoutManager {
    constructor() {
        this.layout = 'tabs';       // Current layout type
        this.panels = [];           // Active panel instances
        this.assignments = {};      // Which view in which slot
    }

    setLayout(type) {
        // Destroy old panels
        // Create new container structure
        // Instantiate panels based on assignments
    }

    assignView(slot, viewType) {
        // Put a specific view type in a slot
    }
}
```

#### 3. CSS Grid-Based Layouts

```css
/* Layout containers */
.layout-tabs { display: flex; }
.layout-hsplit { display: grid; grid-template-columns: 1fr 1fr; }
.layout-vsplit { display: grid; grid-template-rows: 1fr 1fr; }
.layout-grid { display: grid; grid-template: 1fr 1fr / 1fr 1fr; }
.layout-main-side { display: grid; grid-template: 1fr 1fr / 2fr 1fr; }

/* Panel slots */
.panel-slot {
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-width: 200px;
    min-height: 150px;
}

/* Panel header (for identifying/swapping) */
.panel-header {
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 12px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
}
```

#### 4. View Selector Dropdown

Each panel slot gets a dropdown to change what view it displays:

```html
<div class="panel-slot">
    <div class="panel-header">
        <select class="view-selector">
            <option value="guide">Guide</option>
            <option value="global-map">Global Map</option>
            <option value="live">Live Activity</option>
            <option value="memory">Cluster Memory</option>
            <option value="metrics">Metrics</option>
        </select>
    </div>
    <div class="panel-content">
        <!-- Rendered view -->
    </div>
</div>
```

#### 5. Responsive Behavior

- On small screens (< 768px): Force tabs layout
- Allow drag-to-resize split positions (optional, phase 2)
- Remember layout preference in localStorage

### Files to Modify/Create

| File | Changes |
|------|---------|
| `index.html` | Add layout selector, restructure for dynamic panels |
| `style.css` | Add layout grid styles, panel styles |
| `layout-manager.js` | **NEW** - Layout orchestration |
| `panels/*.js` | **NEW** - Individual panel components |
| `visualizer.js` | Refactor into panel-based rendering |
| `app.js` | Integrate LayoutManager |

---

## Part 2: Improved Cluster Anchoring

### Current Problem

When hotspots turn off, the cluster center can shift unexpectedly because:
1. Peak positions are tracked individually but decay independently
2. No concept of "cluster identity" - peaks are merged by proximity only
3. When activity returns, it may create new peaks instead of updating old ones

### Example Scenario

```
Time T1: Four hotspots active
    ●  ●
    ●  ●
    Center: (15, 15)

Time T2: Two hotspots turn off
    ○  ●
    ●  ○
    Current behavior: Center shifts to (14, 16) - wrong!
    Desired behavior: Center stays at (15, 15)

Time T3: Different two turn off
    ●  ○
    ○  ●
    Current behavior: Center shifts again
    Desired behavior: Center remains stable at (15, 15)
```

### Proposed Solution: Cluster Entity Tracking

Instead of tracking individual peaks, track **cluster entities** that contain multiple peaks.

#### 1. Cluster Entity Structure

```javascript
class ClusterEntity {
    constructor(id) {
        this.id = id;
        this.peaks = [];              // Member peaks with positions
        this.centroid = {x: 0, y: 0}; // Weighted center of ALL peaks
        this.boundingBox = null;      // Convex hull or bounding box
        this.createdAt = Date.now();
        this.lastSeen = Date.now();
        this.confidence = 0;

        // Peak positions are ANCHORED even when inactive
        this.anchoredPeaks = [];      // {x, y, intensity, active, lastActive}
    }

    // Update with new frame data
    update(activePeaks) {
        // Match active peaks to anchored positions
        // Update positions with weighted smoothing
        // Mark matched peaks as active
        // Keep unmatched anchored peaks (but mark inactive)
        // Add new peaks if no match found
    }

    // Compute centroid from ALL anchored peaks (active or not)
    computeCentroid() {
        let totalWeight = 0;
        let wx = 0, wy = 0;

        for (const peak of this.anchoredPeaks) {
            // Weight by: historical max intensity * recency factor
            const recency = Math.exp(-(Date.now() - peak.lastActive) / 5000);
            const weight = peak.maxIntensity * recency;

            wx += peak.x * weight;
            wy += peak.y * weight;
            totalWeight += weight;
        }

        this.centroid = {
            x: wx / totalWeight,
            y: wy / totalWeight
        };
    }
}
```

#### 2. Peak-to-Cluster Assignment Algorithm

```javascript
function assignPeaksToClusters(currentPeaks, existingClusters) {
    // 1. For each current peak, find best matching cluster
    //    - Based on distance to cluster centroid
    //    - And distance to any anchored peak in cluster

    // 2. If peak is close to existing cluster, add to it
    //    - Update the matching anchored peak position
    //    - Or add as new anchored peak if no close match

    // 3. If peak is far from all clusters, create new cluster

    // 4. Merge clusters if they've grown together

    // 5. Split clusters if internal peaks have diverged
}
```

#### 3. Anchored Peak Matching

```javascript
function matchPeakToAnchored(newPeak, anchoredPeaks) {
    let bestMatch = null;
    let bestScore = Infinity;

    for (const anchored of anchoredPeaks) {
        // Distance score
        const dist = Math.sqrt(
            Math.pow(newPeak.x - anchored.x, 2) +
            Math.pow(newPeak.y - anchored.y, 2)
        );

        // Intensity similarity score
        const intensityDiff = Math.abs(newPeak.intensity - anchored.maxIntensity);

        // Recency bonus (recently active peaks match better)
        const recency = Date.now() - anchored.lastActive;
        const recencyBonus = recency < 1000 ? 0 : recency / 10000;

        const score = dist + intensityDiff * 2 + recencyBonus;

        if (score < bestScore && dist < 5) {  // Max match distance
            bestScore = score;
            bestMatch = anchored;
        }
    }

    return bestMatch;
}
```

#### 4. Cluster Centroid Stability

Key insight: The centroid should be computed from the **geometric arrangement** of all known peak positions, not just currently active ones.

```javascript
function computeStableCentroid(cluster) {
    // Option A: Simple weighted average (current approach, improved)
    // Weight by: maxIntensity * confidence * recencyFactor

    // Option B: Geometric median (more robust to outliers)
    // Iterative algorithm that finds point minimizing sum of distances

    // Option C: Convex hull centroid
    // Find convex hull of all anchored peaks, compute its centroid

    // Option D: Principal component analysis
    // Find the center along the major axis of the peak distribution
}
```

#### 5. Handling Cluster Lifecycle

```javascript
class ClusterTracker {
    // Cluster creation
    createCluster(seedPeak) {
        const cluster = new ClusterEntity(this.nextId++);
        cluster.addAnchoredPeak(seedPeak);
        this.clusters.push(cluster);
    }

    // Cluster merging (when two clusters overlap)
    mergeClusters(cluster1, cluster2) {
        // Combine anchored peaks
        // Keep the older cluster ID for continuity
        // Recompute centroid
    }

    // Cluster death (no activity for extended period)
    pruneDeadClusters() {
        const timeout = 10000;  // 10 seconds
        this.clusters = this.clusters.filter(c =>
            Date.now() - c.lastSeen < timeout
        );
    }
}
```

### Global Map Integration

The global map should anchor **cluster entities** rather than individual peaks:

```javascript
class GlobalBrainMap {
    // Instead of globalHotspots array, use globalClusters
    globalClusters = [];  // Cluster entities in global coordinates

    updateGlobalClusters(localClusters, arrayPosition) {
        for (const local of localClusters) {
            // Convert local centroid to global coordinates
            const globalCentroid = {
                x: arrayPosition.x + local.centroid.x - 16,
                y: arrayPosition.y + local.centroid.y - 16
            };

            // Find matching global cluster
            const match = this.findMatchingGlobalCluster(globalCentroid);

            if (match) {
                // Update global cluster position (with anchoring smoothing)
                this.updateGlobalCluster(match, local, globalCentroid);
            } else {
                // Create new global cluster
                this.createGlobalCluster(local, globalCentroid);
            }
        }
    }
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `cluster-tracker.js` | Rewrite with ClusterEntity class |
| `global-map.js` | Integrate cluster entities instead of peaks |
| `visualizer.js` | Render cluster bounding boxes, show inactive peaks |

---

## Implementation Order

### Phase 1: Improved Cluster Anchoring
1. Create `ClusterEntity` class
2. Implement peak-to-cluster matching
3. Update `ClusterTracker` to use entities
4. Update `GlobalBrainMap` integration
5. Test with simulated on/off patterns

### Phase 2: Modular Layout System
1. Create `LayoutManager` class
2. Refactor visualizer into separate panel components
3. Implement CSS grid layouts
4. Add layout selector UI
5. Implement view selector dropdowns
6. Add localStorage persistence

### Phase 3: Polish
1. Drag-to-resize splits
2. Preset layout buttons (e.g., "Surgery Mode", "Debug Mode")
3. Keyboard shortcuts for layout switching
4. Mobile-responsive behavior

---

## Technical Considerations

### Performance
- Each panel renders independently
- Only visible panels render (lazy rendering for tabs)
- Use `requestAnimationFrame` properly across multiple canvases
- Consider OffscreenCanvas for heavy rendering

### State Management
- Single source of truth in `app.js`
- Panels receive data via `render(data)` calls
- No direct DOM manipulation across panels

### Testing
- Create test mode with synthetic data patterns
- Specifically test: 4 hotspots with 2 on/off cycling
- Test cluster splitting/merging scenarios
