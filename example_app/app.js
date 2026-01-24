/**
 * BCI Array Placement - Real-time Neural Activity Visualization
 *
 * Connects to the Python backend (ws://localhost:8766) and renders
 * processed neural activity with hotspot detection and surgical guidance.
 */

// State
let ws = null;
let isConnected = false;
let gridSize = 32;

// Array Canvas
let canvas = null;
let ctx = null;
const canvasSize = 576; // 32 * 18
const cellSize = canvasSize / 32;
const padding = 16;

// Global Map Canvas
let globalCanvas = null;
let globalCtx = null;
const globalCanvasSize = 384; // 48 * 8 (downsampled size)
const globalCellSize = globalCanvasSize / 48;
const globalMapSize = 96; // Full map size
const globalDisplaySize = 48; // Downsampled for display

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();
let currentFps = 0;

// Data
let currentData = null;

// Ground truth overlay toggle
let showGroundTruth = true;

// Color interpolation helpers
function lerp(a, b, t) {
    return a + (b - a) * t;
}

function lerpColor(c1, c2, t) {
    return [
        Math.round(lerp(c1[0], c2[0], t)),
        Math.round(lerp(c1[1], c2[1], t)),
        Math.round(lerp(c1[2], c2[2], t))
    ];
}

// Activity colormap (dark purple -> orange -> yellow)
const ACTIVITY_COLORS = [
    [26, 26, 46],      // 0.0 - dark
    [45, 27, 78],      // 0.25 - purple
    [249, 115, 22],    // 0.6 - orange
    [250, 204, 21]     // 1.0 - yellow
];

const ACTIVITY_STOPS = [0, 0.25, 0.6, 1.0];

function activityToColor(value) {
    // Clamp value to [0, 1]
    const v = Math.max(0, Math.min(1, value));

    // Find the two stops we're between
    for (let i = 0; i < ACTIVITY_STOPS.length - 1; i++) {
        if (v <= ACTIVITY_STOPS[i + 1]) {
            const t = (v - ACTIVITY_STOPS[i]) / (ACTIVITY_STOPS[i + 1] - ACTIVITY_STOPS[i]);
            const [r, g, b] = lerpColor(ACTIVITY_COLORS[i], ACTIVITY_COLORS[i + 1], t);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    const [r, g, b] = ACTIVITY_COLORS[ACTIVITY_COLORS.length - 1];
    return `rgb(${r}, ${g}, ${b})`;
}

// Arrow SVG paths for different directions
const ARROW_PATHS = {
    'up': 'M50,20 L75,50 L60,50 L60,80 L40,80 L40,50 L25,50 Z',
    'down': 'M50,80 L75,50 L60,50 L60,20 L40,20 L40,50 L25,50 Z',
    'left': 'M20,50 L50,25 L50,40 L80,40 L80,60 L50,60 L50,75 Z',
    'right': 'M80,50 L50,25 L50,40 L20,40 L20,60 L50,60 L50,75 Z',
    'up-left': 'M20,20 L55,20 L55,35 L35,35 L35,55 L20,55 Z M40,60 L60,40 L70,50 L50,70 Z',
    'up-right': 'M80,20 L45,20 L45,35 L65,35 L65,55 L80,55 Z M60,60 L40,40 L30,50 L50,70 Z',
    'down-left': 'M20,80 L55,80 L55,65 L35,65 L35,45 L20,45 Z M40,40 L60,60 L70,50 L50,30 Z',
    'down-right': 'M80,80 L45,80 L45,65 L65,65 L65,45 L80,45 Z M60,40 L40,60 L30,50 L50,30 Z',
    'center': 'M50,50 m-20,0 a20,20 0 1,1 40,0 a20,20 0 1,1 -40,0'
};

function initCanvas() {
    canvas = document.getElementById('neural-canvas');
    ctx = canvas.getContext('2d');

    // Set canvas size
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvasSize, canvasSize);
}

function initGlobalCanvas() {
    globalCanvas = document.getElementById('global-canvas');
    globalCtx = globalCanvas.getContext('2d');

    // Set canvas size
    globalCanvas.width = globalCanvasSize;
    globalCanvas.height = globalCanvasSize;

    // Clear canvas
    globalCtx.fillStyle = '#0a0a0f';
    globalCtx.fillRect(0, 0, globalCanvasSize, globalCanvasSize);
}

function renderGrid(beliefMap, badChannels) {
    if (!beliefMap) return;

    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    // Draw each cell
    // Flip both axes to match ground truth coordinate system
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            const x = (gridSize - 1 - col) * cellSize;
            const y = (gridSize - 1 - row) * cellSize;
            const value = beliefMap[row][col];
            const isBad = badChannels && badChannels[row][col];

            // Cell size with small gap
            const size = cellSize - 1;

            if (isBad) {
                // Bad channel - gray
                ctx.fillStyle = '#374151';
            } else {
                // Color by activity
                ctx.fillStyle = activityToColor(value);

                // Add glow effect for high activity
                if (value > 0.5) {
                    const glowIntensity = (value - 0.5) * 2;
                    ctx.shadowColor = `rgba(250, 204, 21, ${glowIntensity * 0.5})`;
                    ctx.shadowBlur = 8 * glowIntensity;
                } else {
                    ctx.shadowBlur = 0;
                }
            }

            // Draw rounded rectangle
            const radius = 3;
            ctx.beginPath();
            ctx.roundRect(x + 0.5, y + 0.5, size, size, radius);
            ctx.fill();

            // Reset shadow
            ctx.shadowBlur = 0;
        }
    }
}

function renderGroundTruthOverlay(groundTruth) {
    if (!groundTruth || !showGroundTruth) return;

    // Region colors matching the evaluation visualizer
    const regionColors = {
        vx_pos: '#ef4444',  // red - rightward
        vx_neg: '#3b82f6',  // blue - leftward
        vy_pos: '#22c55e',  // green - upward
        vy_neg: '#a855f7',  // purple - downward
    };

    const regionLabels = {
        vx_pos: 'Vx+',
        vx_neg: 'Vx-',
        vy_pos: 'Vy+',
        vy_neg: 'Vy-',
    };

    // Draw each region center
    for (const [region, coords] of Object.entries(groundTruth)) {
        if (!Array.isArray(coords)) continue;

        const [row, col] = coords;
        const x = (col + 0.5) * cellSize;
        const y = (row + 0.5) * cellSize;
        const color = regionColors[region];
        const label = regionLabels[region];

        if (!color) continue;

        // Draw circle outline
        ctx.beginPath();
        ctx.arc(x, y, cellSize * 1.2, 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();

        // Draw label
        ctx.font = 'bold 12px monospace';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.fillText(label, x, y - cellSize * 1.5);
    }

    // Calculate and draw ground truth array center (yellow star)
    if (groundTruth.vy_pos && groundTruth.vy_neg && groundTruth.vx_pos && groundTruth.vx_neg) {
        const centerRow = (groundTruth.vy_pos[0] + groundTruth.vy_neg[0]) / 2;
        const centerCol = (groundTruth.vx_pos[1] + groundTruth.vx_neg[1]) / 2;
        const cx = (centerCol + 0.5) * cellSize;
        const cy = (centerRow + 0.5) * cellSize;

        // Draw star
        ctx.beginPath();
        const spikes = 5;
        const outerRadius = cellSize * 0.8;
        const innerRadius = cellSize * 0.4;
        for (let i = 0; i < spikes * 2; i++) {
            const radius = i % 2 === 0 ? outerRadius : innerRadius;
            const angle = (i * Math.PI / spikes) - Math.PI / 2;
            const px = cx + Math.cos(angle) * radius;
            const py = cy + Math.sin(angle) * radius;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fillStyle = '#fbbf24';  // yellow
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// Global map colormap (dark -> cyan -> green for visited areas)
function globalEvidenceToColor(value, confidence) {
    // Base color from evidence
    const v = Math.max(0, Math.min(1, value));
    const c = Math.max(0, Math.min(1, confidence));

    if (c < 0.1) {
        // Unexplored - dark gray
        return 'rgb(20, 20, 30)';
    }

    // Blend based on evidence
    if (v < 0.3) {
        // Low evidence - dim cyan
        const intensity = 40 + v * 100;
        return `rgb(${Math.round(intensity * 0.3)}, ${Math.round(intensity * 0.5)}, ${Math.round(intensity * 0.6)})`;
    } else if (v < 0.6) {
        // Medium evidence - cyan to green
        const t = (v - 0.3) / 0.3;
        const r = Math.round(34 * (1 - t) + 34 * t);
        const g = Math.round(211 * (1 - t) + 197 * t);
        const b = Math.round(238 * (1 - t) + 94 * t);
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // High evidence - bright green to yellow
        const t = (v - 0.6) / 0.4;
        const r = Math.round(34 + (250 - 34) * t);
        const g = Math.round(197 + (204 - 197) * t);
        const b = Math.round(94 + (21 - 94) * t);
        return `rgb(${r}, ${g}, ${b})`;
    }
}

function renderGlobalMap(globalMapping) {
    if (!globalMapping || !globalCtx) return;

    const evidence = globalMapping.global_evidence;
    const confidence = globalMapping.global_confidence;

    if (!evidence || !confidence) return;

    // Clear canvas
    globalCtx.fillStyle = '#0a0a0f';
    globalCtx.fillRect(0, 0, globalCanvasSize, globalCanvasSize);

    // Draw each cell
    const displaySize = evidence.length; // 48x48
    const cellDisplaySize = globalCanvasSize / displaySize;

    for (let row = 0; row < displaySize; row++) {
        for (let col = 0; col < displaySize; col++) {
            const x = col * cellDisplaySize;
            const y = row * cellDisplaySize;
            const evidenceVal = evidence[row][col];
            const confidenceVal = confidence[row][col];

            globalCtx.fillStyle = globalEvidenceToColor(evidenceVal, confidenceVal);

            // Draw cell
            globalCtx.fillRect(x, y, cellDisplaySize - 0.5, cellDisplaySize - 0.5);
        }
    }

    // Draw hotspots
    if (globalMapping.hotspots) {
        for (const hotspot of globalMapping.hotspots) {
            const [gRow, gCol] = hotspot.global_position;
            // Convert from 96x96 to 48x48 display coordinates
            const displayRow = gRow / 2;
            const displayCol = gCol / 2;
            const x = displayCol * cellDisplaySize;
            const y = displayRow * cellDisplaySize;

            // Draw hotspot circle
            globalCtx.beginPath();
            globalCtx.arc(x, y, 6, 0, 2 * Math.PI);
            globalCtx.strokeStyle = '#facc15';
            globalCtx.lineWidth = 2;
            globalCtx.stroke();

            // Add glow effect
            globalCtx.shadowColor = 'rgba(250, 204, 21, 0.6)';
            globalCtx.shadowBlur = 8;
            globalCtx.beginPath();
            globalCtx.arc(x, y, 4, 0, 2 * Math.PI);
            globalCtx.fillStyle = '#facc15';
            globalCtx.fill();
            globalCtx.shadowBlur = 0;
        }
    }
}

function updateArrayPositionIndicator(globalMapping) {
    const indicator = document.getElementById('array-indicator');
    const wrapper = document.querySelector('.global-canvas-wrapper');

    if (!globalMapping || !globalMapping.array_bounds) {
        indicator.style.display = 'none';
        return;
    }

    const [r1, c1, r2, c2] = globalMapping.array_bounds;

    // Convert from 96x96 global coords to display coords
    const scale = globalCanvasSize / globalMapSize;
    const x = c1 * scale + 16; // +16 for padding
    const y = r1 * scale + 16;
    const width = (c2 - c1) * scale;
    const height = (r2 - r1) * scale;

    indicator.style.left = `${x}px`;
    indicator.style.top = `${y}px`;
    indicator.style.width = `${width}px`;
    indicator.style.height = `${height}px`;
    indicator.style.display = 'block';
}

function updateExplorationDisplay(globalMapping) {
    if (!globalMapping) return;

    // Update exploration bar
    const coverage = globalMapping.exploration_coverage || 0;
    const coveragePercent = Math.round(coverage * 100);
    document.getElementById('exploration-bar').style.width = `${coveragePercent}%`;
    document.getElementById('exploration-value').textContent = `${coveragePercent}%`;

    // Update exploration suggestion
    const suggestion = globalMapping.exploration_suggestion;
    const suggestionEl = document.getElementById('suggestion-value');
    if (suggestion) {
        suggestionEl.textContent = suggestion.toUpperCase();
        suggestionEl.style.color = '#f97316';
    } else {
        suggestionEl.textContent = 'COMPLETE';
        suggestionEl.style.color = '#22c55e';
    }

    // Update array position display
    if (globalMapping.array_position) {
        const [row, col] = globalMapping.array_position;
        document.getElementById('array-position-value').textContent =
            `(${row.toFixed(1)}, ${col.toFixed(1)})`;
    }

    // Update global hotspot count
    const hotspotCount = globalMapping.hotspots ? globalMapping.hotspots.length : 0;
    document.getElementById('global-hotspot-count').textContent = hotspotCount;
}

function updateCentroidMarker(centroid) {
    const marker = document.getElementById('centroid-marker');
    const wrapper = document.querySelector('.canvas-wrapper');

    if (!centroid || centroid[0] === null) {
        marker.classList.remove('visible');
        return;
    }

    // Convert grid coordinates to pixel position
    // centroid is (row, col), 0-indexed - flip to match neural data display
    const x = padding + (gridSize - 1 - centroid[1] + 0.5) * cellSize;
    const y = padding + (gridSize - 1 - centroid[0] + 0.5) * cellSize;

    marker.style.left = `${x}px`;
    marker.style.top = `${y}px`;
    marker.classList.add('visible');
}

function updateGuidance(guidance) {
    const directionText = document.getElementById('direction-text');
    const arrowSvg = document.getElementById('guidance-arrow');
    const distanceValue = document.getElementById('distance-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');
    const centeredIndicator = document.getElementById('centered-indicator');
    const centerMarker = document.getElementById('center-marker');

    if (!guidance) {
        directionText.textContent = 'SEARCHING...';
        directionText.className = 'direction-text searching';
        arrowSvg.innerHTML = '';
        distanceValue.textContent = '--';
        confidenceBar.style.width = '0%';
        confidenceValue.textContent = '0%';
        centeredIndicator.classList.remove('visible');
        centerMarker.classList.remove('centered');
        return;
    }

    // Update direction text
    directionText.textContent = guidance.direction;

    if (guidance.is_centered) {
        directionText.className = 'direction-text centered';
        arrowSvg.className = 'guidance-arrow centered';
        centeredIndicator.classList.add('visible');
        centerMarker.classList.add('centered');
    } else {
        directionText.className = 'direction-text';
        arrowSvg.className = 'guidance-arrow';
        centeredIndicator.classList.remove('visible');
        centerMarker.classList.remove('centered');
    }

    // Update arrow
    const arrowPath = ARROW_PATHS[guidance.arrow] || '';
    if (arrowPath) {
        arrowSvg.innerHTML = `<path d="${arrowPath}" />`;
    } else {
        arrowSvg.innerHTML = '';
    }

    // Update distance (convert grid units to approximate mm, assuming 0.4mm electrode pitch)
    const distanceMm = (guidance.distance * 0.4).toFixed(1);
    distanceValue.textContent = `${distanceMm} mm`;

    // Update confidence
    const confidencePercent = Math.round(guidance.confidence * 100);
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;
}

function updateClusterCount(numClusters) {
    document.getElementById('cluster-count').textContent = numClusters || 0;
}

function updatePhaseDisplay(groundTruth) {
    // Update phase indicator if element exists
    const phaseEl = document.getElementById('phase-display');
    if (phaseEl && groundTruth.phase) {
        phaseEl.textContent = `Phase: ${groundTruth.phase}`;
    }

    // Update velocity display if element exists
    const velEl = document.getElementById('velocity-display');
    if (velEl && groundTruth.vx !== undefined) {
        velEl.textContent = `Vel: (${groundTruth.vx.toFixed(1)}, ${groundTruth.vy.toFixed(1)})`;
    }
}

function toggleGroundTruth() {
    showGroundTruth = !showGroundTruth;
    const btn = document.getElementById('gt-toggle-btn');
    if (btn) {
        btn.textContent = showGroundTruth ? 'Hide GT' : 'Show GT';
        btn.classList.toggle('active', showGroundTruth);
    }
}

function updateSignalQuality(badChannels) {
    if (!badChannels) {
        document.getElementById('signal-quality').textContent = 'Signal: --';
        return;
    }

    // Count bad channels
    let badCount = 0;
    for (let row of badChannels) {
        for (let val of row) {
            if (val) badCount++;
        }
    }

    const totalChannels = gridSize * gridSize;
    const goodPercent = ((totalChannels - badCount) / totalChannels * 100).toFixed(0);

    let quality;
    if (goodPercent >= 98) quality = 'GOOD';
    else if (goodPercent >= 95) quality = 'OK';
    else quality = 'POOR';

    document.getElementById('signal-quality').textContent = `Signal: ${quality}`;
}

function updateStatus(status, text) {
    const indicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const connectBtn = document.getElementById('connect-btn');

    indicator.className = 'status-indicator';

    switch (status) {
        case 'connected':
            indicator.classList.add('connected');
            statusText.textContent = 'Connected';
            connectBtn.textContent = 'Disconnect';
            connectBtn.classList.add('disconnect');
            connectBtn.disabled = false;
            isConnected = true;
            break;
        case 'connecting':
            indicator.classList.add('connecting');
            statusText.textContent = 'Connecting...';
            connectBtn.disabled = true;
            break;
        case 'disconnected':
        default:
            statusText.textContent = text || 'Disconnected';
            connectBtn.textContent = 'Connect';
            connectBtn.classList.remove('disconnect');
            connectBtn.disabled = false;
            isConnected = false;
            break;
    }
}

function updateFps() {
    frameCount++;
    const now = performance.now();
    if (now - lastFpsUpdate >= 1000) {
        currentFps = frameCount;
        frameCount = 0;
        lastFpsUpdate = now;
        document.getElementById('fps-counter').textContent = `${currentFps} FPS`;
    }
}

function processMessage(data) {
    if (data.type === 'init') {
        gridSize = data.grid_size || 32;
        console.log(`Initialized with grid size ${gridSize}`);

    } else if (data.type === 'processed') {
        currentData = data;

        // Update time
        const timeS = data.time_s || 0;
        document.getElementById('time-display').textContent = `t = ${timeS.toFixed(2)}s`;

        // Render grid using current activity (raw normalized power)
        renderGrid(data.current_activity, data.bad_channels);

        // Render ground truth overlay if available (dev mode)
        if (data.ground_truth) {
            renderGroundTruthOverlay(data.ground_truth);
            updatePhaseDisplay(data.ground_truth);
        }

        // Render global brain map
        if (data.global_mapping) {
            renderGlobalMap(data.global_mapping);
            updateArrayPositionIndicator(data.global_mapping);
            updateExplorationDisplay(data.global_mapping);
        }

        // Update centroid marker
        updateCentroidMarker(data.centroid);

        // Update guidance
        updateGuidance(data.guidance);

        // Update cluster count
        updateClusterCount(data.num_clusters);

        // Update signal quality
        updateSignalQuality(data.bad_channels);

        // Update FPS
        updateFps();
    }
}

function connect() {
    const url = document.getElementById('server-url').value;

    if (isConnected && ws) {
        ws.close();
        return;
    }

    updateStatus('connecting');

    try {
        ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('WebSocket connected');
            updateStatus('connected');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                processMessage(data);
            } catch (err) {
                console.error('Error parsing message:', err);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('disconnected', 'Connection error');
        };

        ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            updateStatus('disconnected', event.wasClean ? 'Disconnected' : 'Connection lost');
            ws = null;
        };

    } catch (err) {
        console.error('Failed to create WebSocket:', err);
        updateStatus('disconnected', 'Failed to connect');
    }
}

function reset() {
    // Clear the array canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    // Clear the global canvas
    if (globalCtx) {
        globalCtx.fillStyle = '#0a0a0f';
        globalCtx.fillRect(0, 0, globalCanvasSize, globalCanvasSize);
    }

    // Reset UI elements
    updateGuidance(null);
    updateClusterCount(0);
    document.getElementById('time-display').textContent = 't = 0.00s';
    document.getElementById('signal-quality').textContent = 'Signal: --';

    // Reset global map UI
    document.getElementById('exploration-bar').style.width = '0%';
    document.getElementById('exploration-value').textContent = '0%';
    document.getElementById('suggestion-value').textContent = '--';
    document.getElementById('array-position-value').textContent = '(0, 0)';
    document.getElementById('global-hotspot-count').textContent = '0';

    // Hide indicators
    document.getElementById('centroid-marker').classList.remove('visible');
    document.getElementById('center-marker').classList.remove('centered');
    document.getElementById('array-indicator').style.display = 'none';
}

function init() {
    initCanvas();
    initGlobalCanvas();

    // Connect button handler
    document.getElementById('connect-btn').addEventListener('click', connect);

    // Reset button handler
    document.getElementById('reset-btn').addEventListener('click', reset);

    // Ground truth toggle button handler
    const gtToggle = document.getElementById('gt-toggle-btn');
    if (gtToggle) {
        gtToggle.addEventListener('click', toggleGroundTruth);
    }

    // Enter key in URL input
    document.getElementById('server-url').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            connect();
        }
    });
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
