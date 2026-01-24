/**
 * Array Placement Guide - Simplified Surgeon UI
 * ONE JOB: Tell the surgeon which way to move
 */

// ============================================
// State
// ============================================

let ws = null;
let isConnected = false;
let gridSize = 32;

// Heatmap state (for debug view)
let liveCanvas, liveCtx, liveOff, liveOffCtx;
let memCanvas, memCtx, memOff, memOffCtx;
let liveMax = 1.0;
let memMax = 1.0;

// DOM elements
const dom = {};

// ============================================
// Utilities
// ============================================

function $(id) {
  return document.getElementById(id);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// ============================================
// Direction Text
// ============================================

function getDirectionText(moveRow, moveCol) {
  const r = moveRow || 0;
  const c = moveCol || 0;
  const mag = Math.hypot(r, c);

  if (mag < 0.08) return "HOLD";

  // Determine direction
  const up = r < -0.1;
  const down = r > 0.1;
  const left = c < -0.1;
  const right = c > 0.1;

  if (up && left) return "UP LEFT";
  if (up && right) return "UP RIGHT";
  if (down && left) return "DOWN LEFT";
  if (down && right) return "DOWN RIGHT";
  if (up) return "UP";
  if (down) return "DOWN";
  if (left) return "LEFT";
  if (right) return "RIGHT";

  return "ADJUST";
}

// ============================================
// App State Management
// ============================================

function setAppState(state) {
  const app = dom.app;
  app.classList.remove("connected", "connecting", "acquiring", "on-target", "show-debug");

  if (state.connected) app.classList.add("connected");
  if (state.connecting) app.classList.add("connecting");
  if (state.acquiring) app.classList.add("acquiring");
  if (state.onTarget) app.classList.add("on-target");
  if (state.showDebug) app.classList.add("show-debug");
}

function setConnectionStatus(status) {
  if (status === "connected") {
    dom.statusText.textContent = "Connected";
    dom.connectBtn.textContent = "Disconnect";
    isConnected = true;
    setAppState({ connected: true, showDebug: dom.showDebug.checked });
  } else if (status === "connecting") {
    dom.statusText.textContent = "Connecting...";
    dom.connectBtn.textContent = "Cancel";
    isConnected = false;
    setAppState({ connecting: true, showDebug: dom.showDebug.checked });
  } else {
    dom.statusText.textContent = "Disconnected";
    dom.connectBtn.textContent = "Connect";
    isConnected = false;
    setAppState({ showDebug: dom.showDebug.checked });
  }
}

// ============================================
// Arrow Rendering
// ============================================

function updateArrow(moveRow, moveCol, onTarget) {
  const dx = clamp(moveCol || 0, -1, 1);
  const dy = clamp(moveRow || 0, -1, 1);
  const mag = Math.min(1, Math.hypot(dx, dy));

  const cx = 100, cy = 100;
  const maxLen = 65;
  const headSize = 15;

  if (onTarget || mag < 0.05) {
    dom.arrowShaft.setAttribute("x2", cx);
    dom.arrowShaft.setAttribute("y2", cy);
    dom.arrowHead.setAttribute("points", "100,100 100,100 100,100");
    return;
  }

  // Calculate direction
  const normDx = dx / (mag || 1);
  const normDy = dy / (mag || 1);
  const len = maxLen * Math.min(1, mag * 1.5);

  const x2 = cx + normDx * len;
  const y2 = cy + normDy * len;

  dom.arrowShaft.setAttribute("x2", x2.toFixed(1));
  dom.arrowShaft.setAttribute("y2", y2.toFixed(1));

  // Arrowhead
  const angle = Math.atan2(normDy, normDx);
  const ha = Math.PI / 5;

  const tipX = x2 + normDx * headSize;
  const tipY = y2 + normDy * headSize;
  const lX = x2 - headSize * 0.6 * Math.cos(angle - ha);
  const lY = y2 - headSize * 0.6 * Math.sin(angle - ha);
  const rX = x2 - headSize * 0.6 * Math.cos(angle + ha);
  const rY = y2 - headSize * 0.6 * Math.sin(angle + ha);

  dom.arrowHead.setAttribute("points", `${tipX},${tipY} ${lX},${lY} ${rX},${rY}`);
}

// ============================================
// Distance Bar
// ============================================

function updateDistanceBar(distance, maxDist = 15) {
  // Distance of 0 = on target (right side), high distance = far (left side)
  const pct = clamp(1 - (distance / maxDist), 0, 1) * 100;
  dom.distanceMarker.style.left = `${pct}%`;
}

// ============================================
// Heatmap Rendering (Debug View)
// ============================================

function heatmapColor(t) {
  const x = clamp(t, 0, 1);
  const colors = [
    [20, 30, 48],
    [30, 80, 100],
    [80, 160, 140],
    [220, 180, 80],
    [255, 250, 240],
  ];

  const n = colors.length - 1;
  const idx = x * n;
  const i = Math.min(Math.floor(idx), n - 1);
  const frac = idx - i;

  return [
    Math.round(lerp(colors[i][0], colors[i + 1][0], frac)),
    Math.round(lerp(colors[i][1], colors[i + 1][1], frac)),
    Math.round(lerp(colors[i][2], colors[i + 1][2], frac)),
  ];
}

function drawHeatmap(ctx, offCtx, offCanvas, arr2d, maxRef) {
  if (!arr2d || !arr2d.length) return maxRef;

  const h = arr2d.length;
  const w = arr2d[0].length;

  let frameMax = 0;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      frameMax = Math.max(frameMax, arr2d[r][c]);
    }
  }
  const targetMax = Math.max(1e-6, frameMax);
  maxRef = Math.max(targetMax, maxRef * 0.99);

  const img = offCtx.createImageData(w, h);
  const d = img.data;

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = arr2d[r][c] / maxRef;
      const [rr, gg, bb] = heatmapColor(v);
      const i = (r * w + c) * 4;
      d[i] = rr;
      d[i + 1] = gg;
      d[i + 2] = bb;
      d[i + 3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(offCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();

  return maxRef;
}

// ============================================
// Main UI Update
// ============================================

function updateUI(frame) {
  const t = frame.t_s ?? 0;
  const conf = clamp(frame.confidence ?? 0, 0, 1);
  const dist = frame.distance ?? 10;
  const mvR = frame.move_row ?? 0;
  const mvC = frame.move_col ?? 0;

  // Determine state
  const onTarget = dist < 1.5 && conf > 0.65;
  const acquiring = conf < 0.35;

  // Update time
  dom.timeDisplay.textContent = `${t.toFixed(1)}s`;

  // Update app state
  setAppState({
    connected: isConnected,
    acquiring: acquiring && !onTarget,
    onTarget: onTarget,
    showDebug: dom.showDebug.checked,
  });

  // Update instruction text
  if (onTarget) {
    dom.instructionText.textContent = "ON TARGET";
    dom.instructionDetail.textContent = "Hold position";
  } else if (acquiring) {
    dom.instructionText.textContent = "ACQUIRING";
    dom.instructionDetail.textContent = "Move slowly";
  } else {
    const dir = getDirectionText(mvR, mvC);
    dom.instructionText.textContent = dir;
    dom.instructionDetail.textContent = `Distance: ${dist.toFixed(1)} units`;
  }

  // Update arrow
  updateArrow(mvR, mvC, onTarget);

  // Update distance bar
  updateDistanceBar(dist);

  // Update debug view if visible
  if (dom.showDebug.checked) {
    dom.debugCenter.textContent = `${(frame.center_row ?? 0).toFixed(1)}, ${(frame.center_col ?? 0).toFixed(1)}`;
    dom.debugConfidence.textContent = `${Math.round(conf * 100)}%`;
    dom.debugDistance.textContent = dist.toFixed(2);

    const regions = frame.regions || {};
    const regionCount = Object.keys(regions).length;
    dom.debugRegions.textContent = `${regionCount} active`;

    // Draw heatmaps
    if (frame.heatmap) {
      liveMax = drawHeatmap(liveCtx, liveOffCtx, liveOff, frame.heatmap, liveMax);
    }
    if (frame.memory) {
      memMax = drawHeatmap(memCtx, memOffCtx, memOff, frame.memory, memMax);
    }
  }
}

// ============================================
// WebSocket
// ============================================

function connect() {
  const url = dom.wsInput.value.trim();
  if (!url) return;

  if (isConnected && ws) {
    ws.close();
    return;
  }

  setConnectionStatus("connecting");

  try {
    ws = new WebSocket(url);

    ws.onopen = () => {
      setConnectionStatus("connected");
    };

    ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      if (data.type === "init") {
        gridSize = data.grid_size ?? 32;
        liveOff.width = gridSize;
        liveOff.height = gridSize;
        memOff.width = gridSize;
        memOff.height = gridSize;
      }

      if (data.type === "compass_frame") {
        updateUI(data);
      }
    };

    ws.onerror = () => {
      setConnectionStatus("disconnected");
    };

    ws.onclose = () => {
      ws = null;
      setConnectionStatus("disconnected");
    };

  } catch (e) {
    console.error("Connection error:", e);
    setConnectionStatus("disconnected");
  }
}

// ============================================
// Reset
// ============================================

function reset() {
  liveMax = 1.0;
  memMax = 1.0;

  if (liveCtx) liveCtx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);
  if (memCtx) memCtx.clearRect(0, 0, memCanvas.width, memCanvas.height);

  dom.instructionText.textContent = isConnected ? "ACQUIRING" : "CONNECT";
  dom.instructionDetail.textContent = isConnected ? "Move slowly" : "Press Connect below";

  updateArrow(0, 0, false);
  updateDistanceBar(10);

  setAppState({
    connected: isConnected,
    acquiring: isConnected,
    showDebug: dom.showDebug.checked,
  });

  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset" }));
    }
  } catch {
    // ignore
  }
}

// ============================================
// Init
// ============================================

function init() {
  // Cache DOM elements
  dom.app = $("app");
  dom.statusDot = $("status-dot");
  dom.statusText = $("status-text");
  dom.timeDisplay = $("time-display");

  dom.instructionText = $("instruction-text");
  dom.instructionDetail = $("instruction-detail");

  dom.arrowShaft = $("arrow-shaft");
  dom.arrowHead = $("arrow-head");

  dom.distanceFill = $("distance-fill");
  dom.distanceMarker = $("distance-marker");

  dom.controlsPanel = $("controls-panel");
  dom.controlsToggle = $("controls-toggle");
  dom.wsInput = $("ws-input");
  dom.connectBtn = $("connect-btn");
  dom.resetBtn = $("reset-btn");
  dom.showDebug = $("show-debug");

  dom.debugCenter = $("debug-center");
  dom.debugConfidence = $("debug-confidence");
  dom.debugDistance = $("debug-distance");
  dom.debugRegions = $("debug-regions");

  // Setup canvases
  liveCanvas = $("heatmap-live");
  memCanvas = $("heatmap-memory");
  liveCtx = liveCanvas.getContext("2d");
  memCtx = memCanvas.getContext("2d");

  liveOff = document.createElement("canvas");
  memOff = document.createElement("canvas");
  liveOff.width = gridSize;
  liveOff.height = gridSize;
  memOff.width = gridSize;
  memOff.height = gridSize;
  liveOffCtx = liveOff.getContext("2d");
  memOffCtx = memOff.getContext("2d");

  // Initial state
  setConnectionStatus("disconnected");
  reset();

  // Event listeners
  dom.connectBtn.addEventListener("click", connect);
  dom.resetBtn.addEventListener("click", reset);
  dom.wsInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") connect();
  });

  // Controls toggle
  dom.controlsToggle.addEventListener("click", () => {
    dom.controlsPanel.classList.toggle("expanded");
  });

  // Debug toggle
  dom.showDebug.addEventListener("change", () => {
    setAppState({
      connected: isConnected,
      showDebug: dom.showDebug.checked,
    });
  });

  // Start with controls expanded
  dom.controlsPanel.classList.add("expanded");
}

document.addEventListener("DOMContentLoaded", init);
