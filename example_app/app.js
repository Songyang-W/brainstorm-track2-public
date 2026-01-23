/**
 * Compass HUD - WebSocket Client
 *
 * Consumes backend messages from scripts/compass_backend.py (brainstorm-compass)
 * and renders a surgeon-ready guidance UI.
 */

let ws = null;
let isConnected = false;
let gridSize = 32;

// DOM
const el = {};

// Canvases
let heatCanvas, heatCtx;
let memCanvas, memCtx;

// Running normalization for heatmaps (prevents flicker).
let heatMax = 1.0;
let memMax = 1.0;

function $(id) {
  return document.getElementById(id);
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function setStatus(mode) {
  const dot = el.statusIndicator;
  dot.className = "dot";

  if (mode === "connected") {
    dot.classList.add("connected");
    el.statusText.textContent = "CONNECTED";
    el.connectBtn.textContent = "DISCONNECT";
    isConnected = true;
  } else if (mode === "connecting") {
    dot.classList.add("connecting");
    el.statusText.textContent = "CONNECTING";
    el.connectBtn.textContent = "CONNECT";
    isConnected = false;
  } else {
    el.statusText.textContent = "DISCONNECTED";
    el.connectBtn.textContent = "CONNECT";
    isConnected = false;
  }
}

// Amber -> green ramp with deep-black floor (designed for OR contrast).
function colorRamp(t) {
  const x = clamp(t, 0, 1);
  const r0 = 10, g0 = 12, b0 = 12;   // floor
  const r1 = 255, g1 = 176, b1 = 32; // amber
  const r2 = 44, g2 = 255, b2 = 179; // green

  // Two-stage ramp: floor->amber->green
  const k = clamp(x * 1.2, 0, 1);
  const mid = 0.55;
  let r, g, b;
  if (k < mid) {
    const tt = k / mid;
    r = lerp(r0, r1, tt);
    g = lerp(g0, g1, tt);
    b = lerp(b0, b1, tt);
  } else {
    const tt = (k - mid) / (1 - mid);
    r = lerp(r1, r2, tt);
    g = lerp(g1, g2, tt);
    b = lerp(b1, b2, tt);
  }
  return [r | 0, g | 0, b | 0];
}

function drawHeatmap(ctx, arr2d, maxRef, overlays) {
  if (!arr2d) return maxRef;
  const h = arr2d.length;
  const w = arr2d[0].length;

  // Find frame max with small decay for stability.
  let frameMax = 0.0;
  for (let r = 0; r < h; r++) {
    const row = arr2d[r];
    for (let c = 0; c < w; c++) frameMax = Math.max(frameMax, row[c]);
  }
  const targetMax = Math.max(1e-6, frameMax);
  maxRef = Math.max(targetMax, maxRef * 0.985);

  const img = ctx.createImageData(w, h);
  const d = img.data;
  for (let r = 0; r < h; r++) {
    const row = arr2d[r];
    for (let c = 0; c < w; c++) {
      const v = row[c] / maxRef;
      const [rr, gg, bb] = colorRamp(v);
      const i = (r * w + c) * 4;
      d[i + 0] = rr;
      d[i + 1] = gg;
      d[i + 2] = bb;
      d[i + 3] = 255;
    }
  }

  // Render into a tiny offscreen canvas, then scale up crisply.
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const offCtx = off.getContext("2d");
  offCtx.putImageData(img, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(off, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();

  // Overlays in canvas coordinate space (scaled).
  const sx = ctx.canvas.width / w;
  const sy = ctx.canvas.height / h;

  if (overlays && overlays.center) {
    const { row, col } = overlays.center;
    ctx.save();
    ctx.translate((col + 0.5) * sx, (row + 0.5) * sy);
    ctx.strokeStyle = "rgba(125, 227, 255, 0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-12, 0);
    ctx.lineTo(12, 0);
    ctx.moveTo(0, -12);
    ctx.lineTo(0, 12);
    ctx.stroke();
    ctx.restore();
  }

  if (overlays && overlays.arms) {
    ctx.save();
    ctx.strokeStyle = "rgba(44, 255, 179, 0.85)";
    ctx.fillStyle = "rgba(44, 255, 179, 0.85)";
    ctx.lineWidth = 2;
    for (const p of overlays.arms) {
      ctx.beginPath();
      ctx.arc((p.col + 0.5) * sx, (p.row + 0.5) * sy, 6, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Grid lines (subtle)
  ctx.save();
  ctx.strokeStyle = "rgba(244, 241, 234, 0.08)";
  ctx.lineWidth = 1;
  for (let k = 0; k <= w; k += 4) {
    ctx.beginPath();
    ctx.moveTo(k * sx, 0);
    ctx.lineTo(k * sx, ctx.canvas.height);
    ctx.stroke();
  }
  for (let k = 0; k <= h; k += 4) {
    ctx.beginPath();
    ctx.moveTo(0, k * sy);
    ctx.lineTo(ctx.canvas.width, k * sy);
    ctx.stroke();
  }
  ctx.restore();

  return maxRef;
}

function frameTo2DArray(frameArr) {
  // frameArr is a nested list from Python (list[list[float]])
  if (!frameArr) return null;
  return frameArr;
}

function updateUI(frame) {
  const t = frame.t_s ?? 0.0;
  const conf = clamp(frame.confidence ?? 0.0, 0, 1);
  const dist = frame.distance ?? null;

  el.timeDisplay.textContent = `${t.toFixed(2)}s`;
  el.confidenceDisplay.textContent = `${Math.round(conf * 100)}%`;
  el.confidenceBar.style.width = `${Math.round(conf * 100)}%`;

  if (dist === null || dist === undefined) {
    el.distanceDisplay.textContent = "--";
  } else {
    el.distanceDisplay.textContent = `${dist.toFixed(1)} cells`;
  }

  // Arrow angle: move_col is x, move_row is y (down positive).
  const mvR = frame.move_row ?? 0.0;
  const mvC = frame.move_col ?? 0.0;
  const angle = Math.atan2(mvR, mvC) * (180 / Math.PI);
  el.arrow.style.setProperty("--angle", `${angle}deg`);

  // Text direction shorthand
  const dirX = mvC > 0.15 ? "RIGHT" : mvC < -0.15 ? "LEFT" : "—";
  const dirY = mvR > 0.15 ? "DOWN" : mvR < -0.15 ? "UP" : "—";
  el.moveDisplay.textContent = `${dirY} / ${dirX}`;

  const locked = (dist !== null && dist < 1.6 && conf > 0.78);
  el.lock.classList.toggle("on", locked);
  el.arrow.style.opacity = locked ? "0.0" : "1.0";

  if (locked) {
    el.modeText.textContent = "LOCKED";
    el.modeText.style.color = "rgba(44, 255, 179, 0.9)";
  } else if (conf < 0.35) {
    el.modeText.textContent = "SEARCHING";
    el.modeText.style.color = "rgba(255, 176, 32, 0.9)";
  } else {
    el.modeText.textContent = "TRACKING";
    el.modeText.style.color = "rgba(244, 241, 234, 0.75)";
  }

  // Heatmaps
  const center = { row: frame.center_row, col: frame.center_col };
  const dx = frame.dx ?? 6.0;
  const dy = frame.dy ?? 6.0;
  const arms = [
    { row: center.row, col: center.col + dx },
    { row: center.row, col: center.col - dx },
    { row: center.row + dy, col: center.col },
    { row: center.row - dy, col: center.col },
  ];

  const heat2d = frameTo2DArray(frame.heatmap);
  const mem2d = frameTo2DArray(frame.memory);

  if (heat2d) heatMax = drawHeatmap(heatCtx, heat2d, heatMax, { center, arms });
  if (mem2d) memMax = drawHeatmap(memCtx, mem2d, memMax, null);
}

function connect() {
  const url = el.serverUrl.value.trim();
  if (!url) return;

  if (isConnected && ws) {
    ws.close();
    return;
  }

  setStatus("connecting");

  try {
    ws = new WebSocket(url);

    ws.onopen = () => {
      setStatus("connected");
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
      }

      if (data.type === "compass_frame") {
        updateUI(data);
      }
    };

    ws.onerror = () => {
      setStatus("disconnected");
    };

    ws.onclose = () => {
      ws = null;
      setStatus("disconnected");
    };
  } catch {
    setStatus("disconnected");
  }
}

function reset() {
  // Soft reset: clear canvases + normalization. Backend reset is a future enhancement.
  heatMax = 1.0;
  memMax = 1.0;
  heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
  memCtx.clearRect(0, 0, memCanvas.width, memCanvas.height);
  el.modeText.textContent = "SEARCHING";
  el.lock.classList.remove("on");
  el.arrow.style.opacity = "1.0";
}

function init() {
  el.statusIndicator = $("status-indicator");
  el.statusText = $("status-text");
  el.timeDisplay = $("time-display");
  el.confidenceDisplay = $("confidence-display");
  el.distanceDisplay = $("distance-display");
  el.connectBtn = $("connect-btn");
  el.resetBtn = $("reset-btn");
  el.serverUrl = $("server-url");
  el.arrow = $("arrow");
  el.lock = $("lock");
  el.moveDisplay = $("move-display");
  el.confidenceBar = $("confidence-bar");
  el.modeText = $("mode-text");

  heatCanvas = $("heatmap-canvas");
  memCanvas = $("memory-canvas");
  heatCtx = heatCanvas.getContext("2d");
  memCtx = memCanvas.getContext("2d");

  setStatus("disconnected");
  reset();

  el.connectBtn.addEventListener("click", connect);
  el.resetBtn.addEventListener("click", reset);
  el.serverUrl.addEventListener("keypress", (e) => {
    if (e.key === "Enter") connect();
  });
}

document.addEventListener("DOMContentLoaded", init);
