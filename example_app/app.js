/**
 * Compass | OR Guidance UI
 * Consumes ws frames from scripts/compass_backend.py (brainstorm-compass).
 */

let ws = null;
let isConnected = false;
let gridSize = 32;

// DOM
const el = {};

// Canvases
let heatCanvas, heatCtx, heatOff, heatOffCtx;
let memCanvas, memCtx, memOff, memOffCtx;

// Running normalization to reduce flicker.
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
  const chip = el.stateChip;
  chip.classList.remove("connected", "connecting", "acquiring", "locked");

  const dot = el.statusIndicator;
  dot.style.opacity = "1";

  if (mode === "connected") {
    chip.classList.add("connected");
    el.statusText.textContent = "CONNECTED";
    el.connectBtn.textContent = "Disconnect";
    isConnected = true;
  } else if (mode === "connecting") {
    chip.classList.add("connecting");
    el.statusText.textContent = "CONNECTING";
    el.connectBtn.textContent = "Connect";
    isConnected = false;
  } else {
    chip.classList.add("acquiring");
    el.statusText.textContent = "DISCONNECTED";
    el.connectBtn.textContent = "Connect";
    isConnected = false;
  }
}

function setMode(kind) {
  // kind: acquiring | tracking | locked
  el.dialWrap.classList.toggle("locked", kind === "locked");
  el.lock.classList.toggle("on", kind === "locked");

  if (kind === "locked") {
    el.stateChip.classList.add("locked");
  } else {
    el.stateChip.classList.remove("locked");
  }
}

// Color ramp tuned for OR: deep black -> amber -> surgical green.
function colorRamp(t) {
  const x = clamp(t, 0, 1);
  const r0 = 6, g0 = 7, b0 = 10;      // floor
  const r1 = 243, g1 = 180, b1 = 75;  // amber
  const r2 = 57, g2 = 246, b2 = 193;  // green

  // Contrast curve (brings out structure without blasting the background)
  const y = Math.pow(x, 0.72);
  const mid = 0.56;
  let r, g, b;
  if (y < mid) {
    const tt = y / mid;
    r = lerp(r0, r1, tt);
    g = lerp(g0, g1, tt);
    b = lerp(b0, b1, tt);
  } else {
    const tt = (y - mid) / (1 - mid);
    r = lerp(r1, r2, tt);
    g = lerp(g1, g2, tt);
    b = lerp(b1, b2, tt);
  }
  return [r | 0, g | 0, b | 0];
}

function drawHeatmap(ctx, offCtx, offCanvas, arr2d, maxRef, overlays) {
  if (!arr2d) return maxRef;
  const h = arr2d.length;
  const w = arr2d[0].length;

  // Frame max + smoothing for stability.
  let frameMax = 0.0;
  for (let r = 0; r < h; r++) {
    const row = arr2d[r];
    for (let c = 0; c < w; c++) frameMax = Math.max(frameMax, row[c]);
  }
  const targetMax = Math.max(1e-6, frameMax);
  maxRef = Math.max(targetMax, maxRef * 0.987);

  const img = offCtx.createImageData(w, h);
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

  offCtx.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(offCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();

  const sx = ctx.canvas.width / w;
  const sy = ctx.canvas.height / h;

  // Subtle grid for orientation (every 4 cells)
  ctx.save();
  ctx.strokeStyle = "rgba(245,246,247,0.07)";
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

  if (overlays && overlays.center) {
    const { row, col } = overlays.center;
    ctx.save();
    ctx.translate((col + 0.5) * sx, (row + 0.5) * sy);
    ctx.strokeStyle = "rgba(122,166,255,0.92)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-14, 0);
    ctx.lineTo(14, 0);
    ctx.moveTo(0, -14);
    ctx.lineTo(0, 14);
    ctx.stroke();
    ctx.restore();
  }

  if (overlays && overlays.anchors && overlays.anchors.length) {
    const anchors = overlays.anchors;
    let maxW = 0;
    for (const a of anchors) maxW = Math.max(maxW, a.w || 0);
    maxW = Math.max(1e-6, maxW);

    ctx.save();
    ctx.lineWidth = 2;
    for (const a of anchors) {
      const rel = clamp((a.w || 0) / maxW, 0, 1);
      const rad = clamp(4 + rel * 7, 4, 11);
      ctx.strokeStyle = `rgba(57,246,193,${0.35 + rel * 0.45})`;
      ctx.fillStyle = `rgba(57,246,193,${0.08 + rel * 0.10})`;
      ctx.beginPath();
      ctx.arc((a.col + 0.5) * sx, (a.row + 0.5) * sy, rad, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  return maxRef;
}

function parse2D(frameArr) {
  return frameArr || null;
}

function vectorText(moveRow, moveCol) {
  const r = moveRow || 0.0;
  const c = moveCol || 0.0;

  const mag = Math.hypot(r, c);
  if (mag < 0.08) return "HOLD";

  const vert = r < -0.15 ? "UP" : r > 0.15 ? "DOWN" : null;
  const horz = c < -0.15 ? "LEFT" : c > 0.15 ? "RIGHT" : null;
  if (vert && horz) return `MOVE ${vert} ${horz}`;
  if (vert) return `MOVE ${vert}`;
  if (horz) return `MOVE ${horz}`;
  return "ADJUST";
}

function updateUI(frame) {
  const t = frame.t_s ?? 0.0;
  const conf = clamp(frame.confidence ?? 0.0, 0, 1);
  const dist = frame.distance ?? null;

  el.timeDisplay.textContent = `${t.toFixed(2)}s`;
  el.confidenceDisplay.textContent = `${Math.round(conf * 100)}%`;
  el.distanceDisplay.textContent = dist === null || dist === undefined ? "--" : `${dist.toFixed(1)} grid`;

  const mvR = frame.move_row ?? 0.0;
  const mvC = frame.move_col ?? 0.0;
  const angle = Math.atan2(mvR, mvC) * (180 / Math.PI);
  el.needle.style.setProperty("--angle", `${angle}deg`);

  const moveText = vectorText(mvR, mvC);
  el.moveText.textContent = moveText;

  const locked = (dist !== null && dist < 1.35 && conf > 0.78);
  setMode(locked ? "locked" : conf < 0.35 ? "acquiring" : "tracking");

  // More explicit operator-facing readouts
  const mid = (gridSize - 1) / 2.0;
  const dRow = (frame.center_row ?? mid) - mid;
  const dCol = (frame.center_col ?? mid) - mid;
  el.offsetDisplay.textContent = `${dRow.toFixed(1)}, ${dCol.toFixed(1)}`;

  const qual = conf > 0.78 ? "HIGH" : conf > 0.5 ? "OK" : conf > 0.35 ? "LOW" : "ACQ";
  el.qualityDisplay.textContent = qual;

  // Heatmaps
  const heat2d = parse2D(frame.heatmap);
  const mem2d = parse2D(frame.memory);

  const anchorsLive = Array.isArray(frame.spots) ? frame.spots : [];
  const anchorsMem = Array.isArray(frame.spots_mem) ? frame.spots_mem : [];

  const live = anchorsLive.slice(0, 8).map((p) => ({ row: p[0], col: p[1], w: p[2] }));
  const mem = anchorsMem.slice(0, 8).map((p) => ({ row: p[0], col: p[1], w: p[2] }));

  if (heat2d) {
    heatMax = drawHeatmap(
      heatCtx,
      heatOffCtx,
      heatOff,
      heat2d,
      heatMax,
      { center: { row: frame.center_row, col: frame.center_col }, anchors: live }
    );
  }

  if (mem2d) {
    const midPt = (gridSize - 1) / 2.0;
    memMax = drawHeatmap(
      memCtx,
      memOffCtx,
      memOff,
      mem2d,
      memMax,
      { center: { row: midPt, col: midPt }, anchors: mem }
    );
  }
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
      setMode("acquiring");
    };
  } catch {
    setStatus("disconnected");
  }
}

function reset() {
  heatMax = 1.0;
  memMax = 1.0;
  heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
  memCtx.clearRect(0, 0, memCanvas.width, memCanvas.height);

  el.moveText.textContent = "ACQUIRING…";
  el.offsetDisplay.textContent = "—";
  el.qualityDisplay.textContent = "—";
  setMode("acquiring");

  // Ask backend to clear tracking memory.
  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset" }));
    }
  } catch {
    // Ignore.
  }
}

function init() {
  el.stateChip = $("state-chip");
  el.statusIndicator = $("status-indicator");
  el.statusText = $("status-text");
  el.timeDisplay = $("time-display");
  el.confidenceDisplay = $("confidence-display");
  el.distanceDisplay = $("distance-display");
  el.connectBtn = $("connect-btn");
  el.resetBtn = $("reset-btn");
  el.serverUrl = $("server-url");

  el.dialWrap = $("dial-wrap");
  el.needle = $("needle");
  el.lock = $("lock");
  el.moveText = $("move-text");
  el.offsetDisplay = $("offset-display");
  el.qualityDisplay = $("quality-display");

  heatCanvas = $("heatmap-canvas");
  memCanvas = $("memory-canvas");
  heatCtx = heatCanvas.getContext("2d");
  memCtx = memCanvas.getContext("2d");

  // Offscreen buffers to avoid recreating canvases every frame.
  heatOff = document.createElement("canvas");
  memOff = document.createElement("canvas");
  heatOff.width = gridSize;
  heatOff.height = gridSize;
  memOff.width = gridSize;
  memOff.height = gridSize;
  heatOffCtx = heatOff.getContext("2d");
  memOffCtx = memOff.getContext("2d");

  setStatus("disconnected");
  setMode("acquiring");
  reset();

  el.connectBtn.addEventListener("click", connect);
  el.resetBtn.addEventListener("click", reset);
  el.serverUrl.addEventListener("keypress", (e) => {
    if (e.key === "Enter") connect();
  });
}

document.addEventListener("DOMContentLoaded", init);

