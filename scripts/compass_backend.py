#!/usr/bin/env python3
"""
Compass Backend (Realtime Processor)

Connects to the Track2 upstream stream, runs causal filtering + feature extraction,
tracks a stable tuned-region center (robust to missing arms + noise), and serves a
browser-friendly WebSocket at a lower frame rate (~10-20 Hz).

Run:
  uv run brainstorm-stream --from-file data/super_easy/
  uv run brainstorm-compass
  uv run brainstorm-serve

Browser UI connects to ws://localhost:8767 (default).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import typer
import websockets
from rich.console import Console
from rich.panel import Panel
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from scripts.hotspot_tracker import PersistentSpotClusterTracker, extract_peak_observations

app = typer.Typer(help="Realtime processing backend for the OR 'Compass' UI")
console = Console()


@dataclass
class FilterBank:
    sos_notch: np.ndarray
    sos_band: np.ndarray
    zi_notch: np.ndarray
    zi_band: np.ndarray


def _make_filterbank(fs: float, n_channels: int) -> FilterBank:
    # Notch 120Hz (line harmonic) and bandpass high-gamma.
    b, a = iirnotch(w0=120.0, Q=30.0, fs=fs)
    sos_notch = tf2sos(b, a).astype(np.float32)
    sos_band = butter(4, [70.0, 150.0], btype="band", fs=fs, output="sos").astype(
        np.float32
    )

    zi_notch = np.zeros((sos_notch.shape[0], 2, n_channels), dtype=np.float32)
    zi_band = np.zeros((sos_band.shape[0], 2, n_channels), dtype=np.float32)
    return FilterBank(
        sos_notch=sos_notch,
        sos_band=sos_band,
        zi_notch=zi_notch,
        zi_band=zi_band,
    )


def _reshape_to_grid(channel_vec: np.ndarray, grid_size: int) -> np.ndarray:
    grid = channel_vec.reshape(grid_size, grid_size)
    # Empirically matches ground_truth coordinates in this repo.
    return np.flipud(np.fliplr(grid))


def _robust_z(vec: np.ndarray) -> np.ndarray:
    med = float(np.median(vec))
    mad = float(np.median(np.abs(vec - med))) + 1e-6
    return (vec - med) / mad


class CompassProcessor:
    def __init__(
        self,
        fs: float,
        n_channels: int,
        grid_size: int,
        ema_tau_s: float = 0.20,
        spatial_sigma: float = 1.0,
    ):
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.grid_size = int(grid_size)
        self.spatial_sigma = float(spatial_sigma)
        self.fb = _make_filterbank(self.fs, self.n_channels)

        self.ema_tau_s = float(ema_tau_s)
        self.ema = np.zeros((self.n_channels,), dtype=np.float32)
        self._last_t: float | None = None

        # Peak-cluster tracker with explicit memory (interpretable, noise-robust).
        self.tracker = PersistentSpotClusterTracker(
            max_tracks=6,
            ema_alpha=0.35,
            max_match_dist=6.0,
            strength_gain=0.3,
            strength_decay=0.98,
            max_age=200,
        )
        self.peak_n = 3
        self.peak_smooth_sigma = 1.0
        self.peak_suppress_radius = 6
        self.peak_com_radius = 2
        self.memory_map: np.ndarray | None = None
        self.memory_half_life_s = 45.0
        self.center_alpha = 0.45
        self.center_ema: np.ndarray | None = None
        self.conf_alpha = 0.35
        self.conf_ema: float | None = None

    def process_batch(self, batch: np.ndarray, t_s: float) -> dict[str, Any] | None:
        if batch.size == 0:
            return None

        x = batch.astype(np.float32, copy=False)

        # Common average reference.
        x = x - x.mean(axis=1, keepdims=True)

        # Causal filtering.
        x, self.fb.zi_notch = sosfilt(self.fb.sos_notch, x, axis=0, zi=self.fb.zi_notch)
        x, self.fb.zi_band = sosfilt(self.fb.sos_band, x, axis=0, zi=self.fb.zi_band)

        # High-gamma power estimate over the batch.
        p = np.mean(x * x, axis=0)
        p = np.log1p(p)

        # EMA smoothing.
        if self._last_t is None:
            dt = float(batch.shape[0] / self.fs)
        else:
            dt = max(1e-3, float(t_s - self._last_t))
        self._last_t = float(t_s)
        alpha = float(1.0 - np.exp(-dt / max(1e-3, self.ema_tau_s)))
        self.ema = (1 - alpha) * self.ema + alpha * p

        # Robust normalize + clamp.
        z = _robust_z(self.ema)
        z = np.maximum(z, 0.0)

        grid = _reshape_to_grid(z, self.grid_size)
        grid = gaussian_filter(grid, sigma=self.spatial_sigma)

        observations = extract_peak_observations(
            grid,
            n_peaks=self.peak_n,
            smooth_sigma=self.peak_smooth_sigma,
            suppress_radius=self.peak_suppress_radius,
            com_radius=self.peak_com_radius,
        )
        cr, cc, _tracks = self.tracker.update(observations)
        if cr is None or cc is None:
            return None

        center_obs = np.array([float(cr), float(cc)], dtype=float)
        if self.center_ema is None:
            self.center_ema = center_obs.copy()
        else:
            self.center_ema = (1.0 - self.center_alpha) * self.center_ema + self.center_alpha * center_obs
        smooth_cr, smooth_cc = float(self.center_ema[0]), float(self.center_ema[1])

        spots = observations
        track_states = self.tracker.track_states()
        spots_mem = [(r, c, s) for (r, c, s, _age) in track_states]

        # Confidence from how many stable tracks we have + their strengths.
        strengths = sorted([s for (_r, _c, s, _age) in track_states], reverse=True)
        if strengths:
            strength_conf = float(np.mean(strengths[:2]))
            count_conf = min(1.0, len(strengths) / 2.0)
            conf = float(np.clip(0.7 * strength_conf + 0.3 * count_conf, 0.0, 1.0))
        else:
            conf = 0.0
        if self.conf_ema is None:
            self.conf_ema = conf
        else:
            self.conf_ema = (1.0 - self.conf_alpha) * self.conf_ema + self.conf_alpha * conf
        conf = float(self.conf_ema)

        if self.memory_map is None:
            self.memory_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        decay = 0.5 ** (dt / max(1e-3, self.memory_half_life_s))
        self.memory_map *= decay
        if track_states:
            upd = np.zeros_like(self.memory_map)
            for r, c, s, _age in track_states:
                rr = int(round(float(r)))
                cc = int(round(float(c)))
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    upd[rr, cc] = max(float(upd[rr, cc]), float(s))
            if float(upd.max()) > 0:
                upd = gaussian_filter(upd, sigma=1.0)
                mx = float(upd.max())
                if mx > 0:
                    upd = upd / mx
                self.memory_map = np.maximum(self.memory_map, upd)

        mid = (self.grid_size - 1) / 2.0
        # Use smoothed center for UI display
        delta_r = float(smooth_cr - mid)
        delta_c = float(smooth_cc - mid)
        move_r = float(-delta_r / mid)
        move_c = float(-delta_c / mid)
        dist = float(np.hypot(delta_r, delta_c))

        mem = self.memory_map

        regions = {}
        track_states_sorted = sorted(track_states, key=lambda x: float(x[2]), reverse=True)
        for idx, (r, c, s, _age) in enumerate(track_states_sorted[:4], start=1):
            regions[f"anchor_{idx}"] = (float(r), float(c), float(s))

        return {
            "t_s": float(t_s),
            "center_row": float(smooth_cr),
            "center_col": float(smooth_cc),
            "confidence": float(conf),
            "distance": dist,
            "move_row": move_r,
            "move_col": move_c,
            "spots": [[float(r), float(c), float(v)] for (r, c, v) in (spots or [])],
            "spots_mem": [[float(r), float(c), float(v)] for (r, c, v) in (spots_mem or [])],
            "regions": {k: [float(v[0]), float(v[1]), float(v[2])] for (k, v) in regions.items()},
            # JSON-friendly payload (32x32 is small enough for local UI at ~15 Hz).
            "heatmap": grid.astype(np.float32).tolist(),
            "memory": None if mem is None else mem.astype(np.float32).tolist(),
        }

    def reset(self) -> None:
        """Reset all causal state (filters, EMA, and tracker memory)."""
        self.fb = _make_filterbank(self.fs, self.n_channels)
        self.ema[:] = 0.0
        self._last_t = None
        self.tracker.reset()
        self.center_ema = None
        self.conf_ema = None
        self.memory_map = None


class CompassServer:
    def __init__(
        self,
        stream_url: str,
        host: str,
        port: int,
        ui_hz: float,
        include_heatmaps: bool,
        ema_tau_s: float,
        spatial_sigma: float,
    ):
        self.stream_url = stream_url
        self.host = host
        self.port = port
        self.ui_hz = ui_hz
        self.include_heatmaps = include_heatmaps
        self.ema_tau_s = float(ema_tau_s)
        self.spatial_sigma = float(spatial_sigma)
        self.clients: set[websockets.WebSocketServerProtocol] = set()
        self.processor: CompassProcessor | None = None
        self._last_send = 0.0

    async def register(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        await ws.send(
            json.dumps(
                {
                    "type": "init",
                    "grid_size": 32,
                    "fs": 500.0,
                    "ui_hz": self.ui_hz,
                }
            )
        )

    async def unregister(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.discard(ws)

    async def ws_handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        await self.register(ws)
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if data.get("type") == "reset" and self.processor is not None:
                    self.processor.reset()
                    await ws.send(json.dumps({"type": "ack", "ok": True, "action": "reset"}))
        finally:
            await self.unregister(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        msg = json.dumps(payload)
        await asyncio.gather(*(c.send(msg) for c in list(self.clients)), return_exceptions=True)

    async def run_stream(self) -> None:
        backoff = 0.2
        while True:
            try:
                async with websockets.connect(self.stream_url) as ws:
                    console.print(f"[green]✓ Connected to stream[/green] {self.stream_url}")
                    backoff = 0.2
                    async for message in ws:
                        data = json.loads(message)
                        if data.get("type") == "init":
                            fs = float(data.get("fs", 500.0))
                            grid_size = int(data.get("grid_size", 32))
                            n_channels = len(data.get("channels_coords", [])) or 1024
                            self.processor = CompassProcessor(
                                fs=fs,
                                n_channels=n_channels,
                                grid_size=grid_size,
                                ema_tau_s=self.ema_tau_s,
                                spatial_sigma=self.spatial_sigma,
                            )
                        elif data.get("type") == "sample_batch" and self.processor is not None:
                            batch = np.asarray(data["neural_data"], dtype=np.float32)
                            start_t = float(data.get("start_time_s", 0.0))
                            fs = float(data.get("fs", self.processor.fs))
                            t_s = start_t + (batch.shape[0] - 1) / fs

                            frame = self.processor.process_batch(batch, t_s)
                            if frame is None:
                                continue

                            now = time.perf_counter()
                            if now - self._last_send < 1.0 / max(1.0, self.ui_hz):
                                continue
                            self._last_send = now

                            if not self.include_heatmaps:
                                frame.pop("heatmap", None)
                                frame.pop("memory", None)

                            await self.broadcast({"type": "compass_frame", **frame})
            except Exception as e:
                console.print(f"[yellow]Stream reconnecting:[/yellow] {e}")
                await asyncio.sleep(backoff)
                backoff = min(5.0, backoff * 1.8)

    async def run(self) -> None:
        console.print()
        console.print(
            Panel.fit(
                f"Compass backend serving ws://{self.host}:{self.port}\n"
                f"Upstream stream: {self.stream_url}",
                border_style="cyan",
            )
        )
        console.print()
        async with websockets.serve(self.ws_handler, self.host, self.port):
            await self.run_stream()


@app.command()
def main(
    stream_url: str = typer.Option(
        "ws://localhost:8765",
        help="Upstream Track2 stream WebSocket URL",
    ),
    host: str = typer.Option("localhost", help="Host to bind the UI websocket"),
    port: int = typer.Option(8767, help="Port for the UI websocket"),
    ui_hz: float = typer.Option(15.0, help="UI update rate (Hz)"),
    heatmaps: bool = typer.Option(
        True, "--heatmaps/--no-heatmaps", help="Send heatmap + memory arrays"
    ),
    ema_tau_s: float = typer.Option(
        0.20, help="Temporal smoothing (EMA time constant, seconds)"
    ),
    spatial_sigma: float = typer.Option(
        1.0, help="Spatial Gaussian smoothing sigma (grid units)"
    ),
) -> None:
    """Run the realtime compass processing backend."""
    server = CompassServer(
        stream_url=stream_url,
        host=host,
        port=port,
        ui_hz=ui_hz,
        include_heatmaps=heatmaps,
        ema_tau_s=ema_tau_s,
        spatial_sigma=spatial_sigma,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    app()
