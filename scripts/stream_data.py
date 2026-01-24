#!/usr/bin/env python3
"""
WebSocket Streaming Server for Track 2 Neural Data (File-Only Mode)

This script streams pre-generated neural data from parquet files via WebSocket
to connected clients at the original sampling rate (500 Hz).

This is the participant-facing version for local development. It reads from
downloaded dataset files and streams them to your visualization app.

Architecture:
    [Dataset Files] -> [stream_data.py] -> WebSocket -> [serve.py] -> [Web App]

Usage:
    # Stream from downloaded dataset directory
    brainstorm-stream --from-file data/easy/

    # With looping disabled (stops at end of file)
    brainstorm-stream --from-file data/easy/ --no-loop
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from websockets.server import WebSocketServerProtocol  # type: ignore

app = typer.Typer(help="WebSocket Streaming Server for Track 2 Neural Data")
console = Console()


# =============================================================================
# Channel Coordinates
# =============================================================================


def generate_channel_coords(grid_size: int = 32) -> np.ndarray:
    """
    Generate channel coordinates for a grid of electrodes.

    Returns 1-indexed coordinates matching the Layer7 array layout.
    """
    coords = []
    for row in range(1, grid_size + 1):
        for col in range(1, grid_size + 1):
            coords.append([row, col])
    return np.array(coords)


# =============================================================================
# File-based Sample Provider
# =============================================================================


class FileSampleProvider:
    """Provides samples from a pre-generated parquet file."""

    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        loop: bool = True,
    ):
        self.data = data
        self.fs = fs
        self.loop = loop
        self.n_samples = data.shape[0]
        self.current_idx = 0

        # Stats tracking
        self._recent_min = 0.0
        self._recent_max = 0.0

    def get_next_batch(self, batch_size: int) -> tuple[list[list[float]], float]:
        """Get next batch of samples from the file."""
        start_time_s = self.current_idx / self.fs

        batch_samples = []
        for _ in range(batch_size):
            sample = self.data[self.current_idx]
            batch_samples.append(sample.tolist())

            # Track stats
            self._recent_min = min(self._recent_min, float(sample.min()))
            self._recent_max = max(self._recent_max, float(sample.max()))

            self.current_idx += 1
            if self.current_idx >= self.n_samples:
                if self.loop:
                    self.current_idx = 0
                else:
                    # Stop at end of file
                    break

        return batch_samples, start_time_s

    def get_stats(self) -> tuple[float, float]:
        """Get min/max stats and reset."""
        stats = (self._recent_min, self._recent_max)
        self._recent_min = float("inf")
        self._recent_max = float("-inf")
        return stats

    @property
    def progress(self) -> float:
        """Get playback progress as fraction [0, 1]."""
        return self.current_idx / self.n_samples

    @property
    def current_time_s(self) -> float:
        """Get current playback time in seconds."""
        return self.current_idx / self.fs

    @property
    def total_duration_s(self) -> float:
        """Get total duration in seconds."""
        return self.n_samples / self.fs

    @property
    def is_finished(self) -> bool:
        """Check if playback has finished (only relevant when loop=False)."""
        return not self.loop and self.current_idx >= self.n_samples


def print_header(
    host: str,
    port: int,
    fs: float,
    data_dir: str,
) -> None:
    """Print nice header with configuration."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Track 2 Neural Data Streaming Server[/bold cyan]\n"
            "[dim]File Playback Mode[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column(style="bold")
    config_table.add_column()
    config_table.add_row("WebSocket", f"ws://{host}:{port}")
    config_table.add_row("Sampling Rate", f"{fs:.0f} Hz")
    config_table.add_row("Data Source", data_dir)

    console.print(Panel(config_table, title="Configuration", border_style="blue"))
    console.print()


def print_controls() -> None:
    """Print keyboard controls."""
    controls_table = Table(show_header=False, box=None, padding=(0, 2))
    controls_table.add_column(style="bold cyan", justify="center")
    controls_table.add_column()
    controls_table.add_row("Ctrl+C", "Stop server")

    console.print(Panel(controls_table, title="Controls", border_style="green"))
    console.print()


class StreamingServer:
    """WebSocket server for streaming neural data at the sampling rate."""

    def __init__(
        self,
        host: str,
        port: int,
        channels_coords: np.ndarray,
        sample_provider: FileSampleProvider,
        fs: float,
        batch_size: int = 10,
    ):
        self.host = host
        self.port = port
        self.channels_coords = channels_coords
        self.sample_provider = sample_provider
        self.fs = fs
        self.dt = 1.0 / fs
        self.batch_size = batch_size

        self.grid_size = int(channels_coords[:, 0].max())
        self.clients: set[WebSocketServerProtocol] = set()
        self.running = False
        self.sample_count = 0
        self.start_time = 0.0

        # Signal statistics (updated periodically)
        self.signal_min: float = 0.0
        self.signal_max: float = 0.0
        self.signal_stats_samples: int = 0

    def make_status_panel(self) -> Panel:
        """Create a status panel with live information."""
        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(style="bold cyan", justify="right")
        status_table.add_column()

        # File playback progress
        provider = self.sample_provider
        progress_pct = provider.progress * 100
        current_t = provider.current_time_s
        total_t = provider.total_duration_s
        status_table.add_row(
            "Playback:",
            f"{progress_pct:.1f}% ({current_t:.1f}s / {total_t:.1f}s)",
        )

        # Clients connected
        status_table.add_row("Clients Connected:", f"[bold]{len(self.clients)}[/bold]")

        # Samples streamed
        status_table.add_row("Samples Streamed:", f"{self.sample_count:,}")

        # Time elapsed
        elapsed = time.perf_counter() - self.start_time if self.start_time > 0 else 0
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        time_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"
        status_table.add_row("Time Elapsed:", time_str)

        # Current sampling rate (actual)
        actual_rate = self.sample_count / elapsed if elapsed > 0 else 0
        status_table.add_row("Actual Rate:", f"{actual_rate:.1f} Hz")

        # Signal statistics
        if self.signal_stats_samples > 0:
            status_table.add_row(
                "Signal Range:",
                f"[{self.signal_min:.3f}, {self.signal_max:.3f}]",
            )
        else:
            status_table.add_row("Signal Range:", "[dim]collecting...[/dim]")

        return Panel(
            status_table,
            title="[bold cyan]Server Status[/bold cyan]",
            border_style="cyan",
        )

    async def register(self, websocket: WebSocketServerProtocol) -> None:
        """Register a new client and send initialization data."""
        self.clients.add(websocket)

        # Send initialization message
        init_message: dict[str, Any] = {
            "type": "init",
            "channels_coords": self.channels_coords.tolist(),
            "grid_size": self.grid_size,
            "fs": self.fs,
            "batch_size": self.batch_size,
        }
        await websocket.send(json.dumps(init_message))

    async def unregister(self, websocket: WebSocketServerProtocol) -> None:
        """Unregister a client."""
        self.clients.discard(websocket)

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a WebSocket connection."""
        await self.register(websocket)
        try:
            async for _ in websocket:
                # Keep connection alive
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True,
            )

    async def stream_loop(self, live: Live) -> None:
        """Main streaming loop that broadcasts neural data at fs."""
        self.start_time = time.perf_counter()
        last_display_update = self.start_time
        last_stats_update = self.start_time
        display_update_interval = 0.1  # Update display at 10 Hz
        stats_update_interval = 1.0  # Update signal stats

        # Batch timing
        batch_dt = self.batch_size * self.dt

        while self.running:
            # Check if finished (non-looping mode)
            if self.sample_provider.is_finished:
                console.print("\n[yellow]⚠[/yellow] Reached end of file. Stopping...")
                break

            # Calculate expected time for this batch
            expected_time = self.start_time + self.sample_count * self.dt
            current_time = time.perf_counter()

            # Get batch of samples from provider
            batch_samples, start_time_s = self.sample_provider.get_next_batch(
                self.batch_size
            )
            self.sample_count += len(batch_samples)

            # Create batch message
            batch_message = {
                "type": "sample_batch",
                "neural_data": batch_samples,
                "start_time_s": start_time_s,
                "sample_count": len(batch_samples),
                "fs": self.fs,
            }

            # Broadcast to all clients
            await self.broadcast(json.dumps(batch_message))

            # Update signal stats periodically
            if current_time - last_stats_update > stats_update_interval:
                self.signal_min, self.signal_max = self.sample_provider.get_stats()
                self.signal_stats_samples = self.sample_count
                last_stats_update = current_time

            # Update display at 10 Hz
            if current_time - last_display_update > display_update_interval:
                live.update(self.make_status_panel())
                last_display_update = current_time

            # Sleep to maintain sampling rate
            sleep_time = expected_time - time.perf_counter() + batch_dt
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def start(self, live: Live) -> None:
        """Start the WebSocket server and streaming loop."""
        self.running = True

        async with websockets.serve(self.handler, self.host, self.port):
            await self.stream_loop(live)

    def stop(self) -> None:
        """Stop the streaming server."""
        self.running = False


def load_data(data_dir: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    """
    Load neural data and metadata from a dataset directory.

    Args:
        data_dir: Path to directory containing track2_data.parquet and metadata.json

    Returns:
        Tuple of (data array, sampling rate, metadata dict)
    """
    data_path = data_dir / "track2_data.parquet"
    metadata_path = data_dir / "metadata.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load metadata
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        fs = metadata.get("sampling_rate_hz", 500.0)
    else:
        fs = 500.0
        console.print(
            f"[yellow]Warning:[/yellow] metadata.json not found, assuming fs={fs} Hz"
        )

    # Load data
    console.print(f"[dim]Loading data from {data_path}...[/dim]")
    data_df = pd.read_parquet(data_path)
    data = data_df.values.astype(np.float32)

    return data, fs, metadata


@app.command()
def main(
    from_file: str = typer.Option(
        ...,
        "--from-file",
        "-f",
        help="Path to dataset directory (containing track2_data.parquet)",
    ),
    host: str = typer.Option("localhost", help="WebSocket server host"),
    port: int = typer.Option(8765, help="WebSocket server port"),
    batch_size: int = typer.Option(10, help="Number of samples per WebSocket message"),
    loop: bool = typer.Option(
        True,
        "--loop/--no-loop",
        help="Loop playback when reaching end of file (default: True)",
    ),
) -> None:
    """
    Stream pre-generated neural data over WebSocket.

    This is for local development - stream data from downloaded dataset files
    to test your visualization application.

    Example:
        brainstorm-stream --from-file data/easy/
    """
    data_dir = Path(from_file)

    if not data_dir.exists():
        console.print(f"[red]✗ Error:[/red] Directory not found: {data_dir}")
        console.print(
            "\n[dim]Download data first with:[/dim]\n"
            "    uv run python -m scripts.download easy"
        )
        raise typer.Exit(code=1)

    # Load data
    try:
        data, fs, metadata = load_data(data_dir)
    except FileNotFoundError as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print(
        f"[dim]Loaded {data.shape[0]:,} samples ({data.shape[0] / fs:.1f}s) "
        f"with {data.shape[1]} channels[/dim]"
    )

    # Create channel coordinates
    grid_size = int(np.sqrt(data.shape[1]))
    channels_coords = generate_channel_coords(grid_size)

    # Create sample provider
    sample_provider = FileSampleProvider(data=data, fs=fs, loop=loop)

    # Print header
    print_header(host, port, fs, str(data_dir))
    print_controls()

    # Create and run server
    server = StreamingServer(
        host=host,
        port=port,
        channels_coords=channels_coords,
        sample_provider=sample_provider,
        fs=fs,
        batch_size=batch_size,
    )

    try:
        with Live(
            server.make_status_panel(),
            console=console,
            refresh_per_second=10,
            transient=False,
        ) as live:
            asyncio.run(server.start(live))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    console.print()
    console.print(
        Panel.fit(
            "[bold]Server stopped[/bold]",
            border_style="yellow",
        )
    )
    console.print()


if __name__ == "__main__":
    app()

