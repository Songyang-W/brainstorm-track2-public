"""
WebSocket bridge server for BCI signal processing.

Receives raw neural data from stream_data.py (ws://localhost:8765)
Processes through signal pipeline and serves to web UI (ws://localhost:8766)

Dev mode: Pass --data-dir to include ground truth overlay data.
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import websockets
from filters import FilterPipeline
from pipeline import SignalPipeline
from tracker import BCITracker, GlobalMapper
from websockets.client import connect
from websockets.server import serve

# Optional encoder pipeline (requires PyTorch)
try:
    from encoder_pipeline import EncoderPipeline

    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class BCIServer:
    """WebSocket server bridging data stream to processed output."""

    def __init__(
        self,
        source_url: str = "ws://localhost:8765",
        serve_port: int = 8766,
        fs: float = 500.0,
        grid_size: int = 32,
        data_dir: Path | None = None,
        use_encoder: bool = False,
        model_path: Path | str = "scripts/compass_model.pth",
    ):
        """
        Initialize server.

        Args:
            source_url: URL of the data source WebSocket
            serve_port: Port to serve processed data
            fs: Sampling frequency
            grid_size: Size of electrode grid
            data_dir: Optional path to data directory for ground truth overlay
            use_encoder: Use U-Net encoder pipeline instead of traditional DSP
            model_path: Path to U-Net model weights (only used if use_encoder=True)
        """
        self.source_url = source_url
        self.serve_port = serve_port
        self.fs = fs
        self.grid_size = grid_size
        self.n_channels = grid_size * grid_size
        self.batch_size = 10  # Samples per batch
        self.use_encoder = use_encoder

        # Load ground truth if data_dir provided (dev mode)
        self.ground_truth: pd.DataFrame | None = None
        if data_dir:
            gt_path = data_dir / "ground_truth.parquet"
            if gt_path.exists():
                self.ground_truth = pd.read_parquet(gt_path)
                logger.info(
                    f"Loaded ground truth from {gt_path} ({len(self.ground_truth)} rows)"
                )
            else:
                logger.warning(f"Ground truth not found at {gt_path}")

        # Initialize processing components based on mode
        if use_encoder:
            if not ENCODER_AVAILABLE:
                raise ImportError(
                    "Encoder mode requires PyTorch. Install with: pip install torch"
                )
            logger.info("Using U-Net encoder pipeline")
            self.encoder_pipeline = EncoderPipeline(
                model_path=model_path,
                fs=fs,
                n_channels=self.n_channels,
            )
            # Set to None to indicate encoder mode
            self.filter_pipeline = None
            self.signal_pipeline = None
        else:
            logger.info("Using traditional DSP pipeline")
            self.encoder_pipeline = None
            self.filter_pipeline = FilterPipeline(
                fs=fs,
                notch_freq=60.0,
                notch_q=30.0,
                bandpass_low=70.0,
                bandpass_high=150.0,
                bandpass_order=4,
            )

            self.signal_pipeline = SignalPipeline(
                n_channels=self.n_channels,
                ema_alpha=0.1,
                dead_threshold=0,  # Disabled bad channel detection
                artifact_std_multiplier=100.0,  # Effectively disabled
            )

        self.tracker = BCITracker(
            grid_size=grid_size,
            accumulation_alpha=0.1,
            activity_threshold=0.15,  # Lowered to detect weaker signals
            decay_rate=0.99,
            spatial_sigma=1.5,
            cluster_threshold=0.1,  # Lowered for easier hotspot detection
            center_tolerance=2.0,
        )

        # Global mapper for brain-fixed coordinate tracking
        self.global_mapper = GlobalMapper(
            grid_size=grid_size,
            global_size=96,  # 96x96 global brain map
        )

        # Connected clients
        self.clients: set[websockets.WebSocketServerProtocol] = set()

        # Initialization data to send to new clients
        self.init_data: dict | None = None

        # Stats
        self.samples_processed = 0
        self.frames_sent = 0
        self.last_time_s = 0.0

    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        # Send init data if available
        if self.init_data:
            try:
                await websocket.send(json.dumps(self.init_data))
            except websockets.exceptions.ConnectionClosed:
                pass

    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        if not self.clients:
            return

        # Create tasks for all clients
        tasks = [
            asyncio.create_task(self.send_to_client(client, message))
            for client in self.clients.copy()
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to_client(
        self, client: websockets.WebSocketServerProtocol, message: str
    ):
        """Send message to a single client."""
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(client)

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle a client WebSocket connection."""
        await self.register_client(websocket)

        try:
            # Keep connection alive, clients are read-only
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def process_source_message(self, message: str):
        """Process a message from the data source."""
        try:
            data = json.loads(message)

            if data["type"] == "init":
                # Store and forward init data
                self.init_data = data
                logger.info(
                    f"Received init: {len(data['channels_coords'])} channels, "
                    f"grid size {data['grid_size']}"
                )
                await self.broadcast(message)

            elif data["type"] == "sample_batch":
                # Process neural data
                neural_data = np.array(
                    data["neural_data"]
                )  # Shape: (batch_size, n_channels)
                start_time_s = data.get("start_time_s", 0.0)

                self.samples_processed += neural_data.shape[0]
                self.last_time_s = start_time_s

                # Process based on mode (encoder vs traditional DSP)
                velocity_data = None
                if self.use_encoder:
                    # Encoder pipeline: bandpass + power + EMA + U-Net + velocity
                    normalized, bad_channels = self.encoder_pipeline.process(
                        neural_data
                    )
                    noise_ratio = self.encoder_pipeline.get_noise_ratio()

                    # Get velocity from encoder pipeline
                    velocity_result = self.encoder_pipeline.get_velocity()
                    if velocity_result is not None:
                        velocity_data = {
                            "vx": velocity_result.vx,
                            "vy": velocity_result.vy,
                            "com_row": velocity_result.com_row,
                            "com_col": velocity_result.com_col,
                            "num_targets": velocity_result.num_targets,
                            "cursor_x": velocity_result.cursor_x,
                            "cursor_y": velocity_result.cursor_y,
                        }
                else:
                    # Traditional pipeline: notch + bandpass + EMA
                    filtered, noise_ratio = self.filter_pipeline.process(neural_data)
                    normalized, bad_channels = self.signal_pipeline.process(filtered)

                # Compute signal strength (mean of normalized power, excluding bad channels)
                good_channels = ~bad_channels
                if np.any(good_channels):
                    signal_strength = float(np.mean(normalized[good_channels]))
                else:
                    signal_strength = 0.0

                # Update global mapper FIRST to get persistent_evidence
                # Note: Global mapper now uses only observation-based tracking (no GT)
                global_mapping = self.global_mapper.update(
                    normalized, bad_channels, start_time_s
                )

                # Extract persistent_evidence for blended tracking
                persistent_evidence = np.array(global_mapping["persistent_evidence"])

                # Update tracker with persistent_evidence for robust tracking
                tracking_result = self.tracker.update(
                    normalized, bad_channels, persistent_evidence
                )

                # Get stability score from hotspot tracker
                stability_score = (
                    self.global_mapper.hotspot_tracker.get_stability_score()
                )

                # Calculate placement score: high only when stable AND centered
                # This is the key metric for confirming good placement
                guidance = tracking_result.get("guidance", {})
                distance = guidance.get("distance", 16.0)
                hotspot_confidence = guidance.get("confidence", 0.0)
                max_distance = self.grid_size / 2  # 16 for 32x32 grid

                # Proximity score: 1.0 at center, 0.0 at edge
                proximity = max(0.0, 1.0 - distance / max_distance)

                # Placement score: stable + centered + strong hotspot
                # All three must be high for good placement
                placement_score = stability_score * proximity * hotspot_confidence

                # Build signal quality metrics for frontend
                signal_quality = {
                    "signal_strength": signal_strength,  # 0-1, mean normalized power
                    "noise_ratio": noise_ratio,  # 0-1, how much 60Hz noise (lower = better)
                    "stability": stability_score,  # 0-1, how stable the centroid is
                    "proximity": proximity,  # 0-1, how close to center (1 = centered)
                    "placement_score": placement_score,  # 0-1, overall placement quality
                }

                # Build output message
                output = {
                    "type": "processed",
                    "time_s": start_time_s,
                    "samples_processed": self.samples_processed,
                    **tracking_result,
                    "global_mapping": global_mapping,
                    "signal_quality": signal_quality,
                }

                # Add velocity data if available (encoder mode)
                if velocity_data is not None:
                    output["velocity"] = velocity_data

                # Add ground truth overlay if available (dev mode)
                if self.ground_truth is not None:
                    gt_data = self._get_ground_truth(start_time_s)
                    if gt_data:
                        output["ground_truth"] = gt_data

                await self.broadcast(json.dumps(output))
                self.frames_sent += 1

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def _get_ground_truth(self, time_s: float) -> dict | None:
        """Get ground truth data for the given time."""
        if self.ground_truth is None:
            return None

        # Find the closest row by time (using sample index)
        sample_idx = int(time_s * self.fs) + self.batch_size // 2
        if sample_idx >= len(self.ground_truth):
            sample_idx = len(self.ground_truth) - 1

        row = self.ground_truth.iloc[sample_idx]

        # Return cluster centers (convert to 0-indexed for frontend)
        return {
            "vx_pos": [
                float(row["vx_pos_center_row"]) - 1,
                float(row["vx_pos_center_col"]) - 1,
            ],
            "vx_neg": [
                float(row["vx_neg_center_row"]) - 1,
                float(row["vx_neg_center_col"]) - 1,
            ],
            "vy_pos": [
                float(row["vy_pos_center_row"]) - 1,
                float(row["vy_pos_center_col"]) - 1,
            ],
            "vy_neg": [
                float(row["vy_neg_center_row"]) - 1,
                float(row["vy_neg_center_col"]) - 1,
            ],
            "vx": float(row["vx"]),
            "vy": float(row["vy"]),
            "phase": str(row["phase"]),
        }

    async def connect_to_source(self):
        """Connect to data source and process messages."""
        while True:
            try:
                logger.info(f"Connecting to data source: {self.source_url}")
                async with connect(self.source_url) as websocket:
                    logger.info("Connected to data source")

                    async for message in websocket:
                        await self.process_source_message(message)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Data source connection closed")
            except ConnectionRefusedError:
                logger.warning("Data source connection refused")
            except Exception as e:
                logger.error(f"Data source error: {e}")

            # Wait before reconnecting
            logger.info("Reconnecting to data source in 2 seconds...")
            await asyncio.sleep(2)

    async def run(self):
        """Run the server."""
        # Start WebSocket server for clients
        async with serve(self.handle_client, "localhost", self.serve_port):
            logger.info(f"Serving processed data on ws://localhost:{self.serve_port}")

            # Connect to data source
            await self.connect_to_source()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="BCI Signal Processing Server")
    parser.add_argument(
        "--source", default="ws://localhost:8765", help="Data source WebSocket URL"
    )
    parser.add_argument(
        "--port", type=int, default=8766, help="Port to serve processed data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory for ground truth overlay (dev mode)",
    )
    parser.add_argument(
        "--encoder",
        action="store_true",
        help="Use U-Net encoder pipeline instead of traditional DSP",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("scripts/compass_model.pth"),
        help="Path to U-Net model weights (only used with --encoder)",
    )
    args = parser.parse_args()

    # Default to port 8767 when using encoder mode (unless explicitly specified)
    port = args.port
    if args.encoder and args.port == 8766:
        port = 8767

    server = BCIServer(
        source_url=args.source,
        serve_port=port,
        data_dir=args.data_dir,
        use_encoder=args.encoder,
        model_path=args.model_path,
    )
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
