"""
Visualization script for BCI evaluation results.

Shows the 32x32 array with:
- Ground truth cluster centers (Vx+, Vx-, Vy+, Vy-)
- Ground truth array center
- Detected centroid
- Neural activity heatmap

Usage:
    cd backend
    uv run python visualize_evaluation.py --data-dir ../data/hard/ --output ./viz_hard/
"""

import argparse
import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filters import FilterPipeline
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from pipeline import SignalPipeline
from tracker import BCITracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """Visualize signal processing pipeline evaluation."""

    def __init__(
        self,
        batch_size: int = 10,
        fs: float = 500.0,
    ):
        self.batch_size = batch_size
        self.fs = fs

        # Initialize pipeline components
        self.filter_pipeline = FilterPipeline(
            fs=fs,
            notch_freq=60.0,
            bandpass_low=70.0,
            bandpass_high=150.0,
        )
        self.signal_pipeline = SignalPipeline(
            n_channels=1024,
            ema_alpha=0.1,
        )
        self.tracker = BCITracker(
            grid_size=32,
            accumulation_alpha=0.05,
            activity_threshold=0.4,
            decay_rate=0.995,
        )

    def load_data(self, data_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
        """Load neural data and ground truth."""
        neural_path = data_dir / "track2_data.parquet"
        gt_path = data_dir / "ground_truth.parquet"

        logger.info(f"Loading data from {data_dir}")
        neural_df = pd.read_parquet(neural_path)
        neural_data = neural_df.values.astype(np.float64)
        ground_truth = pd.read_parquet(gt_path)

        return neural_data, ground_truth

    def process_batch(self, batch_data: np.ndarray) -> tuple[np.ndarray, dict]:
        """Process a single batch through the pipeline."""
        filtered = self.filter_pipeline.process(batch_data)
        normalized, bad_channels = self.signal_pipeline.process(filtered)
        track_result = self.tracker.update(normalized, bad_channels)
        return normalized, track_result

    def create_snapshot(
        self,
        neural_data: np.ndarray,
        ground_truth: pd.DataFrame,
        output_dir: Path,
        batch_indices: list[int] | None = None,
    ):
        """Create snapshot visualizations at specific time points."""
        output_dir.mkdir(parents=True, exist_ok=True)

        n_samples = len(neural_data)
        n_batches = n_samples // self.batch_size

        # Default to evenly spaced snapshots
        if batch_indices is None:
            batch_indices = [
                int(n_batches * 0.1),
                int(n_batches * 0.3),
                int(n_batches * 0.5),
                int(n_batches * 0.7),
                int(n_batches * 0.9),
            ]

        logger.info(f"Creating snapshots at batches: {batch_indices}")

        # Process up to each snapshot point
        for target_batch in batch_indices:
            # Reset pipeline state
            self.filter_pipeline = FilterPipeline(fs=self.fs)
            self.signal_pipeline = SignalPipeline(n_channels=1024, ema_alpha=0.1)
            self.tracker = BCITracker(grid_size=32)

            # Process all batches up to target
            for batch_idx in range(target_batch + 1):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_data = neural_data[start_idx:end_idx]
                normalized, track_result = self.process_batch(batch_data)

            # Get ground truth for this batch
            gt_idx = target_batch * self.batch_size + self.batch_size // 2
            gt_row = ground_truth.iloc[gt_idx]

            # Create visualization
            fig = self._create_array_figure(normalized, track_result, gt_row)

            time_s = gt_row["time_s"]
            snapshot_path = output_dir / f"snapshot_t{time_s:.1f}s.png"
            fig.savefig(snapshot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved snapshot at t={time_s:.1f}s to {snapshot_path}")

    def _create_array_figure(
        self,
        normalized: np.ndarray,
        track_result: dict,
        gt_row: pd.Series,
    ) -> plt.Figure:
        """Create a figure showing the array with all markers."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Get ground truth positions (convert to 0-indexed)
        gt_centers = {
            "Vx+": (gt_row["vx_pos_center_row"] - 1, gt_row["vx_pos_center_col"] - 1),
            "Vx-": (gt_row["vx_neg_center_row"] - 1, gt_row["vx_neg_center_col"] - 1),
            "Vy+": (gt_row["vy_pos_center_row"] - 1, gt_row["vy_pos_center_col"] - 1),
            "Vy-": (gt_row["vy_neg_center_row"] - 1, gt_row["vy_neg_center_col"] - 1),
        }

        # Compute array center from ground truth
        gt_array_center = (
            (gt_row["vy_pos_center_row"] + gt_row["vy_neg_center_row"]) / 2 - 1,
            (gt_row["vx_pos_center_col"] + gt_row["vx_neg_center_col"]) / 2 - 1,
        )

        # Get detected centroid
        detected_centroid = track_result["centroid"]

        # Color scheme for regions
        region_colors = {
            "Vx+": "red",
            "Vx-": "blue",
            "Vy+": "green",
            "Vy-": "purple",
        }

        # === Panel 1: Current Activity ===
        ax1 = axes[0]
        activity = np.array(track_result["current_activity"])
        im1 = ax1.imshow(activity, cmap="hot", vmin=0, vmax=1, origin="upper")
        ax1.set_title(f"Current Activity (t={gt_row['time_s']:.2f}s)", fontsize=12)
        self._add_markers(
            ax1, gt_centers, gt_array_center, detected_centroid, region_colors
        )
        plt.colorbar(im1, ax=ax1, label="Normalized Power")

        # === Panel 2: Belief Map (accumulated evidence) ===
        ax2 = axes[1]
        belief = np.array(track_result["belief_map"])
        im2 = ax2.imshow(belief, cmap="viridis", vmin=0, origin="upper")
        ax2.set_title("Belief Map (Accumulated Evidence)", fontsize=12)
        self._add_markers(
            ax2, gt_centers, gt_array_center, detected_centroid, region_colors
        )
        plt.colorbar(im2, ax=ax2, label="Evidence")

        # === Panel 3: Smoothed Map with Clusters ===
        ax3 = axes[2]
        smoothed = np.array(track_result["smoothed_map"])
        im3 = ax3.imshow(smoothed, cmap="magma", vmin=0, origin="upper")
        ax3.set_title("Smoothed Map + Detection", fontsize=12)
        self._add_markers(
            ax3, gt_centers, gt_array_center, detected_centroid, region_colors
        )
        plt.colorbar(im3, ax=ax3, label="Smoothed Value")

        # Add legend
        legend_elements = [
            mpatches.Patch(color="red", label="Vx+ (rightward)"),
            mpatches.Patch(color="blue", label="Vx- (leftward)"),
            mpatches.Patch(color="green", label="Vy+ (upward)"),
            mpatches.Patch(color="purple", label="Vy- (downward)"),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="yellow",
                markersize=15,
                label="GT Array Center",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="cyan",
                markersize=12,
                label="Detected Centroid",
            ),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=10)

        # Add info text
        phase = gt_row["phase"]
        vx, vy = gt_row["vx"], gt_row["vy"]
        info_text = f"Phase: {phase} | Velocity: vx={vx:.1f}, vy={vy:.1f}"
        if detected_centroid:
            dist = np.sqrt(
                (detected_centroid[0] - gt_array_center[0]) ** 2
                + (detected_centroid[1] - gt_array_center[1]) ** 2
            )
            info_text += f" | Error: {dist:.1f} units"
        fig.suptitle(info_text, fontsize=11, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        return fig

    def _add_markers(
        self,
        ax: plt.Axes,
        gt_centers: dict,
        gt_array_center: tuple,
        detected_centroid: tuple | None,
        region_colors: dict,
    ):
        """Add ground truth and detection markers to an axis."""
        # Plot ground truth region centers
        for name, (row, col) in gt_centers.items():
            color = region_colors[name]
            # Circle for region center
            circle = plt.Circle((col, row), 2, fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
            # Label
            ax.annotate(
                name,
                (col, row),
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )

        # Plot ground truth array center
        ax.plot(
            gt_array_center[1],
            gt_array_center[0],
            "*",
            color="yellow",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

        # Plot detected centroid
        if detected_centroid:
            ax.plot(
                detected_centroid[1],
                detected_centroid[0],
                "X",
                color="cyan",
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=1.5,
            )

        # Grid settings
        ax.set_xlim(-0.5, 31.5)
        ax.set_ylim(31.5, -0.5)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.grid(True, alpha=0.2, color="white")

    def create_animation(
        self,
        neural_data: np.ndarray,
        ground_truth: pd.DataFrame,
        output_path: Path,
        skip_batches: int = 10,
        max_frames: int = 500,
    ):
        """Create an animated visualization."""
        n_samples = len(neural_data)
        n_batches = n_samples // self.batch_size

        # Reset pipeline
        self.filter_pipeline = FilterPipeline(fs=self.fs)
        self.signal_pipeline = SignalPipeline(n_channels=1024, ema_alpha=0.1)
        self.tracker = BCITracker(grid_size=32)

        # Pre-process all batches and store results
        logger.info("Pre-processing batches for animation...")
        frames_data = []

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_data = neural_data[start_idx:end_idx]
            normalized, track_result = self.process_batch(batch_data)

            # Store every Nth batch
            if batch_idx % skip_batches == 0:
                gt_idx = batch_idx * self.batch_size + self.batch_size // 2
                gt_row = ground_truth.iloc[gt_idx]
                frames_data.append(
                    {
                        "normalized": normalized.copy(),
                        "track_result": {
                            k: (np.array(v).copy() if isinstance(v, list) else v)
                            for k, v in track_result.items()
                        },
                        "gt_row": gt_row.copy(),
                    }
                )

                if len(frames_data) >= max_frames:
                    break

            if (batch_idx + 1) % 1000 == 0:
                logger.info(f"Processed {batch_idx + 1}/{n_batches} batches")

        logger.info(f"Creating animation with {len(frames_data)} frames...")

        # Create figure for animation
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        region_colors = {
            "Vx+": "red",
            "Vx-": "blue",
            "Vy+": "green",
            "Vy-": "purple",
        }

        def init():
            for ax in axes:
                ax.clear()
            return []

        def update(frame_idx):
            for ax in axes:
                ax.clear()

            data = frames_data[frame_idx]
            track_result = data["track_result"]
            gt_row = data["gt_row"]

            # Get positions
            gt_centers = {
                "Vx+": (
                    gt_row["vx_pos_center_row"] - 1,
                    gt_row["vx_pos_center_col"] - 1,
                ),
                "Vx-": (
                    gt_row["vx_neg_center_row"] - 1,
                    gt_row["vx_neg_center_col"] - 1,
                ),
                "Vy+": (
                    gt_row["vy_pos_center_row"] - 1,
                    gt_row["vy_pos_center_col"] - 1,
                ),
                "Vy-": (
                    gt_row["vy_neg_center_row"] - 1,
                    gt_row["vy_neg_center_col"] - 1,
                ),
            }
            gt_array_center = (
                (gt_row["vy_pos_center_row"] + gt_row["vy_neg_center_row"]) / 2 - 1,
                (gt_row["vx_pos_center_col"] + gt_row["vx_neg_center_col"]) / 2 - 1,
            )
            detected_centroid = track_result["centroid"]

            # Panel 1: Current Activity
            activity = track_result["current_activity"]
            axes[0].imshow(activity, cmap="hot", vmin=0, vmax=1, origin="upper")
            axes[0].set_title(f"Current Activity (t={gt_row['time_s']:.2f}s)")
            self._add_markers(
                axes[0], gt_centers, gt_array_center, detected_centroid, region_colors
            )

            # Panel 2: Belief Map
            belief = track_result["belief_map"]
            axes[1].imshow(belief, cmap="viridis", vmin=0, origin="upper")
            axes[1].set_title("Belief Map")
            self._add_markers(
                axes[1], gt_centers, gt_array_center, detected_centroid, region_colors
            )

            # Panel 3: Smoothed Map
            smoothed = track_result["smoothed_map"]
            axes[2].imshow(smoothed, cmap="magma", vmin=0, origin="upper")
            axes[2].set_title("Smoothed + Detection")
            self._add_markers(
                axes[2], gt_centers, gt_array_center, detected_centroid, region_colors
            )

            # Update suptitle
            phase = gt_row["phase"]
            error = ""
            if detected_centroid:
                dist = np.sqrt(
                    (detected_centroid[0] - gt_array_center[0]) ** 2
                    + (detected_centroid[1] - gt_array_center[1]) ** 2
                )
                error = f" | Error: {dist:.1f}"
            fig.suptitle(
                f"Phase: {phase} | vx={gt_row['vx']:.1f}, vy={gt_row['vy']:.1f}{error}"
            )

            return []

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(frames_data),
            interval=100,
            blit=False,
        )

        # Save animation
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if str(output_path).endswith(".gif"):
            writer = PillowWriter(fps=10)
        else:
            writer = FFMpegWriter(fps=10)

        logger.info(f"Saving animation to {output_path}...")
        anim.save(str(output_path), writer=writer)
        plt.close(fig)
        logger.info("Animation saved!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BCI evaluation with cluster centers"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./viz"),
        help="Output directory",
    )
    parser.add_argument(
        "--mode",
        choices=["snapshots", "animation", "both"],
        default="snapshots",
        help="Visualization mode",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=20,
        help="Skip N batches between animation frames",
    )
    args = parser.parse_args()

    visualizer = EvaluationVisualizer()
    neural_data, ground_truth = visualizer.load_data(args.data_dir)

    if args.mode in ["snapshots", "both"]:
        visualizer.create_snapshot(neural_data, ground_truth, args.output)

    if args.mode in ["animation", "both"]:
        anim_path = args.output / "tracking_animation.gif"
        visualizer.create_animation(
            neural_data, ground_truth, anim_path, skip_batches=args.skip
        )


if __name__ == "__main__":
    main()
