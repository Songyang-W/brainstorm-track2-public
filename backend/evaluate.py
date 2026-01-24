"""
Offline evaluation script for BCI signal processing pipeline.

Evaluates tracking accuracy against ground truth by:
1. Array center tracking - comparing detected centroid to ground truth array center
2. Velocity correlation - correlating neural activity at region centers with velocity

Usage:
    cd backend
    uv run python evaluate.py --data-dir ../data/hard/ --output ./results/
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filters import FilterPipeline
from pipeline import SignalPipeline
from tracker import BCITracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    # Array center tracking metrics
    mean_distance_error: float
    median_distance_error: float
    max_distance_error: float
    tracking_latency_samples: int
    guidance_accuracy: float
    phase_errors: dict[str, float]

    # Per-region velocity correlation
    region_correlations: dict[str, float]
    active_region_accuracy: float

    # Summary statistics
    total_samples: int
    samples_with_detection: int
    detection_rate: float


class OfflineEvaluator:
    """Evaluate signal processing pipeline against ground truth."""

    def __init__(
        self,
        batch_size: int = 10,
        fs: float = 500.0,
    ):
        """
        Initialize evaluator.

        Args:
            batch_size: Number of samples per processing batch (matches streaming)
            fs: Sampling frequency in Hz
        """
        self.batch_size = batch_size
        self.fs = fs

        # Initialize pipeline components (matching server.py configuration)
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
        """
        Load neural data and ground truth from parquet files.

        Args:
            data_dir: Directory containing track2_data.parquet and ground_truth.parquet

        Returns:
            Tuple of (neural_data array, ground_truth dataframe)
        """
        neural_path = data_dir / "track2_data.parquet"
        gt_path = data_dir / "ground_truth.parquet"

        logger.info(f"Loading neural data from {neural_path}")
        neural_df = pd.read_parquet(neural_path)
        neural_data = neural_df.values.astype(np.float64)

        logger.info(f"Loading ground truth from {gt_path}")
        ground_truth = pd.read_parquet(gt_path)

        logger.info(
            f"Loaded {len(neural_data)} samples, {neural_data.shape[1]} channels"
        )
        return neural_data, ground_truth

    def compute_ground_truth_center(self, gt_row: pd.Series) -> tuple[float, float]:
        """
        Compute array center from ground truth region positions.

        The array center is the average of the 4 velocity-tuned region centers.

        Args:
            gt_row: Single row from ground truth dataframe

        Returns:
            (center_row, center_col) in 1-indexed coordinates
        """
        # Average of Vy+ and Vy- rows gives center row
        center_row = (gt_row["vy_pos_center_row"] + gt_row["vy_neg_center_row"]) / 2
        # Average of Vx+ and Vx- cols gives center col
        center_col = (gt_row["vx_pos_center_col"] + gt_row["vx_neg_center_col"]) / 2
        return center_row, center_col

    def run_pipeline(
        self, neural_data: np.ndarray, ground_truth: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run the full signal processing pipeline on all data.

        Args:
            neural_data: Shape (n_samples, n_channels)
            ground_truth: Ground truth dataframe with same number of rows

        Returns:
            DataFrame with aligned results and ground truth
        """
        n_samples = len(neural_data)
        n_batches = n_samples // self.batch_size

        results = []

        logger.info(f"Processing {n_batches} batches...")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size

            # Get batch data
            batch_data = neural_data[start_idx:end_idx]

            # Filter
            filtered = self.filter_pipeline.process(batch_data)

            # Signal processing
            normalized, bad_channels = self.signal_pipeline.process(filtered)

            # Tracking
            track_result = self.tracker.update(normalized, bad_channels)

            # Get ground truth for this batch (use middle sample)
            gt_idx = start_idx + self.batch_size // 2
            gt_row = ground_truth.iloc[gt_idx]

            # Compute ground truth center
            gt_center_row, gt_center_col = self.compute_ground_truth_center(gt_row)

            # Extract region activities
            region_activities = self.compute_region_activities(normalized, gt_row)

            # Store result
            result = {
                "batch_idx": batch_idx,
                "time_s": gt_row["time_s"],
                "phase": gt_row["phase"],
                # Ground truth
                "gt_center_row": gt_center_row,
                "gt_center_col": gt_center_col,
                "gt_vx": gt_row["vx"],
                "gt_vy": gt_row["vy"],
                # Region centers (for reference)
                "gt_vx_pos_row": gt_row["vx_pos_center_row"],
                "gt_vx_pos_col": gt_row["vx_pos_center_col"],
                "gt_vx_neg_row": gt_row["vx_neg_center_row"],
                "gt_vx_neg_col": gt_row["vx_neg_center_col"],
                "gt_vy_pos_row": gt_row["vy_pos_center_row"],
                "gt_vy_pos_col": gt_row["vy_pos_center_col"],
                "gt_vy_neg_row": gt_row["vy_neg_center_row"],
                "gt_vy_neg_col": gt_row["vy_neg_center_col"],
                # Detected values
                "detected_centroid": track_result["centroid"],
                "num_clusters": track_result["num_clusters"],
                "guidance_direction": track_result["guidance"]["direction"],
                "guidance_is_centered": track_result["guidance"]["is_centered"],
                # Region activities
                **region_activities,
            }
            results.append(result)

            # Progress logging
            if (batch_idx + 1) % 1000 == 0:
                logger.info(f"Processed {batch_idx + 1}/{n_batches} batches")

        logger.info("Pipeline processing complete")
        return pd.DataFrame(results)

    def compute_region_activities(
        self, normalized_power: np.ndarray, gt_row: pd.Series
    ) -> dict[str, float]:
        """
        Extract neural activity at each velocity-tuned region center.

        Args:
            normalized_power: Shape (1024,), normalized power values
            gt_row: Ground truth row with region center coordinates

        Returns:
            Dictionary with activity at each region
        """
        grid = normalized_power.reshape(32, 32)

        # Get region centers (convert to 0-indexed for array access)
        regions = {
            "vx_pos": (
                int(gt_row["vx_pos_center_row"]) - 1,
                int(gt_row["vx_pos_center_col"]) - 1,
            ),
            "vx_neg": (
                int(gt_row["vx_neg_center_row"]) - 1,
                int(gt_row["vx_neg_center_col"]) - 1,
            ),
            "vy_pos": (
                int(gt_row["vy_pos_center_row"]) - 1,
                int(gt_row["vy_pos_center_col"]) - 1,
            ),
            "vy_neg": (
                int(gt_row["vy_neg_center_row"]) - 1,
                int(gt_row["vy_neg_center_col"]) - 1,
            ),
        }

        activities = {}
        for name, (r, c) in regions.items():
            # Clamp to valid range for 3x3 neighborhood
            r = np.clip(r, 1, 30)
            c = np.clip(c, 1, 30)
            # Extract 3x3 neighborhood for robustness
            neighborhood = grid[r - 1 : r + 2, c - 1 : c + 2]
            activities[f"{name}_activity"] = float(np.mean(neighborhood))

        return activities

    def compute_metrics(self, aligned: pd.DataFrame) -> EvaluationResult:
        """
        Compute evaluation metrics from aligned results.

        Args:
            aligned: DataFrame with detection results aligned to ground truth

        Returns:
            EvaluationResult with all metrics
        """
        # Filter to rows with valid detections
        has_detection = aligned["detected_centroid"].notna()
        detected = aligned[has_detection].copy()

        # Extract detected centroids (0-indexed)
        detected["det_row"] = detected["detected_centroid"].apply(
            lambda x: x[0] if x else np.nan
        )
        detected["det_col"] = detected["detected_centroid"].apply(
            lambda x: x[1] if x else np.nan
        )

        # Convert ground truth to 0-indexed for comparison
        detected["gt_row_0idx"] = detected["gt_center_row"] - 1
        detected["gt_col_0idx"] = detected["gt_center_col"] - 1

        # Compute distance errors
        detected["distance_error"] = np.sqrt(
            (detected["det_row"] - detected["gt_row_0idx"]) ** 2
            + (detected["det_col"] - detected["gt_col_0idx"]) ** 2
        )

        # Array center tracking metrics
        mean_distance_error = float(detected["distance_error"].mean())
        median_distance_error = float(detected["distance_error"].median())
        max_distance_error = float(detected["distance_error"].max())

        # Tracking latency via cross-correlation
        tracking_latency = self._compute_tracking_latency(detected)

        # Guidance accuracy (% of time guidance direction is correct)
        guidance_accuracy = self._compute_guidance_accuracy(detected)

        # Phase-wise errors
        phase_errors = {}
        for phase in detected["phase"].unique():
            phase_data = detected[detected["phase"] == phase]
            if len(phase_data) > 0:
                phase_errors[phase] = float(phase_data["distance_error"].mean())

        # Velocity correlations
        region_correlations = self._compute_velocity_correlations(aligned)

        # Active region accuracy
        active_region_accuracy = self._compute_active_region_accuracy(aligned)

        return EvaluationResult(
            mean_distance_error=mean_distance_error,
            median_distance_error=median_distance_error,
            max_distance_error=max_distance_error,
            tracking_latency_samples=tracking_latency,
            guidance_accuracy=guidance_accuracy,
            phase_errors=phase_errors,
            region_correlations=region_correlations,
            active_region_accuracy=active_region_accuracy,
            total_samples=len(aligned),
            samples_with_detection=len(detected),
            detection_rate=len(detected) / len(aligned) if len(aligned) > 0 else 0,
        )

    def _compute_tracking_latency(self, detected: pd.DataFrame) -> int:
        """Estimate tracking latency via cross-correlation."""
        if len(detected) < 100:
            return 0

        # Use detected row position vs ground truth row
        gt_signal = detected["gt_row_0idx"].values
        det_signal = detected["det_row"].values

        # Remove NaNs
        valid = ~np.isnan(gt_signal) & ~np.isnan(det_signal)
        if np.sum(valid) < 100:
            return 0

        gt_clean = gt_signal[valid] - np.mean(gt_signal[valid])
        det_clean = det_signal[valid] - np.mean(det_signal[valid])

        # Cross-correlation
        correlation = np.correlate(det_clean, gt_clean, mode="full")
        lags = np.arange(-len(gt_clean) + 1, len(gt_clean))

        # Find peak in reasonable lag range (-500 to 500 batches)
        valid_lags = np.abs(lags) < 500
        correlation_valid = correlation[valid_lags]
        lags_valid = lags[valid_lags]

        if len(correlation_valid) > 0:
            peak_idx = np.argmax(correlation_valid)
            return int(lags_valid[peak_idx])
        return 0

    def _compute_guidance_accuracy(self, detected: pd.DataFrame) -> float:
        """Compute percentage of correct guidance directions."""
        correct = 0
        total = 0

        for _, row in detected.iterrows():
            if pd.isna(row["det_row"]) or pd.isna(row["det_col"]):
                continue

            # Compute actual offset from center (15.5, 15.5 is 0-indexed center of 32x32)
            center = 15.5
            offset_row = row["det_row"] - center
            offset_col = row["det_col"] - center

            # Ground truth offset
            gt_offset_row = row["gt_row_0idx"] - center
            gt_offset_col = row["gt_col_0idx"] - center

            # Check if guidance direction matches
            # Row: negative = UP, positive = DOWN
            row_match = (offset_row * gt_offset_row > 0) or (
                abs(offset_row) < 2 and abs(gt_offset_row) < 2
            )
            # Col: negative = LEFT, positive = RIGHT
            col_match = (offset_col * gt_offset_col > 0) or (
                abs(offset_col) < 2 and abs(gt_offset_col) < 2
            )

            if row_match and col_match:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

    def _compute_velocity_correlations(self, aligned: pd.DataFrame) -> dict[str, float]:
        """Compute correlation between region activities and velocity."""
        correlations = {}

        # Filter out rows with zero velocity variance
        vx = aligned["gt_vx"].values
        vy = aligned["gt_vy"].values

        for region, _vel_name, vel in [
            ("vx_pos", "vx", vx),
            ("vx_neg", "vx", vx),
            ("vy_pos", "vy", vy),
            ("vy_neg", "vy", vy),
        ]:
            activity = aligned[f"{region}_activity"].values

            # Remove NaNs and compute correlation
            valid = ~np.isnan(activity) & ~np.isnan(vel)
            if np.sum(valid) > 100 and np.std(vel[valid]) > 0.01:
                corr = np.corrcoef(activity[valid], vel[valid])[0, 1]
                correlations[region] = float(corr)
            else:
                correlations[region] = 0.0

        return correlations

    def _compute_active_region_accuracy(self, aligned: pd.DataFrame) -> float:
        """Compute accuracy of detecting which region is most active."""
        correct = 0
        total = 0

        for _, row in aligned.iterrows():
            vx = row["gt_vx"]
            vy = row["gt_vy"]

            # Skip low-velocity samples
            if abs(vx) < 1 and abs(vy) < 1:
                continue

            # Get activities
            activities = {
                "vx_pos": row["vx_pos_activity"],
                "vx_neg": row["vx_neg_activity"],
                "vy_pos": row["vy_pos_activity"],
                "vy_neg": row["vy_neg_activity"],
            }

            # Skip if any NaN
            if any(pd.isna(v) for v in activities.values()):
                continue

            # Find most active region
            detected_region = max(activities, key=activities.get)

            # Determine ground truth active region
            if abs(vx) > abs(vy):
                gt_region = "vx_pos" if vx > 0 else "vx_neg"
            else:
                gt_region = "vy_pos" if vy > 0 else "vy_neg"

            if detected_region == gt_region:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

    def generate_report(
        self,
        result: EvaluationResult,
        aligned: pd.DataFrame,
        output_dir: Path,
    ):
        """
        Generate evaluation report with plots and summary.

        Args:
            result: Computed evaluation metrics
            aligned: Aligned results dataframe
            output_dir: Directory to save outputs
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary JSON
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        # Generate tracking error plot
        self._plot_tracking_error(aligned, result, output_dir)

        # Generate trajectory plot
        self._plot_trajectory(aligned, output_dir)

        # Generate velocity correlation plot
        self._plot_velocity_correlations(aligned, result, output_dir)

        # Print summary
        self._print_summary(result)

    def _plot_tracking_error(
        self,
        aligned: pd.DataFrame,
        result: EvaluationResult,
        output_dir: Path,
    ):
        """Plot tracking error over time."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Extract data
        time = aligned["time_s"].values
        gt_row = aligned["gt_center_row"].values - 1  # 0-indexed
        gt_col = aligned["gt_center_col"].values - 1

        det_row = (
            aligned["detected_centroid"].apply(lambda x: x[0] if x else np.nan).values
        )
        det_col = (
            aligned["detected_centroid"].apply(lambda x: x[1] if x else np.nan).values
        )

        # Plot row tracking
        axes[0].plot(time, gt_row, "b-", label="Ground Truth", alpha=0.7)
        axes[0].plot(time, det_row, "r-", label="Detected", alpha=0.7)
        axes[0].set_ylabel("Row Position")
        axes[0].legend()
        axes[0].set_title("Row Tracking")
        axes[0].grid(True, alpha=0.3)

        # Plot col tracking
        axes[1].plot(time, gt_col, "b-", label="Ground Truth", alpha=0.7)
        axes[1].plot(time, det_col, "r-", label="Detected", alpha=0.7)
        axes[1].set_ylabel("Column Position")
        axes[1].legend()
        axes[1].set_title("Column Tracking")
        axes[1].grid(True, alpha=0.3)

        # Compute and plot error
        distance_error = np.sqrt((det_row - gt_row) ** 2 + (det_col - gt_col) ** 2)
        axes[2].plot(time, distance_error, "k-", alpha=0.7)
        axes[2].axhline(
            result.mean_distance_error,
            color="r",
            linestyle="--",
            label=f"Mean: {result.mean_distance_error:.2f}",
        )
        axes[2].set_ylabel("Distance Error (grid units)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        axes[2].set_title("Tracking Error")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "tracking_error.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved tracking error plot to {plot_path}")

    def _plot_trajectory(self, aligned: pd.DataFrame, output_dir: Path):
        """Plot 2D trajectory comparison."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract data
        gt_row = aligned["gt_center_row"].values - 1
        gt_col = aligned["gt_center_col"].values - 1

        det_row = (
            aligned["detected_centroid"].apply(lambda x: x[0] if x else np.nan).values
        )
        det_col = (
            aligned["detected_centroid"].apply(lambda x: x[1] if x else np.nan).values
        )

        # Plot trajectories
        ax.plot(gt_col, gt_row, "b-", label="Ground Truth", alpha=0.5, linewidth=1)
        ax.plot(det_col, det_row, "r-", label="Detected", alpha=0.5, linewidth=1)

        # Mark start and end
        valid_det = ~np.isnan(det_row) & ~np.isnan(det_col)
        if np.any(valid_det):
            first_valid = np.argmax(valid_det)
            last_valid = len(valid_det) - 1 - np.argmax(valid_det[::-1])
            ax.scatter(
                [det_col[first_valid]],
                [det_row[first_valid]],
                c="green",
                s=100,
                marker="o",
                label="Start",
                zorder=5,
            )
            ax.scatter(
                [det_col[last_valid]],
                [det_row[last_valid]],
                c="red",
                s=100,
                marker="s",
                label="End",
                zorder=5,
            )

        # Grid settings
        ax.set_xlim(-1, 32)
        ax.set_ylim(32, -1)  # Invert y-axis so row 0 is at top
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Array Center Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plot_path = output_dir / "trajectory.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved trajectory plot to {plot_path}")

    def _plot_velocity_correlations(
        self,
        aligned: pd.DataFrame,
        result: EvaluationResult,
        output_dir: Path,
    ):
        """Plot velocity correlations for each region."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        regions = [
            ("vx_pos", "gt_vx", "Vx+ (Rightward)", axes[0, 0]),
            ("vx_neg", "gt_vx", "Vx- (Leftward)", axes[0, 1]),
            ("vy_pos", "gt_vy", "Vy+ (Upward)", axes[1, 0]),
            ("vy_neg", "gt_vy", "Vy- (Downward)", axes[1, 1]),
        ]

        for region, vel_col, title, ax in regions:
            activity = aligned[f"{region}_activity"].values
            velocity = aligned[vel_col].values

            # Subsample for scatter plot
            n_points = min(1000, len(activity))
            indices = np.linspace(0, len(activity) - 1, n_points, dtype=int)

            ax.scatter(
                velocity[indices],
                activity[indices],
                alpha=0.3,
                s=5,
            )

            corr = result.region_correlations.get(region, 0)
            ax.set_title(f"{title}\nCorr: {corr:.3f}")
            ax.set_xlabel(f"{vel_col.upper()} Velocity")
            ax.set_ylabel("Region Activity")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "velocity_correlations.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved velocity correlations plot to {plot_path}")

    def _print_summary(self, result: EvaluationResult):
        """Print evaluation summary to console."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print("\n--- Array Center Tracking ---")
        print(f"Mean Distance Error:    {result.mean_distance_error:.3f} grid units")
        print(f"Median Distance Error:  {result.median_distance_error:.3f} grid units")
        print(f"Max Distance Error:     {result.max_distance_error:.3f} grid units")
        print(f"Tracking Latency:       {result.tracking_latency_samples} batches")
        print(f"Guidance Accuracy:      {result.guidance_accuracy:.1%}")

        print("\n--- Detection Statistics ---")
        print(f"Total Samples:          {result.total_samples}")
        print(f"Samples w/ Detection:   {result.samples_with_detection}")
        print(f"Detection Rate:         {result.detection_rate:.1%}")

        print("\n--- Phase-wise Errors ---")
        for phase, error in sorted(result.phase_errors.items()):
            print(f"  {phase:20s}: {error:.3f} grid units")

        print("\n--- Velocity Correlations ---")
        print("  (Expected: Vx+ positive, Vx- negative, Vy+ positive, Vy- negative)")
        for region, corr in sorted(result.region_correlations.items()):
            expected = "+" if "pos" in region else "-"
            actual = "+" if corr > 0 else "-"
            match = "✓" if expected == actual else "✗"
            print(
                f"  {region:8s}: {corr:+.3f} (expected {expected}, got {actual}) {match}"
            )

        print("\n--- Active Region Detection ---")
        print(f"Active Region Accuracy: {result.active_region_accuracy:.1%}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BCI signal processing pipeline against ground truth"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing track2_data.parquet and ground_truth.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10, matching streaming)",
    )
    args = parser.parse_args()

    # Validate input directory
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Initialize evaluator
    evaluator = OfflineEvaluator(batch_size=args.batch_size)

    # Load data
    neural_data, ground_truth = evaluator.load_data(args.data_dir)

    # Run pipeline
    aligned = evaluator.run_pipeline(neural_data, ground_truth)

    # Compute metrics
    result = evaluator.compute_metrics(aligned)

    # Generate report
    evaluator.generate_report(result, aligned, args.output)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
