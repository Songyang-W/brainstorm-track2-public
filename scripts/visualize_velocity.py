"""
Visualization script: Neural activity heatmap with velocity (vx/vy) arrow and graph.

Displays side-by-side:
- Left: 32x32 neural activity heatmap with hotspot regions marked
- Right: Velocity arrow indicator + time-series graph of vx/vy

Usage:
    # Animated GIF
    uv run python scripts/visualize_velocity.py hard --output velocity.gif --max-frames 300

    # Static snapshot at specific time
    uv run python scripts/visualize_velocity.py hard --snapshot 50.0 --output snapshot.png

    # Interactive window
    uv run python scripts/visualize_velocity.py hard
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrow
from scipy.ndimage import gaussian_filter


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load neural data and ground truth."""
    neural_path = data_dir / "track2_data.parquet"
    gt_path = data_dir / "ground_truth.parquet"

    neural_df = pd.read_parquet(neural_path)
    gt_df = pd.read_parquet(gt_path)

    return neural_df, gt_df


def process_neural_power(neural_row: pd.Series, grid_size: int = 32) -> np.ndarray:
    """
    Process neural data to get power in high-gamma band.

    Uses bandpass filter (70-150 Hz) and power extraction.
    """
    # Extract channel columns (0 to 1023)
    channel_cols = list(range(grid_size * grid_size))
    raw_data = neural_row[channel_cols].values.astype(float)

    # Simple power estimation (squared signal)
    # In production, you'd use proper bandpass filtering
    power = np.abs(raw_data) ** 2

    # Normalize to [0, 1]
    if power.max() > 0:
        power = (power - power.min()) / (power.max() - power.min() + 1e-8)

    return power.reshape(grid_size, grid_size)


def create_static_snapshot(
    neural_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    time_s: float,
    output_path: Path | None = None,
    history_seconds: float = 5.0,
):
    """Create a static snapshot at a specific time."""
    grid_size = 32
    sample_rate = 500  # Hz

    # Find the closest sample to the requested time
    idx = int(time_s * sample_rate)
    idx = max(0, min(idx, len(neural_df) - 1))

    # Get data for this frame
    neural_row = neural_df.iloc[idx]
    gt_row = gt_df.iloc[idx]

    # Get velocity history leading up to this point
    history_samples = int(history_seconds * sample_rate)
    start_idx = max(0, idx - history_samples)
    gt_history = gt_df.iloc[start_idx : idx + 1]

    time_history = gt_history["time_s"].values
    vx_history = gt_history["vx"].values
    vy_history = gt_history["vy"].values

    # Get current velocity
    vx = gt_row["vx"]
    vy = gt_row["vy"]
    current_time = gt_row["time_s"]

    # Process neural data
    activity = process_neural_power(neural_row, grid_size)
    smoothed = gaussian_filter(activity, sigma=1.5)

    # Create figure
    fig = plt.figure(figsize=(16, 7), facecolor="#0a0a14")
    fig.suptitle(
        f"BCI Neural Activity & Velocity @ t={current_time:.2f}s",
        fontsize=14,
        color="white",
        fontweight="bold",
    )

    # Grid spec for layout
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.2, 0.6, 1.2],
        height_ratios=[1, 1],
        hspace=0.3,
        wspace=0.3,
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.1,
    )

    # Left: Heatmap
    ax_heatmap = fig.add_subplot(gs[:, 0])
    ax_heatmap.set_facecolor("#0a0a14")
    ax_heatmap.set_title("Neural Activity (32x32 Grid)", color="white", fontsize=12)

    # Draw heatmap
    ax_heatmap.imshow(
        smoothed,
        cmap="inferno",
        aspect="equal",
        vmin=0,
        vmax=1,
        origin="lower",
    )
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])

    # Draw hotspot regions from ground truth
    region_colors = {
        "vx_pos": "#ef4444",  # red
        "vx_neg": "#3b82f6",  # blue
        "vy_pos": "#22c55e",  # green
        "vy_neg": "#a855f7",  # purple
    }
    region_labels = {"vx_pos": "Vx+", "vx_neg": "Vx-", "vy_pos": "Vy+", "vy_neg": "Vy-"}

    for region, color in region_colors.items():
        row = gt_row[f"{region}_center_row"]
        col = gt_row[f"{region}_center_col"]
        # Convert to 0-indexed for display
        circle = plt.Circle(
            (col - 1, row - 1),
            1.5,
            fill=False,
            color=color,
            linewidth=2,
        )
        ax_heatmap.add_patch(circle)

        # Label
        ax_heatmap.text(
            col - 1,
            row - 1 + 2.5,
            region_labels[region],
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Middle: Velocity arrow
    ax_arrow = fig.add_subplot(gs[0, 1])
    ax_arrow.set_facecolor("#0a0a14")
    ax_arrow.set_title("Velocity Direction", color="white", fontsize=12)
    ax_arrow.set_xlim(-1.5, 1.5)
    ax_arrow.set_ylim(-1.5, 1.5)
    ax_arrow.set_aspect("equal")
    ax_arrow.axis("off")

    # Draw reference circle
    circle = Circle((0, 0), 1.0, fill=False, color="#333344", linewidth=2)
    ax_arrow.add_patch(circle)

    # Cross-hairs
    ax_arrow.axhline(0, color="#333344", linewidth=1, linestyle="--")
    ax_arrow.axvline(0, color="#333344", linewidth=1, linestyle="--")

    # Labels
    ax_arrow.text(1.2, 0, "Vx+", color="#ef4444", fontsize=10, ha="left", va="center")
    ax_arrow.text(-1.2, 0, "Vx-", color="#3b82f6", fontsize=10, ha="right", va="center")
    ax_arrow.text(0, 1.2, "Vy+", color="#22c55e", fontsize=10, ha="center", va="bottom")
    ax_arrow.text(0, -1.2, "Vy-", color="#a855f7", fontsize=10, ha="center", va="top")

    # Draw velocity arrow
    max_vel = 30.0
    norm_vx = np.clip(vx / max_vel, -1, 1)
    norm_vy = np.clip(vy / max_vel, -1, 1)
    vel_mag = np.sqrt(norm_vx**2 + norm_vy**2)

    if vel_mag > 0.05:
        if abs(norm_vx) > abs(norm_vy):
            arrow_color = "#ef4444" if norm_vx > 0 else "#3b82f6"
        else:
            arrow_color = "#22c55e" if norm_vy > 0 else "#a855f7"

        arrow = FancyArrow(
            0,
            0,
            norm_vx * 0.9,
            norm_vy * 0.9,
            width=0.15,
            head_width=0.3,
            head_length=0.15,
            fc=arrow_color,
            ec="white",
            linewidth=1,
        )
        ax_arrow.add_patch(arrow)

    # Middle bottom: Velocity text
    ax_vel_text = fig.add_subplot(gs[1, 1])
    ax_vel_text.set_facecolor("#0a0a14")
    ax_vel_text.axis("off")

    ax_vel_text.text(
        0.5,
        0.7,
        f"Vx: {vx:+.1f}\nVy: {vy:+.1f}",
        color="white",
        fontsize=14,
        ha="center",
        va="center",
        transform=ax_vel_text.transAxes,
        family="monospace",
    )
    ax_vel_text.text(
        0.5,
        0.3,
        f"t = {current_time:.2f}s",
        color="#888888",
        fontsize=11,
        ha="center",
        va="center",
        transform=ax_vel_text.transAxes,
        family="monospace",
    )

    # Right: Velocity time series
    ax_graph = fig.add_subplot(gs[:, 2])
    ax_graph.set_facecolor("#0a0a14")
    ax_graph.set_title("Velocity Over Time", color="white", fontsize=12)
    ax_graph.set_xlabel("Time (s)", color="white", fontsize=10)
    ax_graph.set_ylabel("Velocity", color="white", fontsize=10)
    ax_graph.tick_params(colors="white")
    for spine in ax_graph.spines.values():
        spine.set_color("#333344")
    ax_graph.grid(True, alpha=0.2, color="#333344")

    # Plot velocity history
    ax_graph.plot(time_history, vx_history, color="#ef4444", linewidth=2, label="Vx")
    ax_graph.plot(time_history, vy_history, color="#22c55e", linewidth=2, label="Vy")

    # Mark current time
    ax_graph.axvline(
        current_time, color="#facc15", linewidth=2, linestyle="--", alpha=0.7
    )

    ax_graph.legend(
        loc="upper right", facecolor="#1a1a24", edgecolor="#333344", labelcolor="white"
    )

    if output_path:
        print(f"Saving snapshot to {output_path}...")
        plt.savefig(output_path, dpi=150, facecolor="#0a0a14", edgecolor="none")
        print("Done!")
    else:
        plt.show()

    plt.close()


def create_visualization(
    neural_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    output_path: Path | None = None,
    fps: int = 30,
    max_frames: int | None = None,
    start_time: float = 0.0,
    history_seconds: float = 5.0,
):
    """Create animated visualization."""
    grid_size = 32
    sample_rate = 500  # Hz

    # Calculate start index
    start_idx = int(start_time * sample_rate)
    start_idx = max(0, min(start_idx, len(neural_df) - 1))

    # Downsample for visualization (every N samples)
    step = max(1, sample_rate // fps)
    frame_indices = list(range(start_idx, len(neural_df), step))
    if max_frames:
        frame_indices = frame_indices[:max_frames]

    if not frame_indices:
        print("Error: No frames to render")
        return None

    print(f"Rendering {len(frame_indices)} frames from t={start_time:.2f}s...")

    # History buffer for velocity graph
    history_samples = int(history_seconds * fps)
    vx_history = []
    vy_history = []
    time_history = []

    # Set up figure
    fig = plt.figure(figsize=(16, 7), facecolor="#0a0a14")
    fig.suptitle(
        "BCI Neural Activity & Velocity Visualization",
        fontsize=14,
        color="white",
        fontweight="bold",
    )

    # Grid spec for layout
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.2, 0.6, 1.2],
        height_ratios=[1, 1],
        hspace=0.3,
        wspace=0.3,
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.1,
    )

    # Left: Heatmap
    ax_heatmap = fig.add_subplot(gs[:, 0])
    ax_heatmap.set_facecolor("#0a0a14")
    ax_heatmap.set_title("Neural Activity (32x32 Grid)", color="white", fontsize=12)

    # Middle: Velocity arrow
    ax_arrow = fig.add_subplot(gs[0, 1])
    ax_arrow.set_facecolor("#0a0a14")
    ax_arrow.set_title("Velocity Direction", color="white", fontsize=12)
    ax_arrow.set_xlim(-1.5, 1.5)
    ax_arrow.set_ylim(-1.5, 1.5)
    ax_arrow.set_aspect("equal")
    ax_arrow.axis("off")

    # Draw reference circle
    circle = Circle((0, 0), 1.0, fill=False, color="#333344", linewidth=2)
    ax_arrow.add_patch(circle)

    # Cross-hairs
    ax_arrow.axhline(0, color="#333344", linewidth=1, linestyle="--")
    ax_arrow.axvline(0, color="#333344", linewidth=1, linestyle="--")

    # Labels
    ax_arrow.text(1.2, 0, "Vx+", color="#ef4444", fontsize=10, ha="left", va="center")
    ax_arrow.text(-1.2, 0, "Vx-", color="#3b82f6", fontsize=10, ha="right", va="center")
    ax_arrow.text(0, 1.2, "Vy+", color="#22c55e", fontsize=10, ha="center", va="bottom")
    ax_arrow.text(0, -1.2, "Vy-", color="#a855f7", fontsize=10, ha="center", va="top")

    # Middle bottom: Velocity magnitude text
    ax_vel_text = fig.add_subplot(gs[1, 1])
    ax_vel_text.set_facecolor("#0a0a14")
    ax_vel_text.axis("off")

    # Right: Velocity time series
    ax_graph = fig.add_subplot(gs[:, 2])
    ax_graph.set_facecolor("#0a0a14")
    ax_graph.set_title("Velocity Over Time", color="white", fontsize=12)
    ax_graph.set_xlabel("Time (s)", color="white", fontsize=10)
    ax_graph.set_ylabel("Velocity", color="white", fontsize=10)
    ax_graph.tick_params(colors="white")
    for spine in ax_graph.spines.values():
        spine.set_color("#333344")
    ax_graph.grid(True, alpha=0.2, color="#333344")

    # Initialize plot elements
    heatmap_img = ax_heatmap.imshow(
        np.zeros((grid_size, grid_size)),
        cmap="inferno",
        aspect="equal",
        vmin=0,
        vmax=1,
        origin="lower",
    )
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])

    # Hotspot markers (will be updated)
    hotspot_artists = []

    # Velocity arrow (will be updated)
    arrow_artist = None

    # Velocity text
    vel_text = ax_vel_text.text(
        0.5,
        0.7,
        "",
        color="white",
        fontsize=14,
        ha="center",
        va="center",
        transform=ax_vel_text.transAxes,
        family="monospace",
    )
    time_text = ax_vel_text.text(
        0.5,
        0.3,
        "",
        color="#888888",
        fontsize=11,
        ha="center",
        va="center",
        transform=ax_vel_text.transAxes,
        family="monospace",
    )

    # Graph lines
    (vx_line,) = ax_graph.plot([], [], color="#ef4444", linewidth=2, label="Vx")
    (vy_line,) = ax_graph.plot([], [], color="#22c55e", linewidth=2, label="Vy")
    ax_graph.legend(
        loc="upper right", facecolor="#1a1a24", edgecolor="#333344", labelcolor="white"
    )

    def update(frame_num):
        nonlocal arrow_artist, hotspot_artists, vx_history, vy_history, time_history

        idx = frame_indices[frame_num]
        neural_row = neural_df.iloc[idx]
        gt_row = gt_df.iloc[idx]

        # Get velocity
        vx = gt_row["vx"]
        vy = gt_row["vy"]
        time_s = gt_row["time_s"]

        # Update history
        vx_history.append(vx)
        vy_history.append(vy)
        time_history.append(time_s)

        # Trim history
        if len(vx_history) > history_samples:
            vx_history.pop(0)
            vy_history.pop(0)
            time_history.pop(0)

        # Process neural data
        activity = process_neural_power(neural_row, grid_size)

        # Apply spatial smoothing
        smoothed = gaussian_filter(activity, sigma=1.5)

        # Update heatmap
        heatmap_img.set_data(smoothed)

        # Clear old hotspot markers
        for artist in hotspot_artists:
            artist.remove()
        hotspot_artists = []

        # Draw hotspot regions from ground truth
        region_colors = {
            "vx_pos": "#ef4444",  # red
            "vx_neg": "#3b82f6",  # blue
            "vy_pos": "#22c55e",  # green
            "vy_neg": "#a855f7",  # purple
        }
        region_labels = {
            "vx_pos": "Vx+",
            "vx_neg": "Vx-",
            "vy_pos": "Vy+",
            "vy_neg": "Vy-",
        }

        for region, color in region_colors.items():
            row = gt_row[f"{region}_center_row"]
            col = gt_row[f"{region}_center_col"]
            # Convert to 0-indexed for display
            circle = plt.Circle(
                (col - 1, row - 1),
                1.5,
                fill=False,
                color=color,
                linewidth=2,
            )
            ax_heatmap.add_patch(circle)
            hotspot_artists.append(circle)

            # Label
            text = ax_heatmap.text(
                col - 1,
                row - 1 + 2.5,
                region_labels[region],
                color=color,
                fontsize=8,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            hotspot_artists.append(text)

        # Update velocity arrow
        if arrow_artist is not None:
            arrow_artist.remove()

        # Normalize velocity for display (max velocity ~30)
        max_vel = 30.0
        norm_vx = np.clip(vx / max_vel, -1, 1)
        norm_vy = np.clip(vy / max_vel, -1, 1)
        vel_mag = np.sqrt(norm_vx**2 + norm_vy**2)

        if vel_mag > 0.05:  # Only draw if significant velocity
            # Color based on direction
            if abs(norm_vx) > abs(norm_vy):
                arrow_color = "#ef4444" if norm_vx > 0 else "#3b82f6"
            else:
                arrow_color = "#22c55e" if norm_vy > 0 else "#a855f7"

            arrow_artist = FancyArrow(
                0,
                0,
                norm_vx * 0.9,
                norm_vy * 0.9,
                width=0.15,
                head_width=0.3,
                head_length=0.15,
                fc=arrow_color,
                ec="white",
                linewidth=1,
            )
            ax_arrow.add_patch(arrow_artist)
        else:
            arrow_artist = None

        # Update velocity text
        vel_text.set_text(f"Vx: {vx:+.1f}\nVy: {vy:+.1f}")
        time_text.set_text(f"t = {time_s:.2f}s")

        # Update graph
        if len(time_history) > 1:
            vx_line.set_data(time_history, vx_history)
            vy_line.set_data(time_history, vy_history)

            # Adjust axis limits
            t_min, t_max = min(time_history), max(time_history)
            if t_max > t_min:
                ax_graph.set_xlim(t_min, t_max)
            y_min = min(min(vx_history), min(vy_history)) - 5
            y_max = max(max(vx_history), max(vy_history)) + 5
            if y_max > y_min:
                ax_graph.set_ylim(y_min, y_max)

        return [heatmap_img, vx_line, vy_line, vel_text, time_text] + hotspot_artists

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=False,
    )

    if output_path:
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer="pillow", fps=fps)
        print("Done!")
    else:
        plt.show()

    return anim


def main():
    parser = argparse.ArgumentParser(
        description="Visualize neural activity with velocity overlay"
    )
    parser.add_argument(
        "difficulty",
        choices=["super_easy", "easy", "medium", "hard"],
        help="Dataset difficulty level",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (e.g., output.gif or snapshot.png)",
    )
    parser.add_argument(
        "--snapshot",
        type=float,
        help="Create a static snapshot at specified time (seconds) instead of animation",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to render",
    )
    parser.add_argument(
        "--history",
        type=float,
        default=5.0,
        help="Seconds of velocity history to show in graph (default: 5.0)",
    )

    args = parser.parse_args()

    data_dir = Path("data") / args.difficulty
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} not found.")
        print(f"Run: uv run python -m scripts.download {args.difficulty}")
        return 1

    print(f"Loading data from {data_dir}...")
    neural_df, gt_df = load_data(data_dir)
    print(f"Loaded {len(neural_df)} samples ({len(neural_df)/500:.1f}s)")

    if args.snapshot is not None:
        # Static snapshot mode
        create_static_snapshot(
            neural_df,
            gt_df,
            time_s=args.snapshot,
            output_path=args.output,
            history_seconds=args.history,
        )
    else:
        # Animation mode
        create_visualization(
            neural_df,
            gt_df,
            output_path=args.output,
            fps=args.fps,
            max_frames=args.max_frames,
            start_time=args.start_time,
            history_seconds=args.history,
        )

    return 0


if __name__ == "__main__":
    exit(main())
