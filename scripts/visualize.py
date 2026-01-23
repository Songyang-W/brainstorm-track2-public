"""
Neural Data Visualization for BrainStorm Track 2

Visualizes the 1024-channel neural data as a 32x32 heatmap grid,
showing neural activity patterns that encode cursor velocity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.signal import butter, sosfiltfilt
from pathlib import Path


def bandpass_filter(data, lowcut, highcut, fs=500, order=4):
    """Apply bandpass filter to extract frequency band of interest."""
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=0)


def compute_power(filtered_data, window_ms=100, fs=500):
    """Compute smoothed power envelope."""
    power = filtered_data ** 2
    window_samples = int(window_ms * fs / 1000)
    kernel = np.ones(window_samples) / window_samples
    
    # Handle 2D data (samples x channels)
    if filtered_data.ndim == 2:
        smoothed = np.zeros_like(power)
        for i in range(power.shape[1]):
            smoothed[:, i] = np.convolve(power[:, i], kernel, mode='same')
        return smoothed
    else:
        return np.convolve(power, kernel, mode='same')


def channels_to_grid(channel_data):
    """
    Convert 1024 channels to 32x32 grid.
    Channel 0 is top-left (0,0), Channel 1023 is bottom-right (31,31).
    """
    return np.array(channel_data).reshape(32, 32)


def load_data(difficulty="easy"):
    """Load neural data and ground truth for a given difficulty."""
    data_path = Path(f"data/{difficulty}")
    
    neural_data = pd.read_parquet(data_path / "track2_data.parquet")
    ground_truth = pd.read_parquet(data_path / "ground_truth.parquet")
    
    print(f"Loaded {difficulty} dataset:")
    print(f"  Neural data shape: {neural_data.shape}")
    print(f"  Time range: {neural_data.index[0]:.2f}s to {neural_data.index[-1]:.2f}s")
    print(f"  Ground truth columns: {list(ground_truth.columns)}")
    
    return neural_data, ground_truth


def visualize_static_heatmap(neural_data, ground_truth=None, time_point=5.0, 
                             filter_band=None, title=None):
    """
    Display a static heatmap of neural activity at a specific time point.
    
    Args:
        neural_data: DataFrame with neural signals
        ground_truth: Optional DataFrame with cursor velocity
        time_point: Time in seconds to visualize
        filter_band: Optional tuple (lowcut, highcut) for bandpass filtering
        title: Optional custom title
    """
    # Get closest time index
    idx = np.abs(neural_data.index - time_point).argmin()
    actual_time = neural_data.index[idx]
    
    if filter_band:
        # Apply bandpass filter and compute power over a window
        lowcut, highcut = filter_band
        window_samples = 50  # 100ms window at 500Hz
        start_idx = max(0, idx - window_samples)
        end_idx = min(len(neural_data), idx + window_samples)
        
        data_segment = neural_data.iloc[start_idx:end_idx].values
        filtered = bandpass_filter(data_segment, lowcut, highcut)
        power = compute_power(filtered)
        channel_values = power[idx - start_idx]
    else:
        channel_values = neural_data.iloc[idx].values
    
    # Reshape to 32x32 grid
    grid = channels_to_grid(channel_values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(grid, cmap='hot', aspect='equal', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Signal Amplitude' if not filter_band else 'Power')
    
    # Add grid lines
    ax.set_xticks(np.arange(0, 32, 4))
    ax.set_yticks(np.arange(0, 32, 4))
    ax.grid(True, alpha=0.3, color='white')
    
    # Labels
    ax.set_xlabel('Column (1-32)')
    ax.set_ylabel('Row (1-32)')
    
    # Title with velocity info if available
    if title:
        ax.set_title(title)
    elif ground_truth is not None and 'vx' in ground_truth.columns:
        gt_idx = np.abs(ground_truth['time_s'].values - actual_time).argmin()
        vx = ground_truth['vx'].iloc[gt_idx]
        vy = ground_truth['vy'].iloc[gt_idx]
        ax.set_title(f'Neural Activity at t={actual_time:.2f}s\nCursor velocity: vx={vx:.2f}, vy={vy:.2f}')
    else:
        ax.set_title(f'Neural Activity at t={actual_time:.2f}s')
    
    plt.tight_layout()
    return fig, ax


def visualize_interactive(neural_data, ground_truth=None, filter_band=(70, 150)):
    """
    Interactive visualization with a time slider.
    
    Args:
        neural_data: DataFrame with neural signals
        ground_truth: Optional DataFrame with cursor velocity
        filter_band: Tuple (lowcut, highcut) for bandpass filtering
    """
    # Pre-compute filtered power for entire dataset
    print("Pre-computing filtered power (this may take a moment)...")
    lowcut, highcut = filter_band
    filtered = bandpass_filter(neural_data.values, lowcut, highcut)
    power_data = compute_power(filtered)
    print("Done!")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)
    
    # Initial time
    t_idx = 0
    
    # Left plot: Heatmap
    grid = channels_to_grid(power_data[t_idx])
    im = axes[0].imshow(grid, cmap='hot', aspect='equal', interpolation='bilinear',
                        vmin=np.percentile(power_data, 5), 
                        vmax=np.percentile(power_data, 95))
    plt.colorbar(im, ax=axes[0], label='Power')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    axes[0].set_title(f'Neural Activity Heatmap (t={neural_data.index[0]:.2f}s)')
    
    # Right plot: Velocity
    if ground_truth is not None and 'vx' in ground_truth.columns:
        time_vals = ground_truth['time_s'].values
        vx = ground_truth['vx'].values
        vy = ground_truth['vy'].values
        
        axes[1].plot(time_vals, vx, label='vx (horizontal)', alpha=0.7)
        axes[1].plot(time_vals, vy, label='vy (vertical)', alpha=0.7)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        vline = axes[1].axvline(x=neural_data.index[0], color='r', linestyle='-', lw=2, label='Current time')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Velocity')
        axes[1].set_title('Cursor Velocity')
        axes[1].legend()
        axes[1].set_xlim(time_vals[0], time_vals[-1])
    else:
        axes[1].text(0.5, 0.5, 'No ground truth available', 
                     ha='center', va='center', transform=axes[1].transAxes)
        vline = None
    
    # Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Time (s)', 
                    neural_data.index[0], neural_data.index[-1],
                    valinit=neural_data.index[0], valstep=0.02)
    
    def update(val):
        t = slider.val
        idx = np.abs(neural_data.index - t).argmin()
        
        grid = channels_to_grid(power_data[idx])
        im.set_data(grid)
        axes[0].set_title(f'Neural Activity Heatmap (t={t:.2f}s)')
        
        if vline:
            vline.set_xdata([t, t])
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.suptitle(f'Neural Signal Visualization ({lowcut}-{highcut}Hz Band Power)', fontsize=12)
    plt.show()
    
    return fig


def visualize_animation(neural_data, ground_truth=None, filter_band=(70, 150),
                        start_time=0, duration=10, fps=30, save_path=None):
    """
    Create an animated visualization of neural activity over time.
    
    Args:
        neural_data: DataFrame with neural signals
        ground_truth: Optional DataFrame with cursor velocity
        filter_band: Tuple (lowcut, highcut) for bandpass filtering
        start_time: Start time in seconds
        duration: Duration in seconds
        fps: Frames per second
        save_path: Optional path to save animation
    """
    print("Pre-computing filtered power...")
    lowcut, highcut = filter_band
    filtered = bandpass_filter(neural_data.values, lowcut, highcut)
    power_data = compute_power(filtered)
    print("Done!")
    
    # Get time indices
    start_idx = np.abs(neural_data.index - start_time).argmin()
    end_idx = np.abs(neural_data.index - (start_time + duration)).argmin()
    
    # Subsample for animation
    step = max(1, int(500 / fps))  # 500Hz data, subsample to fps
    indices = range(start_idx, end_idx, step)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    grid = channels_to_grid(power_data[start_idx])
    im = axes[0].imshow(grid, cmap='hot', aspect='equal', interpolation='bilinear',
                        vmin=np.percentile(power_data[start_idx:end_idx], 5),
                        vmax=np.percentile(power_data[start_idx:end_idx], 95))
    plt.colorbar(im, ax=axes[0], label='Power')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    title = axes[0].set_title(f't={start_time:.2f}s')
    
    # Velocity plot
    if ground_truth is not None and 'vx' in ground_truth.columns:
        time_vals = ground_truth['time_s'].values
        vx = ground_truth['vx'].values
        vy = ground_truth['vy'].values
        
        mask = (time_vals >= start_time) & (time_vals <= start_time + duration)
        axes[1].plot(time_vals[mask], vx[mask], label='vx', alpha=0.7)
        axes[1].plot(time_vals[mask], vy[mask], label='vy', alpha=0.7)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        vline = axes[1].axvline(x=start_time, color='r', lw=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Velocity')
        axes[1].set_title('Cursor Velocity')
        axes[1].legend()
    else:
        vline = None
    
    plt.suptitle(f'Neural Activity ({lowcut}-{highcut}Hz Band Power)')
    plt.tight_layout()
    
    def animate(frame):
        idx = list(indices)[frame]
        t = neural_data.index[idx]
        
        grid = channels_to_grid(power_data[idx])
        im.set_data(grid)
        title.set_text(f't={t:.2f}s')
        
        if vline:
            vline.set_xdata([t, t])
        
        return [im, title] + ([vline] if vline else [])
    
    anim = FuncAnimation(fig, animate, frames=len(list(indices)), 
                         interval=1000/fps, blit=True)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps)
        print("Done!")
    
    plt.show()
    return anim


def visualize_multi_timepoint(neural_data, ground_truth=None, 
                              times=[1, 5, 10, 15, 20], filter_band=(70, 150)):
    """
    Show heatmaps at multiple time points for comparison.
    """
    print("Pre-computing filtered power...")
    lowcut, highcut = filter_band
    filtered = bandpass_filter(neural_data.values, lowcut, highcut)
    power_data = compute_power(filtered)
    print("Done!")
    
    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4))
    
    vmin = np.percentile(power_data, 5)
    vmax = np.percentile(power_data, 95)
    
    for i, t in enumerate(times):
        idx = np.abs(neural_data.index - t).argmin()
        grid = channels_to_grid(power_data[idx])
        
        im = axes[i].imshow(grid, cmap='hot', aspect='equal', 
                           interpolation='bilinear', vmin=vmin, vmax=vmax)
        
        # Add velocity info
        subtitle = f't={t:.1f}s'
        if ground_truth is not None and 'vx' in ground_truth.columns:
            gt_idx = np.abs(ground_truth['time_s'].values - t).argmin()
            vx = ground_truth['vx'].iloc[gt_idx]
            vy = ground_truth['vy'].iloc[gt_idx]
            subtitle += f'\nvx={vx:.1f}, vy={vy:.1f}'
        
        axes[i].set_title(subtitle, fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.colorbar(im, ax=axes, label='Power', shrink=0.8)
    plt.suptitle(f'Neural Activity Over Time ({lowcut}-{highcut}Hz Band)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Load easy dataset
    print("=" * 50)
    print("BrainStorm Track 2 - Neural Data Visualization")
    print("=" * 50)
    
    neural_data, ground_truth = load_data("easy")
    
    print("\n[1/3] Showing static heatmap at t=5s...")
    visualize_static_heatmap(neural_data, ground_truth, time_point=5.0, 
                            filter_band=(70, 150))
    plt.show()
    
    print("\n[2/3] Showing multi-timepoint comparison...")
    visualize_multi_timepoint(neural_data, ground_truth, 
                             times=[2, 5, 10, 15, 20], filter_band=(70, 150))
    
    print("\n[3/3] Launching interactive visualization...")
    print("Use the slider to explore different time points.")
    visualize_interactive(neural_data, ground_truth, filter_band=(70, 150))