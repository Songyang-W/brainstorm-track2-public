"""
Hotspot Detection and Tracking Algorithm for BrainStorm Track 2

This module detects neural hotspots (high activity regions) in the 32x32 channel grid
and predicts cursor velocity based on hotspot positions.

Processing Pipeline:
    Raw Data (500 Hz, 1024 channels)
        │
        ▼
    Bandpass Filter (e.g., 70-150 Hz)
        │
        ▼
    Power/Envelope Extraction
        │
        ▼
    Temporal Smoothing (EMA or sliding window)
        │
        ▼
    Reshape to 32×32 Grid
        │
        ▼
    Spatial Smoothing (optional)
        │
        ▼
    Hotspot Detection & Velocity Prediction

Usage:
    python scripts/hotspot_tracker.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ndimage
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import pearsonr
from pathlib import Path


# =============================================================================
# Signal Processing Pipeline
# =============================================================================

class SignalProcessor:
    """
    Signal processing pipeline for neural data.
    
    Pipeline:
    1. Bandpass filter (extract frequency band of interest)
    2. Power/envelope extraction (Hilbert transform or squaring)
    3. Temporal smoothing (EMA or sliding window)
    4. Reshape to 32x32 grid
    5. Spatial smoothing (Gaussian blur)
    """
    
    def __init__(self, fs=500, lowcut=70, highcut=150, 
                 ema_alpha=0.1, spatial_sigma=1.0,
                 envelope_method='hilbert'):
        """
        Initialize the signal processor.
        
        Args:
            fs: Sampling frequency (Hz)
            lowcut: Low cutoff for bandpass filter (Hz)
            highcut: High cutoff for bandpass filter (Hz)
            ema_alpha: Exponential moving average smoothing factor (0-1)
                       Higher = more responsive, Lower = more smooth
            spatial_sigma: Gaussian smoothing sigma for spatial filtering
            envelope_method: 'hilbert' or 'square' for envelope extraction
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.ema_alpha = ema_alpha
        self.spatial_sigma = spatial_sigma
        self.envelope_method = envelope_method
        
        # Pre-compute filter coefficients
        self.sos = butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
        
        # State for real-time EMA smoothing
        self.ema_state = None

        
    def subtract_mean(self, data):
        for row in data:
            data[row, :] -= np.mean(data[row, :])
        return data

    def bandpass_filter(self, data):
        """
        Apply bandpass filter to extract frequency band of interest.
        
        Args:
            data: numpy array (n_samples, n_channels) or (n_samples,)
            
        Returns:
            Filtered data with same shape
        """
        return sosfiltfilt(self.sos, data, axis=0)
    
    def extract_envelope(self, filtered_data):
        """
        Extract power envelope from filtered signal.
        
        Args:
            filtered_data: Bandpass filtered data
            
        Returns:
            Power envelope (amplitude squared or Hilbert envelope)
        """
        if self.envelope_method == 'hilbert':
            # Hilbert transform gives instantaneous amplitude
            analytic_signal = hilbert(filtered_data, axis=0)
            envelope = np.abs(analytic_signal)
        else:
            # Simple squaring (power)
            envelope = filtered_data ** 2
            
        return envelope
    
    def temporal_smooth_window(self, envelope, window_ms=100):
        """
        Apply sliding window temporal smoothing.
        
        Args:
            envelope: Power envelope data (n_samples, n_channels)
            window_ms: Window size in milliseconds
            
        Returns:
            Temporally smoothed data
        """
        window_samples = int(window_ms * self.fs / 1000)
        kernel = np.ones(window_samples) / window_samples
        
        if envelope.ndim == 1:
            return np.convolve(envelope, kernel, mode='same')
        else:
            smoothed = np.zeros_like(envelope)
            for ch in range(envelope.shape[1]):
                smoothed[:, ch] = np.convolve(envelope[:, ch], kernel, mode='same')
            return smoothed
    
    def temporal_smooth_ema(self, envelope, reset=False):
        """
        Apply exponential moving average for real-time smoothing.
        
        Args:
            envelope: Current envelope values (n_channels,) for single sample
                      or (n_samples, n_channels) for batch
            reset: Reset EMA state
            
        Returns:
            EMA smoothed values
        """
        if reset or self.ema_state is None:
            if envelope.ndim == 1:
                self.ema_state = envelope.copy()
            else:
                self.ema_state = envelope[0].copy()
        
        if envelope.ndim == 1:
            # Single sample update
            self.ema_state = self.ema_alpha * envelope + (1 - self.ema_alpha) * self.ema_state
            return self.ema_state
        else:
            # Batch processing
            smoothed = np.zeros_like(envelope)
            for i in range(envelope.shape[0]):
                self.ema_state = self.ema_alpha * envelope[i] + (1 - self.ema_alpha) * self.ema_state
                smoothed[i] = self.ema_state
            return smoothed
    
    def reshape_to_grid(self, channel_data):
        """
        Reshape 1024 channels to 32x32 grid.
        
        Args:
            channel_data: 1D array of 1024 values
            
        Returns:
            32x32 numpy array
        """
        return np.array(channel_data).reshape(32, 32)
    
    def spatial_smooth(self, grid):
        """
        Apply Gaussian spatial smoothing to the grid.
        
        Args:
            grid: 32x32 numpy array
            
        Returns:
            Spatially smoothed 32x32 array
        """
        if self.spatial_sigma > 0:
            return ndimage.gaussian_filter(grid, sigma=self.spatial_sigma)
        return grid
    
    
    def process_window(self, raw_data, use_ema=False, window_ms=100):
        """
        Full processing pipeline for a time window of data.
        
        Args:
            raw_data: DataFrame or array (n_samples, 1024 channels)
            use_ema: Use EMA instead of sliding window for temporal smoothing
            window_ms: Window size for sliding window smoothing
            
        Returns:
            Processed 32x32 grid
        """
        # Convert DataFrame to numpy if needed
        if isinstance(raw_data, pd.DataFrame):
            data = raw_data.values
        else:
            data = raw_data

            
        # Step 1: Bandpass filter
        filtered = self.bandpass_filter(data)
        
        # Step 2: Extract envelope
        envelope = self.extract_envelope(filtered)
        
        # Step 3: Temporal smoothing
        if use_ema:
            smoothed = self.temporal_smooth_ema(envelope)
            # Take the last sample (most recent)
            channel_values = smoothed[-1] if smoothed.ndim > 1 else smoothed
        else:
            smoothed = self.temporal_smooth_window(envelope, window_ms)
            # Take the mean across time window
            channel_values = smoothed.mean(axis=0)
        
        # Step 4: Reshape to grid
        grid = self.reshape_to_grid(channel_values)
        
        # Step 5: Spatial smoothing
        grid = self.spatial_smooth(grid)

        
        return grid
    
    def process_single_sample(self, sample_data, reset_ema=False):
        """
        Process a single sample for real-time streaming.
        
        Note: Bandpass filtering requires multiple samples, so this method
        assumes pre-filtered data or uses a simple approach.
        
        Args:
            sample_data: 1D array of 1024 channel values (already filtered)
            reset_ema: Reset EMA state
            
        Returns:
            Processed 32x32 grid
        """
        # For real-time, we assume data is already filtered
        # Just do envelope + EMA + reshape + spatial smooth
        envelope = sample_data ** 2 if self.envelope_method != 'hilbert' else np.abs(sample_data)
        smoothed = self.temporal_smooth_ema(envelope, reset=reset_ema)
        grid = self.reshape_to_grid(smoothed)
        grid = self.spatial_smooth(grid)
        return grid


class HotspotTracker:
    """
    Tracks neural hotspots and predicts cursor movement based on activity patterns.
    
    The algorithm:
    1. Computes RMS power for each channel over a time window
    2. Detects hotspots using thresholding + connected component labeling
    3. Finds the centroid (weighted center of mass) of each hotspot
    4. Predicts velocity based on hotspot position relative to grid center
    """
    
    def __init__(self, threshold_percentile=85, min_spot_size=3, smoothing_sigma=1.5, max_gap=10):
        """
        Initialize the HotspotTracker.
        
        Parameters:
        - threshold_percentile: Percentile above which activity is considered a hotspot
        - min_spot_size: Minimum number of pixels to be considered a valid hotspot
        - smoothing_sigma: Gaussian smoothing applied to the grid before detection
        """
        self.threshold_percentile = threshold_percentile
        self.min_spot_size = min_spot_size
        self.smoothing_sigma = smoothing_sigma
        self.previous_hotspots = []
        self.tracked_hotspots = []   # persistent list
        self.max_gap = max_gap       # frames allowed to disappear
        self.frame_count = 0
    def detect_hotspots(self, grid):
        """
        Detect hotspots in the 32x32 grid.
        
        Args:
            grid: 32x32 numpy array of power values
            
        Returns:
            List of hotspots, each with: center_row, center_col, intensity, size
        """
        # Smooth the grid to reduce noise
        smoothed = ndimage.gaussian_filter(grid, sigma=self.smoothing_sigma)
        
        # Threshold to find high activity regions
        threshold = np.percentile(smoothed, self.threshold_percentile)
        binary_mask = smoothed > threshold
        
        # Label connected regions
        labeled, num_features = ndimage.label(binary_mask)
        
        hotspots = []
        for i in range(1, num_features + 1):
            spot_mask = labeled == i
            spot_size = np.sum(spot_mask)
            
            # Skip tiny spots (noise)
            if spot_size < self.min_spot_size:
                continue
            
            # Calculate weighted centroid
            rows, cols = np.where(spot_mask)
            intensities = smoothed[spot_mask]
            total_intensity = np.sum(intensities)
            
            center_row = np.sum(rows * intensities) / total_intensity
            center_col = np.sum(cols * intensities) / total_intensity
            # total_intensity = np.sum(intensities)

            # if total_intensity <= 0 or not np.isfinite(total_intensity):
            #     continue
            
            hotspots.append({
                'center_row': center_row,
                'center_col': center_col,
                'intensity': np.mean(intensities),
                'size': spot_size,
                'max_intensity': np.max(intensities)
            })
            
        
        # Sort by intensity (strongest first)
        hotspots.sort(key=lambda x: x['intensity'], reverse=True)
        return hotspots
    


    def _match_hotspots(self, detected, max_dist=5.0):

    # Match detected hotspots to existing tracked ones by proximity.

        for h in self.tracked_hotspots:
            h['active'] = False

        for d in detected:
            best = None
            best_dist = np.inf

            for h in self.tracked_hotspots:
                dist = np.hypot(d['center_row'] - h['row'],
                                d['center_col'] - h['col'])
                if dist < best_dist and dist < max_dist:
                    best = h
                    best_dist = dist

            if best is not None:
                # Update existing hotspot
                best['intensity'] = d['intensity']
                best['last_seen'] = self.frame_count
                best['active'] = True
            else:
                # New hotspot
                self.tracked_hotspots.append({
                    'row': d['center_row'],
                    'col': d['center_col'],
                    'intensity': d['intensity'],
                    'last_seen': self.frame_count,
                    'active': True
                })



    def _prune_hotspots(self):
        self.tracked_hotspots = [
            h for h in self.tracked_hotspots
            if self.frame_count - h['last_seen'] <= self.max_gap
        ]



    def get_average_hotspot_center(self, hotspots, top_n=8):
        """
        Get the weighted average center of the top N hotspots.
        
        Args:
            hotspots: List of detected hotspots
            top_n: Number of top hotspots to average
            
        Returns:
            (avg_row, avg_col) or (None, None) if no hotspots
        """
        if not hotspots:
            return None, None
        
        top_hotspots = hotspots[:top_n]
        total_weight = sum(h['intensity'] for h in top_hotspots)
        
        avg_row = sum(h['center_row'] * h['intensity'] for h in top_hotspots) / total_weight
        avg_col = sum(h['center_col'] * h['intensity'] for h in top_hotspots) / total_weight
        
        return avg_row, avg_col
    

    
    def track_and_predict(self, grid, top_n=6):
        """
        Main function: detect hotspots and predict cursor velocity.
        
        Args:
            grid: 32x32 numpy array of power values
            
        Returns:
            - hotspots: List of detected hotspots
            - avg_center: (row, col) average center of hotspots
            - velocity_prediction: (vx, vy) predicted cursor velocity (-1 to 1)
        """
        # hotspots = self.detect_hotspots(grid)
        # avg_row, avg_col = self.get_average_hotspot_center(hotspots, 8)
        
        # if avg_row is not None:
        #     # Convert grid position to velocity
        #     # Hotspot right of center -> positive vx (moving right)
        #     # Hotspot above center -> positive vy (moving up)
        #     center = 15.5  # Center of 0-31 grid
        #     vx_pred = (avg_col - center) / center  # Normalized -1 to 1
        #     vy_pred = -(avg_row - center) / center  # Negative because row 0 is top
        #     velocity_prediction = (vx_pred, vy_pred)
        # else:
        #     velocity_prediction = (0, 0)
        
        # self.previous_hotspots = hotspots
        # return hotspots, (avg_row, avg_col), velocity_prediction
    
        self.frame_count += 1

        detected = self.detect_hotspots(grid)
        self._match_hotspots(detected)
        self._prune_hotspots()

        if not self.tracked_hotspots:
            return [], (None, None), (0, 0)

        # Weighted COM of ALL tracked hotspots (active or not)
        weights = np.array([h['intensity'] for h in self.tracked_hotspots[:top_n]])
        rows = np.array([h['row'] for h in self.tracked_hotspots[:top_n]])
        cols = np.array([h['col'] for h in self.tracked_hotspots[:top_n]])

        avg_row = np.sum(rows * weights) / np.sum(weights)
        avg_col = np.sum(cols * weights) / np.sum(weights)

        center = 15.5
        vx = (avg_col - center) / center
        vy = -(avg_row - center) / center

        return self.tracked_hotspots, (avg_row, avg_col), (vx, vy)


def load_data(difficulty="super_easy"):
    """Load neural data and ground truth for a given difficulty."""
    data_path = Path(f"data/{difficulty}")
    
    neural_data = pd.read_parquet(data_path / "track2_data.parquet")
    ground_truth = pd.read_parquet(data_path / "ground_truth.parquet")
    
    print(f"Loaded {difficulty} dataset:")
    print(f"  Neural data shape: {neural_data.shape}")
    print(f"  Time range: {neural_data.index[0]:.2f}s to {neural_data.index[-1]:.2f}s")
    
    return neural_data, ground_truth


def compute_power_grid(df, t_start, window_size=0.5, processor=None):
    """
    Compute processed power grid for a time window using full pipeline.
    
    Args:
        df: Neural data DataFrame
        t_start: Start time in seconds
        window_size: Window size in seconds
        processor: SignalProcessor instance (creates default if None)
        
    Returns:
        32x32 processed grid
    """
    if processor is None:
        processor = SignalProcessor()
    
    subset = df.loc[t_start:t_start + window_size]
    return processor.process_window(subset)


def visualize_hotspot_detection(df, tracker, t_start, window_size=0.5, processor=None):
    """Visualize hotspot detection on a single time window."""
    power = compute_power_grid(df, t_start, window_size, processor)
    hotspots, avg_center, velocity = tracker.track_and_predict(power)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Power grid with hotspots
    im1 = axes[0].imshow(power, cmap='hot', aspect='equal')
    plt.colorbar(im1, ax=axes[0], label='RMS Power')
    
    # Mark hotspot centers
    for spot in hotspots[:4]:
        axes[0].scatter(spot['center_col'], spot['center_row'], 
                       c='cyan', s=100, marker='x', linewidths=3)
    
    # Mark average center
    if avg_center[0] is not None:
        axes[0].scatter(avg_center[1], avg_center[0], 
                       c='lime', s=200, marker='+', linewidths=4, label='Avg Center')
    
    axes[0].set_title(f'Hotspot Detection (t={t_start:.2f}s)')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    axes[0].legend(loc='upper right')
    
    # Right: Velocity arrow
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    vx, vy = velocity
    axes[1].arrow(0, 0, vx, vy, head_width=0.1, head_length=0.05, 
                  fc='blue', ec='blue', linewidth=2)
    axes[1].scatter([0], [0], c='black', s=100, zorder=5)
    
    axes[1].set_title(f'Predicted Velocity: vx={vx:.3f}, vy={vy:.3f}')
    axes[1].set_xlabel('X Velocity')
    axes[1].set_ylabel('Y Velocity')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, hotspots, avg_center, velocity


def track_hotspots_over_time(df, ground_truth, tracker, t_start=0, t_end=30, 
                              window_size=0.2, step=0.1, processor=None):
    """
    Track hotspots over time and compare to ground truth.
    
    Args:
        df: Neural data DataFrame
        ground_truth: Ground truth DataFrame
        tracker: HotspotTracker instance
        t_start: Start time in seconds
        t_end: End time in seconds
        window_size: Processing window size in seconds
        step: Time step between windows in seconds
        processor: SignalProcessor instance (creates default if None)
        
    Returns:
        DataFrame with predictions and ground truth
    """
    if processor is None:
        processor = SignalProcessor()
        
    times = np.arange(t_start, t_end - window_size, step)
    
    results = {
        'time': [], 'pred_vx': [], 'pred_vy': [],
        'true_vx': [], 'true_vy': [],
        'avg_row': [], 'avg_col': []
    }
    
    for t in times:
        power = compute_power_grid(df, t, window_size, processor)
        hotspots, avg_center, velocity = tracker.track_and_predict(power)
        
        gt_idx = np.abs(ground_truth['time_s'].values - t).argmin()
        
        results['time'].append(t)
        results['pred_vx'].append(velocity[0])
        results['pred_vy'].append(velocity[1])
        results['true_vx'].append(ground_truth['vx'].iloc[gt_idx])
        results['true_vy'].append(ground_truth['vy'].iloc[gt_idx])
        results['avg_row'].append(avg_center[0] if avg_center[0] else 16)
        results['avg_col'].append(avg_center[1] if avg_center[1] else 16)
    
    return pd.DataFrame(results)


def plot_tracking_comparison(tracking_results):
    """Plot predicted vs ground truth velocity."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # X velocity
    axes[0].plot(tracking_results['time'], tracking_results['true_vx'], 
                 label='Ground Truth Vx', alpha=0.7, linewidth=2)
    axes[0].plot(tracking_results['time'], tracking_results['pred_vx'] * 80, 
                 label='Predicted Vx (scaled)', alpha=0.7, linewidth=2)
    axes[0].set_ylabel('X Velocity')
    axes[0].set_title('Hotspot-Based Velocity Prediction vs Ground Truth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y velocity
    axes[1].plot(tracking_results['time'], tracking_results['true_vy'], 
                 label='Ground Truth Vy', alpha=0.7, linewidth=2)
    axes[1].plot(tracking_results['time'], tracking_results['pred_vy'] * 80, 
                 label='Predicted Vy (scaled)', alpha=0.7, linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Y Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Calculate correlation
    corr_vx, _ = pearsonr(tracking_results['true_vx'], tracking_results['pred_vx'])
    corr_vy, _ = pearsonr(tracking_results['true_vy'], tracking_results['pred_vy'])
    
    return fig, corr_vx, corr_vy


def animate_hotspot_tracking(df, ground_truth, tracker, t_start=0, t_end=20, 
                              window_size=0.5, fps=10, processor=None):
    """
    Animated visualization showing hotspot detection and cursor simulation.
    
    Args:
        df: Neural data DataFrame
        ground_truth: Ground truth DataFrame
        tracker: HotspotTracker instance
        t_start: Start time in seconds
        t_end: End time in seconds
        window_size: Processing window size in seconds
        fps: Frames per second for animation
        processor: SignalProcessor instance (creates default if None)
    
    Returns:
        (figure, FuncAnimation) tuple
    """
    if processor is None:
        processor = SignalProcessor()
        
    dt = 1 / fps
    time_steps = np.arange(t_start, t_end - window_size, dt)
    
    # Pre-compute for consistent color scaling
    print("  Pre-computing color scale...")
    sample_grids = []
    for t in np.linspace(t_start, t_end - window_size, 20):
        sample_grids.append(compute_power_grid(df, t, window_size, processor))
    global_max = np.max([g.max() for g in sample_grids]) * 1.2
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Initialize first frame
    power = compute_power_grid(df, t_start, window_size, processor)
    
    # Left: Power grid
    im = axes[0].imshow(power, cmap='hot', aspect='equal', vmin=0, vmax=global_max)
    plt.colorbar(im, ax=axes[0], label='Power')
    hotspot_scatter = axes[0].scatter([], [], c='cyan', s=100, marker='x', linewidths=2)
    center_scatter = axes[0].scatter([], [], c='lime', s=200, marker='+', linewidths=3)
    title0 = axes[0].set_title(f't={t_start:.2f}s')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    
    # Middle: Velocity arrows
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('Vx')
    axes[1].set_ylabel('Vy')
    title1 = axes[1].set_title('Velocity')
    pred_arrow = axes[1].quiver([0], [0], [0], [0], color='blue', scale=1, 
                                 scale_units='xy', angles='xy', width=0.02)
    
    # Right: Cursor trajectory
    axes[2].set_xlim(-100, 100)
    axes[2].set_ylim(-100, 100)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    title2 = axes[2].set_title('Cursor')
    
    # Cursor state
    cursor_state = {'x': 0, 'y': 0, 'history_x': [0], 'history_y': [0]}
    cursor_trail, = axes[2].plot([], [], 'b-', alpha=0.5, linewidth=1)
    cursor_dot = axes[2].scatter([0], [0], c='blue', s=100, zorder=5)
    
    def update(t):
        power = compute_power_grid(df, t, window_size, processor)
        hotspots, avg_center, pred_vel = tracker.track_and_predict(power)
        
        # Update power grid
        im.set_array(power)
        
        # Update hotspot markers
        if hotspots:
            # spots = np.array([[h['center_col'], h['center_row']] for h in hotspots[:4]])
            spots = np.array([[h['col'], h['row']] for h in hotspots[:4]])

            hotspot_scatter.set_offsets(spots)
        
        if avg_center[0] is not None:
            center_scatter.set_offsets([[avg_center[1], avg_center[0]]])
        
        # Update velocity arrow
        pred_arrow.set_UVC([pred_vel[0]], [pred_vel[1]])
        
        # Update cursor position
        cursor_state['x'] = np.clip(cursor_state['x'] + pred_vel[0] * 5, -100, 100)
        cursor_state['y'] = np.clip(cursor_state['y'] + pred_vel[1] * 5, -100, 100)
        cursor_state['history_x'].append(cursor_state['x'])
        cursor_state['history_y'].append(cursor_state['y'])
        
        # Keep last 100 points
        if len(cursor_state['history_x']) > 100:
            cursor_state['history_x'] = cursor_state['history_x'][-100:]
            cursor_state['history_y'] = cursor_state['history_y'][-100:]
        
        cursor_trail.set_data(cursor_state['history_x'], cursor_state['history_y'])
        cursor_dot.set_offsets([[cursor_state['x'], cursor_state['y']]])
        
        # Update titles
        title0.set_text(f'Neural Activity (t={t:.2f}s)')
        title1.set_text(f'Vel: ({pred_vel[0]:.2f}, {pred_vel[1]:.2f})')
        title2.set_text(f'Cursor: ({cursor_state["x"]:.1f}, {cursor_state["y"]:.1f})')
        
        return []
    
    anim = FuncAnimation(fig, update, frames=time_steps, interval=1000/fps, blit=False)
    return fig, anim


if __name__ == "__main__":
    print("=" * 60)
    print("BrainStorm Track 2 - Hotspot Tracking Algorithm")
    print("=" * 60)
    
    # Load data
    neural_data, ground_truth = load_data("medium")
    
    # Initialize signal processor with full pipeline
    print("\nInitializing signal processing pipeline:")
    print("  1. Bandpass filter: 70-150 Hz (high-gamma band)")
    print("  2. Envelope extraction: Hilbert transform")
    print("  3. Temporal smoothing: 100ms sliding window")
    print("  4. Spatial smoothing: Gaussian σ=1.0")
    
    processor = SignalProcessor(
        fs=500,
        lowcut=70,
        highcut=150,
        ema_alpha=0.1,
        spatial_sigma=1.0,
        envelope_method='hilbert'
    )
    
    # Initialize tracker
    tracker = HotspotTracker(threshold_percentile=95, min_spot_size=3, smoothing_sigma=0)
    # Note: spatial smoothing is now done in SignalProcessor, so set smoothing_sigma=0 in tracker
    print("\nHotspotTracker initialized!")
    
    # Test single frame detection
    # print("\n[1/3] Testing hotspot detection at t=5s...")
    # fig1, hotspots, center, vel = visualize_hotspot_detection(
    #     neural_data, tracker, t_start=5.0, processor=processor
    # )
    # print(f"  Detected {len(hotspots)} hotspots")
    # if center[0] is not None:
    #     print(f"  Average center: row={center[0]:.1f}, col={center[1]:.1f}")
    # print(f"  Predicted velocity: vx={vel[0]:.3f}, vy={vel[1]:.3f}")
    # plt.show()
    
    # # Track over time
    # print("\n[2/3] Tracking hotspots over 30 seconds...")
    # tracking_results = track_hotspots_over_time(
    #     neural_data, ground_truth, tracker, t_start=0, t_end=30, processor=processor
    # )
    # print(f"  Tracked {len(tracking_results)} time points")
    
    # Plot comparison
    # fig2, corr_vx, corr_vy = plot_tracking_comparison(tracking_results)
    # print(f"\n  Correlation with ground truth:")
    # print(f"    Vx: {corr_vx:.3f}")
    # print(f"    Vy: {corr_vy:.3f}")
    # plt.show()
    
    # Animation
    print("\n[3/3] Creating animated visualization...")
    print("  Close the plot window when done viewing.")
    fig3, anim = animate_hotspot_tracking(
        neural_data, ground_truth, tracker, t_start=0, t_end=20, fps=10, processor=processor
    )
    plt.show()
    
    print("\nDone!")
