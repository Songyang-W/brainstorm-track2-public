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
import torch
import torch.nn as nn


# =============================================================================
# Neural Network Denoising Model (UNet Autoencoder)
# =============================================================================

class MedicalUNet(nn.Module):
    """
    U-Net style autoencoder for denoising neural activity grids.
    
    Takes a 32x32 neural activity grid and outputs a cleaned version
    with noise filtered out and hotspots enhanced.
    """
    def __init__(self):
        super(MedicalUNet, self).__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        self.e2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU())
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.upconv2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d2 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.e1(x)
        p1 = self.pool1(x1)
        x2 = self.e2(p1)
        p2 = self.pool2(x2)
        b = self.bottleneck(p2)
        d1 = self.upconv1(b)
        d1 = torch.cat((d1, x2), dim=1)
        d1 = self.d1(d1)
        d2 = self.upconv2(d1)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.d2(d2)
        return self.sigmoid(d2)


def load_denoising_model(model_path="scripts/compass_model.pth"):
    """
    Load the pre-trained UNet denoising model.
    
    Args:
        model_path: Path to the .pth model file
        
    Returns:
        (model, device) tuple, or (None, None) if loading fails
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalUNet().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"  ✓ Loaded denoising model from {model_path} (device: {device})")
        return model, device
    except Exception as e:
        print(f"  ⚠️ Could not load model: {e}")
        return None, device


def apply_denoising_model(grid, model, device):
    """
    Apply the UNet model to denoise/enhance a 32x32 grid.
    
    Args:
        grid: 32x32 numpy array of neural activity
        model: Loaded MedicalUNet model
        device: torch device
        
    Returns:
        32x32 numpy array of denoised/enhanced activity
    """
    if model is None:
        return grid
    
    # Normalize input (99.5th percentile clipping)
    p99 = np.percentile(grid, 99.5)
    input_norm = np.clip(grid, 0, p99) / (p99 + 1e-9)
    
    # Convert to tensor
    tensor = torch.FloatTensor(input_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output_tensor = model(tensor)
        output_grid = output_tensor.squeeze().cpu().numpy()
    
    return output_grid


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
    
    def __init__(self, threshold_percentile=85, min_spot_size=3, smoothing_sigma=1.5):
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
    
    def get_average_hotspot_center(self, hotspots, top_n=4):
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
    
    def track_and_predict(self, grid):
        """
        Main function: detect hotspots and predict cursor velocity.
        
        Args:
            grid: 32x32 numpy array of power values
            
        Returns:
            - hotspots: List of detected hotspots
            - avg_center: (row, col) average center of hotspots
            - velocity_prediction: (vx, vy) predicted cursor velocity (-1 to 1)
        """
        hotspots = self.detect_hotspots(grid)
        avg_row, avg_col = self.get_average_hotspot_center(hotspots)
        
        if avg_row is not None:
            # Convert grid position to velocity
            # Hotspot right of center -> positive vx (moving right)
            # Hotspot above center -> positive vy (moving up)
            center = 15.5  # Center of 0-31 grid
            vx_pred = (avg_col - center) / center  # Normalized -1 to 1
            vy_pred = -(avg_row - center) / center  # Negative because row 0 is top
            velocity_prediction = (vx_pred, vy_pred)
        else:
            velocity_prediction = (0, 0)
        
        self.previous_hotspots = hotspots
        return hotspots, (avg_row, avg_col), velocity_prediction


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
            spots = np.array([[h['center_col'], h['center_row']] for h in hotspots[:4]])
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


def animate_with_denoising(df, ground_truth, tracker, t_start=0, t_end=20, 
                           window_size=0.5, fps=10, processor=None, 
                           model_path="scripts/compass_model.pth"):
    """
    Animated visualization showing AI-denoised neural activity.
    
    Args:
        df: Neural data DataFrame
        ground_truth: Ground truth DataFrame
        tracker: HotspotTracker instance
        t_start: Start time in seconds
        t_end: End time in seconds
        window_size: Processing window size in seconds
        fps: Frames per second for animation
        processor: SignalProcessor instance (creates default if None)
        model_path: Path to the UNet model weights
    
    Returns:
        (figure, FuncAnimation) tuple
    """
    if processor is None:
        processor = SignalProcessor()
    
    # Load denoising model
    print("  Loading denoising model...")
    model, device = load_denoising_model(model_path)
        
    dt = 1 / fps
    time_steps = np.arange(t_start, t_end - window_size, dt)
    
    # Create figure with 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('AI-Denoised Neural Activity', fontsize=14, fontweight='bold')
    
    # Initialize first frame
    power = compute_power_grid(df, t_start, window_size, processor)
    denoised = apply_denoising_model(power, model, device) if model else power
    
    # Left: Denoised power grid
    im_denoised = axes[0].imshow(denoised, cmap='inferno', aspect='equal', vmin=0, vmax=1)
    plt.colorbar(im_denoised, ax=axes[0], label='Confidence')
    denoised_scatter = axes[0].scatter([], [], c='lime', s=150, marker='o', 
                                        linewidths=2, edgecolors='white')
    # Add center of mass marker
    com_scatter = axes[0].scatter([], [], c='cyan', s=200, marker='+', linewidths=3)
    title_denoised = axes[0].set_title('AI DENOISED')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    
    # Middle: Cursor trajectory
    axes[1].set_xlim(-100, 100)
    axes[1].set_ylim(-100, 100)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    title_cursor = axes[1].set_title('Predicted Cursor')
    
    cursor_state = {'x': 0, 'y': 0, 'history_x': [0], 'history_y': [0]}
    cursor_trail, = axes[1].plot([], [], 'b-', alpha=0.5, linewidth=1)
    cursor_dot = axes[1].scatter([0], [0], c='blue', s=100, zorder=5)
    
    # Right: Velocity comparison
    axes[2].set_xlim(-1.5, 1.5)
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('Vx')
    axes[2].set_ylabel('Vy')
    title_vel = axes[2].set_title('Velocity')
    
    pred_arrow = axes[2].quiver([0], [0], [0], [0], color='blue', scale=1, 
                                    scale_units='xy', angles='xy', width=0.02,
                                    label='Predicted')
    gt_arrow = axes[2].quiver([0], [0], [0], [0], color='red', scale=1, 
                                  scale_units='xy', angles='xy', width=0.02,
                                  alpha=0.5, label='Ground Truth')
    axes[2].legend(loc='upper right')
    
    def update(t):
        # Get raw processed grid
        power = compute_power_grid(df, t, window_size, processor)
        
        # Apply denoising
        denoised = apply_denoising_model(power, model, device) if model else power
        
        # Detect hotspots from denoised output (for visualization)
        denoised_hotspots = []
        if model is not None:
            from scipy.ndimage import label, center_of_mass
            mask = denoised > 0.2
            labeled, num = label(mask)
            for i in range(1, num + 1):
                region = labeled == i
                if np.sum(region) > 3:
                    com = center_of_mass(denoised, labeled, i)
                    intensity = np.mean(denoised[region])
                    denoised_hotspots.append({
                        'center_row': com[0],
                        'center_col': com[1],
                        'intensity': intensity
                    })
        
        # Update image
        im_denoised.set_array(denoised)
        
        # Update hotspot markers
        if denoised_hotspots:
            spots = np.array([[h['center_col'], h['center_row']] for h in denoised_hotspots[:4]])
            denoised_scatter.set_offsets(spots)
        else:
            denoised_scatter.set_offsets(np.empty((0, 2)))
        
        # Calculate velocity using INTENSITY-WEIGHTED center of mass of entire grid
        # This is the key fix - use weighted average like the original algorithm
        total_intensity = denoised.sum()
        if total_intensity > 0.01:  # Avoid division by zero
            # Create coordinate grids
            rows, cols = np.mgrid[0:32, 0:32]
            
            # Weighted center of mass
            avg_row = (rows * denoised).sum() / total_intensity
            avg_col = (cols * denoised).sum() / total_intensity
            
            # Update center of mass marker on plot
            com_scatter.set_offsets([[avg_col, avg_row]])
            
            # Convert to velocity (center is 15.5, 15.5)
            # Offset from center determines velocity direction
            pred_vx = (avg_col - 15.5) / 15.5
            pred_vy = -(avg_row - 15.5) / 15.5
        else:
            com_scatter.set_offsets(np.empty((0, 2)))
            pred_vx, pred_vy = 0, 0
        
        # Update velocity arrows
        pred_arrow.set_UVC([pred_vx], [pred_vy])
        
        # Get ground truth
        gt_idx = np.abs(ground_truth['time_s'].values - t).argmin()
        true_vx = ground_truth.iloc[gt_idx]['vx']
        true_vy = ground_truth.iloc[gt_idx]['vy']
        true_vx_norm = np.clip(true_vx / 50, -1, 1)
        true_vy_norm = np.clip(true_vy / 50, -1, 1)
        gt_arrow.set_UVC([true_vx_norm], [true_vy_norm])
        
        # Update cursor position
        cursor_state['x'] = np.clip(cursor_state['x'] + pred_vx * 5, -100, 100)
        cursor_state['y'] = np.clip(cursor_state['y'] + pred_vy * 5, -100, 100)
        cursor_state['history_x'].append(cursor_state['x'])
        cursor_state['history_y'].append(cursor_state['y'])
        
        if len(cursor_state['history_x']) > 100:
            cursor_state['history_x'] = cursor_state['history_x'][-100:]
            cursor_state['history_y'] = cursor_state['history_y'][-100:]
        
        cursor_trail.set_data(cursor_state['history_x'], cursor_state['history_y'])
        cursor_dot.set_offsets([[cursor_state['x'], cursor_state['y']]])
        
        # Update titles
        title_denoised.set_text(f'AI DENOISED (t={t:.2f}s) - {len(denoised_hotspots)} targets')
        title_cursor.set_text(f'Cursor: ({cursor_state["x"]:.1f}, {cursor_state["y"]:.1f})')
        title_vel.set_text(f'Predicted: ({pred_vx:.2f}, {pred_vy:.2f})')
        
        return []
    
    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=time_steps, interval=1000/fps, blit=False)
    return fig, anim


if __name__ == "__main__":
    print("=" * 60)
    print("BrainStorm Track 2 - Hotspot Tracking with AI Denoising")
    print("=" * 60)
    
    # Load data
    neural_data, ground_truth = load_data("super_easy")
    
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
    tracker = HotspotTracker(threshold_percentile=85, min_spot_size=3, smoothing_sigma=0)
    print("\nHotspotTracker initialized!")
    
    # NEW: Animation with AI denoising comparison
    print("\n[1/2] Creating RAW vs AI-DENOISED comparison animation...")
    print("  This shows how the UNet autoencoder cleans up the neural signal.")
    print("  Left: Traditional DSP processing")
    print("  Right: AI-enhanced (cleaner hotspots)")
    print("\n  Close the plot window when done viewing.")
    
    fig1, anim1 = animate_with_denoising(
        neural_data, ground_truth, tracker, 
        t_start=0, t_end=30, fps=15, 
        processor=processor,
        model_path="scripts/compass_model.pth"
    )
    plt.show()
    
    # Track over time and show correlation
    print("\n[2/2] Tracking hotspots over 30 seconds...")
    tracking_results = track_hotspots_over_time(
        neural_data, ground_truth, tracker, t_start=0, t_end=30, processor=processor
    )
    print(f"  Tracked {len(tracking_results)} time points")
    
    # Plot comparison
    fig2, corr_vx, corr_vy = plot_tracking_comparison(tracking_results)
    print(f"\n  Correlation with ground truth:")
    print(f"    Vx: {corr_vx:.3f}")
    print(f"    Vy: {corr_vy:.3f}")
    plt.show()
    
    print("\nDone!")
