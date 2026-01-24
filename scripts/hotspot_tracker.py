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
from scipy.signal import butter, hilbert, iirnotch, sosfiltfilt, tf2sos
from scipy.optimize import linear_sum_assignment
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
                 envelope_method='hilbert',
                 grid_transform="rot180",
                 notch_freqs=(120.0,),
                 notch_q=30.0,
                 common_average_reference=True,
                 normalize="robust_z",
                 clip_negative=True):
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
            grid_transform: 'none' or 'rot180'. Empirically, the provided parquet
                channel ordering maps to a grid rotated 180 degrees relative to
                the ground_truth coordinates in this repo.
            notch_freqs: Frequencies to notch filter (Hz) before bandpass
            notch_q: Q factor for notch filters (higher = narrower)
            common_average_reference: Subtract mean across channels per sample
            normalize: 'none' or 'robust_z' (median/MAD normalization per frame)
            clip_negative: If True, clamp normalized grid to >= 0 (good for heatmaps)
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.ema_alpha = ema_alpha
        self.spatial_sigma = spatial_sigma
        self.envelope_method = envelope_method
        self.grid_transform = grid_transform
        self.notch_freqs = tuple(notch_freqs) if notch_freqs else tuple()
        self.notch_q = notch_q
        self.common_average_reference = common_average_reference
        self.normalize = normalize
        self.clip_negative = clip_negative
        
        # Pre-compute filter coefficients
        self.sos = butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
        self.notch_sos = []
        for f0 in self.notch_freqs:
            b, a = iirnotch(w0=f0, Q=self.notch_q, fs=self.fs)
            self.notch_sos.append(tf2sos(b, a))
        
        # State for real-time EMA smoothing
        self.ema_state = None
        
    def bandpass_filter(self, data):
        """
        Apply notch filter(s) then bandpass filter to extract frequency band of interest.
        
        Args:
            data: numpy array (n_samples, n_channels) or (n_samples,)
            
        Returns:
            Filtered data with same shape
        """
        x = data
        for ns in self.notch_sos:
            x = sosfiltfilt(ns, x, axis=0)
        return sosfiltfilt(self.sos, x, axis=0)
    
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
        grid = np.array(channel_data).reshape(32, 32)
        if self.grid_transform == "rot180":
            # Match ground_truth coordinate system for this dataset.
            grid = np.flipud(np.fliplr(grid))
        return grid
    
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
            
        # Optional CAR to remove common-mode noise.
        if self.common_average_reference and data.ndim == 2 and data.shape[1] > 1:
            data = data - data.mean(axis=1, keepdims=True)

        # Step 1: Notch + bandpass
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

        # Step 6: Per-frame robust normalization for stability across noise/drift.
        if self.normalize == "robust_z":
            med = float(np.median(grid))
            mad = float(np.median(np.abs(grid - med))) + 1e-6
            grid = (grid - med) / mad
            if self.clip_negative:
                grid = np.maximum(grid, 0.0)
        
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


class DirectionalAccumulator:
    """
    Accumulate direction-specific activity maps to recover all 4 hotspot centers.

    Maintains a persistent presence map per direction (vx+/vx-/vy+/vy-) using
    velocity gating. The presence map is updated from a binary activation mask,
    so the algorithm remembers historically active cluster areas even when they
    are temporarily silent.
    """

    def __init__(
        self,
        speed_thresh=0.1,
        axis_dominance=1.2,
        presence_decay=0.98,
        presence_gain=0.2,
        mask_z=2.5,
        mask_percentile=90.0,
        presence_sigma=1.0,
        presence_percentile=70.0,
        presence_min_threshold=0.05,
        min_component_size=8,
    ):
        """
        Initialize the directional accumulator.

        Args:
            speed_thresh: Minimum |velocity| to update a direction map
            axis_dominance: Ratio needed to treat vx or vy as dominant
            presence_decay: Per-step decay for persistence (0-1)
            presence_gain: Gain for new masks (higher = faster memory update)
            mask_z: Robust z-score threshold for activation mask
            mask_percentile: Fallback percentile threshold for activation mask
            presence_sigma: Gaussian smoothing for presence map
            presence_percentile: Percentile threshold for cluster extraction
            presence_min_threshold: Minimum threshold for presence mask
            min_component_size: Minimum pixels to use for a cluster
        """
        self.speed_thresh = speed_thresh
        self.axis_dominance = axis_dominance
        self.presence_decay = presence_decay
        self.presence_gain = presence_gain
        self.mask_z = mask_z
        self.mask_percentile = mask_percentile
        self.presence_sigma = presence_sigma
        self.presence_percentile = presence_percentile
        self.presence_min_threshold = presence_min_threshold
        self.min_component_size = min_component_size
        self.maps = {
            "vx_pos": None,
            "vx_neg": None,
            "vy_pos": None,
            "vy_neg": None,
        }

    def reset(self):
        """Reset all accumulated maps."""
        for key in self.maps:
            self.maps[key] = None

    def _decay_all(self):
        for key, grid in self.maps.items():
            if grid is not None:
                self.maps[key] = grid * self.presence_decay

    def _robust_zscore(self, grid):
        median = np.median(grid)
        mad = np.median(np.abs(grid - median)) + 1e-6
        return (grid - median) / mad

    def _activation_mask(self, grid):
        z = self._robust_zscore(grid)
        mask = z >= self.mask_z
        if mask.sum() < self.min_component_size:
            threshold = np.percentile(z, self.mask_percentile)
            mask = z >= threshold
        if mask.sum() >= self.min_component_size * 4:
            mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
        return mask

    def _update_presence(self, key, mask):
        if self.maps[key] is None:
            self.maps[key] = mask.astype(float)
            return
        updated = self.maps[key] + self.presence_gain * mask.astype(float)
        updated = np.clip(updated, 0.0, 1.0)
        self.maps[key] = updated

    def update(self, grid, vx, vy):
        """Update directional maps based on current velocity."""
        self._decay_all()

        vx_val = 0.0 if vx is None else float(vx)
        vy_val = 0.0 if vy is None else float(vy)

        ax = abs(vx_val)
        ay = abs(vy_val)
        if ax < self.speed_thresh and ay < self.speed_thresh:
            return None

        mask = self._activation_mask(grid)

        if ax >= self.axis_dominance * ay and ax > self.speed_thresh:
            if vx_val > 0:
                self._update_presence("vx_pos", mask)
            else:
                self._update_presence("vx_neg", mask)

        if ay >= self.axis_dominance * ax and ay > self.speed_thresh:
            if vy_val > 0:
                self._update_presence("vy_pos", mask)
            else:
                self._update_presence("vy_neg", mask)

    def _weighted_centroid(self, weights):
        total = weights.sum()
        if total <= 0:
            return None
        rows, cols = np.indices(weights.shape)
        row = np.sum(rows * weights) / total
        col = np.sum(cols * weights) / total
        return row, col

    def _center_from_map(self, grid):
        if grid is None:
            return None
        smoothed = ndimage.gaussian_filter(grid, sigma=self.presence_sigma)
        max_val = float(np.max(smoothed))
        if max_val <= 0:
            return None
        threshold = np.percentile(smoothed, self.presence_percentile)
        threshold = max(threshold, self.presence_min_threshold)
        mask = smoothed >= threshold

        labeled, num = ndimage.label(mask)
        if num == 0:
            return self._weighted_centroid(smoothed)

        sums = ndimage.sum(smoothed, labeled, range(1, num + 1))
        best = int(np.argmax(sums)) + 1
        component = labeled == best
        if component.sum() < self.min_component_size:
            return self._weighted_centroid(smoothed)

        return self._weighted_centroid(smoothed * component)

    def get_centers(self, tracker):
        """Detect cluster centers from each directional map."""
        centers = {}
        for key, grid in self.maps.items():
            if grid is None:
                continue
            center = self._center_from_map(grid)
            if center is not None:
                centers[key] = center
        return centers

    def estimate_center(self, tracker):
        """
        Estimate the overall center from opposing directional centers.

        Returns:
            (row, col, centers_dict)
        """
        centers = self.get_centers(tracker)

        row = None
        col = None

        if "vy_pos" in centers and "vy_neg" in centers:
            row = (centers["vy_pos"][0] + centers["vy_neg"][0]) / 2.0
        elif centers:
            rows = [val[0] for val in centers.values()]
            row = sum(rows) / len(rows)

        if "vx_pos" in centers and "vx_neg" in centers:
            col = (centers["vx_pos"][1] + centers["vx_neg"][1]) / 2.0
        elif centers:
            cols = [val[1] for val in centers.values()]
            col = sum(cols) / len(cols)

        return row, col, centers


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


def _find_gt_column(ground_truth, candidates):
    for name in candidates:
        if name in ground_truth.columns:
            return name
    return None


def _get_gt_time_array(ground_truth):
    if "time_s" in ground_truth.columns:
        return ground_truth["time_s"].values
    return ground_truth.index.values


def _get_gt_center_columns(ground_truth):
    return {
        "vx_pos": (
            _find_gt_column(ground_truth, ["vx_pos_center_row", "vx_pos_row"]),
            _find_gt_column(ground_truth, ["vx_pos_center_col", "vx_pos_col"]),
        ),
        "vx_neg": (
            _find_gt_column(ground_truth, ["vx_neg_center_row", "vx_neg_row"]),
            _find_gt_column(ground_truth, ["vx_neg_center_col", "vx_neg_col"]),
        ),
        "vy_pos": (
            _find_gt_column(ground_truth, ["vy_pos_center_row", "vy_pos_row"]),
            _find_gt_column(ground_truth, ["vy_pos_center_col", "vy_pos_col"]),
        ),
        "vy_neg": (
            _find_gt_column(ground_truth, ["vy_neg_center_row", "vy_neg_row"]),
            _find_gt_column(ground_truth, ["vy_neg_center_col", "vy_neg_col"]),
        ),
    }


def compute_ground_truth_center(ground_truth, gt_idx):
    """
    Compute true center from ground truth region centers.

    Prefer click center if present. Otherwise average available region centers.

    Ground truth coordinates are sometimes already 0-indexed (this repo's
    super_easy has values hitting 0 and 32). We detect indexing and only shift
    if everything looks 1-indexed.
    """
    if "click_center_row" in ground_truth.columns and "click_center_col" in ground_truth.columns:
        row = float(ground_truth["click_center_row"].iloc[gt_idx])
        col = float(ground_truth["click_center_col"].iloc[gt_idx])
        return row, col, {"click": (row, col)}

    centers = {}
    columns = _get_gt_center_columns(ground_truth)

    # Detect whether centers are 1-indexed. If any center reaches 0, treat as 0-indexed.
    all_vals = []
    for row_col, col_col in columns.values():
        if row_col:
            all_vals.append(float(ground_truth[row_col].iloc[gt_idx]))
        if col_col:
            all_vals.append(float(ground_truth[col_col].iloc[gt_idx]))
    is_one_indexed = bool(all_vals) and min(all_vals) >= 1.0
    shift = 1.0 if is_one_indexed else 0.0

    for key, (row_col, col_col) in columns.items():
        if row_col and col_col:
            row_val = float(ground_truth[row_col].iloc[gt_idx]) - shift
            col_val = float(ground_truth[col_col].iloc[gt_idx]) - shift
            centers[key] = (row_val, col_val)

    if not centers:
        return None, None, centers

    rows = [val[0] for val in centers.values()]
    cols = [val[1] for val in centers.values()]
    center_row = sum(rows) / len(rows)
    center_col = sum(cols) / len(cols)

    return center_row, center_col, centers


def extract_peak_observations(
    grid,
    n_peaks=2,
    smooth_sigma=1.0,
    suppress_radius=6,
    com_radius=2,
    min_abs=0.0,
):
    """
    Robustly extract the top-N peak locations from a 32x32 activity grid.

    This avoids connected-component merging and produces stable observations for
    tracking even when only 1-2 spots are lit in a frame.
    min_abs filters out weak peaks after normalization.
    """
    g = ndimage.gaussian_filter(grid, sigma=smooth_sigma)
    work = g.copy()

    peaks = []
    for _ in range(n_peaks):
        idx = np.argmax(work)
        peak_val = float(work.flat[idx])
        if not np.isfinite(peak_val) or peak_val <= 0 or peak_val < float(min_abs):
            break

        r0, c0 = np.unravel_index(idx, work.shape)

        r1 = max(0, r0 - com_radius)
        r2 = min(work.shape[0], r0 + com_radius + 1)
        c1 = max(0, c0 - com_radius)
        c2 = min(work.shape[1], c0 + com_radius + 1)
        patch = g[r1:r2, c1:c2]
        pr, pc = np.indices(patch.shape)
        total = float(patch.sum())
        if total > 0:
            rr = (pr * patch).sum() / total + r1
            cc = (pc * patch).sum() / total + c1
        else:
            rr, cc = float(r0), float(c0)

        peaks.append((float(rr), float(cc), peak_val))

        # Suppress a neighborhood so we don't pick the same spot again.
        sr1 = max(0, r0 - suppress_radius)
        sr2 = min(work.shape[0], r0 + suppress_radius + 1)
        sc1 = max(0, c0 - suppress_radius)
        sc2 = min(work.shape[1], c0 + suppress_radius + 1)
        work[sr1:sr2, sc1:sc2] = float("-inf")

    return peaks


class PersistentSpotClusterTracker:
    """
    Track a set of hotspot components across time with explicit memory.

    Key idea: estimate frame-to-frame drift as a rigid translation from the spots
    we *do* see, then apply that translation to spots we *don't* see so they
    remain in the correct (moving) coordinate frame.
    """

    def __init__(
        self,
        max_tracks=4,
        ema_alpha=0.25,
        max_match_dist=6.0,
        strength_gain=0.3,
        strength_decay=0.98,
        max_age=200,
    ):
        self.max_tracks = max_tracks
        self.ema_alpha = ema_alpha
        self.max_match_dist = max_match_dist
        self.strength_gain = strength_gain
        self.strength_decay = strength_decay
        self.max_age = max_age
        self._tracks = []  # list of dict(pos=np.array([r,c]), strength=float, age=int)

    def reset(self):
        self._tracks = []

    def _prune(self):
        kept = []
        for tr in self._tracks:
            if tr["age"] <= self.max_age and tr["strength"] > 1e-3:
                kept.append(tr)
        self._tracks = kept[: self.max_tracks]

    def update(self, observations):
        """
        Update tracker with a list of observed (row, col) points.

        Returns:
            (center_row, center_col, tracks_positions)
        """
        obs = [np.array([r, c], dtype=float) for (r, c, _v) in observations]

        # Age/decay all tracks (memory persists but fades if never reinforced).
        for tr in self._tracks:
            tr["age"] += 1
            tr["strength"] *= self.strength_decay

        if not self._tracks:
            for p in obs[: self.max_tracks]:
                self._tracks.append({"pos": p, "strength": 0.8, "age": 0})
            return self.center()

        if not obs:
            self._prune()
            return self.center()

        # Assignment: tracks <-> observations.
        T = len(self._tracks)
        O = len(obs)
        cost = np.zeros((T, O), dtype=float)
        for i, tr in enumerate(self._tracks):
            for j, p in enumerate(obs):
                cost[i, j] = np.linalg.norm(p - tr["pos"])

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        unmatched_obs = set(range(O))
        matched_tracks = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= self.max_match_dist:
                matches.append((i, j))
                unmatched_obs.discard(j)
                matched_tracks.add(i)

        # Estimate rigid translation drift from matched pairs.
        if matches:
            deltas = [obs[j] - self._tracks[i]["pos"] for (i, j) in matches]
            delta = np.mean(deltas, axis=0)
        else:
            delta = np.array([0.0, 0.0], dtype=float)

        # Apply drift prediction to all tracks (including unseen ones).
        if np.any(delta):
            for tr in self._tracks:
                tr["pos"] = tr["pos"] + delta

        # Correct matched tracks with EMA toward their observations.
        for i, j in matches:
            tr = self._tracks[i]
            tr["pos"] = (1 - self.ema_alpha) * tr["pos"] + self.ema_alpha * obs[j]
            tr["strength"] = min(1.0, tr["strength"] + self.strength_gain)
            tr["age"] = 0

        # Spawn new tracks for unmatched observations.
        for j in list(unmatched_obs):
            if len(self._tracks) >= self.max_tracks:
                break
            self._tracks.append({"pos": obs[j], "strength": 0.6, "age": 0})

        self._prune()
        return self.center()

    def center(self):
        if not self._tracks:
            return None, None, []
        weights = np.array([tr["strength"] for tr in self._tracks], dtype=float)
        positions = np.stack([tr["pos"] for tr in self._tracks], axis=0)
        wsum = float(weights.sum())
        if wsum <= 0:
            center = positions.mean(axis=0)
        else:
            center = (positions * weights[:, None]).sum(axis=0) / wsum
        return float(center[0]), float(center[1]), [tuple(p.tolist()) for p in positions]

    def track_states(self) -> list[tuple[float, float, float, int]]:
        """Return tracked positions with strength and age for interpretability."""
        out = []
        for tr in self._tracks:
            r, c = float(tr["pos"][0]), float(tr["pos"][1])
            out.append((r, c, float(tr["strength"]), int(tr["age"])))
        return out


class InterpretableClusterTracker:
    """
    Interpretable tracker for noisy data using peak observations + persistent memory.

    Returns a stable center, a confidence score, and a long-memory map of anchors
    without assuming four tuned regions.
    """

    def __init__(
        self,
        grid_size: int = 32,
        peak_n: int = 6,
        peak_smooth_sigma: float = 1.0,
        peak_suppress_radius: int = 7,
        peak_com_radius: int = 2,
        peak_min_abs: float = 0.15,
        max_tracks: int = 10,
        ema_alpha: float = 0.4,
        max_match_dist: float = 7.5,
        strength_gain: float = 0.3,
        strength_decay: float = 0.98,
        max_age: int = 240,
        memory_half_life_s: float = 45.0,
        memory_sigma: float = 1.0,
        age_tau_updates: float = 120.0,
        conf_strength_weight: float = 0.7,
        conf_count_weight: float = 0.3,
    ):
        self.grid_size = int(grid_size)
        self.peak_n = int(peak_n)
        self.peak_smooth_sigma = float(peak_smooth_sigma)
        self.peak_suppress_radius = int(peak_suppress_radius)
        self.peak_com_radius = int(peak_com_radius)
        self.peak_min_abs = float(peak_min_abs)
        self.tracker = PersistentSpotClusterTracker(
            max_tracks=max_tracks,
            ema_alpha=ema_alpha,
            max_match_dist=max_match_dist,
            strength_gain=strength_gain,
            strength_decay=strength_decay,
            max_age=max_age,
        )
        self.memory_half_life_s = float(memory_half_life_s)
        self.memory_sigma = float(memory_sigma)
        self.age_tau_updates = float(age_tau_updates)
        self.conf_strength_weight = float(conf_strength_weight)
        self.conf_count_weight = float(conf_count_weight)
        self.memory_map: np.ndarray | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self.memory_map = None

    def _age_weight(self, age: int) -> float:
        return float(np.exp(-float(age) / max(1.0, self.age_tau_updates)))

    def _confidence(self, track_states: list[tuple[float, float, float, int]]) -> float:
        if not track_states:
            return 0.0
        weighted = [float(s) * self._age_weight(age) for (_r, _c, s, age) in track_states]
        weighted.sort(reverse=True)
        strength_conf = float(np.mean(weighted[:2]))
        count_conf = min(1.0, len(weighted) / 2.0)
        return float(np.clip(
            self.conf_strength_weight * strength_conf + self.conf_count_weight * count_conf,
            0.0,
            1.0,
        ))

    def _update_memory(self, track_states: list[tuple[float, float, float, int]], dt_s: float) -> np.ndarray:
        if self.memory_map is None:
            self.memory_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        decay = 0.5 ** (dt_s / max(1e-3, self.memory_half_life_s))
        self.memory_map *= decay

        if not track_states:
            return self.memory_map

        upd = np.zeros_like(self.memory_map)
        for r, c, s, age in track_states:
            rr = int(round(float(r)))
            cc = int(round(float(c)))
            if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                val = float(s) * self._age_weight(int(age))
                if val > upd[rr, cc]:
                    upd[rr, cc] = val

        if float(upd.max()) > 0:
            if self.memory_sigma > 0:
                upd = ndimage.gaussian_filter(upd, sigma=self.memory_sigma)
            mx = float(upd.max())
            if mx > 0:
                upd = upd / mx
            self.memory_map = np.maximum(self.memory_map, upd)
        return self.memory_map

    def update(
        self, grid: np.ndarray, dt_s: float = 0.05
    ) -> tuple[
        float | None,
        float | None,
        float,
        np.ndarray | None,
        list[tuple[float, float, float]],
        list[tuple[float, float, float, int]],
    ]:
        g = np.maximum(grid.astype(float), 0.0)
        scale = float(np.percentile(g, 99.5)) + 1e-6
        g_norm = np.clip(g / scale, 0.0, 1.0)
        observations = extract_peak_observations(
            g_norm,
            n_peaks=self.peak_n,
            smooth_sigma=self.peak_smooth_sigma,
            suppress_radius=self.peak_suppress_radius,
            com_radius=self.peak_com_radius,
            min_abs=self.peak_min_abs,
        )
        cr, cc, _tracks = self.tracker.update(observations)
        track_states = self.tracker.track_states()
        conf = self._confidence(track_states)
        memory = self._update_memory(track_states, dt_s)
        return cr, cc, conf, memory, observations, track_states


def track_persistent_cluster_center_over_time(
    df,
    ground_truth,
    tracker,
    cluster_tracker=None,
    t_start=0,
    t_end=30,
    window_size=0.2,
    step=0.1,
    processor=None,
):
    """
    Track cluster center from neural data alone (no vx/vy gating).

    This is designed for the "only 2 lit at a time" case: it tracks multiple spot
    components with memory and rigid drift propagation.
    """
    if processor is None:
        processor = SignalProcessor()
    if cluster_tracker is None:
        cluster_tracker = PersistentSpotClusterTracker(max_tracks=4)

    times = np.arange(t_start, t_end - window_size, step)
    gt_time = _get_gt_time_array(ground_truth)

    results = {
        "time": [],
        "est_row": [],
        "est_col": [],
        "true_row": [],
        "true_col": [],
        "n_obs": [],
        "n_tracks": [],
    }

    for t in times:
        power = compute_power_grid(df, t, window_size, processor)
        observations = extract_peak_observations(
            power, n_peaks=2, smooth_sigma=1.0, suppress_radius=6
        )
        est_row, est_col, tracks_pos = cluster_tracker.update(observations)

        gt_idx = np.abs(gt_time - t).argmin()
        true_row, true_col, _ = compute_ground_truth_center(ground_truth, gt_idx)

        results["time"].append(t)
        results["est_row"].append(est_row if est_row is not None else np.nan)
        results["est_col"].append(est_col if est_col is not None else np.nan)
        results["true_row"].append(true_row if true_row is not None else np.nan)
        results["true_col"].append(true_col if true_col is not None else np.nan)
        results["n_obs"].append(len(observations))
        results["n_tracks"].append(len(tracks_pos))

    return pd.DataFrame(results)


class CrossCenterTracker:
    """
    Estimate the center of a 4-spot cross from observations where only 1-2 arms
    are active at any moment.

    This uses a simple geometric model:
      horizontal arm: (center_row, center_col +/- dx)
      vertical arm:   (center_row +/- dy, center_col)

    When two spots are visible (one horizontal + one vertical), the center is
    solvable directly by trying the (sign_x, sign_y) combinations and choosing
    the most self-consistent solution.
    """

    def __init__(
        self,
        center_alpha=0.35,
        offset_alpha=0.1,
        min_offset=2.0,
        max_offset=20.0,
    ):
        self.center_alpha = center_alpha
        self.offset_alpha = offset_alpha
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.center = None  # np.array([row, col])
        self.dx = None
        self.dy = None

    def reset(self):
        self.center = None
        self.dx = None
        self.dy = None

    def _clip_offset(self, v):
        return float(np.clip(v, self.min_offset, self.max_offset))

    def _update_offsets(self, dx_obs, dy_obs):
        dx_obs = self._clip_offset(dx_obs)
        dy_obs = self._clip_offset(dy_obs)
        if self.dx is None:
            self.dx = dx_obs
        else:
            self.dx = (1 - self.offset_alpha) * self.dx + self.offset_alpha * dx_obs
        if self.dy is None:
            self.dy = dy_obs
        else:
            self.dy = (1 - self.offset_alpha) * self.dy + self.offset_alpha * dy_obs

    def _center_from_two(self, p_h, p_v):
        """
        Compute best center candidate assuming p_h is horizontal-arm spot and
        p_v is vertical-arm spot.
        """
        if self.dx is None or self.dy is None:
            return None, float("inf")

        best = None
        best_score = float("inf")
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                # horizontal: (cr, cc+sx*dx) -> cc = ch - sx*dx, cr = rh
                cr1 = float(p_h[0])
                cc1 = float(p_h[1] - sx * self.dx)
                # vertical: (cr+sy*dy, cc) -> cr = rv - sy*dy, cc = cv
                cr2 = float(p_v[0] - sy * self.dy)
                cc2 = float(p_v[1])
                score = abs(cr1 - cr2) + abs(cc1 - cc2)
                if score < best_score:
                    best_score = score
                    best = np.array([(cr1 + cr2) / 2.0, (cc1 + cc2) / 2.0], dtype=float)
        return best, best_score

    def update(self, observations):
        obs = [(r, c, v) for (r, c, v) in observations]
        obs.sort(key=lambda x: x[2], reverse=True)
        obs = obs[:2]

        if len(obs) == 2:
            p1 = np.array([obs[0][0], obs[0][1]], dtype=float)
            p2 = np.array([obs[1][0], obs[1][1]], dtype=float)

            dx_obs = abs(p1[1] - p2[1])
            dy_obs = abs(p1[0] - p2[0])
            self._update_offsets(dx_obs, dy_obs)

            # Try both assignments (p1 as horizontal vs p2 as horizontal).
            c12, s12 = self._center_from_two(p1, p2)
            c21, s21 = self._center_from_two(p2, p1)

            # Pick the candidate closest to previous center (if we have one),
            # otherwise choose the one with in-bounds coordinates.
            candidates = [(c, s) for (c, s) in ((c12, s12), (c21, s21)) if c is not None]
            if not candidates:
                return self.center_state()

            if self.center is None:
                # Prefer candidate inside the grid.
                def in_bounds(c):
                    return 0 <= c[0] <= 31 and 0 <= c[1] <= 31
                candidates.sort(key=lambda cs: (not in_bounds(cs[0]), cs[1]))
                self.center = candidates[0][0]
            else:
                # Prefer lower self-consistency score; break ties by proximity.
                candidates.sort(
                    key=lambda cs: (cs[1], float(np.linalg.norm(cs[0] - self.center)))
                )
                self.center = (1 - self.center_alpha) * self.center + self.center_alpha * candidates[0][0]

        elif len(obs) == 1 and self.center is not None and self.dx is not None and self.dy is not None:
            p = np.array([obs[0][0], obs[0][1]], dtype=float)
            # Snap to the nearest predicted arm and update center accordingly.
            cr, cc = float(self.center[0]), float(self.center[1])
            preds = {
                "vx_pos": np.array([cr, cc + self.dx]),
                "vx_neg": np.array([cr, cc - self.dx]),
                "vy_pos": np.array([cr - self.dy, cc]),
                "vy_neg": np.array([cr + self.dy, cc]),
            }
            best_key = min(preds.keys(), key=lambda k: float(np.linalg.norm(p - preds[k])))
            if best_key == "vx_pos":
                cand = np.array([p[0], p[1] - self.dx])
            elif best_key == "vx_neg":
                cand = np.array([p[0], p[1] + self.dx])
            elif best_key == "vy_pos":
                cand = np.array([p[0] + self.dy, p[1]])
            else:
                cand = np.array([p[0] - self.dy, p[1]])
            self.center = (1 - self.center_alpha) * self.center + self.center_alpha * cand

        return self.center_state()

    def center_state(self):
        if self.center is None:
            return None, None, None, None
        return float(self.center[0]), float(self.center[1]), self.dx, self.dy


def _bilinear_sample(grid, r, c, fill_value=float("-inf")):
    """Bilinear sample from grid at floating (r, c)."""
    h, w = grid.shape
    if r < 0 or c < 0 or r > h - 1 or c > w - 1:
        return fill_value
    r0 = int(np.floor(r))
    c0 = int(np.floor(c))
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)
    dr = r - r0
    dc = c - c0
    v00 = grid[r0, c0]
    v01 = grid[r0, c1]
    v10 = grid[r1, c0]
    v11 = grid[r1, c1]
    return (
        (1 - dr) * (1 - dc) * v00
        + (1 - dr) * dc * v01
        + dr * (1 - dc) * v10
        + dr * dc * v11
    )


def _gaussian_patch_energy(grid, r, c, radius=3, sigma=1.25, fill_value=0.0):
    """
    Sum grid energy in a small Gaussian-weighted patch centered at (r, c).

    This is much more robust than point sampling when each region is a cluster
    of scattered spots (medium/hard).
    """
    h, w = grid.shape
    rr = int(round(r))
    cc = int(round(c))
    if rr + radius < 0 or rr - radius >= h or cc + radius < 0 or cc - radius >= w:
        return float(fill_value)
    r1 = rr - radius
    c1 = cc - radius
    size = 2 * radius + 1

    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-9)

    patch = np.full((size, size), fill_value, dtype=float)
    pr1 = max(0, r1)
    pr2 = max(0, min(h, rr + radius + 1))
    pc1 = max(0, c1)
    pc2 = max(0, min(w, cc + radius + 1))
    if pr2 <= pr1 or pc2 <= pc1:
        return float(fill_value)
    patch[(pr1 - r1) : (pr2 - r1), (pc1 - c1) : (pc2 - c1)] = grid[pr1:pr2, pc1:pc2]
    return float(np.sum(patch * kernel))


class CrossMatchedFilterTracker:
    """
    Robust cross-center tracker that does NOT rely on 'remembering' off-spots.

    Once it has learned dx/dy (arm offsets), it estimates the center each frame by
    searching for the center whose predicted arm locations best match the current
    activity (matched filter / template score).
    """

    def __init__(
        self,
        search_radius=6,
        center_alpha=0.6,
        fixed_dx=6.0,
        fixed_dy=6.0,
        patch_radius=3,
        patch_sigma=1.25,
        global_search=True,
    ):
        self.search_radius = search_radius
        self.center_alpha = center_alpha
        self.dx = fixed_dx
        self.dy = fixed_dy
        self.patch_radius = patch_radius
        self.patch_sigma = patch_sigma
        self.global_search = global_search
        self.center = None  # np.array([row, col])
        self.last_score = None
        self.last_confidence = None
        # Precompute Gaussian kernel for patch scoring.
        r = self.patch_radius
        y, x = np.mgrid[-r : r + 1, -r : r + 1]
        kernel = np.exp(-(x * x + y * y) / (2 * self.patch_sigma * self.patch_sigma))
        self._kernel = kernel / (kernel.sum() + 1e-9)

    def reset(self):
        self.center = None
        self.last_score = None
        self.last_confidence = None

    def _template_score(self, grid, cr, cc):
        # Best of the 4 sign combinations (vx sign, vy sign).
        best = float("-inf")
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                s = self._patch_energy(grid, cr, cc + sx * self.dx) + self._patch_energy(
                    grid, cr + sy * self.dy, cc
                )
                if s > best:
                    best = s
        return best

    def _patch_energy(self, grid, r, c):
        radius = self.patch_radius
        h, w = grid.shape
        rr = int(round(r))
        cc = int(round(c))
        if rr + radius < 0 or rr - radius >= h or cc + radius < 0 or cc - radius >= w:
            return 0.0
        r1 = rr - radius
        c1 = cc - radius
        size = 2 * radius + 1
        patch = np.zeros((size, size), dtype=float)
        pr1 = max(0, r1)
        pr2 = min(h, rr + radius + 1)
        pc1 = max(0, c1)
        pc2 = min(w, cc + radius + 1)
        patch[(pr1 - r1) : (pr2 - r1), (pc1 - c1) : (pc2 - c1)] = grid[pr1:pr2, pc1:pc2]
        return float(np.sum(patch * self._kernel))

    def update(self, grid):
        if self.dx is None or self.dy is None:
            return None, None, None, None

        if self.center is None:
            self.center = np.array([15.5, 15.5], dtype=float)

        r0 = int(round(float(self.center[0])))
        c0 = int(round(float(self.center[1])))
        best_score = float("-inf")
        best_rc = np.array([r0, c0], dtype=float)
        scores = []

        if self.global_search:
            r_range = range(0, 32)
            c_range = range(0, 32)
        else:
            r_range = range(r0 - self.search_radius, r0 + self.search_radius + 1)
            c_range = range(c0 - self.search_radius, c0 + self.search_radius + 1)

        for r in r_range:
            for c in c_range:
                s = self._template_score(grid, float(r), float(c))
                scores.append(s)
                if s > best_score:
                    best_score = s
                    best_rc = np.array([float(r), float(c)], dtype=float)

        self.center = (1 - self.center_alpha) * self.center + self.center_alpha * best_rc
        self.last_score = float(best_score)
        if scores:
            med = float(np.median(scores))
            mad = float(np.median(np.abs(np.array(scores) - med))) + 1e-6
            z = (float(best_score) - med) / mad
            # Squash to [0,1] for UI; 0~no separation, 1~very confident.
            self.last_confidence = float(1.0 / (1.0 + np.exp(-0.6 * (z - 2.0))))
        else:
            self.last_confidence = None
        return float(self.center[0]), float(self.center[1]), self.dx, self.dy


class CenterAlignedPersistence:
    """
    Maintain a long-memory map in a center-aligned frame.

    This is the simplest way to 'remember' all 4 arms: align each frame by the
    estimated cross center, then take a leaky max (or EMA) over time.
    """

    def __init__(self, ref_center=(15.5, 15.5), decay=0.995, mode="leaky_max"):
        self.ref_center = np.array(ref_center, dtype=float)
        self.decay = decay
        self.mode = mode
        self.state = None

    def reset(self):
        self.state = None

    def update(self, grid, center_rc):
        if center_rc is None or center_rc[0] is None or center_rc[1] is None:
            return self.state
        center = np.array([center_rc[0], center_rc[1]], dtype=float)
        shift = self.ref_center - center
        aligned = ndimage.shift(
            grid,
            shift=(float(shift[0]), float(shift[1])),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        if self.state is None:
            self.state = aligned
        else:
            if self.mode == "ema":
                alpha = 1.0 - self.decay
                self.state = (1 - alpha) * self.state + alpha * aligned
            else:
                self.state = np.maximum(self.state * self.decay, aligned)
        return self.state


def _shift2d_int(arr: np.ndarray, dr: int, dc: int, fill: float = 0.0) -> np.ndarray:
    """Integer shift with zero padding (no wrap). dr/dc are in array coordinates."""
    out = np.full_like(arr, fill, dtype=float)
    h, w = arr.shape

    r_src1 = max(0, -dr)
    r_src2 = min(h, h - dr)  # exclusive
    r_dst1 = max(0, dr)
    r_dst2 = r_dst1 + (r_src2 - r_src1)

    c_src1 = max(0, -dc)
    c_src2 = min(w, w - dc)
    c_dst1 = max(0, dc)
    c_dst2 = c_dst1 + (c_src2 - c_src1)

    if r_src2 <= r_src1 or c_src2 <= c_src1:
        return out

    out[r_dst1:r_dst2, c_dst1:c_dst2] = arr[r_src1:r_src2, c_src1:c_src2]
    return out


class ClusterTemplateTracker:
    """
    Track an arbitrary cluster with explicit memory, without assuming 4 fixed arms.

    Maintains a template ("memory map") in a centered coordinate frame. Each frame:
    - find the integer translation that best aligns the template to the current grid
      (within a search radius)
    - shift current grid into the template frame and update the template with a leaky-max
      (only using the brightest part of the current frame to avoid accumulating noise)

    The reported center is the stable centroid of the template (in live grid coords),
    so it stays consistent even when subsets of the cluster turn off.
    """

    def __init__(
        self,
        grid_size: int = 32,
        search_radius: int = 10,
        shift_alpha: float = 0.35,
        # "template" is used primarily for alignment; keep a moderate half-life.
        template_half_life_s: float = 20.0,
        # "presence" is the long-memory map used for stable center/spots.
        presence_half_life_s: float = 60.0,
        presence_gain: float = 0.10,
        update_percentile: float = 92.0,
        centroid_power: float = 1.2,
        min_update_pixels: int = 16,
        max_update_pixels: int = 64,
        roi_radius: float | None = 14.0,
        peak_suppress_radius: int = 5,
        frame_n_peaks: int = 12,
        peak_min_rel: float = 0.55,
        peak_abs_min: float = 0.18,
        update_blob_sigma: float = 1.2,
        update_min_rel: float = 0.55,
        update_dog_small_sigma: float = 0.6,
        update_dog_large_sigma: float = 1.6,
        update_dog_large_scale: float = 0.65,
        update_max_filter_size: int = 5,
        exclude_center_radius: float = 2.2,
        center_percentile: float = 92.0,
        center_min_rel: float = 0.55,
        min_update_conf: float = 0.45,
        presence_floor: float = 0.02,
        anchor_max: int = 8,
        anchor_percentile: float = 99.0,
        anchor_min_rel: float = 0.85,
        anchor_min_size: int = 14,
        anchor_keep_rel: float = 0.82,
        anchor_prefer_quadrants: bool = True,
        anchor_quadrant_min_rel: float = 0.55,
    ):
        self.grid_size = grid_size
        self.mid = (grid_size - 1) / 2.0
        self.search_radius = search_radius
        self.shift_alpha = shift_alpha
        self.template_half_life_s = float(template_half_life_s)
        self.presence_half_life_s = float(presence_half_life_s)
        self.presence_gain = float(presence_gain)
        self.update_percentile = update_percentile
        self.centroid_power = centroid_power
        self.min_update_pixels = min_update_pixels
        self.max_update_pixels = int(max_update_pixels)
        self.roi_radius = roi_radius
        self.peak_suppress_radius = int(peak_suppress_radius)
        self.frame_n_peaks = int(frame_n_peaks)
        self.peak_min_rel = float(peak_min_rel)
        self.peak_abs_min = float(peak_abs_min)
        self.update_blob_sigma = float(update_blob_sigma)
        self.update_min_rel = float(update_min_rel)
        self.update_dog_small_sigma = float(update_dog_small_sigma)
        self.update_dog_large_sigma = float(update_dog_large_sigma)
        self.update_dog_large_scale = float(update_dog_large_scale)
        self.update_max_filter_size = int(update_max_filter_size)
        self.exclude_center_radius = float(exclude_center_radius)
        self.center_percentile = float(center_percentile)
        self.center_min_rel = float(center_min_rel)
        self.min_update_conf = float(min_update_conf)
        self.presence_floor = float(presence_floor)
        self.anchor_max = int(anchor_max)
        self.anchor_percentile = float(anchor_percentile)
        self.anchor_min_rel = float(anchor_min_rel)
        self.anchor_min_size = int(anchor_min_size)
        self.anchor_keep_rel = float(anchor_keep_rel)
        self.anchor_prefer_quadrants = bool(anchor_prefer_quadrants)
        self.anchor_quadrant_min_rel = float(anchor_quadrant_min_rel)

        self.template: np.ndarray | None = None
        self.presence: np.ndarray | None = None
        self.shift = np.array([0.0, 0.0], dtype=float)  # (dr, dc) relative to mid
        self.last_confidence: float | None = None

    def reset(self):
        self.template = None
        self.presence = None
        self.shift[:] = 0.0
        self.last_confidence = None

    def _decay_from_half_life(self, half_life_s: float, dt_s: float) -> float:
        # Half-life decay: value *= 0.5 ** (dt / half_life)
        hl = max(1e-3, float(half_life_s))
        dt = max(0.0, float(dt_s))
        return float(0.5 ** (dt / hl))

    def _normalize_frame(self, grid: np.ndarray) -> np.ndarray:
        g = np.maximum(grid.astype(float), 0.0)
        # Per-frame scale normalization makes alignment + memory robust across sessions
        # and across amplitude drift (esp. medium/hard).
        scale = float(np.percentile(g, 99.5)) + 1e-6
        g = np.clip(g / scale, 0.0, 1.0)
        return g

    def _centroid(self, weights: np.ndarray) -> np.ndarray | None:
        w = np.maximum(weights, 0.0).astype(float)
        if self.centroid_power != 1.0:
            w = w ** self.centroid_power
        total = float(w.sum())
        if total <= 0:
            return None
        rows, cols = np.indices(w.shape)
        r = float((rows * w).sum() / total)
        c = float((cols * w).sum() / total)
        return np.array([r, c], dtype=float)

    def _score_shift(self, grid: np.ndarray, dr: int, dc: int) -> float:
        # Score = normalized dot product (cosine similarity) between grid and shifted template.
        assert self.template is not None
        t_shift = _shift2d_int(self.template, dr, dc, fill=0.0)
        num = float(np.sum(grid * t_shift))
        den = float(np.sqrt(np.sum(grid * grid) * np.sum(t_shift * t_shift))) + 1e-9
        return num / den

    def _best_shift(self, grid: np.ndarray) -> tuple[np.ndarray, float]:
        # Search around previous shift (integerized) to find best translation.
        base = np.round(self.shift).astype(int)
        scores = []
        best_score = float("-inf")
        best = base.copy()
        for dr in range(base[0] - self.search_radius, base[0] + self.search_radius + 1):
            for dc in range(base[1] - self.search_radius, base[1] + self.search_radius + 1):
                s = self._score_shift(grid, int(dr), int(dc))
                scores.append(s)
                if s > best_score:
                    best_score = s
                    best[:] = (dr, dc)

        med = float(np.median(scores))
        mad = float(np.median(np.abs(np.array(scores) - med))) + 1e-6
        z = (best_score - med) / mad
        # Squash to [0,1]
        conf = float(1.0 / (1.0 + np.exp(-0.7 * (z - 2.0))))
        self.last_confidence = conf
        return best.astype(float), best_score

    def update(
        self, grid: np.ndarray, dt_s: float = 1.0
    ) -> tuple[float | None, float | None, np.ndarray | None]:
        """
        Args:
            grid: 32x32 non-negative activity map (already filtered + normalized)
            dt_s: Time delta since last update (seconds). Used to turn half-life
                parameters into a consistent decay regardless of update rate.

        Returns:
            (center_row, center_col, memory_map)
        """
        g = self._normalize_frame(grid)

        if self.template is None:
            # Initialize memory in a centered frame (no shift). We keep a sparse,
            # long-lived presence map and derive the match template from it.
            # Presence starts as a conservative mask of the first frame.
            thr0 = float(np.percentile(g, self.update_percentile))
            mask0 = g >= thr0
            if int(mask0.sum()) < self.min_update_pixels:
                flat = g.ravel()
                order = np.argsort(flat)[::-1]
                k = min(self.min_update_pixels, order.size)
                mask0 = np.zeros_like(flat, dtype=bool)
                mask0[order[:k]] = True
                mask0 = mask0.reshape(g.shape)
            self.presence = mask0.astype(float)
            self.template = ndimage.gaussian_filter(self.presence, sigma=1.0)

            centroid = self._centroid(self.presence)
            if centroid is not None:
                self.shift[:] = np.array([centroid[0] - self.mid, centroid[1] - self.mid])
            return float(self.mid + self.shift[0]), float(self.mid + self.shift[1]), self.presence

        # Derive the match template from long-memory presence. This makes alignment
        # robust even when only a subset of cluster parts are active in the current frame.
        assert self.presence is not None
        self.template = ndimage.gaussian_filter(self.presence, sigma=1.0)

        best_shift, _best_score = self._best_shift(g)
        # If alignment is uncertain, be conservative with shift updates to avoid drift.
        conf = float(self.last_confidence or 0.0)
        shift_alpha = self.shift_alpha if conf >= self.min_update_conf else (self.shift_alpha * 0.15)
        self.shift = (1.0 - shift_alpha) * self.shift + shift_alpha * best_shift

        # Align current frame into template coordinates by shifting by -best_shift.
        aligned = _shift2d_int(g, int(-round(best_shift[0])), int(-round(best_shift[1])), fill=0.0)

        # Anchor/presence update (sparse): DoG local maxima (splits close blobs and avoids
        # writing midpoint peaks into memory when only 2-of-4 are active).
        a_small = ndimage.gaussian_filter(aligned, sigma=self.update_dog_small_sigma)
        a_large = ndimage.gaussian_filter(aligned, sigma=self.update_dog_large_sigma)
        dog = a_small - self.update_dog_large_scale * a_large
        dog = np.maximum(dog, 0.0)

        mx_d = float(np.max(dog))
        thr = float(np.percentile(dog, self.update_percentile))
        if mx_d > 0:
            thr = max(thr, self.update_min_rel * mx_d, self.peak_abs_min)
        maxima = dog == ndimage.maximum_filter(dog, size=self.update_max_filter_size)
        mask = maxima & (dog >= thr)

        if self.roi_radius is not None:
            rr, cc = np.indices(aligned.shape)
            roi = (rr - self.mid) ** 2 + (cc - self.mid) ** 2 <= float(self.roi_radius) ** 2
            mask = mask & roi

        upd_sparse = np.zeros_like(aligned, dtype=float)
        if mask.any():
            coords = np.argwhere(mask)
            # Sort maxima by DoG score.
            scores = dog[mask]
            order = np.argsort(scores)[::-1]
            taken: list[tuple[int, int]] = []
            for idx in order:
                r, c = map(int, coords[idx])
                if len(taken) >= self.frame_n_peaks:
                    break
                # Do not write the cluster center itself into memory; we want memory to
                # represent the parts of the cluster, not the midpoint between parts.
                if (r - self.mid) ** 2 + (c - self.mid) ** 2 < self.exclude_center_radius ** 2:
                    continue
                ok = True
                for (tr, tc) in taken:
                    if (r - tr) ** 2 + (c - tc) ** 2 < float(self.peak_suppress_radius) ** 2:
                        ok = False
                        break
                if not ok:
                    continue
                taken.append((r, c))
                upd_sparse[r, c] = max(upd_sparse[r, c], float(aligned[r, c]))

        if self.update_blob_sigma > 0 and float(upd_sparse.max()) > 0:
            upd_sparse = ndimage.gaussian_filter(upd_sparse, sigma=self.update_blob_sigma)

        # Fallback if peak-based update produced nothing (very weak / flat frame).
        if float(upd_sparse.max()) <= 0:
            thr = float(np.percentile(aligned, self.update_percentile))
            mask = aligned >= thr
            if int(mask.sum()) < self.min_update_pixels:
                flat = aligned.ravel()
                order = np.argsort(flat)[::-1]
                k = min(self.min_update_pixels, order.size)
                mask = np.zeros_like(flat, dtype=bool)
                mask[order[:k]] = True
                mask = mask.reshape(aligned.shape)
            if self.roi_radius is not None:
                rr, cc = np.indices(aligned.shape)
                roi = (rr - self.mid) ** 2 + (cc - self.mid) ** 2 <= float(self.roi_radius) ** 2
                mask = mask & roi
            if int(mask.sum()) > self.max_update_pixels:
                flat = (aligned * mask.astype(float)).ravel()
                order = np.argsort(flat)[::-1]
                k = min(self.max_update_pixels, order.size)
                keep = np.zeros_like(flat, dtype=bool)
                keep[order[:k]] = True
                mask = keep.reshape(aligned.shape) & mask
            upd_sparse = aligned * mask.astype(float)

        if self.roi_radius is not None:
            rr, cc = np.indices(upd_sparse.shape)
            roi = (rr - self.mid) ** 2 + (cc - self.mid) ** 2 <= float(self.roi_radius) ** 2
            upd_sparse = upd_sparse * roi.astype(float)

        # Time-based decay (half-life parameters are more intuitive than per-tick decay).
        pdec = self._decay_from_half_life(self.presence_half_life_s, dt_s)

        # Always decay; only incorporate new evidence when we're confident in alignment.
        self.presence = self.presence * pdec
        if conf >= self.min_update_conf:
            # Presence update: accumulate stable support over time (even if currently off).
            umx = float(upd_sparse.max())
            if umx > 0:
                upd = np.clip(upd_sparse / (umx + 1e-6), 0.0, 1.0)
            else:
                upd = np.zeros_like(upd_sparse, dtype=float)
            self.presence = np.clip(self.presence + self.presence_gain * upd, 0.0, 1.0)

        if self.presence_floor > 0:
            self.presence[self.presence < self.presence_floor] = 0.0

        # Recenter template to keep its centroid near the grid midpoint; absorb the
        # recentering into the accumulated shift so live coordinates remain consistent.
        # Use the presence centroid for center stability.
        pres_sm = ndimage.gaussian_filter(self.presence, sigma=1.0)
        anchors = self._anchors_template(max_anchors=8)
        # Prefer center from anchors (stable parts), fall back to core presence centroid.
        if len(anchors) >= 2:
            ar = np.array([[a[0], a[1]] for a in anchors], dtype=float)
            centroid = ar.mean(axis=0)
        else:
            mx = float(np.max(pres_sm))
            if mx > 0:
                thr_c = float(np.percentile(pres_sm, self.center_percentile))
                thr_c = max(thr_c, self.center_min_rel * mx)
                mask_c = pres_sm >= thr_c
                centroid = self._centroid(pres_sm * mask_c.astype(float))
            else:
                centroid = None
        if centroid is None:
            return None, None, self.presence
        recenter = centroid - np.array([self.mid, self.mid], dtype=float)
        if abs(float(recenter[0])) > 0.6 or abs(float(recenter[1])) > 0.6:
            dr = int(round(float(recenter[0])))
            dc = int(round(float(recenter[1])))
            self.presence = _shift2d_int(self.presence, -dr, -dc, fill=0.0)
            self.shift += np.array([dr, dc], dtype=float)
            centroid = centroid - np.array([dr, dc], dtype=float)

        # Refresh the match template after any presence shift/recenter.
        self.template = ndimage.gaussian_filter(self.presence, sigma=1.0)

        # Stable center is the long-memory presence centroid + accumulated shift.
        center_rc = centroid + self.shift
        return float(center_rc[0]), float(center_rc[1]), self.presence

    def _anchors_template(self, max_anchors: int | None = None) -> list[tuple[float, float, float]]:
        """
        Extract stable "anchor" points from the long-memory presence map in template coords.

        Prefer connected components of the high-confidence core; fall back to local maxima.
        """
        if self.presence is None:
            return []
        p = ndimage.gaussian_filter(self.presence, sigma=1.0)
        mx = float(np.max(p))
        if mx <= 0:
            return []

        k = self.anchor_max if max_anchors is None else int(max_anchors)
        thr = max(float(np.percentile(p, self.anchor_percentile)), self.anchor_min_rel * mx)
        mask = p >= thr
        labeled, num = ndimage.label(mask)

        anchors: list[tuple[float, float, float]] = []
        if num > 0:
            sums = ndimage.sum(p, labeled, range(1, num + 1))
            order = np.argsort(np.array(sums))[::-1]
            for idx in order[: max(1, min(k, len(order)))]:
                lab = int(idx) + 1
                comp = labeled == lab
                if int(comp.sum()) < self.anchor_min_size:
                    continue
                w = p * comp.astype(float)
                centroid = self._centroid(w)
                if centroid is None:
                    continue
                if (float(centroid[0]) - self.mid) ** 2 + (float(centroid[1]) - self.mid) ** 2 < self.exclude_center_radius ** 2:
                    continue
                # Weight = component peak value (for UI sizing).
                peak = float(np.max(w))
                anchors.append((float(centroid[0]), float(centroid[1]), peak))

        # If components are insufficient, fill with local maxima (with suppression).
        if len(anchors) < min(4, k):
            peaks = extract_peak_observations(
                p,
                n_peaks=k,
                smooth_sigma=0.5,
                suppress_radius=self.peak_suppress_radius,
                com_radius=2,
            )
            for (r, c, v) in peaks:
                if len(anchors) >= k:
                    break
                if (float(r) - self.mid) ** 2 + (float(c) - self.mid) ** 2 < self.exclude_center_radius ** 2:
                    continue
                # Avoid duplicates near existing anchors.
                ok = True
                for (ar, ac, _av) in anchors:
                    if (r - ar) ** 2 + (c - ac) ** 2 < float(self.peak_suppress_radius) ** 2:
                        ok = False
                        break
                if not ok:
                    continue
                anchors.append((float(r), float(c), float(v)))

        anchors.sort(key=lambda x: float(x[2]), reverse=True)
        if not anchors:
            return []

        # Prefer 4 anchors (one per quadrant) when the structure supports it.
        # This matches the "super_easy" regime where the tuned region is 4 stable parts
        # that toggle on/off with direction. For denser clusters, we still keep memory,
        # but limit displayed anchors to the dominant representatives.
        vmax = float(anchors[0][2])
        if self.anchor_prefer_quadrants and k >= 4 and len(anchors) >= 4 and vmax > 0:
            quad_best: dict[int, tuple[float, float, float]] = {}
            for (r, c, v) in anchors:
                if float(v) < self.anchor_quadrant_min_rel * vmax:
                    continue
                dr = float(r - self.mid)
                dc = float(c - self.mid)
                if abs(dr) < 0.7 or abs(dc) < 0.7:
                    # Too close to an axis -> ambiguous quadrant.
                    continue
                q = (0 if dr < 0 else 2) + (0 if dc < 0 else 1)  # 0..3
                prev = quad_best.get(q)
                if prev is None or float(v) > float(prev[2]):
                    quad_best[q] = (float(r), float(c), float(v))
            if len(quad_best) == 4:
                quad = [quad_best[i] for i in range(4)]
                quad.sort(key=lambda x: float(x[2]), reverse=True)
                return quad[:4]

        kept = [a for a in anchors if float(a[2]) >= self.anchor_keep_rel * vmax]
        anchors = kept if kept else anchors[: min(2, len(anchors))]
        return anchors[:k]

    def top_spots_live(self, n_peaks: int = 8) -> list[tuple[float, float, float]]:
        """
        Return top peaks from the long-memory presence map, transformed into live
        grid coordinates. These are the "relevant bright spots" even if currently off.
        """
        peaks = self._anchors_template(max_anchors=n_peaks)
        out = []
        for (r, c, v) in peaks:
            rr = float(r + self.shift[0])
            cc = float(c + self.shift[1])
            # Discard offscreen peaks rather than clamping (clamping creates
            # artificial "edge peaks" when a spot is just outside the grid).
            if rr < 0.0 or cc < 0.0 or rr > self.grid_size - 1.0 or cc > self.grid_size - 1.0:
                continue
            out.append((rr, cc, float(v)))
        return out

    def top_spots_memory(self, n_peaks: int = 8) -> list[tuple[float, float, float]]:
        """Stable anchor points in the memory/template coordinate frame."""
        return self._anchors_template(max_anchors=n_peaks)


class DriftCompensatedClusterTracker:
    """
    Track a cluster center with explicit memory while the cluster *moves* over time.

    Key differences vs ClusterTemplateTracker:
    - estimates per-step translation (frame-to-frame) via small-radius correlation
      and integrates it to get global motion; this aligns better with ground_truth
      movement in the provided datasets.
    - maintains a long-lived presence map in a registered (moving) frame so parts
      that turn off remain remembered.
    """

    def __init__(
        self,
        grid_size: int = 32,
        # Per-step translation search radius (cells). Keep small for stability.
        step_search_radius: int = 3,
        shift_alpha: float = 0.55,
        min_update_conf: float = 0.40,
        presence_half_life_s: float = 60.0,
        presence_gain: float = 0.10,
        presence_floor: float = 0.0,
        update_percentile: float = 92.0,
        update_min_rel: float = 0.40,
        update_dog_small_sigma: float = 0.6,
        update_dog_large_sigma: float = 1.6,
        update_dog_large_scale: float = 0.65,
        update_max_filter_size: int = 5,
        update_blob_sigma: float = 1.0,
        peak_suppress_radius: int = 5,
        exclude_center_radius: float = 2.0,
        roi_radius: float | None = 14.0,
        # Anchor extraction
        anchor_max: int = 8,
        anchor_percentile: float = 99.0,
        anchor_min_rel: float = 0.85,
        anchor_min_size: int = 14,
        anchor_keep_rel: float = 0.82,
        anchor_prefer_quadrants: bool = True,
        anchor_quadrant_min_rel: float = 0.55,
        centroid_power: float = 1.2,
    ):
        self.grid_size = int(grid_size)
        self.mid = (self.grid_size - 1) / 2.0
        self.step_search_radius = int(step_search_radius)
        self.shift_alpha = float(shift_alpha)
        self.min_update_conf = float(min_update_conf)

        self.presence_half_life_s = float(presence_half_life_s)
        self.presence_gain = float(presence_gain)
        self.presence_floor = float(presence_floor)

        self.update_percentile = float(update_percentile)
        self.update_min_rel = float(update_min_rel)
        self.update_dog_small_sigma = float(update_dog_small_sigma)
        self.update_dog_large_sigma = float(update_dog_large_sigma)
        self.update_dog_large_scale = float(update_dog_large_scale)
        self.update_max_filter_size = int(update_max_filter_size)
        self.update_blob_sigma = float(update_blob_sigma)
        self.peak_suppress_radius = int(peak_suppress_radius)
        self.exclude_center_radius = float(exclude_center_radius)
        self.roi_radius = roi_radius

        self.anchor_max = int(anchor_max)
        self.anchor_percentile = float(anchor_percentile)
        self.anchor_min_rel = float(anchor_min_rel)
        self.anchor_min_size = int(anchor_min_size)
        self.anchor_keep_rel = float(anchor_keep_rel)
        self.anchor_prefer_quadrants = bool(anchor_prefer_quadrants)
        self.anchor_quadrant_min_rel = float(anchor_quadrant_min_rel)

        self.centroid_power = float(centroid_power)

        self.shift = np.array([0.0, 0.0], dtype=float)  # registered -> live
        self.prev_live: np.ndarray | None = None
        self.presence: np.ndarray | None = None
        self.last_confidence: float | None = None

    def reset(self) -> None:
        self.shift[:] = 0.0
        self.prev_live = None
        self.presence = None
        self.last_confidence = None

    def _decay_from_half_life(self, half_life_s: float, dt_s: float) -> float:
        hl = max(1e-3, float(half_life_s))
        dt = max(0.0, float(dt_s))
        return float(0.5 ** (dt / hl))

    def _normalize_frame(self, grid: np.ndarray) -> np.ndarray:
        g = np.maximum(grid.astype(float), 0.0)
        scale = float(np.percentile(g, 99.5)) + 1e-6
        return np.clip(g / scale, 0.0, 1.0)

    def _centroid(self, weights: np.ndarray) -> np.ndarray | None:
        w = np.maximum(weights, 0.0).astype(float)
        if self.centroid_power != 1.0:
            w = w ** self.centroid_power
        total = float(w.sum())
        if total <= 0:
            return None
        rows, cols = np.indices(w.shape)
        r = float((rows * w).sum() / total)
        c = float((cols * w).sum() / total)
        return np.array([r, c], dtype=float)

    def _score_shift(self, prev: np.ndarray, curr: np.ndarray, dr: int, dc: int) -> float:
        # Align curr -> prev by shifting curr by (-dr, -dc).
        shifted = _shift2d_int(curr, -int(dr), -int(dc), fill=0.0)
        num = float(np.sum(prev * shifted))
        den = float(np.sqrt(np.sum(prev * prev) * np.sum(shifted * shifted))) + 1e-9
        return num / den

    def _estimate_delta(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        r = self.step_search_radius
        best = np.array([0.0, 0.0], dtype=float)
        best_score = float("-inf")
        scores = []
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                s = self._score_shift(prev, curr, dr, dc)
                scores.append(s)
                if s > best_score:
                    best_score = s
                    best[:] = (dr, dc)

        med = float(np.median(scores))
        mad = float(np.median(np.abs(np.array(scores) - med))) + 1e-6
        z = (best_score - med) / mad
        conf = float(1.0 / (1.0 + np.exp(-0.75 * (z - 2.0))))
        self.last_confidence = conf
        if conf < self.min_update_conf:
            return np.array([0.0, 0.0], dtype=float)
        return best

    def _anchors_template(self) -> list[tuple[float, float, float]]:
        if self.presence is None:
            return []
        p = ndimage.gaussian_filter(self.presence, sigma=1.0)
        mx = float(np.max(p))
        if mx <= 0:
            return []

        k = self.anchor_max
        thr = max(float(np.percentile(p, self.anchor_percentile)), self.anchor_min_rel * mx)
        mask = p >= thr
        labeled, num = ndimage.label(mask)

        anchors: list[tuple[float, float, float]] = []
        if num > 0:
            sums = ndimage.sum(p, labeled, range(1, num + 1))
            order = np.argsort(np.array(sums))[::-1]
            for idx in order[: max(1, min(k, len(order)))]:
                lab = int(idx) + 1
                comp = labeled == lab
                if int(comp.sum()) < self.anchor_min_size:
                    continue
                w = p * comp.astype(float)
                centroid = self._centroid(w)
                if centroid is None:
                    continue
                if (float(centroid[0]) - self.mid) ** 2 + (float(centroid[1]) - self.mid) ** 2 < self.exclude_center_radius ** 2:
                    continue
                anchors.append((float(centroid[0]), float(centroid[1]), float(np.max(w))))

        if not anchors:
            return []

        anchors.sort(key=lambda x: float(x[2]), reverse=True)
        vmax = float(anchors[0][2])

        if self.anchor_prefer_quadrants and len(anchors) >= 4 and vmax > 0:
            quad_best: dict[int, tuple[float, float, float]] = {}
            for (r0, c0, v0) in anchors:
                if float(v0) < self.anchor_quadrant_min_rel * vmax:
                    continue
                dr = float(r0 - self.mid)
                dc = float(c0 - self.mid)
                if abs(dr) < 0.7 or abs(dc) < 0.7:
                    continue
                q = (0 if dr < 0 else 2) + (0 if dc < 0 else 1)
                prev = quad_best.get(q)
                if prev is None or float(v0) > float(prev[2]):
                    quad_best[q] = (float(r0), float(c0), float(v0))
            if len(quad_best) == 4:
                quad = [quad_best[i] for i in range(4)]
                quad.sort(key=lambda x: float(x[2]), reverse=True)
                return quad[:4]

        kept = [a for a in anchors if float(a[2]) >= self.anchor_keep_rel * vmax]
        return (kept if kept else anchors[: min(2, len(anchors))])[:k]

    def update(self, grid: np.ndarray, dt_s: float = 1.0) -> tuple[float | None, float | None, np.ndarray | None]:
        g = self._normalize_frame(grid)

        if self.prev_live is None:
            # Initialize shift so the current cluster roughly lands near mid in registered coords.
            peaks = extract_peak_observations(g, n_peaks=2, smooth_sigma=1.0, suppress_radius=6)
            if peaks:
                r0 = float(np.mean([p[0] for p in peaks]))
                c0 = float(np.mean([p[1] for p in peaks]))
                self.shift[:] = np.array([r0 - self.mid, c0 - self.mid], dtype=float)

            aligned = _shift2d_int(g, int(-round(self.shift[0])), int(-round(self.shift[1])), fill=0.0)
            thr = float(np.percentile(aligned, self.update_percentile))
            mask = aligned >= thr
            if int(mask.sum()) < 16:
                flat = aligned.ravel()
                order = np.argsort(flat)[::-1]
                k = min(16, order.size)
                mask = np.zeros_like(flat, dtype=bool)
                mask[order[:k]] = True
                mask = mask.reshape(aligned.shape)
            self.presence = mask.astype(float)
            self.prev_live = g

        else:
            delta = self._estimate_delta(self.prev_live, g)
            # Integrate translation (registered -> live).
            self.shift = self.shift + self.shift_alpha * delta

            aligned = _shift2d_int(g, int(-round(self.shift[0])), int(-round(self.shift[1])), fill=0.0)

            assert self.presence is not None
            pdec = self._decay_from_half_life(self.presence_half_life_s, dt_s)
            self.presence = self.presence * pdec

            if float(self.last_confidence or 0.0) >= self.min_update_conf:
                a_small = ndimage.gaussian_filter(aligned, sigma=self.update_dog_small_sigma)
                a_large = ndimage.gaussian_filter(aligned, sigma=self.update_dog_large_sigma)
                dog = np.maximum(a_small - self.update_dog_large_scale * a_large, 0.0)

                mx_d = float(np.max(dog))
                thr = float(np.percentile(dog, self.update_percentile))
                if mx_d > 0:
                    thr = max(thr, self.update_min_rel * mx_d)

                maxima = dog == ndimage.maximum_filter(dog, size=self.update_max_filter_size)
                mask = maxima & (dog >= thr)

                if self.roi_radius is not None:
                    rr, cc = np.indices(aligned.shape)
                    roi = (rr - self.mid) ** 2 + (cc - self.mid) ** 2 <= float(self.roi_radius) ** 2
                    mask = mask & roi

                # Do not write the cluster center itself into memory.
                if self.exclude_center_radius > 0:
                    rr, cc = np.indices(aligned.shape)
                    center_mask = (rr - self.mid) ** 2 + (cc - self.mid) ** 2 >= self.exclude_center_radius ** 2
                    mask = mask & center_mask

                upd_sparse = np.zeros_like(aligned, dtype=float)
                if mask.any():
                    coords = np.argwhere(mask)
                    scores = dog[mask]
                    order = np.argsort(scores)[::-1]
                    taken: list[tuple[int, int]] = []
                    for idx in order:
                        r, c = map(int, coords[idx])
                        if len(taken) >= self.anchor_max:
                            break
                        ok = True
                        for (tr, tc) in taken:
                            if (r - tr) ** 2 + (c - tc) ** 2 < float(self.peak_suppress_radius) ** 2:
                                ok = False
                                break
                        if not ok:
                            continue
                        taken.append((r, c))
                        upd_sparse[r, c] = max(upd_sparse[r, c], float(aligned[r, c]))

                if self.update_blob_sigma > 0 and float(upd_sparse.max()) > 0:
                    upd_sparse = ndimage.gaussian_filter(upd_sparse, sigma=self.update_blob_sigma)

                umx = float(upd_sparse.max())
                if umx > 0:
                    upd = np.clip(upd_sparse / (umx + 1e-6), 0.0, 1.0)
                    self.presence = np.clip(self.presence + self.presence_gain * upd, 0.0, 1.0)

            if self.presence_floor > 0:
                self.presence[self.presence < self.presence_floor] = 0.0

            self.prev_live = g

        # Center from anchors when available; otherwise from presence.
        anchors = self._anchors_template()
        if len(anchors) >= 2:
            centroid = np.mean(np.array([[a[0], a[1]] for a in anchors], dtype=float), axis=0)
        else:
            pres_sm = ndimage.gaussian_filter(self.presence, sigma=1.0) if self.presence is not None else None
            centroid = self._centroid(pres_sm) if pres_sm is not None else None
        if centroid is None:
            return None, None, self.presence

        # Recenter registered frame around mid to keep numbers stable.
        recenter = centroid - np.array([self.mid, self.mid], dtype=float)
        if abs(float(recenter[0])) > 0.6 or abs(float(recenter[1])) > 0.6:
            dr = int(round(float(recenter[0])))
            dc = int(round(float(recenter[1])))
            assert self.presence is not None
            self.presence = _shift2d_int(self.presence, -dr, -dc, fill=0.0)
            self.shift += np.array([dr, dc], dtype=float)
            centroid = centroid - np.array([dr, dc], dtype=float)

        center_live = centroid + self.shift
        return float(center_live[0]), float(center_live[1]), self.presence

    def top_spots_memory(self, n_peaks: int = 8) -> list[tuple[float, float, float]]:
        out = self._anchors_template()
        return out[: int(n_peaks)]

    def top_spots_live(self, n_peaks: int = 8) -> list[tuple[float, float, float]]:
        out = []
        for (r, c, v) in self.top_spots_memory(n_peaks=n_peaks):
            rr = float(r + self.shift[0])
            cc = float(c + self.shift[1])
            if rr < 0.0 or cc < 0.0 or rr > self.grid_size - 1.0 or cc > self.grid_size - 1.0:
                continue
            out.append((rr, cc, float(v)))
        return out


class SimpleKalmanFilter2D:
    """
    Simple 2D Kalman filter with constant-velocity model for smooth center tracking.

    State: [row, col, vel_row, vel_col]
    Assumes smooth, continuous motion of the array.
    """

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 2.0,
        initial_pos: tuple[float, float] = (15.5, 15.5),
    ):
        # State: [row, col, vel_row, vel_col]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)

        # State covariance
        self.P = np.eye(4, dtype=float) * 10.0

        # Process noise (how much we expect state to change)
        self.Q = np.eye(4, dtype=float)
        self.Q[0, 0] = process_noise * 0.5
        self.Q[1, 1] = process_noise * 0.5
        self.Q[2, 2] = process_noise
        self.Q[3, 3] = process_noise

        # Measurement noise (how much we trust observations)
        self.R = np.eye(2, dtype=float) * measurement_noise

        # Measurement matrix (we only observe position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self._initialized = False

    def reset(self, pos: tuple[float, float] | None = None):
        if pos is not None:
            self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        else:
            self.x = np.array([15.5, 15.5, 0.0, 0.0], dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0
        self._initialized = False

    def predict(self, dt: float = 0.05):
        """Predict next state based on constant velocity model."""
        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, measurement: np.ndarray, confidence: float = 1.0):
        """Update state with new measurement. Lower confidence = trust measurement less."""
        if not self._initialized:
            self.x[0] = measurement[0]
            self.x[1] = measurement[1]
            self._initialized = True
            return

        # Adjust measurement noise based on confidence
        R_adj = self.R / max(0.1, confidence)

        # Innovation
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R_adj

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    @property
    def position(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))


class MotionCompensatedTracker:
    """
    Track the 4 tuned regions with motion-compensated memory and Kalman-smoothed center.

    Key improvements over FourTunedRegionTracker:
    1. Kalman filter smooths center position (no jumping)
    2. Frame-to-frame motion estimation keeps memory aligned with array movement
    3. Region positions are smoothed with outlier rejection
    4. Memory map shifts with estimated motion
    """

    REGION_KEYS = ("vx_pos", "vx_neg", "vy_pos", "vy_neg")

    def __init__(
        self,
        grid_size: int = 32,
        # Region tracking
        ema_alpha: float = 0.25,
        smooth_sigma: float = 1.2,
        detect_percentile: float = 90.0,
        min_component_size: int = 5,
        max_components: int = 6,
        min_offset_for_assignment: float = 2.0,
        # Memory
        memory_half_life_s: float = 60.0,
        memory_gain: float = 0.15,
        # Kalman filter
        kalman_process_noise: float = 0.3,
        kalman_measurement_noise: float = 1.5,
        # Motion estimation
        motion_search_radius: int = 4,
        motion_alpha: float = 0.6,
        # Outlier rejection
        max_region_jump: float = 4.0,
    ):
        self.grid_size = int(grid_size)
        self.mid = (self.grid_size - 1) / 2.0

        self.ema_alpha = float(ema_alpha)
        self.smooth_sigma = float(smooth_sigma)
        self.detect_percentile = float(detect_percentile)
        self.min_component_size = int(min_component_size)
        self.max_components = int(max_components)
        self.min_offset_for_assignment = float(min_offset_for_assignment)

        self.memory_half_life_s = float(memory_half_life_s)
        self.memory_gain = float(memory_gain)

        self.motion_search_radius = int(motion_search_radius)
        self.motion_alpha = float(motion_alpha)
        self.max_region_jump = float(max_region_jump)

        # State
        self.kalman = SimpleKalmanFilter2D(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        )
        self.regions: dict[str, np.ndarray | None] = {k: None for k in self.REGION_KEYS}
        self.region_strength: dict[str, float] = {k: 0.0 for k in self.REGION_KEYS}
        self.memory_map: np.ndarray | None = None
        self.prev_grid: np.ndarray | None = None
        self.estimated_motion: np.ndarray = np.array([0.0, 0.0], dtype=float)
        self.accumulated_motion: np.ndarray = np.array([0.0, 0.0], dtype=float)

    def reset(self) -> None:
        self.kalman.reset()
        self.regions = {k: None for k in self.REGION_KEYS}
        self.region_strength = {k: 0.0 for k in self.REGION_KEYS}
        self.memory_map = None
        self.prev_grid = None
        self.estimated_motion = np.array([0.0, 0.0], dtype=float)
        self.accumulated_motion = np.array([0.0, 0.0], dtype=float)

    def _decay_from_half_life(self, half_life_s: float, dt_s: float) -> float:
        hl = max(1e-3, float(half_life_s))
        dt = max(0.0, float(dt_s))
        return float(0.5 ** (dt / hl))

    def _estimate_motion(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Estimate frame-to-frame motion using cross-correlation."""
        if prev is None:
            return np.array([0.0, 0.0], dtype=float)

        r = self.motion_search_radius
        best_dr, best_dc = 0, 0
        best_score = float("-inf")

        # Normalize both frames
        p_norm = prev - np.mean(prev)
        c_norm = curr - np.mean(curr)

        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                # Shift current frame
                shifted = _shift2d_int(c_norm, -dr, -dc, fill=0.0)
                # Compute correlation
                score = float(np.sum(p_norm * shifted))
                if score > best_score:
                    best_score = score
                    best_dr, best_dc = dr, dc

        return np.array([float(best_dr), float(best_dc)], dtype=float)

    def _detect_components(self, grid: np.ndarray) -> list[dict[str, float]]:
        g = ndimage.gaussian_filter(np.maximum(grid.astype(float), 0.0), sigma=self.smooth_sigma)
        thr = float(np.percentile(g, self.detect_percentile))
        mask = g >= thr
        labeled, num = ndimage.label(mask)
        if num <= 0:
            return []

        comps: list[dict[str, float]] = []
        for lab in range(1, num + 1):
            comp = labeled == lab
            size = int(comp.sum())
            if size < self.min_component_size:
                continue
            vals = g[comp]
            total = float(vals.sum()) + 1e-9
            rows, cols = np.where(comp)
            r = float(np.sum(rows * vals) / total)
            c = float(np.sum(cols * vals) / total)
            peak = float(np.max(vals))
            comps.append({"row": r, "col": c, "peak": peak, "mass": total, "size": float(size)})

        comps.sort(key=lambda x: float(x["mass"]), reverse=True)
        return comps[: self.max_components]

    def _assign_region(self, comp_row: float, comp_col: float, center: np.ndarray) -> str | None:
        dr = float(comp_row - float(center[0]))
        dc = float(comp_col - float(center[1]))
        if abs(dr) < self.min_offset_for_assignment and abs(dc) < self.min_offset_for_assignment:
            return None
        if abs(dc) >= abs(dr):
            return "vx_pos" if dc > 0 else "vx_neg"
        return "vy_pos" if dr < 0 else "vy_neg"

    def _infer_center_from_components(self, comps: list[dict[str, float]]) -> np.ndarray | None:
        if len(comps) < 2:
            return None

        c1, c2 = comps[0], comps[1]
        r1, c1c = float(c1["row"]), float(c1["col"])
        r2, c2c = float(c2["row"]), float(c2["col"])

        dr = abs(r1 - r2)
        dc = abs(c1c - c2c)

        if dr <= 0.5 * dc:
            return np.array([(r1 + r2) / 2.0, (c1c + c2c) / 2.0], dtype=float)
        if dc <= 0.5 * dr:
            return np.array([(r1 + r2) / 2.0, (c1c + c2c) / 2.0], dtype=float)

        cand_a = np.array([r1, c2c], dtype=float)
        cand_b = np.array([r2, c1c], dtype=float)
        prev_pos = self.kalman.position
        prev = np.array([prev_pos[0], prev_pos[1]], dtype=float)
        da = float(np.linalg.norm(cand_a - prev))
        db = float(np.linalg.norm(cand_b - prev))
        return cand_a if da <= db else cand_b

    def update(
        self, grid: np.ndarray, dt_s: float = 0.05
    ) -> tuple[
        float | None,
        float | None,
        float,
        np.ndarray | None,
        list[tuple[float, float, float]],
        list[tuple[float, float, float]],
        dict[str, tuple[float, float, float]],
    ]:
        # Smooth input grid
        g = ndimage.gaussian_filter(np.maximum(grid.astype(float), 0.0), sigma=0.8)

        # Estimate frame-to-frame motion
        if self.prev_grid is not None:
            raw_motion = self._estimate_motion(self.prev_grid, g)
            self.estimated_motion = (1 - self.motion_alpha) * self.estimated_motion + self.motion_alpha * raw_motion
            self.accumulated_motion += self.estimated_motion
        self.prev_grid = g.copy()

        # Kalman predict step
        self.kalman.predict(dt=dt_s)

        # Detect components
        comps = self._detect_components(grid)

        # Decay region strengths
        decay = self._decay_from_half_life(self.memory_half_life_s, dt_s)
        for k in self.REGION_KEYS:
            self.region_strength[k] *= decay

        # Infer center from components
        center_obs = self._infer_center_from_components(comps)

        # Compute confidence based on how many regions we're seeing
        n_active = len(comps)
        obs_confidence = min(1.0, n_active / 2.0)  # 2+ components = full confidence

        # Update Kalman filter with observation
        if center_obs is not None:
            self.kalman.update(center_obs, confidence=obs_confidence)

        # Get smoothed center from Kalman filter
        center = np.array(self.kalman.position, dtype=float)

        # Assign components to regions with outlier rejection
        best: dict[str, dict[str, float]] = {}
        for comp in comps:
            key = self._assign_region(comp["row"], comp["col"], center)
            if key is None:
                continue
            prev = best.get(key)
            if prev is None or float(comp["mass"]) > float(prev["mass"]):
                best[key] = comp

        # Update region positions with outlier rejection
        for key, comp in best.items():
            new_pos = np.array([float(comp["row"]), float(comp["col"])], dtype=float)

            if self.regions[key] is None:
                self.regions[key] = new_pos
            else:
                # Reject if jump is too large (outlier)
                jump = float(np.linalg.norm(new_pos - self.regions[key]))
                if jump < self.max_region_jump:
                    self.regions[key] = (1.0 - self.ema_alpha) * self.regions[key] + self.ema_alpha * new_pos

            self.region_strength[key] = min(1.0, self.region_strength[key] + self.memory_gain)

        # If no direct center observation, use average of remembered regions
        if center_obs is None:
            pts = [v for v in self.regions.values() if v is not None]
            if len(pts) >= 2:
                region_center = np.mean(np.stack(pts, axis=0), axis=0)
                self.kalman.update(region_center, confidence=0.3)
                center = np.array(self.kalman.position, dtype=float)

        # Confidence based on region strengths and current observations
        strengths = np.array([self.region_strength[k] for k in self.REGION_KEYS], dtype=float)
        seen_now = min(2, len(best)) / 2.0
        conf = float(np.clip(0.5 * np.mean(np.sort(strengths)[-2:]) + 0.5 * seen_now, 0.0, 1.0))

        # Update motion-compensated memory map
        if self.memory_map is None:
            self.memory_map = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # Shift memory by estimated motion to keep it aligned
        if abs(self.estimated_motion[0]) > 0.3 or abs(self.estimated_motion[1]) > 0.3:
            dr = int(round(self.estimated_motion[0]))
            dc = int(round(self.estimated_motion[1]))
            if dr != 0 or dc != 0:
                self.memory_map = _shift2d_int(self.memory_map, dr, dc, fill=0.0)

        # Add current frame to memory (center-aligned)
        dr = int(round(float(center[0] - self.mid)))
        dc = int(round(float(center[1] - self.mid)))
        aligned = _shift2d_int(np.maximum(grid.astype(float), 0.0), -dr, -dc, fill=0.0)
        thr_m = float(np.percentile(aligned, self.detect_percentile))
        mask_m = aligned >= thr_m
        self.memory_map = np.maximum(self.memory_map * decay, aligned * mask_m.astype(float))

        # Build spots output (stable region positions)
        spots_live: list[tuple[float, float, float]] = []
        spots_mem: list[tuple[float, float, float]] = []
        regions_out: dict[str, tuple[float, float, float]] = {}

        for k in self.REGION_KEYS:
            p = self.regions[k]
            if p is None:
                continue
            w = float(self.region_strength[k])
            if w > 0.1:  # Only output regions with meaningful strength
                spots_live.append((float(p[0]), float(p[1]), w))
                spots_mem.append((float(p[0] - dr), float(p[1] - dc), w))
                regions_out[k] = (float(p[0]), float(p[1]), w)

        spots_live.sort(key=lambda x: float(x[2]), reverse=True)
        spots_mem.sort(key=lambda x: float(x[2]), reverse=True)

        return float(center[0]), float(center[1]), conf, self.memory_map, spots_live, spots_mem, regions_out


class FourTunedRegionTracker:
    """
    Track the 4 velocity-tuned subregions (Vx+/Vx-/Vy+/Vy-) without using ground truth.

    Assumption (matches Track2 generator): at any moment, activity concentrates into
    1-2 blobs corresponding to horizontal and/or vertical tuning. Over time, these
    belong to 4 stable subregions around a common center. We:
    - detect blobs each frame
    - assign each blob to one of 4 regions based on position relative to current center
    - EMA each region center and keep it as memory when it's "off"
    - compute overall center as average of the 4 remembered region centers

    This aligns well with ground_truth center movement and avoids brittle global
    template alignment.
    """

    REGION_KEYS = ("vx_pos", "vx_neg", "vy_pos", "vy_neg")

    def __init__(
        self,
        grid_size: int = 32,
        ema_alpha: float = 0.35,
        smooth_sigma: float = 1.0,
        detect_percentile: float = 92.0,
        min_component_size: int = 6,
        max_components: int = 6,
        min_offset_for_assignment: float = 1.0,
        memory_half_life_s: float = 80.0,
        memory_gain: float = 0.12,
        center_alpha: float = 0.35,
    ):
        self.grid_size = int(grid_size)
        self.mid = (self.grid_size - 1) / 2.0
        self.ema_alpha = float(ema_alpha)
        self.smooth_sigma = float(smooth_sigma)
        self.detect_percentile = float(detect_percentile)
        self.min_component_size = int(min_component_size)
        self.max_components = int(max_components)
        self.min_offset_for_assignment = float(min_offset_for_assignment)
        self.memory_half_life_s = float(memory_half_life_s)
        self.memory_gain = float(memory_gain)
        self.center_alpha = float(center_alpha)

        self.regions: dict[str, np.ndarray | None] = {k: None for k in self.REGION_KEYS}
        self.region_strength: dict[str, float] = {k: 0.0 for k in self.REGION_KEYS}
        self.memory_map: np.ndarray | None = None  # registered to center
        self.center: np.ndarray | None = None

    def reset(self) -> None:
        self.regions = {k: None for k in self.REGION_KEYS}
        self.region_strength = {k: 0.0 for k in self.REGION_KEYS}
        self.memory_map = None
        self.center = None

    def _decay_from_half_life(self, half_life_s: float, dt_s: float) -> float:
        hl = max(1e-3, float(half_life_s))
        dt = max(0.0, float(dt_s))
        return float(0.5 ** (dt / hl))

    def _detect_components(self, grid: np.ndarray) -> list[dict[str, float]]:
        g = ndimage.gaussian_filter(np.maximum(grid.astype(float), 0.0), sigma=self.smooth_sigma)
        thr = float(np.percentile(g, self.detect_percentile))
        mask = g >= thr
        labeled, num = ndimage.label(mask)
        if num <= 0:
            return []

        comps: list[dict[str, float]] = []
        for lab in range(1, num + 1):
            comp = labeled == lab
            size = int(comp.sum())
            if size < self.min_component_size:
                continue
            vals = g[comp]
            total = float(vals.sum()) + 1e-9
            rows, cols = np.where(comp)
            r = float(np.sum(rows * vals) / total)
            c = float(np.sum(cols * vals) / total)
            peak = float(np.max(vals))
            comps.append({"row": r, "col": c, "peak": peak, "mass": total, "size": float(size)})

        comps.sort(key=lambda x: float(x["mass"]), reverse=True)
        return comps[: self.max_components]

    def _current_center_estimate(self) -> np.ndarray:
        if self.center is not None:
            return self.center.copy()
        return np.array([self.mid, self.mid], dtype=float)

    def _infer_center_from_components(self, comps: list[dict[str, float]]) -> np.ndarray | None:
        # Use simple cross geometry with temporal continuity.
        if len(comps) < 2:
            return None
        # Use two strongest components.
        c1, c2 = comps[0], comps[1]
        r1, c1c = float(c1["row"]), float(c1["col"])
        r2, c2c = float(c2["row"]), float(c2["col"])

        dr = abs(r1 - r2)
        dc = abs(c1c - c2c)

        # If the pair is mostly horizontal (vx+/vx-), rows are similar.
        if dr <= 0.5 * dc:
            return np.array([(r1 + r2) / 2.0, (c1c + c2c) / 2.0], dtype=float)
        # If the pair is mostly vertical (vy+/vy-), cols are similar.
        if dc <= 0.5 * dr:
            return np.array([(r1 + r2) / 2.0, (c1c + c2c) / 2.0], dtype=float)

        # Orthogonal pair: choose the L-shape corner closest to previous center.
        cand_a = np.array([r1, c2c], dtype=float)
        cand_b = np.array([r2, c1c], dtype=float)
        prev = self.center if self.center is not None else np.array([self.mid, self.mid], dtype=float)
        da = float(np.linalg.norm(cand_a - prev))
        db = float(np.linalg.norm(cand_b - prev))
        return cand_a if da <= db else cand_b

    def _assign_region(self, comp_row: float, comp_col: float, center: np.ndarray) -> str | None:
        dr = float(comp_row - float(center[0]))
        dc = float(comp_col - float(center[1]))
        if abs(dr) < self.min_offset_for_assignment and abs(dc) < self.min_offset_for_assignment:
            return None
        if abs(dc) >= abs(dr):
            return "vx_pos" if dc > 0 else "vx_neg"
        # row smaller = up on image => vy_pos
        return "vy_pos" if dr < 0 else "vy_neg"

    def update(
        self, grid: np.ndarray, dt_s: float = 1.0
    ) -> tuple[
        float | None,
        float | None,
        float,
        np.ndarray | None,
        list[tuple[float, float, float]],
        list[tuple[float, float, float]],
        dict[str, tuple[float, float, float]],
    ]:
        comps = self._detect_components(grid)
        center = self._current_center_estimate()

        decay = self._decay_from_half_life(self.memory_half_life_s, dt_s)
        for k in self.REGION_KEYS:
            self.region_strength[k] *= decay

        center_obs = self._infer_center_from_components(comps)
        if center_obs is not None:
            if self.center is None:
                self.center = center_obs
            else:
                self.center = (1.0 - self.center_alpha) * self.center + self.center_alpha * center_obs
            center = self._current_center_estimate()

        # For each region, pick the strongest assigned component this frame.
        best: dict[str, dict[str, float]] = {}
        for comp in comps:
            key = self._assign_region(comp["row"], comp["col"], center)
            if key is None:
                continue
            prev = best.get(key)
            if prev is None or float(comp["mass"]) > float(prev["mass"]):
                best[key] = comp

        for key, comp in best.items():
            p = np.array([float(comp["row"]), float(comp["col"])], dtype=float)
            if self.regions[key] is None:
                self.regions[key] = p
            else:
                self.regions[key] = (1.0 - self.ema_alpha) * self.regions[key] + self.ema_alpha * p
            self.region_strength[key] = min(1.0, self.region_strength[key] + self.memory_gain)

        # If we did not get a direct center observation, fall back to remembered regions.
        if center_obs is None:
            pts = [v for v in self.regions.values() if v is not None]
            if pts:
                center_from_regions = np.mean(np.stack(pts, axis=0), axis=0)
                if self.center is None:
                    self.center = center_from_regions
                else:
                    self.center = (1.0 - self.center_alpha) * self.center + self.center_alpha * center_from_regions
            center = self._current_center_estimate()

        # Confidence: how many regions are currently remembered with decent strength.
        strengths = np.array([self.region_strength[k] for k in self.REGION_KEYS], dtype=float)
        seen_now = min(2, len(best)) / 2.0
        conf = float(np.clip(0.55 * np.mean(np.sort(strengths)[-2:]) + 0.45 * seen_now, 0.0, 1.0))

        # Update registered memory map (aligned so center lives at mid).
        if self.memory_map is None:
            self.memory_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        dr = int(round(float(center[0] - self.mid)))
        dc = int(round(float(center[1] - self.mid)))
        aligned = _shift2d_int(np.maximum(grid.astype(float), 0.0), -dr, -dc, fill=0.0)
        # Use a conservative mask to avoid accumulating background.
        thr_m = float(np.percentile(aligned, self.detect_percentile))
        mask_m = aligned >= thr_m
        self.memory_map = np.maximum(self.memory_map * decay, aligned * mask_m.astype(float))

        # Spots live + spots_mem (template coords)
        spots_live: list[tuple[float, float, float]] = []
        spots_mem: list[tuple[float, float, float]] = []
        regions_out: dict[str, tuple[float, float, float]] = {}
        for k in self.REGION_KEYS:
            p = self.regions[k]
            if p is None:
                continue
            w = float(self.region_strength[k])
            spots_live.append((float(p[0]), float(p[1]), w))
            spots_mem.append((float(p[0] - dr), float(p[1] - dc), w))
            regions_out[k] = (float(p[0]), float(p[1]), w)

        # Sort for display.
        spots_live.sort(key=lambda x: float(x[2]), reverse=True)
        spots_mem.sort(key=lambda x: float(x[2]), reverse=True)

        return float(center[0]), float(center[1]), conf, self.memory_map, spots_live, spots_mem, regions_out


def visualize_cross_center_with_memory(
    df,
    t_start,
    window_size=0.2,
    processor=None,
    cross_tracker=None,
    memory=None,
):
    """
    Visualize: current activity, inferred cross center/arms, and the center-aligned
    persistence map (should show all 4 arms over time).
    """
    if processor is None:
        processor = SignalProcessor()
    if cross_tracker is None:
        cross_tracker = CrossMatchedFilterTracker()
    if memory is None:
        memory = CenterAlignedPersistence()

    grid = compute_power_grid(df, t_start, window_size, processor)
    est_row, est_col, dx, dy = cross_tracker.update(grid)
    mem = memory.update(grid, (est_row, est_col))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(grid, cmap="hot", aspect="equal")
    axes[0].set_title(f"Current Activity (t={t_start:.2f}s)")
    if est_row is not None and dx is not None and dy is not None:
        cr, cc = est_row, est_col
        axes[0].scatter([cc], [cr], c="cyan", s=140, marker="*", linewidths=1.5)
        arms = [
            (cr, cc + dx),
            (cr, cc - dx),
            (cr + dy, cc),
            (cr - dy, cc),
        ]
        for (r, c) in arms:
            axes[0].scatter([c], [r], c="lime", s=80, marker="o", edgecolors="black")
    axes[0].set_xlabel("Col")
    axes[0].set_ylabel("Row")

    if mem is not None:
        axes[1].imshow(mem, cmap="hot", aspect="equal")
        axes[1].set_title("Center-Aligned Persistence (memory)")
    else:
        axes[1].text(0.5, 0.5, "No memory yet", ha="center", va="center")
        axes[1].set_title("Center-Aligned Persistence (memory)")
    axes[1].set_xlabel("Col")
    axes[1].set_ylabel("Row")

    plt.tight_layout()
    return fig


def track_cross_center_over_time(
    df,
    ground_truth,
    tracker,
    cross_tracker=None,
    t_start=0,
    t_end=30,
    window_size=0.2,
    step=0.1,
    processor=None,
):
    if processor is None:
        processor = SignalProcessor()
    if cross_tracker is None:
        cross_tracker = CrossMatchedFilterTracker()

    times = np.arange(t_start, t_end - window_size, step)
    gt_time = _get_gt_time_array(ground_truth)

    results = {
        "time": [],
        "est_row": [],
        "est_col": [],
        "true_row": [],
        "true_col": [],
        "dx": [],
        "dy": [],
    }

    for t in times:
        power = compute_power_grid(df, t, window_size, processor)
        est_row, est_col, dx, dy = cross_tracker.update(power)

        gt_idx = np.abs(gt_time - t).argmin()
        true_row, true_col, _ = compute_ground_truth_center(ground_truth, gt_idx)

        results["time"].append(t)
        results["est_row"].append(est_row if est_row is not None else np.nan)
        results["est_col"].append(est_col if est_col is not None else np.nan)
        results["true_row"].append(true_row if true_row is not None else np.nan)
        results["true_col"].append(true_col if true_col is not None else np.nan)
        results["dx"].append(dx if dx is not None else np.nan)
        results["dy"].append(dy if dy is not None else np.nan)

    return pd.DataFrame(results)


def track_directional_centers_over_time(
    df,
    ground_truth,
    tracker,
    accumulator=None,
    t_start=0,
    t_end=30,
    window_size=0.2,
    step=0.1,
    processor=None,
):
    """
    Track direction-accumulated center over time and compare to ground truth.

    Uses ground truth vx/vy to gate updates (dev only). Direction maps are only
    updated when an axis dominates to avoid mixing vx/vy regions.
    """
    if processor is None:
        processor = SignalProcessor()
    if accumulator is None:
        accumulator = DirectionalAccumulator()

    times = np.arange(t_start, t_end - window_size, step)
    gt_time = _get_gt_time_array(ground_truth)

    results = {
        "time": [],
        "est_row": [],
        "est_col": [],
        "true_row": [],
        "true_col": [],
        "vx": [],
        "vy": [],
    }

    for t in times:
        power = compute_power_grid(df, t, window_size, processor)

        gt_idx = np.abs(gt_time - t).argmin()
        vx = ground_truth["vx"].iloc[gt_idx] if "vx" in ground_truth.columns else None
        vy = ground_truth["vy"].iloc[gt_idx] if "vy" in ground_truth.columns else None

        accumulator.update(power, vx, vy)
        est_row, est_col, _ = accumulator.estimate_center(tracker)

        true_row, true_col, _ = compute_ground_truth_center(ground_truth, gt_idx)

        results["time"].append(t)
        results["est_row"].append(est_row if est_row is not None else np.nan)
        results["est_col"].append(est_col if est_col is not None else np.nan)
        results["true_row"].append(true_row if true_row is not None else np.nan)
        results["true_col"].append(true_col if true_col is not None else np.nan)
        results["vx"].append(vx if vx is not None else np.nan)
        results["vy"].append(vy if vy is not None else np.nan)

    return pd.DataFrame(results)


def plot_center_comparison(results):
    """Plot estimated vs ground truth center over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(results["time"], results["true_col"], label="True center col", alpha=0.7)
    axes[0].plot(results["time"], results["est_col"], label="Estimated center col", alpha=0.7)
    axes[0].set_ylabel("Column (0-31)")
    axes[0].set_title("Center Column Tracking")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results["time"], results["true_row"], label="True center row", alpha=0.7)
    axes[1].plot(results["time"], results["est_row"], label="Estimated center row", alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Row (0-31)")
    axes[1].set_title("Center Row Tracking")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    col_err = np.nanmean(np.abs(results["est_col"] - results["true_col"]))
    row_err = np.nanmean(np.abs(results["est_row"] - results["true_row"]))

    return fig, row_err, col_err


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


if __name__ == "__main__":
    print("=" * 60)
    print("BrainStorm Track 2 - Hotspot Tracking Algorithm")
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
        envelope_method='hilbert',
        grid_transform="rot180",
    )
    
    # Initialize tracker
    tracker = HotspotTracker(threshold_percentile=85, min_spot_size=3, smoothing_sigma=0)
    # Note: spatial smoothing is now done in SignalProcessor, so set smoothing_sigma=0 in tracker
    print("\nHotspotTracker initialized!")
    
    # Test single frame detection
    print("\n[1/5] Testing hotspot detection at t=5s...")
    fig1, hotspots, center, vel = visualize_hotspot_detection(
        neural_data, tracker, t_start=5.0, processor=processor
    )
    print(f"  Detected {len(hotspots)} hotspots")
    if center[0] is not None:
        print(f"  Average center: row={center[0]:.1f}, col={center[1]:.1f}")
    print(f"  Predicted velocity: vx={vel[0]:.3f}, vy={vel[1]:.3f}")
    plt.show()
    
    # Track over time
    print("\n[2/5] Tracking hotspots over 30 seconds...")
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
    
    # Cross center tracking (neural-only; solves center from 2-of-4 arms)
    print("\n[3/5] Tracking cross center over 30 seconds...")
    cross_tracker = CrossMatchedFilterTracker(
        search_radius=6,
        center_alpha=0.6,
        fixed_dx=6.0,
        fixed_dy=6.0,
        patch_radius=3,
        patch_sigma=1.25,
    )
    center_results = track_cross_center_over_time(
        neural_data,
        ground_truth,
        tracker,
        cross_tracker=cross_tracker,
        t_start=0,
        t_end=30,
        processor=processor,
    )
    fig3, row_err, col_err = plot_center_comparison(center_results)
    print(f"  Mean abs error: row={row_err:.2f}, col={col_err:.2f} (grid units)")
    plt.show()

    print("\n[4/5] Visualizing center-aligned memory map...")
    cross_tracker2 = CrossMatchedFilterTracker(
        search_radius=6,
        center_alpha=0.6,
        fixed_dx=6.0,
        fixed_dy=6.0,
        patch_radius=3,
        patch_sigma=1.25,
    )
    memory = CenterAlignedPersistence(ref_center=(15.5, 15.5), decay=0.997, mode="leaky_max")
    # Warm up the memory for a few seconds to show all 4 arms.
    for t in np.arange(0, 10.0, 0.1):
        grid = compute_power_grid(neural_data, float(t), 0.2, processor)
        est_row, est_col, _dx, _dy = cross_tracker2.update(grid)
        memory.update(grid, (est_row, est_col))
    fig4 = visualize_cross_center_with_memory(
        neural_data,
        t_start=18.1,
        window_size=0.2,
        processor=processor,
        cross_tracker=cross_tracker2,
        memory=memory,
    )
    plt.show()

    # Animation
    print("\n[5/5] Creating animated visualization...")
    print("  Close the plot window when done viewing.")
    fig5, anim = animate_hotspot_tracking(
        neural_data, ground_truth, tracker, t_start=0, t_end=20, fps=10, processor=processor
    )
    plt.show()
    
    print("\nDone!")
