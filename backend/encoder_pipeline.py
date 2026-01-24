"""
U-Net Encoder pipeline for BCI signal processing.

Uses a pre-trained MedicalUNet model to denoise and enhance neural activity maps.
Includes velocity tracking from denoised neural activity grids.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt


class MedicalUNet(nn.Module):
    """U-Net architecture for neural signal denoising."""

    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.e2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU())
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.upconv2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 1, 1)
        )
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


@dataclass
class VelocityResult:
    """Result from velocity computation."""

    vx: float
    vy: float
    com_row: float
    com_col: float
    num_targets: int
    cursor_x: float
    cursor_y: float


class VelocityTracker:
    """
    Computes cursor velocity from denoised neural activity grid.

    Uses intensity-weighted center of mass of the UNet-denoised activity
    to infer intended movement direction. This is purely derived from
    neural data - no ground truth is used.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        threshold: float = 0.2,
        exaggeration: float = 0.6,
    ):
        """
        Initialize velocity tracker.

        Args:
            smoothing_alpha: EMA smoothing factor for velocity (higher = more responsive)
            threshold: Threshold for detecting active zones (0-1)
            exaggeration: Power for exaggerating small displacements (lower = more exaggeration)
        """
        self.smoothing_alpha = smoothing_alpha
        self.threshold = threshold
        self.exaggeration = exaggeration

        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.cursor_x = 0.0
        self.cursor_y = 0.0

    def compute_velocity(
        self, denoised_grid: np.ndarray
    ) -> tuple[float, float, float, float, int]:
        """
        Compute velocity from intensity-weighted center of mass.

        Args:
            denoised_grid: 32x32 numpy array (0-1 confidence values)

        Returns:
            (vx, vy, com_row, com_col, num_targets)
        """
        # Threshold to find active zones
        _, mask = cv2.threshold(
            denoised_grid.astype(np.float32), self.threshold, 1.0, cv2.THRESH_BINARY
        )
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Find contours (targets)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        num_targets = len(contours)

        # Use SQUARED intensity for center of mass (exaggerates bright hotspots)
        weighted_grid = denoised_grid**2

        total_intensity = weighted_grid.sum()

        if total_intensity > 0.01:
            rows, cols = np.mgrid[0:32, 0:32]
            avg_row = (rows * weighted_grid).sum() / total_intensity
            avg_col = (cols * weighted_grid).sum() / total_intensity

            # Convert to velocity (center is 15.5, 15.5)
            raw_vx = (avg_col - 15.5) / 15.5
            raw_vy = -(avg_row - 15.5) / 15.5  # Negative because row 0 is top

            # Exaggerate small displacements (sign-preserving power < 1)
            raw_vx = np.sign(raw_vx) * (np.abs(raw_vx) ** self.exaggeration)
            raw_vy = np.sign(raw_vy) * (np.abs(raw_vy) ** self.exaggeration)

            # Smooth velocity
            vx = (
                self.smoothing_alpha * raw_vx
                + (1 - self.smoothing_alpha) * self.prev_vx
            )
            vy = (
                self.smoothing_alpha * raw_vy
                + (1 - self.smoothing_alpha) * self.prev_vy
            )

            self.prev_vx = vx
            self.prev_vy = vy

            return vx, vy, float(avg_row), float(avg_col), num_targets
        else:
            # Decay velocity when no activity
            self.prev_vx *= 0.9
            self.prev_vy *= 0.9
            return self.prev_vx, self.prev_vy, 15.5, 15.5, 0

    def update_cursor(
        self, vx: float, vy: float, speed_scale: float = 5.0
    ) -> tuple[float, float]:
        """Update cursor position based on velocity."""
        self.cursor_x = float(np.clip(self.cursor_x + vx * speed_scale, -100, 100))
        self.cursor_y = float(np.clip(self.cursor_y + vy * speed_scale, -100, 100))
        return self.cursor_x, self.cursor_y

    def process(
        self, denoised_grid: np.ndarray, speed_scale: float = 5.0
    ) -> VelocityResult:
        """
        Process denoised grid and return full velocity result.

        Args:
            denoised_grid: 32x32 numpy array (0-1 confidence values)
            speed_scale: Cursor movement speed multiplier

        Returns:
            VelocityResult with all computed values
        """
        vx, vy, com_row, com_col, num_targets = self.compute_velocity(denoised_grid)
        cursor_x, cursor_y = self.update_cursor(vx, vy, speed_scale)

        return VelocityResult(
            vx=vx,
            vy=vy,
            com_row=com_row,
            com_col=com_col,
            num_targets=num_targets,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
        )

    def reset(self):
        """Reset tracker state."""
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.cursor_x = 0.0
        self.cursor_y = 0.0


class EncoderPipeline:
    """
    Neural signal processing pipeline using U-Net encoder.

    Processing flow:
    1. Bandpass filter (70-150Hz high-gamma)
    2. Power extraction (squared signal)
    3. EMA smoothing
    4. Normalize to 99.5th percentile
    5. U-Net model inference
    """

    def __init__(
        self,
        model_path: Path | str = "scripts/compass_model.pth",
        fs: float = 500.0,
        n_channels: int = 1024,
        bandpass_low: float = 70.0,
        bandpass_high: float = 150.0,
        ema_alpha: float = 0.2,
        enable_velocity: bool = True,
        velocity_smoothing: float = 0.3,
    ):
        """
        Initialize the encoder pipeline.

        Args:
            model_path: Path to pre-trained U-Net model weights
            fs: Sampling frequency
            n_channels: Number of channels (default 1024 for 32x32 grid)
            bandpass_low: Low cutoff for bandpass filter
            bandpass_high: High cutoff for bandpass filter
            ema_alpha: Exponential moving average smoothing factor
            enable_velocity: Whether to enable velocity tracking
            velocity_smoothing: EMA alpha for velocity smoothing
        """
        self.fs = fs
        self.n_channels = n_channels
        self.grid_size = int(np.sqrt(n_channels))
        self.ema_alpha = ema_alpha

        # Setup bandpass filter
        self.sos = butter(
            4, [bandpass_low, bandpass_high], btype="band", fs=fs, output="sos"
        )
        n_sections = self.sos.shape[0]
        self.zi = np.zeros((n_sections, 2, n_channels))

        # EMA state
        self.prev_smooth: np.ndarray | None = None

        # Setup device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MedicalUNet().to(self.device)

        # Load model weights
        model_path = Path(model_path)
        if model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            self._model_loaded = True
        else:
            self._model_loaded = False
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Please train the MedicalUNet model first."
            )

        # Velocity tracker
        self.enable_velocity = enable_velocity
        self.velocity_tracker = (
            VelocityTracker(smoothing_alpha=velocity_smoothing)
            if enable_velocity
            else None
        )
        self._last_velocity: VelocityResult | None = None

    def process(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process raw neural data through the encoder pipeline.

        Args:
            data: Raw neural data, shape (batch_size, n_channels)

        Returns:
            Tuple of:
            - normalized: Output from U-Net, shape (n_channels,), values 0-1
            - bad_channels: Boolean mask of bad channels, shape (n_channels,)
        """
        # 1. Bandpass filter
        filtered, self.zi = sosfilt(self.sos, data, axis=0, zi=self.zi)

        # 2. Power extraction (squared)
        power = filtered**2

        # 3. Average power across batch
        avg_power = np.mean(power, axis=0)

        # 4. Reshape to grid
        grid = avg_power.reshape(self.grid_size, self.grid_size)

        # 5. EMA smoothing
        if self.prev_smooth is None:
            self.prev_smooth = grid
        else:
            self.prev_smooth = (
                self.ema_alpha * grid + (1 - self.ema_alpha) * self.prev_smooth
            )

        # 6. Normalize to 99.5th percentile
        p99 = np.percentile(self.prev_smooth, 99.5)
        input_norm = np.clip(self.prev_smooth, 0, p99) / (p99 + 1e-9)

        # 7. U-Net inference
        tensor = torch.FloatTensor(input_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output_tensor = self.model(tensor)
            output_grid = output_tensor.squeeze().cpu().numpy()

        # 8. Velocity tracking (if enabled)
        if self.velocity_tracker is not None:
            self._last_velocity = self.velocity_tracker.process(output_grid)

        # Flatten for compatibility with tracker
        normalized = output_grid.flatten()

        # No bad channel detection in encoder mode (model handles it)
        bad_channels = np.zeros(self.n_channels, dtype=bool)

        return normalized, bad_channels

    def get_velocity(self) -> VelocityResult | None:
        """Get the last computed velocity result."""
        return self._last_velocity

    def get_noise_ratio(self) -> float:
        """Return noise ratio estimate. Encoder mode doesn't compute this."""
        return 0.0

    def reset(self):
        """Reset filter state and velocity tracker."""
        n_sections = self.sos.shape[0]
        self.zi = np.zeros((n_sections, 2, self.n_channels))
        self.prev_smooth = None
        if self.velocity_tracker is not None:
            self.velocity_tracker.reset()
        self._last_velocity = None
