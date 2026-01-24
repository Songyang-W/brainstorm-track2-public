"""
Signal processing pipeline for BCI neural data.

Handles:
- Power estimation (squared signal)
- Temporal smoothing (EMA)
- Normalization (z-score + sigmoid)
- Bad channel detection
"""

import numpy as np


class PowerEstimator:
    """Estimate signal power using squared signal with EMA smoothing."""

    def __init__(self, alpha: float = 0.1, n_channels: int = 1024):
        """
        Initialize power estimator.

        Args:
            alpha: EMA smoothing factor (0-1, lower = smoother)
            n_channels: Number of channels
        """
        self.alpha = alpha
        self.n_channels = n_channels
        self.smoothed_power = np.zeros(n_channels)
        self.initialized = False

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Estimate power from filtered signal.

        Args:
            data: Shape (n_samples, n_channels), filtered signal

        Returns:
            Power estimate per channel, shape (n_channels,)
        """
        # Compute instantaneous power (squared signal)
        power = np.mean(data**2, axis=0)

        # Light EMA smoothing only to reduce frame-to-frame jitter
        if not self.initialized:
            self.smoothed_power = power.copy()
            self.initialized = True
        else:
            # EMA smoothing: alpha controls responsiveness vs stability
            self.smoothed_power = (
                self.alpha * power + (1 - self.alpha) * self.smoothed_power
            )

        return self.smoothed_power.copy()


class BadChannelDetector:
    """Detect dead and artifact channels based on signal variance."""

    def __init__(
        self,
        dead_threshold: float = 1e-10,
        artifact_std_multiplier: float = 5.0,
        n_channels: int = 1024,
        update_interval: int = 50,  # Update every N batches
    ):
        """
        Initialize bad channel detector.

        Args:
            dead_threshold: Variance threshold below which channel is dead
            artifact_std_multiplier: How many std devs above median = artifact
            n_channels: Number of channels
            update_interval: How often to recompute bad channels
        """
        self.dead_threshold = dead_threshold
        self.artifact_std_multiplier = artifact_std_multiplier
        self.n_channels = n_channels
        self.update_interval = update_interval

        self.variance_history = []
        self.bad_channels = np.zeros(n_channels, dtype=bool)
        self.dead_channels = np.zeros(n_channels, dtype=bool)
        self.artifact_channels = np.zeros(n_channels, dtype=bool)
        self.batch_count = 0

    def update(self, data: np.ndarray) -> np.ndarray:
        """
        Update bad channel detection.

        Args:
            data: Shape (n_samples, n_channels)

        Returns:
            Boolean mask of bad channels, shape (n_channels,)
        """
        self.batch_count += 1

        # Compute variance per channel for this batch
        variance = np.var(data, axis=0)

        # Accumulate variance history for robust estimation
        self.variance_history.append(variance)
        if len(self.variance_history) > 100:
            self.variance_history.pop(0)

        # Only update detection every N batches for stability
        if (
            self.batch_count % self.update_interval == 0
            and len(self.variance_history) >= 10
        ):
            # Use median variance over recent history
            var_array = np.array(self.variance_history)
            median_var = np.median(var_array, axis=0)

            # Dead channels: very low variance (no signal)
            self.dead_channels = median_var < self.dead_threshold

            # Artifact channels: variance much higher than typical
            # Use median of all channels as reference
            channel_median = np.median(median_var)
            channel_std = np.std(median_var)
            artifact_threshold = (
                channel_median + self.artifact_std_multiplier * channel_std
            )
            self.artifact_channels = median_var > artifact_threshold

            # Combined bad channels
            self.bad_channels = self.dead_channels | self.artifact_channels

        return self.bad_channels.copy()


class Normalizer:
    """Normalize power values using per-frame percentile scaling."""

    def __init__(self, n_channels: int = 1024):
        """
        Initialize normalizer.

        Args:
            n_channels: Number of channels
        """
        self.n_channels = n_channels

    def normalize(
        self, power: np.ndarray, bad_channels: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Normalize power values to 0-1 range using 5th and 95th percentile scaling.

        Maps the 5th percentile to 0 and 95th percentile to 1, then clips to [0, 1].

        Args:
            power: Power per channel, shape (n_channels,)
            bad_channels: Boolean mask of bad channels to exclude from stats

        Returns:
            Normalized values, shape (n_channels,)
        """
        # Use all channels for percentile calculation (more robust)
        if len(power) == 0:
            return np.zeros_like(power)

        # Per-frame percentile normalization
        p5 = np.percentile(power, 5)
        p95 = np.percentile(power, 95)

        range_val = p95 - p5
        if range_val < 1e-12:
            # All values nearly identical - use min/max instead
            p_min, p_max = np.min(power), np.max(power)
            if p_max - p_min < 1e-12:
                return np.full_like(power, 0.5)
            normalized = (power - p_min) / (p_max - p_min)
        else:
            # Normalize: p5 -> 0, p95 -> 1
            normalized = (power - p5) / range_val

        # Clip to [0, 1]
        normalized = np.clip(normalized, 0, 1)

        return normalized


class SignalPipeline:
    """Complete signal processing pipeline."""

    def __init__(
        self,
        n_channels: int = 1024,
        ema_alpha: float = 0.1,
        dead_threshold: float = 1e-10,
        artifact_std_multiplier: float = 5.0,
    ):
        """
        Initialize pipeline.

        Args:
            n_channels: Number of channels
            ema_alpha: EMA smoothing factor for power estimation
            dead_threshold: Variance threshold for dead channels
            artifact_std_multiplier: Std multiplier for artifact detection
        """
        self.n_channels = n_channels

        self.power_estimator = PowerEstimator(alpha=ema_alpha, n_channels=n_channels)
        self.bad_channel_detector = BadChannelDetector(
            dead_threshold=dead_threshold,
            artifact_std_multiplier=artifact_std_multiplier,
            n_channels=n_channels,
        )
        self.normalizer = Normalizer(n_channels=n_channels)

    def process(self, filtered_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process filtered data through pipeline.

        Args:
            filtered_data: Shape (n_samples, n_channels), bandpass filtered

        Returns:
            Tuple of:
                - Normalized power, shape (n_channels,)
                - Bad channel mask, shape (n_channels,)
        """
        # Detect bad channels
        bad_channels = self.bad_channel_detector.update(filtered_data)

        # Estimate power with smoothing
        power = self.power_estimator.process(filtered_data)

        # Debug logging (every 100 calls)
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count % 100 == 1:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Pipeline debug - filtered: min={filtered_data.min():.6f}, max={filtered_data.max():.6f}"
            )
            logger.info(
                f"Pipeline debug - power: min={power.min():.6f}, max={power.max():.6f}, mean={power.mean():.6f}"
            )

        # Normalize
        normalized = self.normalizer.normalize(power, bad_channels)

        if self._debug_count % 100 == 1:
            logger.info(
                f"Pipeline debug - normalized: min={normalized.min():.4f}, max={normalized.max():.4f}, mean={normalized.mean():.4f}"
            )

        return normalized, bad_channels
