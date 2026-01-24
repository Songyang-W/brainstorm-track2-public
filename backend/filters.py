"""
Signal processing filters for BCI neural data.

Implements:
- 60Hz notch filter to remove line noise
- 70-150Hz bandpass filter for high-gamma extraction
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi


class NotchFilter:
    """60Hz notch filter to remove power line interference."""

    def __init__(self, freq: float = 60.0, q: float = 30.0, fs: float = 500.0):
        """
        Initialize notch filter.

        Args:
            freq: Notch frequency in Hz
            q: Quality factor (higher = narrower notch)
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.freq = freq
        self.q = q

        # Design notch filter
        self.b, self.a = iirnotch(freq, q, fs)

        # Convert to second-order sections for numerical stability
        # For a simple notch, we can use the b, a form directly
        self.n_channels = None
        self.zi = None

    def initialize(self, n_channels: int):
        """Initialize filter state for given number of channels."""
        self.n_channels = n_channels
        # Initial conditions for steady-state
        # For each channel, we need filter state
        from scipy.signal import lfilter_zi

        zi_single = lfilter_zi(self.b, self.a)
        self.zi = np.tile(zi_single, (n_channels, 1))

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to data.

        Args:
            data: Shape (n_samples, n_channels)

        Returns:
            Filtered data, same shape
        """
        from scipy.signal import lfilter

        if self.zi is None or self.n_channels != data.shape[1]:
            self.initialize(data.shape[1])

        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch], self.zi[ch] = lfilter(
                self.b, self.a, data[:, ch], zi=self.zi[ch]
            )

        return filtered


class BandpassFilter:
    """70-150Hz bandpass filter for high-gamma band extraction."""

    def __init__(
        self,
        low_freq: float = 70.0,
        high_freq: float = 150.0,
        order: int = 4,
        fs: float = 500.0,
    ):
        """
        Initialize bandpass filter.

        Args:
            low_freq: Lower cutoff frequency in Hz
            high_freq: Upper cutoff frequency in Hz
            order: Filter order
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order

        # Design bandpass filter using second-order sections
        self.sos = butter(
            order, [low_freq, high_freq], btype="band", fs=fs, output="sos"
        )

        self.n_channels = None
        self.zi = None

    def initialize(self, n_channels: int):
        """Initialize filter state for given number of channels."""
        self.n_channels = n_channels
        # Get initial conditions for each channel
        zi_single = sosfilt_zi(self.sos)
        # zi_single shape: (n_sections, 2)
        # We need (n_channels, n_sections, 2)
        self.zi = np.tile(zi_single[np.newaxis, :, :], (n_channels, 1, 1))

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to data.

        Args:
            data: Shape (n_samples, n_channels)

        Returns:
            Filtered data, same shape
        """
        if self.zi is None or self.n_channels != data.shape[1]:
            self.initialize(data.shape[1])

        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch], self.zi[ch] = sosfilt(
                self.sos, data[:, ch], zi=self.zi[ch]
            )

        return filtered


class FilterPipeline:
    """Combined filter pipeline: notch + bandpass."""

    def __init__(
        self,
        fs: float = 500.0,
        notch_freq: float = 60.0,
        notch_q: float = 30.0,
        bandpass_low: float = 70.0,
        bandpass_high: float = 150.0,
        bandpass_order: int = 4,
    ):
        """
        Initialize filter pipeline.

        Args:
            fs: Sampling frequency
            notch_freq: Notch filter frequency
            notch_q: Notch filter quality factor
            bandpass_low: Bandpass lower cutoff
            bandpass_high: Bandpass upper cutoff
            bandpass_order: Bandpass filter order
        """
        self.notch_60 = NotchFilter(freq=notch_freq, q=notch_q, fs=fs)
        # Also notch 120Hz harmonic (within our 70-150Hz band)
        self.notch_120 = NotchFilter(freq=120.0, q=notch_q, fs=fs)
        self.bandpass = BandpassFilter(
            low_freq=bandpass_low,
            high_freq=bandpass_high,
            order=bandpass_order,
            fs=fs,
        )

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply full filter pipeline.

        Args:
            data: Shape (n_samples, n_channels)

        Returns:
            Filtered data
        """
        # Remove 60Hz line noise
        data = self.notch_60.process(data)
        # Remove 120Hz harmonic (within our 70-150Hz band)
        data = self.notch_120.process(data)
        # Then extract high-gamma band
        data = self.bandpass.process(data)
        return data
