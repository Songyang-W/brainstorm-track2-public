import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

class NeuralPreprocessor:
    def __init__(self, fs=500, low_cut=70, high_cut=150):
        self.fs = fs
        # Design the Bandpass Filter (High Gamma: 70-150Hz)
        nyquist = 0.5 * fs
        low = low_cut / nyquist
        high = high_cut / nyquist
        self.b, self.a = butter(N=4, Wn=[low, high], btype='bandpass')

    def process_chunk(self, raw_data):
        """
        Input: DataFrame or Numpy Array (Time x 1024 Channels)
        Output: Tensor-ready Array (Time x 1 x 32 x 32)
        """
        # 1. Bandpass Filter (Apply along time axis, axis=0)
        filtered = filtfilt(self.b, self.a, raw_data, axis=0)

        # 2. Extract Envelope (Analytic Signal Power)
        # Using Hilbert transform to get the instantaneous power
        analytic_signal = hilbert(filtered, axis=0)
        envelope = np.abs(analytic_signal)

        # 3. Reshape 1024 channels -> 32x32 Grid
        # We assume standard row-major mapping. 
        # Shape becomes: (Time_Steps, Height, Width)
        grid_data = envelope.reshape(-1, 32, 32)
        
        # 4. Normalize (Crucial for Autoencoder)
        # We scale between 0 and 1 based on this chunk's max
        # In production, use a fixed global max from calibration
        grid_data = grid_data / (np.max(grid_data) + 1e-9)
        
        # Add "Channel" dimension for PyTorch (N, Channels, H, W)
        return grid_data[:, np.newaxis, :, :]