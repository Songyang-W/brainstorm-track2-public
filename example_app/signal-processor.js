/**
 * Signal Processor Module
 * Implements real-time signal processing for neural data
 * - Bandpass filtering (70-150 Hz high-gamma band)
 * - Power computation
 * - Temporal smoothing (EMA)
 * - Spatial smoothing (Gaussian)
 */

class SignalProcessor {
    constructor(config = {}) {
        this.fs = config.fs || 500;  // Sampling frequency
        this.nChannels = config.nChannels || 1024;
        this.gridSize = config.gridSize || 32;

        // Bandpass filter parameters (70-150 Hz)
        this.lowcut = config.lowcut || 70;
        this.highcut = config.highcut || 150;

        // EMA smoothing parameter
        this.alpha = config.alpha || 0.2;

        // State for temporal smoothing
        this.smoothedPower = null;

        // Pre-compute filter coefficients (Butterworth approximation for browser)
        this.initFilter();

        // Gaussian kernel for spatial smoothing
        this.spatialKernel = this.createGaussianKernel(config.spatialSigma || 1.5);
    }

    /**
     * Initialize bandpass filter coefficients
     * Using a simplified IIR filter design for browser compatibility
     */
    initFilter() {
        // Pre-computed Butterworth bandpass coefficients for 70-150Hz at 500Hz sampling
        // 4th order bandpass = cascade of 2nd order sections
        // These are approximated for the specific frequency range

        const fs = this.fs;
        const f1 = this.lowcut / (fs / 2);  // Normalized frequency
        const f2 = this.highcut / (fs / 2);

        // Simple biquad bandpass approximation
        // For more accurate filtering, a proper IIR design would be needed
        this.filterState = new Float32Array(this.nChannels * 4);  // 2 states per channel for 2nd order

        // Store filter parameters for simplified filtering
        this.centerFreq = (this.lowcut + this.highcut) / 2;
        this.bandwidth = this.highcut - this.lowcut;
        this.Q = this.centerFreq / this.bandwidth;

        // Biquad coefficients for bandpass
        const w0 = 2 * Math.PI * this.centerFreq / fs;
        const alpha = Math.sin(w0) / (2 * this.Q);

        this.b0 = alpha;
        this.b1 = 0;
        this.b2 = -alpha;
        this.a0 = 1 + alpha;
        this.a1 = -2 * Math.cos(w0);
        this.a2 = 1 - alpha;

        // Normalize coefficients
        this.b0 /= this.a0;
        this.b1 /= this.a0;
        this.b2 /= this.a0;
        this.a1 /= this.a0;
        this.a2 /= this.a0;
    }

    /**
     * Create Gaussian kernel for spatial smoothing
     */
    createGaussianKernel(sigma) {
        const size = Math.ceil(sigma * 3) * 2 + 1;
        const kernel = [];
        const center = Math.floor(size / 2);
        let sum = 0;

        for (let i = 0; i < size; i++) {
            kernel[i] = [];
            for (let j = 0; j < size; j++) {
                const dx = i - center;
                const dy = j - center;
                const value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel[i][j] = value;
                sum += value;
            }
        }

        // Normalize
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                kernel[i][j] /= sum;
            }
        }

        return kernel;
    }

    /**
     * Apply bandpass filter to a batch of samples
     * @param {Array<Array<number>>} batch - Array of samples, each with nChannels values
     * @returns {Array<Array<number>>} Filtered samples
     */
    applyBandpass(batch) {
        const filtered = [];

        for (let s = 0; s < batch.length; s++) {
            const sample = batch[s];
            const filteredSample = new Float32Array(this.nChannels);

            for (let ch = 0; ch < this.nChannels; ch++) {
                const stateIdx = ch * 4;
                const x = sample[ch];

                // IIR filter: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
                const x1 = this.filterState[stateIdx];
                const x2 = this.filterState[stateIdx + 1];
                const y1 = this.filterState[stateIdx + 2];
                const y2 = this.filterState[stateIdx + 3];

                const y = this.b0 * x + this.b1 * x1 + this.b2 * x2 - this.a1 * y1 - this.a2 * y2;

                // Update state
                this.filterState[stateIdx + 1] = x1;
                this.filterState[stateIdx] = x;
                this.filterState[stateIdx + 3] = y1;
                this.filterState[stateIdx + 2] = y;

                filteredSample[ch] = y;
            }

            filtered.push(filteredSample);
        }

        return filtered;
    }

    /**
     * Compute power (squared amplitude) and average across batch
     * @param {Array<Array<number>>} filteredBatch - Filtered samples
     * @returns {Float32Array} Average power per channel
     */
    computePower(filteredBatch) {
        const power = new Float32Array(this.nChannels);

        for (let s = 0; s < filteredBatch.length; s++) {
            const sample = filteredBatch[s];
            for (let ch = 0; ch < this.nChannels; ch++) {
                power[ch] += sample[ch] * sample[ch];
            }
        }

        // Average
        for (let ch = 0; ch < this.nChannels; ch++) {
            power[ch] /= filteredBatch.length;
        }

        return power;
    }

    /**
     * Apply exponential moving average smoothing
     * @param {Float32Array} power - Current power values
     * @returns {Float32Array} Smoothed power
     */
    applyTemporalSmoothing(power) {
        if (!this.smoothedPower) {
            this.smoothedPower = new Float32Array(power);
        } else {
            for (let ch = 0; ch < this.nChannels; ch++) {
                this.smoothedPower[ch] = this.alpha * power[ch] + (1 - this.alpha) * this.smoothedPower[ch];
            }
        }
        return this.smoothedPower;
    }

    /**
     * Reshape flat array to 32x32 grid
     * @param {Float32Array} flat - 1024-element array
     * @returns {Array<Array<number>>} 32x32 grid
     */
    reshapeToGrid(flat) {
        const grid = [];
        for (let row = 0; row < this.gridSize; row++) {
            grid[row] = [];
            for (let col = 0; col < this.gridSize; col++) {
                const idx = row * this.gridSize + col;
                grid[row][col] = flat[idx];
            }
        }
        return grid;
    }

    /**
     * Apply Gaussian spatial smoothing
     * @param {Array<Array<number>>} grid - 32x32 grid
     * @returns {Array<Array<number>>} Smoothed grid
     */
    applySpatialSmoothing(grid) {
        const kernel = this.spatialKernel;
        const kSize = kernel.length;
        const kCenter = Math.floor(kSize / 2);
        const smoothed = [];

        for (let row = 0; row < this.gridSize; row++) {
            smoothed[row] = [];
            for (let col = 0; col < this.gridSize; col++) {
                let sum = 0;
                let weightSum = 0;

                for (let ki = 0; ki < kSize; ki++) {
                    for (let kj = 0; kj < kSize; kj++) {
                        const r = row + ki - kCenter;
                        const c = col + kj - kCenter;

                        if (r >= 0 && r < this.gridSize && c >= 0 && c < this.gridSize) {
                            sum += grid[r][c] * kernel[ki][kj];
                            weightSum += kernel[ki][kj];
                        }
                    }
                }

                smoothed[row][col] = sum / weightSum;
            }
        }

        return smoothed;
    }

    /**
     * Normalize grid values to 0-1 range
     * @param {Array<Array<number>>} grid - Input grid
     * @returns {{grid: Array<Array<number>>, min: number, max: number}} Normalized grid and range
     */
    normalizeGrid(grid) {
        let min = Infinity;
        let max = -Infinity;

        // Find min/max
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                const val = grid[row][col];
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }

        // Use percentile-based normalization for robustness
        const values = [];
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                values.push(grid[row][col]);
            }
        }
        values.sort((a, b) => a - b);

        const p05 = values[Math.floor(values.length * 0.05)];
        const p95 = values[Math.floor(values.length * 0.95)];

        const range = p95 - p05 || 1;

        // Normalize
        const normalized = [];
        for (let row = 0; row < this.gridSize; row++) {
            normalized[row] = [];
            for (let col = 0; col < this.gridSize; col++) {
                const val = (grid[row][col] - p05) / range;
                normalized[row][col] = Math.max(0, Math.min(1, val));
            }
        }

        return { grid: normalized, min: p05, max: p95 };
    }

    /**
     * Main processing pipeline
     * @param {Array<Array<number>>} batch - Raw neural data batch
     * @returns {{grid: Array<Array<number>>, normalized: Array<Array<number>>, stats: Object}}
     */
    processBatch(batch) {
        // 1. Bandpass filter
        const filtered = this.applyBandpass(batch);

        // 2. Compute power
        const power = this.computePower(filtered);

        // 3. Temporal smoothing
        const smoothed = this.applyTemporalSmoothing(power);

        // 4. Reshape to grid
        const grid = this.reshapeToGrid(smoothed);

        // 5. Spatial smoothing
        const spatialSmoothed = this.applySpatialSmoothing(grid);

        // 6. Normalize
        const { grid: normalized, min, max } = this.normalizeGrid(spatialSmoothed);

        return {
            grid: spatialSmoothed,
            normalized: normalized,
            stats: {
                min: min,
                max: max,
                mean: smoothed.reduce((a, b) => a + b, 0) / smoothed.length
            }
        };
    }

    /**
     * Reset filter state (call when reconnecting)
     */
    reset() {
        this.filterState = new Float32Array(this.nChannels * 4);
        this.smoothedPower = null;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SignalProcessor;
}
