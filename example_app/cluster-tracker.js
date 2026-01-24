/**
 * Enhanced Cluster Tracker Module
 *
 * Key improvement: Tracks hotspot POSITIONS even when they turn off.
 * The center of a cluster with 4 hotspots should be in the middle of all 4,
 * even when only 2 are currently active.
 *
 * Uses a persistent hotspot map that decays slowly, allowing the system
 * to remember where activity was seen before.
 */

class ClusterTracker {
    constructor(config = {}) {
        this.gridSize = config.gridSize || 32;
        this.threshold = config.threshold || 0.35;
        this.minClusterSize = config.minClusterSize || 3;

        // Persistent hotspot memory - decays slowly to remember off hotspots
        this.hotspotMemory = new Float32Array(this.gridSize * this.gridSize);
        this.hotspotDecay = config.hotspotDecay || 0.98;  // Slow decay

        // Peak detection - remember local maxima positions
        this.peakPositions = [];  // [{row, col, intensity, lastSeen}]
        this.peakMergeDistance = config.peakMergeDistance || 3;
        this.peakTimeout = config.peakTimeout || 3000;  // ms before forgetting a peak

        // Cluster center tracking with memory
        this.trackedCenter = null;
        this.centerSmoothing = config.centerSmoothing || 0.15;

        // Movement inference
        this.prevCenter = null;
        this.movementHistory = [];
        this.movementHistoryLength = 20;
    }

    /**
     * Update hotspot memory with current activity
     * Memory decays but retains positions of previously active areas
     */
    updateHotspotMemory(normalizedGrid) {
        const now = Date.now();

        // Decay existing memory
        for (let i = 0; i < this.hotspotMemory.length; i++) {
            this.hotspotMemory[i] *= this.hotspotDecay;
        }

        // Add current activity to memory (max, not replace)
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                const idx = row * this.gridSize + col;
                const currentVal = normalizedGrid[row][col];

                // Memory is the max of decayed old value and current value
                this.hotspotMemory[idx] = Math.max(
                    this.hotspotMemory[idx],
                    currentVal
                );
            }
        }

        // Detect peaks (local maxima) in current frame
        this.updatePeakPositions(normalizedGrid, now);
    }

    /**
     * Find and track local maxima (peaks) in the activity
     */
    updatePeakPositions(grid, now) {
        const newPeaks = [];

        // Find local maxima in current grid
        for (let row = 1; row < this.gridSize - 1; row++) {
            for (let col = 1; col < this.gridSize - 1; col++) {
                const val = grid[row][col];

                if (val < this.threshold) continue;

                // Check if local maximum (8-connected)
                let isMax = true;
                for (let dr = -1; dr <= 1 && isMax; dr++) {
                    for (let dc = -1; dc <= 1 && isMax; dc++) {
                        if (dr === 0 && dc === 0) continue;
                        if (grid[row + dr][col + dc] > val) {
                            isMax = false;
                        }
                    }
                }

                if (isMax) {
                    newPeaks.push({ row, col, intensity: val });
                }
            }
        }

        // Merge new peaks with existing tracked peaks
        for (const newPeak of newPeaks) {
            let merged = false;

            for (const existing of this.peakPositions) {
                const dist = Math.sqrt(
                    Math.pow(newPeak.row - existing.row, 2) +
                    Math.pow(newPeak.col - existing.col, 2)
                );

                if (dist < this.peakMergeDistance) {
                    // Update existing peak with weighted average
                    const w = 0.3;
                    existing.row = w * newPeak.row + (1 - w) * existing.row;
                    existing.col = w * newPeak.col + (1 - w) * existing.col;
                    existing.intensity = Math.max(existing.intensity * 0.9, newPeak.intensity);
                    existing.lastSeen = now;
                    merged = true;
                    break;
                }
            }

            if (!merged) {
                this.peakPositions.push({
                    row: newPeak.row,
                    col: newPeak.col,
                    intensity: newPeak.intensity,
                    lastSeen: now
                });
            }
        }

        // Decay intensity of peaks not seen recently
        for (const peak of this.peakPositions) {
            if (now - peak.lastSeen > 100) {
                peak.intensity *= 0.95;
            }
        }

        // Remove old peaks
        this.peakPositions = this.peakPositions.filter(p =>
            now - p.lastSeen < this.peakTimeout && p.intensity > 0.1
        );
    }

    /**
     * Get the memory grid (for visualization)
     */
    getMemoryGrid() {
        const grid = [];
        for (let row = 0; row < this.gridSize; row++) {
            grid[row] = [];
            for (let col = 0; col < this.gridSize; col++) {
                grid[row][col] = this.hotspotMemory[row * this.gridSize + col];
            }
        }
        return grid;
    }

    /**
     * Compute cluster center using BOTH current activity AND remembered peaks
     * This ensures the center stays stable even when some hotspots turn off
     */
    computeClusterCenter(normalizedGrid) {
        this.updateHotspotMemory(normalizedGrid);

        // Use peaks for center calculation (includes remembered positions)
        if (this.peakPositions.length === 0) {
            return null;
        }

        // Weighted centroid of all tracked peaks
        let totalWeight = 0;
        let weightedRow = 0;
        let weightedCol = 0;

        for (const peak of this.peakPositions) {
            // Weight by intensity (brighter peaks matter more)
            const weight = peak.intensity * peak.intensity;
            weightedRow += peak.row * weight;
            weightedCol += peak.col * weight;
            totalWeight += weight;
        }

        if (totalWeight < 0.01) {
            return null;
        }

        const rawCenter = {
            row: weightedRow / totalWeight,
            col: weightedCol / totalWeight
        };

        // Smooth the center position
        if (this.trackedCenter === null) {
            this.trackedCenter = rawCenter;
        } else {
            this.trackedCenter.row =
                this.centerSmoothing * rawCenter.row +
                (1 - this.centerSmoothing) * this.trackedCenter.row;
            this.trackedCenter.col =
                this.centerSmoothing * rawCenter.col +
                (1 - this.centerSmoothing) * this.trackedCenter.col;
        }

        return {
            row: this.trackedCenter.row,
            col: this.trackedCenter.col,
            peakCount: this.peakPositions.length,
            activePeaks: this.peakPositions.filter(p => Date.now() - p.lastSeen < 200).length
        };
    }

    /**
     * Detect active clusters in current frame (for visualization)
     */
    detectActiveClusters(normalizedGrid) {
        // Binary mask of current activity
        const mask = [];
        const visited = [];
        for (let i = 0; i < this.gridSize; i++) {
            mask[i] = [];
            visited[i] = [];
            for (let j = 0; j < this.gridSize; j++) {
                mask[i][j] = normalizedGrid[i][j] >= this.threshold ? 1 : 0;
                visited[i][j] = false;
            }
        }

        const clusters = [];

        // Flood fill to find connected components
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                if (mask[row][col] === 1 && !visited[row][col]) {
                    const cells = this.floodFill(mask, visited, row, col);
                    if (cells.length >= this.minClusterSize) {
                        clusters.push(this.computeClusterStats(cells, normalizedGrid));
                    }
                }
            }
        }

        clusters.sort((a, b) => b.totalIntensity - a.totalIntensity);
        return clusters;
    }

    /**
     * Flood fill helper
     */
    floodFill(mask, visited, startRow, startCol) {
        const cells = [];
        const stack = [[startRow, startCol]];

        while (stack.length > 0) {
            const [row, col] = stack.pop();

            if (row < 0 || row >= this.gridSize ||
                col < 0 || col >= this.gridSize) continue;
            if (visited[row][col] || mask[row][col] === 0) continue;

            visited[row][col] = true;
            cells.push({ row, col });

            // 4-connectivity for cleaner clusters
            stack.push([row - 1, col]);
            stack.push([row + 1, col]);
            stack.push([row, col - 1]);
            stack.push([row, col + 1]);
        }

        return cells;
    }

    /**
     * Compute stats for a single cluster
     */
    computeClusterStats(cells, grid) {
        let totalIntensity = 0;
        let weightedRow = 0;
        let weightedCol = 0;
        let maxIntensity = 0;

        for (const cell of cells) {
            const intensity = grid[cell.row][cell.col];
            totalIntensity += intensity;
            weightedRow += cell.row * intensity;
            weightedCol += cell.col * intensity;
            if (intensity > maxIntensity) maxIntensity = intensity;
        }

        return {
            centroid: {
                row: weightedRow / totalIntensity,
                col: weightedCol / totalIntensity
            },
            totalIntensity,
            maxIntensity,
            area: cells.length,
            cells
        };
    }

    /**
     * Infer array movement from center movement
     * If the cluster center moves LEFT, array moved RIGHT
     */
    inferArrayMovement() {
        if (!this.trackedCenter || !this.prevCenter) {
            this.prevCenter = this.trackedCenter ?
                { ...this.trackedCenter } : null;
            return { dx: 0, dy: 0, confidence: 0 };
        }

        const dRow = this.trackedCenter.row - this.prevCenter.row;
        const dCol = this.trackedCenter.col - this.prevCenter.col;

        // Store movement
        this.movementHistory.push({ dRow, dCol, time: Date.now() });
        while (this.movementHistory.length > this.movementHistoryLength) {
            this.movementHistory.shift();
        }

        // Average recent movements
        let avgDRow = 0, avgDCol = 0;
        for (const m of this.movementHistory) {
            avgDRow += m.dRow;
            avgDCol += m.dCol;
        }
        avgDRow /= this.movementHistory.length;
        avgDCol /= this.movementHistory.length;

        // Update previous center
        this.prevCenter = { ...this.trackedCenter };

        // Invert: cluster moving left = array moving right
        const magnitude = Math.sqrt(avgDRow * avgDRow + avgDCol * avgDCol);

        return {
            dx: -avgDCol,  // Inverted
            dy: -avgDRow,  // Inverted
            confidence: Math.min(1, magnitude * 2)
        };
    }

    /**
     * Get distance from array center (16,16) to tracked cluster center
     */
    getDistanceToCenter() {
        if (!this.trackedCenter) return null;

        const arrayCenter = (this.gridSize - 1) / 2;  // 15.5
        const dx = this.trackedCenter.col - arrayCenter;
        const dy = this.trackedCenter.row - arrayCenter;

        return {
            dx,
            dy,
            distance: Math.sqrt(dx * dx + dy * dy),
            distanceMm: Math.sqrt(dx * dx + dy * dy) * 0.3  // ~0.3mm per cell
        };
    }

    /**
     * Get direction to move array to center the cluster
     */
    getGuidanceDirection() {
        const dist = this.getDistanceToCenter();
        if (!dist) return null;

        // To center the cluster, move array OPPOSITE to where cluster is
        // If cluster is to the RIGHT of center, move array LEFT
        const moveX = -dist.dx;
        const moveY = -dist.dy;

        const angle = Math.atan2(-moveY, moveX) * 180 / Math.PI;
        const magnitude = dist.distance / 16;  // Normalized to grid half-size

        return {
            angle,
            magnitude: Math.min(1, magnitude),
            distance: dist.distance,
            distanceMm: dist.distanceMm,
            isOnTarget: dist.distance < 3
        };
    }

    /**
     * Reset tracker
     */
    reset() {
        this.hotspotMemory = new Float32Array(this.gridSize * this.gridSize);
        this.peakPositions = [];
        this.trackedCenter = null;
        this.prevCenter = null;
        this.movementHistory = [];
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ClusterTracker;
}
