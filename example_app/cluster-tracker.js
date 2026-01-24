/**
 * Enhanced Cluster Tracker Module v2
 *
 * Key improvement: Tracks CLUSTER ENTITIES with anchored peak positions.
 * When hotspots turn off, their positions are remembered and the cluster
 * centroid remains stable based on ALL known peak positions.
 *
 * Example: If a cluster has 4 hotspots but only 2 are active at a time,
 * the center stays in the middle of all 4, not just the 2 active ones.
 */

/**
 * Represents a single anchored peak within a cluster
 */
class AnchoredPeak {
    constructor(row, col, intensity) {
        this.row = row;
        this.col = col;
        this.intensity = intensity;
        this.maxIntensity = intensity;  // Historical max
        this.active = true;
        this.lastActive = Date.now();
        this.createdAt = Date.now();
        this.activationCount = 1;  // How many times this peak has been seen
    }

    /**
     * Update peak with new observation
     */
    update(row, col, intensity) {
        // Smooth position update (anchoring)
        const smoothing = 0.3;
        this.row = smoothing * row + (1 - smoothing) * this.row;
        this.col = smoothing * col + (1 - smoothing) * this.col;

        this.intensity = intensity;
        this.maxIntensity = Math.max(this.maxIntensity, intensity);
        this.active = true;
        this.lastActive = Date.now();
        this.activationCount++;
    }

    /**
     * Mark as inactive (decays but position remembered)
     */
    deactivate() {
        this.active = false;
        this.intensity *= 0.95;
    }

    /**
     * Get weight for centroid calculation
     * Based on historical importance and recency
     */
    getWeight() {
        const now = Date.now();
        const recencyMs = now - this.lastActive;

        // Recency factor: full weight if seen in last 500ms, decays over 5 seconds
        const recencyFactor = Math.exp(-recencyMs / 5000);

        // Confidence from activation count (more observations = more reliable)
        const confidenceFactor = Math.min(1, this.activationCount / 10);

        // Combined weight
        return this.maxIntensity * recencyFactor * (0.5 + 0.5 * confidenceFactor);
    }

    /**
     * Check if peak should be pruned
     */
    shouldPrune(timeout = 10000) {
        const age = Date.now() - this.lastActive;
        return age > timeout && this.intensity < 0.1;
    }
}

/**
 * Represents a cluster entity containing multiple anchored peaks
 */
class ClusterEntity {
    constructor(id) {
        this.id = id;
        this.peaks = [];  // AnchoredPeak instances
        this.centroid = { row: 0, col: 0 };
        this.smoothedCentroid = null;  // For extra stability
        this.createdAt = Date.now();
        this.lastSeen = Date.now();
        this.confidence = 0.3;
    }

    /**
     * Add a new anchored peak to this cluster
     */
    addPeak(row, col, intensity) {
        const peak = new AnchoredPeak(row, col, intensity);
        this.peaks.push(peak);
        this.recomputeCentroid();
        return peak;
    }

    /**
     * Find the best matching anchored peak for a new observation
     */
    findMatchingPeak(row, col, maxDistance = 4) {
        let bestMatch = null;
        let bestScore = Infinity;

        for (const peak of this.peaks) {
            const dist = Math.sqrt(
                Math.pow(row - peak.row, 2) +
                Math.pow(col - peak.col, 2)
            );

            if (dist < maxDistance && dist < bestScore) {
                bestScore = dist;
                bestMatch = peak;
            }
        }

        return bestMatch;
    }

    /**
     * Update cluster with new frame's peaks
     * Returns unmatched peaks that might belong elsewhere
     */
    updateWithPeaks(newPeaks) {
        const matched = new Set();
        const unmatched = [];

        // First pass: match new peaks to existing anchored peaks
        for (const newPeak of newPeaks) {
            const match = this.findMatchingPeak(newPeak.row, newPeak.col);

            if (match) {
                match.update(newPeak.row, newPeak.col, newPeak.intensity);
                matched.add(match);
            } else {
                unmatched.push(newPeak);
            }
        }

        // Mark unmatched anchored peaks as inactive
        for (const peak of this.peaks) {
            if (!matched.has(peak)) {
                peak.deactivate();
            }
        }

        // Update cluster state
        if (matched.size > 0) {
            this.lastSeen = Date.now();
            this.confidence = Math.min(1, this.confidence + 0.05);
        } else {
            this.confidence *= 0.98;
        }

        // Recompute centroid from ALL peaks (active and inactive)
        this.recomputeCentroid();

        // Prune very old peaks
        this.peaks = this.peaks.filter(p => !p.shouldPrune());

        return unmatched;
    }

    /**
     * Compute centroid from ALL anchored peaks (the key improvement!)
     */
    recomputeCentroid() {
        if (this.peaks.length === 0) {
            return;
        }

        let totalWeight = 0;
        let weightedRow = 0;
        let weightedCol = 0;

        for (const peak of this.peaks) {
            const weight = peak.getWeight();
            weightedRow += peak.row * weight;
            weightedCol += peak.col * weight;
            totalWeight += weight;
        }

        if (totalWeight > 0.01) {
            const newCentroid = {
                row: weightedRow / totalWeight,
                col: weightedCol / totalWeight
            };

            // Smooth centroid updates for stability
            if (this.smoothedCentroid === null) {
                this.smoothedCentroid = { ...newCentroid };
            } else {
                const smoothing = 0.2;
                this.smoothedCentroid.row =
                    smoothing * newCentroid.row + (1 - smoothing) * this.smoothedCentroid.row;
                this.smoothedCentroid.col =
                    smoothing * newCentroid.col + (1 - smoothing) * this.smoothedCentroid.col;
            }

            this.centroid = this.smoothedCentroid;
        }
    }

    /**
     * Get bounding box of all peaks
     */
    getBoundingBox() {
        if (this.peaks.length === 0) return null;

        let minRow = Infinity, maxRow = -Infinity;
        let minCol = Infinity, maxCol = -Infinity;

        for (const peak of this.peaks) {
            minRow = Math.min(minRow, peak.row);
            maxRow = Math.max(maxRow, peak.row);
            minCol = Math.min(minCol, peak.col);
            maxCol = Math.max(maxCol, peak.col);
        }

        return { minRow, maxRow, minCol, maxCol };
    }

    /**
     * Get active peak count
     */
    getActivePeakCount() {
        return this.peaks.filter(p => p.active).length;
    }

    /**
     * Get total intensity
     */
    getTotalIntensity() {
        return this.peaks.reduce((sum, p) => sum + p.intensity, 0);
    }

    /**
     * Check if cluster should be removed
     */
    shouldPrune(timeout = 15000) {
        const age = Date.now() - this.lastSeen;
        return age > timeout || this.peaks.length === 0;
    }

    /**
     * Check if a point is within this cluster's region
     */
    containsPoint(row, col, margin = 5) {
        const box = this.getBoundingBox();
        if (!box) return false;

        return row >= box.minRow - margin && row <= box.maxRow + margin &&
               col >= box.minCol - margin && col <= box.maxCol + margin;
    }

    /**
     * Distance from point to cluster centroid
     */
    distanceToCentroid(row, col) {
        return Math.sqrt(
            Math.pow(row - this.centroid.row, 2) +
            Math.pow(col - this.centroid.col, 2)
        );
    }
}

/**
 * Main Cluster Tracker
 */
class ClusterTracker {
    constructor(config = {}) {
        this.gridSize = config.gridSize || 32;
        this.threshold = config.threshold || 0.35;
        this.minClusterSize = config.minClusterSize || 3;

        // Cluster entities
        this.clusters = [];
        this.nextClusterId = 1;

        // Legacy compatibility: maintain peakPositions array for visualizer
        this.peakPositions = [];

        // Hotspot memory for visualization
        this.hotspotMemory = new Float32Array(this.gridSize * this.gridSize);
        this.hotspotDecay = config.hotspotDecay || 0.98;

        // Tracked center (primary cluster)
        this.trackedCenter = null;
        this.centerSmoothing = config.centerSmoothing || 0.15;

        // Movement inference
        this.prevCenter = null;
        this.movementHistory = [];
        this.movementHistoryLength = 20;

        // Config
        this.peakMergeDistance = config.peakMergeDistance || 3;
        this.clusterMergeDistance = config.clusterMergeDistance || 8;
        this.peakTimeout = config.peakTimeout || 10000;
    }

    /**
     * Detect local maxima (peaks) in the current grid
     */
    detectPeaks(grid) {
        const peaks = [];

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
                    peaks.push({ row, col, intensity: val });
                }
            }
        }

        return peaks;
    }

    /**
     * Assign peaks to existing clusters or create new ones
     */
    assignPeaksToClusters(peaks) {
        const assignedPeaks = new Set();
        const peakAssignments = new Map();  // peak -> cluster

        // Sort clusters by confidence (process strongest first)
        const sortedClusters = [...this.clusters].sort(
            (a, b) => b.confidence - a.confidence
        );

        // First pass: assign peaks to nearby clusters
        for (const cluster of sortedClusters) {
            const nearbyPeaks = peaks.filter(p => {
                if (assignedPeaks.has(p)) return false;

                // Check if peak is near cluster centroid or any anchored peak
                const distToCentroid = cluster.distanceToCentroid(p.row, p.col);
                if (distToCentroid < 12) return true;

                // Or near any existing peak in cluster
                const match = cluster.findMatchingPeak(p.row, p.col, 5);
                return match !== null;
            });

            // Update cluster with nearby peaks
            const unmatched = cluster.updateWithPeaks(nearbyPeaks);

            // Mark peaks as assigned
            for (const p of nearbyPeaks) {
                if (!unmatched.includes(p)) {
                    assignedPeaks.add(p);
                    peakAssignments.set(p, cluster);
                }
            }

            // Add new peaks that are near centroid but didn't match existing anchors
            for (const p of unmatched) {
                if (cluster.distanceToCentroid(p.row, p.col) < 8) {
                    cluster.addPeak(p.row, p.col, p.intensity);
                    assignedPeaks.add(p);
                    peakAssignments.set(p, cluster);
                }
            }
        }

        // Second pass: create new clusters for unassigned peaks
        const unassignedPeaks = peaks.filter(p => !assignedPeaks.has(p));
        this.createClustersFromPeaks(unassignedPeaks);

        // Merge overlapping clusters
        this.mergeClusters();

        // Prune dead clusters
        this.clusters = this.clusters.filter(c => !c.shouldPrune());
    }

    /**
     * Create new clusters from unassigned peaks using connected components
     */
    createClustersFromPeaks(peaks) {
        if (peaks.length === 0) return;

        // Group nearby peaks
        const groups = [];
        const used = new Set();

        for (const peak of peaks) {
            if (used.has(peak)) continue;

            const group = [peak];
            used.add(peak);

            // Find all peaks connected to this one
            let i = 0;
            while (i < group.length) {
                const current = group[i];

                for (const other of peaks) {
                    if (used.has(other)) continue;

                    const dist = Math.sqrt(
                        Math.pow(current.row - other.row, 2) +
                        Math.pow(current.col - other.col, 2)
                    );

                    if (dist < 6) {
                        group.push(other);
                        used.add(other);
                    }
                }
                i++;
            }

            if (group.length >= 1) {
                groups.push(group);
            }
        }

        // Create cluster for each group
        for (const group of groups) {
            const cluster = new ClusterEntity(this.nextClusterId++);

            for (const peak of group) {
                cluster.addPeak(peak.row, peak.col, peak.intensity);
            }

            this.clusters.push(cluster);
        }
    }

    /**
     * Merge clusters that have grown together
     */
    mergeClusters() {
        const merged = new Set();

        for (let i = 0; i < this.clusters.length; i++) {
            if (merged.has(i)) continue;

            for (let j = i + 1; j < this.clusters.length; j++) {
                if (merged.has(j)) continue;

                const c1 = this.clusters[i];
                const c2 = this.clusters[j];

                // Check if centroids are close
                const dist = Math.sqrt(
                    Math.pow(c1.centroid.row - c2.centroid.row, 2) +
                    Math.pow(c1.centroid.col - c2.centroid.col, 2)
                );

                if (dist < this.clusterMergeDistance) {
                    // Merge c2 into c1
                    for (const peak of c2.peaks) {
                        // Check if peak already exists in c1
                        const existing = c1.findMatchingPeak(peak.row, peak.col, 2);
                        if (!existing) {
                            c1.peaks.push(peak);
                        }
                    }
                    c1.confidence = Math.max(c1.confidence, c2.confidence);
                    c1.recomputeCentroid();
                    merged.add(j);
                }
            }
        }

        // Remove merged clusters
        this.clusters = this.clusters.filter((_, i) => !merged.has(i));
    }

    /**
     * Update hotspot memory for visualization
     */
    updateHotspotMemory(normalizedGrid) {
        // Decay existing memory
        for (let i = 0; i < this.hotspotMemory.length; i++) {
            this.hotspotMemory[i] *= this.hotspotDecay;
        }

        // Add current activity
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                const idx = row * this.gridSize + col;
                const currentVal = normalizedGrid[row][col];
                this.hotspotMemory[idx] = Math.max(this.hotspotMemory[idx], currentVal);
            }
        }
    }

    /**
     * Get memory grid for visualization
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
     * Main entry point: compute cluster center from normalized grid
     */
    computeClusterCenter(normalizedGrid) {
        // Update hotspot memory
        this.updateHotspotMemory(normalizedGrid);

        // Detect peaks in current frame
        const peaks = this.detectPeaks(normalizedGrid);

        // Assign to clusters
        this.assignPeaksToClusters(peaks);

        // Update legacy peakPositions for visualizer compatibility
        this.updatePeakPositions();

        // Find primary cluster (strongest/most confident)
        const primaryCluster = this.getPrimaryCluster();

        if (!primaryCluster) {
            this.trackedCenter = null;
            return null;
        }

        // Use the cluster's stable centroid
        const center = primaryCluster.centroid;

        // Additional smoothing for tracked center
        if (this.trackedCenter === null) {
            this.trackedCenter = { ...center };
        } else {
            this.trackedCenter.row =
                this.centerSmoothing * center.row +
                (1 - this.centerSmoothing) * this.trackedCenter.row;
            this.trackedCenter.col =
                this.centerSmoothing * center.col +
                (1 - this.centerSmoothing) * this.trackedCenter.col;
        }

        return {
            row: this.trackedCenter.row,
            col: this.trackedCenter.col,
            peakCount: primaryCluster.peaks.length,
            activePeaks: primaryCluster.getActivePeakCount(),
            clusterId: primaryCluster.id,
            clusterConfidence: primaryCluster.confidence
        };
    }

    /**
     * Get the primary (strongest) cluster
     */
    getPrimaryCluster() {
        if (this.clusters.length === 0) return null;

        // Score by: confidence * total intensity * peak count
        return this.clusters.reduce((best, cluster) => {
            const score = cluster.confidence *
                          cluster.getTotalIntensity() *
                          Math.sqrt(cluster.peaks.length);

            const bestScore = best ?
                best.confidence * best.getTotalIntensity() * Math.sqrt(best.peaks.length) : 0;

            return score > bestScore ? cluster : best;
        }, null);
    }

    /**
     * Update legacy peakPositions array for visualizer
     */
    updatePeakPositions() {
        this.peakPositions = [];

        for (const cluster of this.clusters) {
            for (const peak of cluster.peaks) {
                this.peakPositions.push({
                    row: peak.row,
                    col: peak.col,
                    intensity: peak.intensity,
                    active: peak.active,
                    clusterId: cluster.id,
                    lastSeen: peak.lastActive
                });
            }
        }
    }

    /**
     * Get all cluster entities (for global map)
     */
    getClusters() {
        return this.clusters;
    }

    /**
     * Infer array movement from center movement
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
            dx: -avgDCol,
            dy: -avgDRow,
            confidence: Math.min(1, magnitude * 2)
        };
    }

    /**
     * Get distance from array center to tracked cluster center
     */
    getDistanceToCenter() {
        if (!this.trackedCenter) return null;

        const arrayCenter = (this.gridSize - 1) / 2;
        const dx = this.trackedCenter.col - arrayCenter;
        const dy = this.trackedCenter.row - arrayCenter;

        return {
            dx,
            dy,
            distance: Math.sqrt(dx * dx + dy * dy),
            distanceMm: Math.sqrt(dx * dx + dy * dy) * 0.3
        };
    }

    /**
     * Get guidance direction for surgeon
     */
    getGuidanceDirection() {
        const dist = this.getDistanceToCenter();
        if (!dist) return null;

        // To center the cluster, move array OPPOSITE to where cluster is
        const moveX = -dist.dx;
        const moveY = -dist.dy;

        const angle = Math.atan2(-moveY, moveX) * 180 / Math.PI;
        const magnitude = dist.distance / 16;

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
        this.clusters = [];
        this.nextClusterId = 1;
        this.peakPositions = [];
        this.hotspotMemory = new Float32Array(this.gridSize * this.gridSize);
        this.trackedCenter = null;
        this.prevCenter = null;
        this.movementHistory = [];
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ClusterTracker, ClusterEntity, AnchoredPeak };
}
