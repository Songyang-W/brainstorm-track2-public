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
     * ENHANCED: Age-based anchoring - older peaks drift less
     */
    update(row, col, intensity) {
        // Calculate age-based anchoring strength
        const age = Date.now() - this.createdAt;
        const ageSeconds = age / 1000;

        // As peak ages, it becomes more anchored (less responsive to new positions)
        // Starts at 0.3 smoothing, decreases to 0.05 over 10 seconds
        const anchoringFactor = Math.min(0.9, ageSeconds / 10);
        const smoothing = 0.3 * (1 - anchoringFactor * 0.85);

        // Distance penalty: if observation is far from current position, trust it less
        const dist = Math.sqrt(Math.pow(row - this.row, 2) + Math.pow(col - this.col, 2));
        const distancePenalty = dist > 3 ? 0.3 : 1.0;
        const effectiveSmoothing = smoothing * distancePenalty;

        this.row = effectiveSmoothing * row + (1 - effectiveSmoothing) * this.row;
        this.col = effectiveSmoothing * col + (1 - effectiveSmoothing) * this.col;

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
     * ENHANCED: Age-based stability and distance-limited updates
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
                // Age-based anchoring: older clusters are more stable
                const age = Date.now() - this.createdAt;
                const ageSeconds = age / 1000;
                const anchoringFactor = Math.min(0.95, ageSeconds / 8);

                // Base smoothing decreases as cluster ages (more anchored)
                const baseSmoothing = 0.15;
                const smoothing = baseSmoothing * (1 - anchoringFactor * 0.9);

                // Distance penalty: large jumps are suspicious
                const dist = Math.sqrt(
                    Math.pow(newCentroid.row - this.smoothedCentroid.row, 2) +
                    Math.pow(newCentroid.col - this.smoothedCentroid.col, 2)
                );
                const distancePenalty = dist > 4 ? 0.2 : 1.0;
                const effectiveSmoothing = smoothing * distancePenalty;

                this.smoothedCentroid.row =
                    effectiveSmoothing * newCentroid.row +
                    (1 - effectiveSmoothing) * this.smoothedCentroid.row;
                this.smoothedCentroid.col =
                    effectiveSmoothing * newCentroid.col +
                    (1 - effectiveSmoothing) * this.smoothedCentroid.col;
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
        this.smoothedMovement = { dx: 0, dy: 0 };

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
     * Compute center of mass using SQUARED intensity weighting
     * IDENTICAL to hotspot_tracker.py compute_velocity() logic
     *
     * Uses weighted_grid = grid ** 2 to exaggerate bright hotspots
     * Center is at (15.5, 15.5) for a 32x32 grid
     */
    computeSquaredIntensityCOM(normalizedGrid) {
        const gridSize = this.gridSize;
        const gridCenter = (gridSize - 1) / 2;  // 15.5 for 32x32

        // Square the intensity values (key insight from hotspot_tracker.py)
        // This exaggerates bright hotspots and makes COM pull more strongly toward them
        let totalIntensity = 0;
        let weightedRow = 0;
        let weightedCol = 0;

        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                const val = normalizedGrid[row][col];
                const squaredVal = val * val;  // ** 2 weighting
                weightedRow += row * squaredVal;
                weightedCol += col * squaredVal;
                totalIntensity += squaredVal;
            }
        }

        if (totalIntensity > 0.01) {
            const avgRow = weightedRow / totalIntensity;
            const avgCol = weightedCol / totalIntensity;

            // Convert to velocity-like values (matching hotspot_tracker.py)
            // raw_vx = (avg_col - 15.5) / 15.5
            // raw_vy = -(avg_row - 15.5) / 15.5  (negative because row 0 is top)
            const rawVx = (avgCol - gridCenter) / gridCenter;
            const rawVy = -(avgRow - gridCenter) / gridCenter;

            return {
                row: avgRow,
                col: avgCol,
                vx: rawVx,
                vy: rawVy,
                totalIntensity
            };
        }

        return null;
    }

    /**
     * Main entry point: compute cluster center from normalized grid
     * Now uses SQUARED intensity weighted center of mass (matching hotspot_tracker.py)
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

        // STEP 1: Compute global squared-intensity weighted COM
        // This is IDENTICAL to hotspot_tracker.py logic
        const squaredCOM = this.computeSquaredIntensityCOM(normalizedGrid);

        if (!squaredCOM) {
            this.trackedCenter = null;
            return null;
        }

        // STEP 2: Find primary cluster near the squared COM
        // This combines the Python approach with our cluster tracking
        const primaryCluster = this.getPrimaryClusterNearCOM(squaredCOM);

        // STEP 3: If we have a cluster, use its peaks to refine the center
        // but anchor it to the squared COM for stability
        let finalRow = squaredCOM.row;
        let finalCol = squaredCOM.col;

        if (primaryCluster && primaryCluster.peaks.length > 0) {
            // Blend cluster centroid with squared COM (cluster provides structure, COM provides stability)
            const clusterWeight = 0.3;  // Prefer squared COM for stability
            finalRow = clusterWeight * primaryCluster.centroid.row + (1 - clusterWeight) * squaredCOM.row;
            finalCol = clusterWeight * primaryCluster.centroid.col + (1 - clusterWeight) * squaredCOM.col;
        }

        // STEP 4: Apply smoothing to tracked center
        if (this.trackedCenter === null) {
            this.trackedCenter = { row: finalRow, col: finalCol };
        } else {
            this.trackedCenter.row =
                this.centerSmoothing * finalRow +
                (1 - this.centerSmoothing) * this.trackedCenter.row;
            this.trackedCenter.col =
                this.centerSmoothing * finalCol +
                (1 - this.centerSmoothing) * this.trackedCenter.col;
        }

        return {
            row: this.trackedCenter.row,
            col: this.trackedCenter.col,
            vx: squaredCOM.vx,
            vy: squaredCOM.vy,
            peakCount: primaryCluster ? primaryCluster.peaks.length : 0,
            activePeaks: primaryCluster ? primaryCluster.getActivePeakCount() : 0,
            clusterId: primaryCluster ? primaryCluster.id : null,
            clusterConfidence: primaryCluster ? primaryCluster.confidence : 0,
            totalIntensity: squaredCOM.totalIntensity
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
     * Get the primary cluster near the squared-intensity COM
     * Prefers clusters close to the computed center of mass
     */
    getPrimaryClusterNearCOM(squaredCOM) {
        if (this.clusters.length === 0) return null;

        // Score clusters by proximity to COM and strength
        let bestCluster = null;
        let bestScore = -Infinity;

        for (const cluster of this.clusters) {
            // Distance from cluster centroid to squared COM
            const dist = Math.sqrt(
                Math.pow(cluster.centroid.row - squaredCOM.row, 2) +
                Math.pow(cluster.centroid.col - squaredCOM.col, 2)
            );

            // Proximity factor: closer to COM = higher score
            const proximityFactor = Math.exp(-dist / 8);  // Decay over ~8 grid units

            // Strength factor
            const strengthFactor = cluster.confidence *
                                   cluster.getTotalIntensity() *
                                   Math.sqrt(cluster.peaks.length);

            // Combined score prioritizes clusters near the COM
            const score = proximityFactor * strengthFactor;

            if (score > bestScore) {
                bestScore = score;
                bestCluster = cluster;
            }
        }

        return bestCluster;
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
     *
     * NOTE: For interactive mode (surgeon moving array), this is less useful
     * because the surgeon IS the one moving the array. This is mainly for:
     * 1. Global map visualization (showing trajectory)
     * 2. Potentially detecting unintended drift
     *
     * The approach: if neural activity center moves left on the grid,
     * the array must have moved right in brain space.
     *
     * ENHANCED: Uses exponential smoothing, velocity clamping, and
     * recency-weighted history for much more stable movement inference.
     */
    inferArrayMovement() {
        if (!this.trackedCenter || !this.prevCenter) {
            this.prevCenter = this.trackedCenter ?
                { ...this.trackedCenter } : null;
            // Initialize smoothed velocity if needed
            if (!this.smoothedMovement) {
                this.smoothedMovement = { dx: 0, dy: 0 };
            }
            return { dx: 0, dy: 0, confidence: 0 };
        }

        const dRow = this.trackedCenter.row - this.prevCenter.row;
        const dCol = this.trackedCenter.col - this.prevCenter.col;
        const now = Date.now();

        // Raw movement magnitude
        const rawMagnitude = Math.sqrt(dRow * dRow + dCol * dCol);

        // Moderate noise threshold - balance between responsiveness and stability
        const noiseThreshold = 0.3;

        // Clamp maximum movement per frame to reject outliers
        const maxMovementPerFrame = 4.0;

        // Initialize smoothed movement if needed
        if (!this.smoothedMovement) {
            this.smoothedMovement = { dx: 0, dy: 0 };
        }

        if (rawMagnitude < noiseThreshold) {
            // Below noise threshold - slowly decay smoothed movement
            this.smoothedMovement.dx *= 0.95;
            this.smoothedMovement.dy *= 0.95;
            this.prevCenter = { ...this.trackedCenter };

            // Still return smoothed value (momentum keeps array moving smoothly)
            const smoothedMag = Math.sqrt(
                this.smoothedMovement.dx * this.smoothedMovement.dx +
                this.smoothedMovement.dy * this.smoothedMovement.dy
            );
            return {
                dx: this.smoothedMovement.dx,
                dy: this.smoothedMovement.dy,
                confidence: Math.min(0.5, smoothedMag * 0.3)
            };
        }

        // Clamp movement to reject outliers (sudden jumps)
        let clampedDRow = dRow;
        let clampedDCol = dCol;
        if (rawMagnitude > maxMovementPerFrame) {
            const scale = maxMovementPerFrame / rawMagnitude;
            clampedDRow *= scale;
            clampedDCol *= scale;
        }

        // Store movement with timestamp
        this.movementHistory.push({
            dRow: clampedDRow,
            dCol: clampedDCol,
            time: now,
            magnitude: Math.sqrt(clampedDRow * clampedDRow + clampedDCol * clampedDCol)
        });

        // Keep limited history
        while (this.movementHistory.length > this.movementHistoryLength) {
            this.movementHistory.shift();
        }

        // Compute recency-weighted average (more recent = higher weight)
        let totalWeight = 0;
        let weightedDRow = 0;
        let weightedDCol = 0;

        for (const m of this.movementHistory) {
            const age = now - m.time;
            // Exponential decay: half-life of 500ms
            const recencyWeight = Math.exp(-age / 500);
            // Also weight by magnitude (stronger movements more reliable)
            const magnitudeWeight = Math.min(1, m.magnitude / 1.5);
            const combinedWeight = recencyWeight * (0.5 + 0.5 * magnitudeWeight);

            weightedDRow += m.dRow * combinedWeight;
            weightedDCol += m.dCol * combinedWeight;
            totalWeight += combinedWeight;
        }

        let avgDRow = 0, avgDCol = 0;
        if (totalWeight > 0.1) {
            avgDRow = weightedDRow / totalWeight;
            avgDCol = weightedDCol / totalWeight;
        }

        // Apply exponential smoothing to the averaged movement
        // Higher factor = more responsive to new movements
        const smoothingFactor = 0.4;
        this.smoothedMovement.dx = smoothingFactor * (-avgDCol) +
                                   (1 - smoothingFactor) * this.smoothedMovement.dx;
        this.smoothedMovement.dy = smoothingFactor * (-avgDRow) +
                                   (1 - smoothingFactor) * this.smoothedMovement.dy;

        // Update previous center
        this.prevCenter = { ...this.trackedCenter };

        // Compute final magnitude for confidence
        const smoothedMagnitude = Math.sqrt(
            this.smoothedMovement.dx * this.smoothedMovement.dx +
            this.smoothedMovement.dy * this.smoothedMovement.dy
        );

        // Confidence based on:
        // 1. Smoothed magnitude (consistent movement = higher confidence)
        // 2. History consistency (all movements in same direction)
        let directionConsistency = 0;
        if (this.movementHistory.length >= 3) {
            // Check if recent movements agree on direction
            const recentMoves = this.movementHistory.slice(-5);
            let sameDirection = 0;
            for (const m of recentMoves) {
                // Check if movement direction matches smoothed direction
                const dotProduct = (-m.dCol) * this.smoothedMovement.dx +
                                  (-m.dRow) * this.smoothedMovement.dy;
                if (dotProduct > 0) sameDirection++;
            }
            directionConsistency = sameDirection / recentMoves.length;
        }

        // Confidence based on magnitude and direction consistency
        // Higher base multiplier to get meaningful confidence values
        const confidence = Math.min(1,
            smoothedMagnitude * 0.5 * (0.6 + 0.4 * directionConsistency)
        );

        return {
            dx: this.smoothedMovement.dx,
            dy: this.smoothedMovement.dy,
            confidence: confidence
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
     * Get guidance direction for surgeon - SIMPLIFIED for interactive mode
     *
     * This is the key method for the OR display. It tells the surgeon
     * which way to move the array to center the neural activity.
     *
     * The approach is simple and robust:
     * 1. Use the squared-intensity weighted center of mass (already computed)
     * 2. Arrow points TOWARD where activity is (surgeon moves array that direction)
     * 3. When activity is centered, show "on target"
     */
    getGuidanceDirection() {
        const dist = this.getDistanceToCenter();
        if (!dist) return null;

        // Arrow points TOWARD activity (same direction as offset from center)
        // If activity is at row=20, col=25 (lower-right of center),
        // arrow points lower-right, telling surgeon to move array that way
        const moveX = dist.dx;   // Point toward activity
        const moveY = dist.dy;   // Point toward activity

        // Angle for arrow: standard math angle (0=right, 90=up)
        // Note: moveY is positive downward (row increases), so negate for screen coords
        const angle = Math.atan2(-moveY, moveX) * 180 / Math.PI;

        // Magnitude: normalized 0-1 based on how far from center
        // At 16 grid units (half the grid), magnitude = 1
        const magnitude = Math.min(1, dist.distance / 16);

        // On target threshold: within ~3 grid units of center (~1mm)
        const onTargetThreshold = 3;

        return {
            angle,
            magnitude,
            distance: dist.distance,
            distanceMm: dist.distanceMm,
            isOnTarget: dist.distance < onTargetThreshold,
            // Additional info for display
            activityRow: this.trackedCenter.row,
            activityCol: this.trackedCenter.col,
            arrayCenter: (this.gridSize - 1) / 2
        };
    }

    /**
     * ALTERNATIVE: Get guidance using raw squared-COM (bypass cluster tracking)
     * This is more responsive but less stable. Use for fast feedback.
     */
    getDirectGuidance(normalizedGrid) {
        const com = this.computeSquaredIntensityCOM(normalizedGrid);
        if (!com) return null;

        const arrayCenter = (this.gridSize - 1) / 2;
        const dx = com.col - arrayCenter;
        const dy = com.row - arrayCenter;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Arrow points toward activity
        const angle = Math.atan2(-dy, dx) * 180 / Math.PI;

        return {
            angle,
            magnitude: Math.min(1, distance / 16),
            distance,
            distanceMm: distance * 0.3,
            isOnTarget: distance < 3,
            // Raw velocity from COM (useful for other purposes)
            vx: com.vx,
            vy: com.vy
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
        this.smoothedMovement = { dx: 0, dy: 0 };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ClusterTracker, ClusterEntity, AnchoredPeak };
}
