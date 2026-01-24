/**
 * Global Brain Map Module v3
 *
 * Maintains a persistent map of accumulated neural activity as the array moves.
 * Now uses ClusterEntity integration for better hotspot anchoring in global coordinates.
 *
 * Key improvements v3:
 * - Supports MULTIPLE clusters as anchor points
 * - Anchors entire cluster entities (not just individual peaks)
 * - Maintains cluster identity across movements
 * - Remembers off hotspots within clusters for structure preservation
 * - Uses squared-intensity weighted COM from cluster-tracker.js
 */

/**
 * Global cluster representation
 * Remembers ALL peaks (active and inactive) for structure preservation
 */
class GlobalCluster {
    constructor(id, localCluster, arrayPosition, localGridSize) {
        this.id = id;
        this.localClusterId = localCluster.id;

        // Convert local centroid to global coordinates
        const halfSize = localGridSize / 2;
        this.centroid = {
            x: arrayPosition.x - halfSize + localCluster.centroid.col,
            y: arrayPosition.y - halfSize + localCluster.centroid.row
        };

        // Smoothed centroid for extra stability
        this.smoothedCentroid = { ...this.centroid };

        // Store anchored peaks in global coordinates
        // IMPORTANT: Store ALL peaks including inactive ones for structure memory
        this.globalPeaks = localCluster.peaks.map(peak => ({
            x: arrayPosition.x - halfSize + peak.col,
            y: arrayPosition.y - halfSize + peak.row,
            intensity: peak.intensity,
            maxIntensity: peak.maxIntensity,
            active: peak.active,
            lastActive: peak.lastActive || Date.now(),
            activationCount: peak.activationCount || 1
        }));

        this.confidence = localCluster.confidence;
        this.lastSeen = Date.now();
        this.createdAt = Date.now();
        this.totalIntensity = localCluster.getTotalIntensity();
        this.peakCount = localCluster.peaks.length;
        this.activePeakCount = localCluster.getActivePeakCount();

        // Track structure stability (how consistent the peak arrangement is)
        this.structureStability = 0.5;
    }

    /**
     * Update with new local cluster observation
     * ENHANCED: Much stronger anchoring with minimal drift
     * Preserves off-hotspot positions for structure memory
     */
    update(localCluster, arrayPosition, localGridSize) {
        const halfSize = localGridSize / 2;

        // Compute new centroid position
        const newX = arrayPosition.x - halfSize + localCluster.centroid.col;
        const newY = arrayPosition.y - halfSize + localCluster.centroid.row;

        // Calculate age-based anchoring strength
        // Older clusters are MORE stable (less responsive to new observations)
        const age = Date.now() - this.createdAt;
        const ageSeconds = age / 1000;

        // Anchoring strength increases with age and observations
        // After 10 seconds, smoothing factor becomes very small (0.01)
        const anchoringFactor = Math.min(0.99, ageSeconds / 10);
        const centroidSmoothing = 0.05 * (1 - anchoringFactor * 0.8);  // 0.05 -> 0.01 over time

        // Only update centroid if observation is reasonably close to current position
        // This prevents drifting due to array movement noise
        const observationDist = Math.sqrt(
            Math.pow(newX - this.centroid.x, 2) +
            Math.pow(newY - this.centroid.y, 2)
        );

        // If observation is too far (>8 units), likely due to noise or array movement
        // Apply even less weight to prevent drift
        const distancePenalty = observationDist > 8 ? 0.1 : 1.0;
        const effectiveSmoothing = centroidSmoothing * distancePenalty;

        this.centroid.x = effectiveSmoothing * newX + (1 - effectiveSmoothing) * this.centroid.x;
        this.centroid.y = effectiveSmoothing * newY + (1 - effectiveSmoothing) * this.centroid.y;

        // Smoothed centroid is even more stable (for visualization stability)
        const smoothedAlpha = 0.02 * distancePenalty;
        this.smoothedCentroid.x = smoothedAlpha * newX + (1 - smoothedAlpha) * this.smoothedCentroid.x;
        this.smoothedCentroid.y = smoothedAlpha * newY + (1 - smoothedAlpha) * this.smoothedCentroid.y;

        // Peak position smoothing - also age-dependent
        const peakSmoothing = 0.08 * (1 - anchoringFactor * 0.5);

        // Track which global peaks got matched
        const matchedGlobalPeaks = new Set();

        // Update or add global peaks
        for (const localPeak of localCluster.peaks) {
            const globalX = arrayPosition.x - halfSize + localPeak.col;
            const globalY = arrayPosition.y - halfSize + localPeak.row;

            // Find matching global peak - use TIGHTER radius (3 units instead of 5)
            let matched = false;
            let bestMatchIdx = -1;
            let bestMatchDist = Infinity;

            for (let i = 0; i < this.globalPeaks.length; i++) {
                const globalPeak = this.globalPeaks[i];
                const dist = Math.sqrt(
                    Math.pow(globalX - globalPeak.x, 2) +
                    Math.pow(globalY - globalPeak.y, 2)
                );

                // Tighter match radius for better anchoring
                if (dist < 3 && dist < bestMatchDist) {
                    bestMatchDist = dist;
                    bestMatchIdx = i;
                }
            }

            if (bestMatchIdx >= 0) {
                const globalPeak = this.globalPeaks[bestMatchIdx];

                // Peaks that have been seen many times are MORE anchored
                const peakAnchoringFactor = Math.min(0.9, (globalPeak.activationCount || 1) / 20);
                const effectivePeakSmoothing = peakSmoothing * (1 - peakAnchoringFactor * 0.5);

                // Update with smoothing (preserves position even when temporarily off)
                globalPeak.x = effectivePeakSmoothing * globalX + (1 - effectivePeakSmoothing) * globalPeak.x;
                globalPeak.y = effectivePeakSmoothing * globalY + (1 - effectivePeakSmoothing) * globalPeak.y;
                globalPeak.intensity = localPeak.intensity;
                globalPeak.maxIntensity = Math.max(globalPeak.maxIntensity, localPeak.maxIntensity);
                globalPeak.active = localPeak.active;
                globalPeak.lastActive = localPeak.active ? Date.now() : globalPeak.lastActive;
                globalPeak.activationCount = (globalPeak.activationCount || 1) + (localPeak.active ? 1 : 0);
                matchedGlobalPeaks.add(bestMatchIdx);
                matched = true;
            }

            // Add new peaks with lower threshold to capture structure
            if (!matched && localPeak.intensity > 0.25) {  // Slightly higher threshold
                this.globalPeaks.push({
                    x: globalX,
                    y: globalY,
                    intensity: localPeak.intensity,
                    maxIntensity: localPeak.maxIntensity,
                    active: localPeak.active,
                    lastActive: Date.now(),
                    activationCount: 1
                });
            }
        }

        // Decay unmatched peaks but DON'T remove them quickly (preserve structure)
        for (let i = 0; i < this.globalPeaks.length; i++) {
            if (!matchedGlobalPeaks.has(i)) {
                const peak = this.globalPeaks[i];
                peak.active = false;
                peak.intensity *= 0.985;  // Even slower decay for stability
            }
        }

        // Update cluster metrics
        this.confidence = Math.max(this.confidence, localCluster.confidence);
        this.lastSeen = Date.now();
        this.totalIntensity = localCluster.getTotalIntensity();
        this.peakCount = localCluster.peaks.length;
        this.activePeakCount = localCluster.getActivePeakCount();

        // Update structure stability (consistent peak count = stable structure)
        const expectedPeaks = this.globalPeaks.length;
        const observedActive = localCluster.getActivePeakCount();
        const stabilityDelta = observedActive / Math.max(1, expectedPeaks);
        this.structureStability = 0.95 * this.structureStability + 0.05 * stabilityDelta;  // Slower stability updates
    }

    /**
     * Decay when not observed
     * Preserves structure by keeping peak POSITIONS even when intensity fades
     */
    decay() {
        this.confidence *= 0.995;
        for (const peak of this.globalPeaks) {
            peak.intensity *= 0.99;
            peak.active = false;
        }
        // Only remove peaks that have been inactive for a long time AND are very weak
        // This preserves cluster structure even when parts turn off temporarily
        const now = Date.now();
        this.globalPeaks = this.globalPeaks.filter(p => {
            const age = now - (p.lastActive || now);
            // Keep if: recently active, high max intensity history, or frequently activated
            return p.intensity > 0.03 ||
                   age < 15000 ||
                   p.maxIntensity > 0.5 ||
                   (p.activationCount || 1) > 5;
        });
    }

    /**
     * Check if should be pruned
     */
    shouldPrune(timeout = 30000) {
        const age = Date.now() - this.lastSeen;
        return age > timeout || this.confidence < 0.05 || this.globalPeaks.length === 0;
    }

    /**
     * Distance to a point
     */
    distanceTo(x, y) {
        return Math.sqrt(
            Math.pow(x - this.centroid.x, 2) +
            Math.pow(y - this.centroid.y, 2)
        );
    }

    /**
     * Get bounding box
     */
    getBoundingBox() {
        if (this.globalPeaks.length === 0) return null;

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        for (const peak of this.globalPeaks) {
            minX = Math.min(minX, peak.x);
            maxX = Math.max(maxX, peak.x);
            minY = Math.min(minY, peak.y);
            maxY = Math.max(maxY, peak.y);
        }

        return { minX, maxX, minY, maxY };
    }
}

/**
 * Global Brain Map
 */
class GlobalBrainMap {
    constructor(config = {}) {
        // Map dimensions (larger than 32x32 to accommodate movement)
        this.mapSize = config.mapSize || 128;
        this.localGridSize = 32;

        // Global activity accumulation
        this.activityMap = new Float32Array(this.mapSize * this.mapSize);
        this.visitMap = new Float32Array(this.mapSize * this.mapSize);

        // Array position in global coordinates
        this.arrayPosition = {
            x: this.mapSize / 2,
            y: this.mapSize / 2
        };

        // Position history for trajectory
        this.positionHistory = [];
        this.maxHistory = 300;

        // Global clusters (replacing globalHotspots)
        this.globalClusters = [];
        this.nextGlobalClusterId = 1;

        // Legacy: maintain globalHotspots for visualizer compatibility
        this.globalHotspots = [];

        // Best discovered region
        this.bestRegion = null;

        // Movement integration
        this.accumulatedMovement = { x: 0, y: 0 };
        this.movementScale = config.movementScale || 1.5;

        // Decay for activity persistence
        this.decayRate = config.decayRate || 0.998;
    }

    /**
     * Update global map with new local activity and movement
     *
     * KEY INSIGHT: When a cluster moves LEFT on the local grid, the array
     * has moved RIGHT in brain space. The inferred movement already handles
     * this inversion, so we apply it directly to array position.
     */
    update(normalizedGrid, movement, peaks, localClusters = null) {
        // Initialize velocity tracking if not present
        if (!this.smoothedVelocity) {
            this.smoothedVelocity = { x: 0, y: 0 };
        }

        // 1. Update array position based on inferred movement
        // Lower threshold to ensure movement is detected
        if (movement && movement.confidence > 0.05) {
            // Smooth the velocity - but be responsive enough to show movement
            const velocitySmoothing = 0.4;  // Higher = more responsive
            this.smoothedVelocity.x = velocitySmoothing * movement.dx + (1 - velocitySmoothing) * this.smoothedVelocity.x;
            this.smoothedVelocity.y = velocitySmoothing * movement.dy + (1 - velocitySmoothing) * this.smoothedVelocity.y;

            // Compute smoothed magnitude
            const smoothedMagnitude = Math.sqrt(
                this.smoothedVelocity.x * this.smoothedVelocity.x +
                this.smoothedVelocity.y * this.smoothedVelocity.y
            );

            // Lower threshold - any meaningful movement should register
            if (smoothedMagnitude > 0.1) {
                // Scale movement - use smoothed velocity directly
                const scaledMovement = this.movementScale;
                this.arrayPosition.x += this.smoothedVelocity.x * scaledMovement;
                this.arrayPosition.y += this.smoothedVelocity.y * scaledMovement;
            }

            // Clamp to boundaries
            const margin = this.localGridSize / 2 + 5;
            this.arrayPosition.x = Math.max(margin, Math.min(this.mapSize - margin, this.arrayPosition.x));
            this.arrayPosition.y = Math.max(margin, Math.min(this.mapSize - margin, this.arrayPosition.y));
        } else {
            // Slowly decay velocity when no movement detected
            this.smoothedVelocity.x *= 0.95;
            this.smoothedVelocity.y *= 0.95;
        }

        // 2. Store position history
        this.positionHistory.push({
            x: this.arrayPosition.x,
            y: this.arrayPosition.y,
            time: Date.now()
        });
        while (this.positionHistory.length > this.maxHistory) {
            this.positionHistory.shift();
        }

        // 3. Decay existing activity
        for (let i = 0; i < this.activityMap.length; i++) {
            this.activityMap[i] *= this.decayRate;
        }

        // 4. Place current local grid onto global map
        this.placeLocalActivity(normalizedGrid);

        // 5. Update global clusters (if cluster entities provided)
        if (localClusters && localClusters.length > 0) {
            this.updateGlobalClusters(localClusters);
        } else if (peaks && peaks.length > 0) {
            // Fallback to legacy peak-based updates
            this.updateGlobalHotspots(peaks);
        }

        // 6. Decay unobserved clusters
        for (const cluster of this.globalClusters) {
            if (Date.now() - cluster.lastSeen > 100) {
                cluster.decay();
            }
        }

        // 7. Prune dead clusters
        this.globalClusters = this.globalClusters.filter(c => !c.shouldPrune());

        // 8. Update legacy globalHotspots from clusters
        this.syncHotspotsFromClusters();

        // 9. Find best region
        this.updateBestRegion();
    }

    /**
     * Place local 32x32 activity onto global map at current array position
     */
    placeLocalActivity(grid) {
        const halfSize = this.localGridSize / 2;
        const startX = Math.floor(this.arrayPosition.x - halfSize);
        const startY = Math.floor(this.arrayPosition.y - halfSize);

        for (let row = 0; row < this.localGridSize; row++) {
            for (let col = 0; col < this.localGridSize; col++) {
                const globalX = startX + col;
                const globalY = startY + row;

                if (globalX >= 0 && globalX < this.mapSize &&
                    globalY >= 0 && globalY < this.mapSize) {
                    const globalIdx = globalY * this.mapSize + globalX;
                    const localVal = grid[row][col];

                    // Blend with existing value
                    const existing = this.activityMap[globalIdx];
                    this.activityMap[globalIdx] = Math.max(existing, localVal * 0.8 + existing * 0.2);
                    this.visitMap[globalIdx]++;
                }
            }
        }
    }

    /**
     * Update global clusters from local cluster entities
     */
    updateGlobalClusters(localClusters) {
        for (const localCluster of localClusters) {
            // Convert local centroid to global for matching
            const halfSize = this.localGridSize / 2;
            const globalX = this.arrayPosition.x - halfSize + localCluster.centroid.col;
            const globalY = this.arrayPosition.y - halfSize + localCluster.centroid.row;

            // Find matching global cluster
            let matched = null;
            let bestDist = Infinity;

            for (const globalCluster of this.globalClusters) {
                const dist = globalCluster.distanceTo(globalX, globalY);

                // Match by distance or by local cluster ID (if same cluster seen again)
                if (dist < 10 || globalCluster.localClusterId === localCluster.id) {
                    if (dist < bestDist) {
                        bestDist = dist;
                        matched = globalCluster;
                    }
                }
            }

            if (matched) {
                matched.update(localCluster, this.arrayPosition, this.localGridSize);
            } else if (localCluster.confidence > 0.2) {
                // Create new global cluster
                const newCluster = new GlobalCluster(
                    this.nextGlobalClusterId++,
                    localCluster,
                    this.arrayPosition,
                    this.localGridSize
                );
                this.globalClusters.push(newCluster);
            }
        }

        // Merge overlapping global clusters
        this.mergeGlobalClusters();
    }

    /**
     * Merge global clusters that have converged
     */
    mergeGlobalClusters() {
        const merged = new Set();

        for (let i = 0; i < this.globalClusters.length; i++) {
            if (merged.has(i)) continue;

            for (let j = i + 1; j < this.globalClusters.length; j++) {
                if (merged.has(j)) continue;

                const c1 = this.globalClusters[i];
                const c2 = this.globalClusters[j];

                const dist = c1.distanceTo(c2.centroid.x, c2.centroid.y);

                if (dist < 6) {
                    // Merge c2 into c1
                    for (const peak of c2.globalPeaks) {
                        // Check if already exists
                        let exists = false;
                        for (const p1 of c1.globalPeaks) {
                            const d = Math.sqrt(
                                Math.pow(peak.x - p1.x, 2) +
                                Math.pow(peak.y - p1.y, 2)
                            );
                            if (d < 3) {
                                exists = true;
                                p1.intensity = Math.max(p1.intensity, peak.intensity);
                                break;
                            }
                        }
                        if (!exists) {
                            c1.globalPeaks.push({ ...peak });
                        }
                    }

                    // Recompute centroid
                    if (c1.globalPeaks.length > 0) {
                        let sumX = 0, sumY = 0, sumW = 0;
                        for (const p of c1.globalPeaks) {
                            const w = p.maxIntensity;
                            sumX += p.x * w;
                            sumY += p.y * w;
                            sumW += w;
                        }
                        if (sumW > 0) {
                            c1.centroid.x = sumX / sumW;
                            c1.centroid.y = sumY / sumW;
                        }
                    }

                    c1.confidence = Math.max(c1.confidence, c2.confidence);
                    merged.add(j);
                }
            }
        }

        this.globalClusters = this.globalClusters.filter((_, i) => !merged.has(i));
    }

    /**
     * Legacy: update global hotspots from peaks (fallback)
     */
    updateGlobalHotspots(peaks) {
        const now = Date.now();
        const halfSize = this.localGridSize / 2;

        for (const peak of peaks) {
            const globalX = this.arrayPosition.x - halfSize + peak.col;
            const globalY = this.arrayPosition.y - halfSize + peak.row;

            // Check if matches existing hotspot
            let matched = false;
            for (const hotspot of this.globalHotspots) {
                const dist = Math.sqrt(
                    Math.pow(globalX - hotspot.x, 2) +
                    Math.pow(globalY - hotspot.y, 2)
                );

                if (dist < 5) {
                    const w = 0.2;
                    hotspot.x = w * globalX + (1 - w) * hotspot.x;
                    hotspot.y = w * globalY + (1 - w) * hotspot.y;
                    hotspot.intensity = Math.max(hotspot.intensity * 0.95, peak.intensity);
                    hotspot.lastSeen = now;
                    hotspot.confidence = Math.min(1, hotspot.confidence + 0.05);
                    matched = true;
                    break;
                }
            }

            if (!matched && peak.intensity > 0.3) {
                this.globalHotspots.push({
                    x: globalX,
                    y: globalY,
                    intensity: peak.intensity,
                    lastSeen: now,
                    confidence: 0.3
                });
            }
        }

        // Decay old hotspots
        for (const hotspot of this.globalHotspots) {
            if (now - hotspot.lastSeen > 500) {
                hotspot.confidence *= 0.98;
                hotspot.intensity *= 0.99;
            }
        }

        // Remove weak hotspots
        this.globalHotspots = this.globalHotspots.filter(h =>
            h.confidence > 0.1 && h.intensity > 0.15
        );

        // Keep top hotspots
        this.globalHotspots.sort((a, b) =>
            (b.intensity * b.confidence) - (a.intensity * a.confidence)
        );
        if (this.globalHotspots.length > 30) {
            this.globalHotspots = this.globalHotspots.slice(0, 30);
        }
    }

    /**
     * Sync legacy globalHotspots from cluster data
     */
    syncHotspotsFromClusters() {
        if (this.globalClusters.length === 0) return;

        this.globalHotspots = [];

        for (const cluster of this.globalClusters) {
            for (const peak of cluster.globalPeaks) {
                this.globalHotspots.push({
                    x: peak.x,
                    y: peak.y,
                    intensity: peak.intensity,
                    lastSeen: cluster.lastSeen,
                    confidence: cluster.confidence,
                    clusterId: cluster.id,
                    active: peak.active
                });
            }
        }

        // Sort by intensity * confidence
        this.globalHotspots.sort((a, b) =>
            (b.intensity * b.confidence) - (a.intensity * a.confidence)
        );

        // Keep top 30
        if (this.globalHotspots.length > 30) {
            this.globalHotspots = this.globalHotspots.slice(0, 30);
        }
    }

    /**
     * Find the best region (strongest cluster)
     */
    updateBestRegion() {
        // Prefer global clusters
        if (this.globalClusters.length > 0) {
            // Find strongest cluster
            let best = null;
            let bestScore = 0;

            for (const cluster of this.globalClusters) {
                const score = cluster.confidence * cluster.totalIntensity *
                              Math.sqrt(cluster.peakCount);
                if (score > bestScore) {
                    bestScore = score;
                    best = cluster;
                }
            }

            if (best) {
                this.bestRegion = {
                    x: best.centroid.x,
                    y: best.centroid.y,
                    intensity: best.totalIntensity,
                    hotspotCount: best.peakCount,
                    clusterId: best.id
                };
                return;
            }
        }

        // Fallback to hotspot-based calculation
        if (this.globalHotspots.length === 0) {
            this.bestRegion = null;
            return;
        }

        let totalWeight = 0;
        let weightedX = 0;
        let weightedY = 0;
        let maxIntensity = 0;

        for (const h of this.globalHotspots) {
            const weight = h.intensity * h.confidence;
            weightedX += h.x * weight;
            weightedY += h.y * weight;
            totalWeight += weight;
            if (h.intensity > maxIntensity) maxIntensity = h.intensity;
        }

        if (totalWeight > 0) {
            this.bestRegion = {
                x: weightedX / totalWeight,
                y: weightedY / totalWeight,
                intensity: maxIntensity,
                hotspotCount: this.globalHotspots.length
            };
        }
    }

    /**
     * Get direction from current array position to best region
     */
    getDirectionToBest() {
        if (!this.bestRegion) return null;

        const dx = this.bestRegion.x - this.arrayPosition.x;
        const dy = this.bestRegion.y - this.arrayPosition.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        return {
            dx,
            dy,
            distance,
            distanceMm: distance * 0.3,
            angle: Math.atan2(-dy, dx) * 180 / Math.PI,
            isInView: distance < this.localGridSize / 2
        };
    }

    /**
     * Get array bounds in global coordinates
     */
    getArrayBounds() {
        const halfSize = this.localGridSize / 2;
        return {
            x: this.arrayPosition.x - halfSize,
            y: this.arrayPosition.y - halfSize,
            width: this.localGridSize,
            height: this.localGridSize
        };
    }

    /**
     * Calculate explored percentage
     */
    getExploredPercentage() {
        let visited = 0;
        for (let i = 0; i < this.visitMap.length; i++) {
            if (this.visitMap[i] > 0) visited++;
        }
        return (visited / this.visitMap.length) * 100;
    }

    /**
     * Get all global clusters (for visualization)
     */
    getGlobalClusters() {
        return this.globalClusters;
    }

    /**
     * Reset the global map
     */
    reset() {
        this.activityMap = new Float32Array(this.mapSize * this.mapSize);
        this.visitMap = new Float32Array(this.mapSize * this.mapSize);
        this.arrayPosition = { x: this.mapSize / 2, y: this.mapSize / 2 };
        this.positionHistory = [];
        this.globalClusters = [];
        this.nextGlobalClusterId = 1;
        this.globalHotspots = [];
        this.bestRegion = null;
        this.accumulatedMovement = { x: 0, y: 0 };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GlobalBrainMap, GlobalCluster };
}
