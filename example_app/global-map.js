/**
 * Global Brain Map Module
 *
 * Maintains a persistent map of accumulated neural activity as the array moves.
 * Uses cluster tracking to infer array position and builds a comprehensive
 * view of the explored brain area.
 */

class GlobalBrainMap {
    constructor(config = {}) {
        // Map dimensions (larger than 32x32 to accommodate movement)
        this.mapSize = config.mapSize || 128;
        this.localGridSize = 32;

        // Global activity accumulation
        this.activityMap = new Float32Array(this.mapSize * this.mapSize);
        this.visitMap = new Float32Array(this.mapSize * this.mapSize);  // How many times visited

        // Array position in global coordinates
        this.arrayPosition = {
            x: this.mapSize / 2,
            y: this.mapSize / 2
        };

        // Position history for trajectory
        this.positionHistory = [];
        this.maxHistory = 300;

        // Anchored hotspots in global coordinates
        this.globalHotspots = [];  // {x, y, intensity, lastSeen, confidence}

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
     * @param {Array} normalizedGrid - Current 32x32 normalized activity
     * @param {Object} movement - Inferred array movement {dx, dy, confidence}
     * @param {Object} clusterCenter - Current cluster center {row, col}
     * @param {Array} peaks - Tracked peak positions from cluster tracker
     */
    update(normalizedGrid, movement, peaks) {
        // 1. Update array position based on inferred movement
        if (movement && movement.confidence > 0.1) {
            this.arrayPosition.x += movement.dx * this.movementScale;
            this.arrayPosition.y += movement.dy * this.movementScale;

            // Clamp to boundaries
            const margin = this.localGridSize / 2 + 5;
            this.arrayPosition.x = Math.max(margin, Math.min(this.mapSize - margin, this.arrayPosition.x));
            this.arrayPosition.y = Math.max(margin, Math.min(this.mapSize - margin, this.arrayPosition.y));
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

        // 5. Update global hotspots from current peaks
        if (peaks && peaks.length > 0) {
            this.updateGlobalHotspots(peaks);
        }

        // 6. Find best region
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
     * Convert local peak positions to global and anchor them
     */
    updateGlobalHotspots(peaks) {
        const now = Date.now();
        const halfSize = this.localGridSize / 2;

        for (const peak of peaks) {
            // Convert to global coordinates
            const globalX = this.arrayPosition.x - halfSize + peak.col;
            const globalY = this.arrayPosition.y - halfSize + peak.row;

            // Check if this matches an existing global hotspot
            let matched = false;
            for (const hotspot of this.globalHotspots) {
                const dist = Math.sqrt(
                    Math.pow(globalX - hotspot.x, 2) +
                    Math.pow(globalY - hotspot.y, 2)
                );

                if (dist < 5) {  // Match threshold
                    // Update with weighted average (anchoring)
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

        // Decay confidence of hotspots not seen recently
        for (const hotspot of this.globalHotspots) {
            if (now - hotspot.lastSeen > 500) {
                hotspot.confidence *= 0.98;
                hotspot.intensity *= 0.99;
            }
        }

        // Remove very old or weak hotspots
        this.globalHotspots = this.globalHotspots.filter(h =>
            h.confidence > 0.1 && h.intensity > 0.15
        );

        // Keep only top hotspots
        this.globalHotspots.sort((a, b) =>
            (b.intensity * b.confidence) - (a.intensity * a.confidence)
        );
        if (this.globalHotspots.length > 30) {
            this.globalHotspots = this.globalHotspots.slice(0, 30);
        }
    }

    /**
     * Find the best region (cluster of high-confidence hotspots)
     */
    updateBestRegion() {
        if (this.globalHotspots.length === 0) {
            this.bestRegion = null;
            return;
        }

        // Weighted centroid of all hotspots
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
            distanceMm: distance * 0.3,  // Rough conversion
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
     * Reset the global map
     */
    reset() {
        this.activityMap = new Float32Array(this.mapSize * this.mapSize);
        this.visitMap = new Float32Array(this.mapSize * this.mapSize);
        this.arrayPosition = { x: this.mapSize / 2, y: this.mapSize / 2 };
        this.positionHistory = [];
        this.globalHotspots = [];
        this.bestRegion = null;
        this.accumulatedMovement = { x: 0, y: 0 };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GlobalBrainMap;
}
