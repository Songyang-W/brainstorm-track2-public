/**
 * Visualizer Module - Clean Surgeon UI
 *
 * Renders:
 * - Main view: SVG arrow + instruction text
 * - Global map: Canvas with accumulated activity
 * - Debug view: Heatmaps and metrics
 */

class Visualizer {
    constructor() {
        // Canvas contexts
        this.globalMapCtx = null;
        this.heatmapLiveCtx = null;
        this.heatmapClustersCtx = null;

        // Inferno-inspired colormap
        this.colormap = this.generateColormap();

        // Grid config
        this.gridSize = 32;
        this.mapSize = 128;
    }

    /**
     * Initialize canvases
     */
    init() {
        const globalMapCanvas = document.getElementById('global-map-canvas');
        const heatmapLive = document.getElementById('heatmap-live');
        const heatmapClusters = document.getElementById('heatmap-clusters');

        if (globalMapCanvas) {
            this.globalMapCtx = globalMapCanvas.getContext('2d');
        }
        if (heatmapLive) {
            this.heatmapLiveCtx = heatmapLive.getContext('2d');
        }
        if (heatmapClusters) {
            this.heatmapClustersCtx = heatmapClusters.getContext('2d');
        }
    }

    /**
     * Generate Inferno-like colormap
     */
    generateColormap() {
        const colors = [];
        const anchors = [
            [0, 0, 4],
            [40, 11, 84],
            [101, 21, 110],
            [159, 42, 99],
            [212, 72, 66],
            [245, 125, 21],
            [250, 193, 39],
            [252, 255, 164]
        ];

        for (let i = 0; i < 256; i++) {
            const t = i / 255 * (anchors.length - 1);
            const idx = Math.floor(t);
            const frac = t - idx;

            if (idx >= anchors.length - 1) {
                colors.push(anchors[anchors.length - 1]);
            } else {
                colors.push([
                    Math.round(anchors[idx][0] + frac * (anchors[idx + 1][0] - anchors[idx][0])),
                    Math.round(anchors[idx][1] + frac * (anchors[idx + 1][1] - anchors[idx][1])),
                    Math.round(anchors[idx][2] + frac * (anchors[idx + 1][2] - anchors[idx][2]))
                ]);
            }
        }
        return colors;
    }

    /**
     * Get color for normalized value (0-1)
     */
    getColor(value) {
        const idx = Math.max(0, Math.min(255, Math.floor(value * 255)));
        const [r, g, b] = this.colormap[idx];
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Update the main surgeon view (SVG arrow + text)
     */
    updateMainView(guidance) {
        const app = document.getElementById('app');
        const arrowGroup = document.getElementById('arrow-group');
        const arrowShaft = document.getElementById('arrow-shaft');
        const arrowHead = document.getElementById('arrow-head');
        const instructionText = document.getElementById('instruction-text');
        const instructionDetail = document.getElementById('instruction-detail');
        const distanceMarker = document.getElementById('distance-marker');
        const distanceFill = document.getElementById('distance-fill');

        if (!guidance) {
            // No guidance - searching state
            app.classList.remove('on-target', 'acquiring');
            if (instructionText) instructionText.textContent = 'SEARCHING';
            if (instructionDetail) instructionDetail.textContent = 'Looking for activity...';
            if (arrowGroup) arrowGroup.style.display = 'none';
            return;
        }

        const { direction, isOnTarget, distance, distanceMm } = guidance;

        // Update state classes
        app.classList.remove('on-target', 'acquiring');
        if (isOnTarget) {
            app.classList.add('on-target');
        } else if (distance < 10) {
            app.classList.add('acquiring');
        }

        // Update arrow
        if (arrowGroup && arrowShaft && arrowHead) {
            if (isOnTarget) {
                arrowGroup.style.display = 'none';
            } else {
                arrowGroup.style.display = 'block';

                // Calculate arrow rotation
                const angle = direction?.angle || 0;
                const magnitude = Math.min(1, (direction?.magnitude || 0) * 1.5);
                const length = 30 + magnitude * 40;  // 30-70 range

                // Arrow points in direction to move
                const radians = angle * Math.PI / 180;
                const endX = 100 + Math.cos(radians) * length;
                const endY = 100 - Math.sin(radians) * length;

                arrowShaft.setAttribute('x2', endX);
                arrowShaft.setAttribute('y2', endY);

                // Rotate arrowhead
                const headSize = 15;
                const tipX = 100 + Math.cos(radians) * (length + 10);
                const tipY = 100 - Math.sin(radians) * (length + 10);
                const leftX = tipX - headSize * Math.cos(radians - 0.5);
                const leftY = tipY + headSize * Math.sin(radians - 0.5);
                const rightX = tipX - headSize * Math.cos(radians + 0.5);
                const rightY = tipY + headSize * Math.sin(radians + 0.5);

                arrowHead.setAttribute('points', `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`);
            }
        }

        // Update instruction text
        if (instructionText) {
            if (isOnTarget) {
                instructionText.textContent = 'ON TARGET';
            } else {
                instructionText.textContent = this.getDirectionText(direction?.angle || 0);
            }
        }

        if (instructionDetail) {
            if (isOnTarget) {
                instructionDetail.textContent = 'Maintain position';
            } else {
                const mm = distanceMm?.toFixed(1) || '--';
                instructionDetail.textContent = `Move ${mm}mm`;
            }
        }

        // Update distance bar
        if (distanceMarker && distanceFill) {
            // Distance of 0 = right side (on target), distance of 20+ = left side (far)
            const maxDist = 20;
            const normalizedDist = Math.min(1, (distance || 0) / maxDist);
            const position = (1 - normalizedDist) * 100;

            distanceMarker.style.left = `${position}%`;
            distanceFill.style.width = `${100 - position}%`;
        }
    }

    /**
     * Convert angle to direction text
     */
    getDirectionText(angle) {
        // Normalize angle to 0-360
        let a = angle;
        while (a < 0) a += 360;
        while (a >= 360) a -= 360;

        if (a >= 337.5 || a < 22.5) return 'RIGHT';
        if (a >= 22.5 && a < 67.5) return 'UP-RIGHT';
        if (a >= 67.5 && a < 112.5) return 'UP';
        if (a >= 112.5 && a < 157.5) return 'UP-LEFT';
        if (a >= 157.5 && a < 202.5) return 'LEFT';
        if (a >= 202.5 && a < 247.5) return 'DOWN-LEFT';
        if (a >= 247.5 && a < 292.5) return 'DOWN';
        return 'DOWN-RIGHT';
    }

    /**
     * Render global map
     */
    renderGlobalMap(globalMap) {
        if (!this.globalMapCtx || !globalMap) return;

        const ctx = this.globalMapCtx;
        const canvas = ctx.canvas;
        const w = canvas.width;
        const h = canvas.height;

        // Clear
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, w, h);

        const mapSize = globalMap.mapSize;
        const scaleX = w / mapSize;
        const scaleY = h / mapSize;

        // Draw activity
        const imageData = ctx.createImageData(w, h);
        const data = imageData.data;

        // Find max for normalization
        let max = 0;
        for (let i = 0; i < globalMap.activityMap.length; i++) {
            if (globalMap.activityMap[i] > max) max = globalMap.activityMap[i];
        }
        if (max === 0) max = 1;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const mapX = Math.floor(x / scaleX);
                const mapY = Math.floor(y / scaleY);
                const mapIdx = mapY * mapSize + mapX;
                const value = globalMap.activityMap[mapIdx] / max;

                const colorIdx = Math.max(0, Math.min(255, Math.floor(value * 255)));
                const [r, g, b] = this.colormap[colorIdx];

                const pixelIdx = (y * w + x) * 4;
                data[pixelIdx] = r;
                data[pixelIdx + 1] = g;
                data[pixelIdx + 2] = b;
                data[pixelIdx + 3] = 255;
            }
        }
        ctx.putImageData(imageData, 0, 0);

        // Draw array position (cyan rectangle)
        const bounds = globalMap.getArrayBounds();
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(
            bounds.x * scaleX,
            bounds.y * scaleY,
            bounds.width * scaleX,
            bounds.height * scaleY
        );
        ctx.setLineDash([]);

        // Draw trajectory
        const history = globalMap.positionHistory;
        if (history.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.4)';
            ctx.lineWidth = 1;
            for (let i = 0; i < history.length; i++) {
                const pos = history[i];
                if (i === 0) {
                    ctx.moveTo(pos.x * scaleX, pos.y * scaleY);
                } else {
                    ctx.lineTo(pos.x * scaleX, pos.y * scaleY);
                }
            }
            ctx.stroke();
        }

        // Draw global hotspots
        for (let i = 0; i < Math.min(globalMap.globalHotspots.length, 10); i++) {
            const h = globalMap.globalHotspots[i];
            const hx = h.x * scaleX;
            const hy = h.y * scaleY;

            ctx.beginPath();
            ctx.arc(hx, hy, 4 + h.confidence * 4, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? '#00ff88' : `rgba(255, 136, 0, ${h.confidence})`;
            ctx.fill();
        }

        // Draw best region marker
        if (globalMap.bestRegion) {
            const bx = globalMap.bestRegion.x * scaleX;
            const by = globalMap.bestRegion.y * scaleY;

            ctx.beginPath();
            ctx.arc(bx, by, 8, 0, Math.PI * 2);
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 3;
            ctx.stroke();

            // Crosshair
            ctx.beginPath();
            ctx.moveTo(bx - 12, by);
            ctx.lineTo(bx + 12, by);
            ctx.moveTo(bx, by - 12);
            ctx.lineTo(bx, by + 12);
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Update map info displays
        const arrayPosEl = document.getElementById('map-array-pos');
        const bestClusterEl = document.getElementById('map-best-cluster');
        const exploredEl = document.getElementById('map-explored');

        if (arrayPosEl) {
            const ax = globalMap.arrayPosition.x.toFixed(0);
            const ay = globalMap.arrayPosition.y.toFixed(0);
            arrayPosEl.textContent = `${ax}, ${ay}`;
        }

        if (bestClusterEl && globalMap.bestRegion) {
            const dist = globalMap.getDirectionToBest();
            bestClusterEl.textContent = dist ? `${dist.distanceMm.toFixed(1)}mm away` : '--';
        }

        if (exploredEl) {
            exploredEl.textContent = `${globalMap.getExploredPercentage().toFixed(1)}%`;
        }
    }

    /**
     * Render live heatmap (debug view)
     */
    renderLiveHeatmap(normalizedGrid, clusterCenter, peaks) {
        if (!this.heatmapLiveCtx || !normalizedGrid) return;

        const ctx = this.heatmapLiveCtx;
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        const cellW = w / this.gridSize;
        const cellH = h / this.gridSize;

        // Draw heatmap
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                const value = normalizedGrid[row][col];
                ctx.fillStyle = this.getColor(value);
                ctx.fillRect(col * cellW, row * cellH, cellW + 1, cellH + 1);
            }
        }

        // Draw cluster center
        if (clusterCenter) {
            const cx = (clusterCenter.col + 0.5) * cellW;
            const cy = (clusterCenter.row + 0.5) * cellH;

            ctx.beginPath();
            ctx.arc(cx, cy, 8, 0, Math.PI * 2);
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 3;
            ctx.stroke();

            // Crosshair
            ctx.beginPath();
            ctx.moveTo(cx - 15, cy);
            ctx.lineTo(cx + 15, cy);
            ctx.moveTo(cx, cy - 15);
            ctx.lineTo(cx, cy + 15);
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Draw peaks
        if (peaks) {
            for (const peak of peaks) {
                const px = (peak.col + 0.5) * cellW;
                const py = (peak.row + 0.5) * cellH;

                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#ff8800';
                ctx.fill();
            }
        }

        // Center marker
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(w / 2, 0);
        ctx.lineTo(w / 2, h);
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    /**
     * Render tracked clusters heatmap (memory view)
     */
    renderClustersHeatmap(memoryGrid) {
        if (!this.heatmapClustersCtx || !memoryGrid) return;

        const ctx = this.heatmapClustersCtx;
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        const cellW = w / this.gridSize;
        const cellH = h / this.gridSize;

        // Find max
        let max = 0;
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                if (memoryGrid[row][col] > max) max = memoryGrid[row][col];
            }
        }
        if (max === 0) max = 1;

        // Draw
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                const value = memoryGrid[row][col] / max;
                ctx.fillStyle = this.getColor(value);
                ctx.fillRect(col * cellW, row * cellH, cellW + 1, cellH + 1);
            }
        }
    }

    /**
     * Update debug metrics
     */
    updateDebugMetrics(data) {
        const setEl = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        };

        if (data.center) {
            setEl('debug-center', `${data.center.row?.toFixed(1)}, ${data.center.col?.toFixed(1)}`);
        }
        setEl('debug-confidence', data.confidence ? `${(data.confidence * 100).toFixed(0)}%` : '--%');
        setEl('debug-distance', data.distance ? `${data.distance.toFixed(1)}` : '--');
        setEl('debug-vx', data.vx?.toFixed(1) || '--');
        setEl('debug-vy', data.vy?.toFixed(1) || '--');
        setEl('debug-hotspots', data.activeHotspots?.toString() || '--');
        setEl('debug-tracked', data.trackedHotspots?.toString() || '--');
        setEl('debug-movement', data.movement || '--');
    }

    /**
     * Update status bar
     */
    updateStatus(data) {
        const app = document.getElementById('app');
        const statusText = document.getElementById('status-text');
        const timeDisplay = document.getElementById('time-display');
        const fpsDisplay = document.getElementById('fps-display');
        const connectBtn = document.getElementById('connect-btn');

        if (data.connected !== undefined) {
            app.classList.remove('connected', 'connecting');
            if (data.connected === true) {
                app.classList.add('connected');
                if (statusText) statusText.textContent = 'Connected';
                if (connectBtn) connectBtn.textContent = 'Disconnect';
            } else if (data.connected === 'connecting') {
                app.classList.add('connecting');
                if (statusText) statusText.textContent = 'Connecting...';
            } else {
                if (statusText) statusText.textContent = 'Disconnected';
                if (connectBtn) connectBtn.textContent = 'Connect';
            }
        }

        if (data.time !== undefined && timeDisplay) {
            timeDisplay.textContent = `${data.time.toFixed(2)}s`;
        }

        if (data.fps !== undefined && fpsDisplay) {
            fpsDisplay.textContent = `${data.fps} FPS`;
        }
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Visualizer;
}
