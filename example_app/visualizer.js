/**
 * Visualizer Module - Modular Panel Rendering
 *
 * Works with LayoutManager to render content into dynamically created panels.
 * Each panel type (guide, global-map, live, memory, metrics) can appear
 * in multiple slots simultaneously.
 */

class Visualizer {
    constructor() {
        // Colormap for heatmaps
        this.colormap = this.generateColormap();

        // Grid config
        this.gridSize = 32;
        this.mapSize = 128;

        // Layout manager reference
        this.layoutManager = null;

        // Canvas contexts cache
        this.canvasContexts = new Map();
    }

    /**
     * Initialize with layout manager
     */
    init(layoutManager) {
        this.layoutManager = layoutManager;
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
     * Get canvas context, creating if needed
     */
    getContext(canvas) {
        if (!canvas) return null;

        if (!this.canvasContexts.has(canvas)) {
            this.canvasContexts.set(canvas, canvas.getContext('2d'));
        }
        return this.canvasContexts.get(canvas);
    }

    /**
     * Update all guide panels
     */
    updateMainView(guidance) {
        if (!this.layoutManager) return;

        const panels = this.layoutManager.getPanelsOfType('guide');

        for (const panel of panels) {
            this.updateGuidePanel(panel, guidance);
        }
    }

    /**
     * Update a single guide panel
     */
    updateGuidePanel(panel, guidance) {
        const content = panel.content;
        const slotId = panel.slotId;

        // Get elements
        const arrowGroup = content.querySelector('.arrow-group');
        const arrowShaft = content.querySelector('.arrow-shaft');
        const arrowHead = content.querySelector('.arrow-head');
        const instructionText = content.querySelector('.instruction-text');
        const instructionDetail = content.querySelector('.instruction-detail');
        const distanceMarker = content.querySelector('.distance-marker');
        const distanceFill = content.querySelector('.distance-fill');

        // Update panel state class
        content.classList.remove('on-target', 'acquiring');

        if (!guidance) {
            if (instructionText) instructionText.textContent = 'SEARCHING';
            if (instructionDetail) instructionDetail.textContent = 'Looking for activity...';
            if (arrowGroup) arrowGroup.style.display = 'none';
            return;
        }

        const { direction, isOnTarget, distance, distanceMm } = guidance;

        // Update state classes
        if (isOnTarget) {
            content.classList.add('on-target');
        } else if (distance < 10) {
            content.classList.add('acquiring');
        }

        // Update arrow
        if (arrowGroup && arrowShaft && arrowHead) {
            if (isOnTarget) {
                arrowGroup.style.display = 'none';
            } else {
                arrowGroup.style.display = 'block';

                const angle = direction?.angle || 0;
                const magnitude = Math.min(1, (direction?.magnitude || 0) * 1.5);
                const length = 30 + magnitude * 40;

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
     * Render all global map panels
     */
    renderGlobalMap(globalMap) {
        if (!this.layoutManager || !globalMap) return;

        const panels = this.layoutManager.getPanelsOfType('global-map');

        for (const panel of panels) {
            this.renderGlobalMapPanel(panel, globalMap);
        }
    }

    /**
     * Render a single global map panel
     */
    renderGlobalMapPanel(panel, globalMap) {
        const canvas = panel.canvas;
        if (!canvas) return;

        const ctx = this.getContext(canvas);
        if (!ctx) return;

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
            ctx.arc(hx, hy, 4 + (h.confidence || 0.5) * 4, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? '#00ff88' : `rgba(255, 136, 0, ${h.confidence || 0.5})`;
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

        // Update info displays
        const content = panel.content;
        const posEl = content.querySelector('[data-target^="map-pos"]');
        const bestEl = content.querySelector('[data-target^="map-best"]');
        const exploredEl = content.querySelector('[data-target^="map-explored"]');

        if (posEl) {
            const ax = globalMap.arrayPosition.x.toFixed(0);
            const ay = globalMap.arrayPosition.y.toFixed(0);
            posEl.textContent = `${ax}, ${ay}`;
        }

        if (bestEl && globalMap.bestRegion) {
            const dist = globalMap.getDirectionToBest();
            bestEl.textContent = dist ? `${dist.distanceMm.toFixed(1)}mm away` : '--';
        }

        if (exploredEl) {
            exploredEl.textContent = `${globalMap.getExploredPercentage().toFixed(1)}%`;
        }
    }

    /**
     * Render all live heatmap panels
     */
    renderLiveHeatmap(normalizedGrid, clusterCenter, peaks) {
        if (!this.layoutManager || !normalizedGrid) return;

        const panels = this.layoutManager.getPanelsOfType('live');

        for (const panel of panels) {
            this.renderLiveHeatmapPanel(panel, normalizedGrid, clusterCenter, peaks);
        }
    }

    /**
     * Render a single live heatmap panel
     */
    renderLiveHeatmapPanel(panel, normalizedGrid, clusterCenter, peaks) {
        const canvas = panel.canvas;
        if (!canvas) return;

        const ctx = this.getContext(canvas);
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;
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
                ctx.fillStyle = peak.active ? '#ff8800' : 'rgba(255, 136, 0, 0.4)';
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
     * Render all cluster memory panels
     */
    renderClustersHeatmap(memoryGrid) {
        if (!this.layoutManager || !memoryGrid) return;

        const panels = this.layoutManager.getPanelsOfType('memory');

        for (const panel of panels) {
            this.renderMemoryHeatmapPanel(panel, memoryGrid);
        }
    }

    /**
     * Render a single memory heatmap panel
     */
    renderMemoryHeatmapPanel(panel, memoryGrid) {
        const canvas = panel.canvas;
        if (!canvas) return;

        const ctx = this.getContext(canvas);
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;
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
     * Update all metrics panels
     */
    updateDebugMetrics(data) {
        if (!this.layoutManager) return;

        const panels = this.layoutManager.getPanelsOfType('metrics');

        for (const panel of panels) {
            this.updateMetricsPanel(panel, data);
        }
    }

    /**
     * Update a single metrics panel
     */
    updateMetricsPanel(panel, data) {
        const content = panel.content;

        const setEl = (selector, value) => {
            const el = content.querySelector(`[data-target^="${selector}"]`);
            if (el) el.textContent = value;
        };

        if (data.center) {
            setEl('metric-center', `${data.center.row?.toFixed(1)}, ${data.center.col?.toFixed(1)}`);
        }
        setEl('metric-confidence', data.confidence ? `${(data.confidence * 100).toFixed(0)}%` : '--%');
        setEl('metric-distance', data.distance ? `${data.distance.toFixed(1)}` : '--');
        setEl('metric-vx', data.vx?.toFixed(1) || '--');
        setEl('metric-vy', data.vy?.toFixed(1) || '--');
        setEl('metric-active', data.activeHotspots?.toString() || '--');
        setEl('metric-tracked', data.trackedHotspots?.toString() || '--');
        setEl('metric-movement', data.movement || '--');
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
