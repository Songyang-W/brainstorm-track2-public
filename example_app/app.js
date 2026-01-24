/**
 * Array Placement Guide - Main Application
 *
 * Modular, surgeon-focused UI with flexible layouts.
 * ONE JOB: Tell the surgeon which way to move.
 */

class ArrayPlacementApp {
    constructor() {
        // WebSocket
        this.ws = null;
        this.isConnected = false;
        this.serverUrl = 'ws://localhost:8765';

        // Config from init message
        this.channelsCoords = null;
        this.gridSize = 32;
        this.fs = 500;

        // Processing modules
        this.signalProcessor = null;
        this.clusterTracker = null;
        this.globalMap = null;

        // UI modules
        this.layoutManager = null;
        this.visualizer = null;

        // Frame management
        this.targetFps = 30;
        this.frameInterval = 1000 / this.targetFps;
        this.lastFrameTime = 0;
        this.sampleBuffer = [];
        this.timeBuffer = [];

        // FPS tracking
        this.frameCount = 0;
        this.lastFpsUpdate = performance.now();
        this.currentFps = 0;

        // State
        this.currentTime = 0;
        this.cursorVelocity = { vx: 0, vy: 0 };
    }

    /**
     * Initialize the application
     */
    init() {
        console.log('Array Placement Guide initializing...');

        // Initialize processing modules
        this.signalProcessor = new SignalProcessor({
            fs: this.fs,
            nChannels: 1024,
            gridSize: this.gridSize,
            lowcut: 70,
            highcut: 150,
            alpha: 0.2,
            spatialSigma: 1.5
        });

        this.clusterTracker = new ClusterTracker({
            gridSize: this.gridSize,
            threshold: 0.35,
            minClusterSize: 3,
            hotspotDecay: 0.98,
            peakTimeout: 10000
        });

        this.globalMap = new GlobalBrainMap({
            mapSize: 128,
            decayRate: 0.998,
            movementScale: 1.5
        });

        // Initialize layout manager
        this.layoutManager = new LayoutManager('layout-container');
        this.layoutManager.init();

        // Handle layout changes - re-init canvases
        this.layoutManager.onLayoutChange = () => {
            this.visualizer.canvasContexts.clear();
        };

        this.layoutManager.onPanelChange = () => {
            this.visualizer.canvasContexts.clear();
        };

        // Initialize visualizer with layout manager
        this.visualizer = new Visualizer();
        this.visualizer.init(this.layoutManager);

        // Setup UI handlers
        this.setupUIHandlers();

        // Get initial URL
        const wsInput = document.getElementById('ws-input');
        if (wsInput) {
            this.serverUrl = wsInput.value;
        }

        console.log('Array Placement Guide ready.');
    }

    /**
     * Setup UI event handlers
     */
    setupUIHandlers() {
        // Controls panel toggle
        const controlsToggle = document.getElementById('controls-toggle');
        const controlsPanel = document.getElementById('controls-panel');
        if (controlsToggle && controlsPanel) {
            controlsToggle.addEventListener('click', () => {
                controlsPanel.classList.toggle('expanded');
            });
        }

        // Connect button
        const connectBtn = document.getElementById('connect-btn');
        if (connectBtn) {
            connectBtn.addEventListener('click', () => {
                if (this.isConnected) {
                    this.disconnect();
                } else {
                    this.connect();
                }
            });
        }

        // Preset button (live server)
        const presetBtn = document.getElementById('preset-live');
        const wsInput = document.getElementById('ws-input');
        if (presetBtn && wsInput) {
            presetBtn.addEventListener('click', () => {
                wsInput.value = 'ws://192.168.1.152:8765/stream';
                this.serverUrl = wsInput.value;
            });
        }

        // URL input
        if (wsInput) {
            wsInput.addEventListener('change', () => {
                this.serverUrl = wsInput.value;
            });
            wsInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.serverUrl = wsInput.value;
                    this.connect();
                }
            });
        }

        // Reset button
        const resetBtn = document.getElementById('reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetState();
            });
        }

        // Audio toggle button
        const audioBtn = document.getElementById('audio-btn');
        if (audioBtn) {
            audioBtn.addEventListener('click', () => {
                const isMuted = audioBtn.classList.toggle('muted');
                audioBtn.textContent = isMuted ? 'Sound: OFF' : 'Sound: ON';
                if (this.visualizer) {
                    this.visualizer.toggleAudio(!isMuted);
                }
            });
        }

        // Prevent arrow keys from navigating DOM elements
        // This stops Chrome from moving focus between elements when arrow keys are pressed
        document.addEventListener('keydown', (e) => {
            const arrowKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
            if (arrowKeys.includes(e.key)) {
                // Only prevent if not in an input/textarea
                const tag = e.target.tagName.toLowerCase();
                if (tag !== 'input' && tag !== 'textarea' && tag !== 'select') {
                    e.preventDefault();
                }
            }
        });
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        if (this.isConnected) {
            this.disconnect();
            return;
        }

        const wsInput = document.getElementById('ws-input');
        if (wsInput) {
            this.serverUrl = wsInput.value;
        }

        console.log(`Connecting to ${this.serverUrl}...`);
        this.visualizer.updateStatus({ connected: 'connecting' });

        try {
            this.ws = new WebSocket(this.serverUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.visualizer.updateStatus({ connected: true });
                this.resetState();
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.visualizer.updateStatus({ connected: false });
            };

            this.ws.onclose = () => {
                console.log('WebSocket closed');
                this.isConnected = false;
                this.visualizer.updateStatus({ connected: false });
                this.ws = null;
            };

        } catch (err) {
            console.error('Failed to connect:', err);
            this.visualizer.updateStatus({ connected: false });
        }
    }

    /**
     * Disconnect
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
        this.visualizer.updateStatus({ connected: false });
    }

    /**
     * Reset processing state
     */
    resetState() {
        this.sampleBuffer = [];
        this.timeBuffer = [];
        this.frameCount = 0;
        this.lastFpsUpdate = performance.now();
        this.lastFrameTime = 0;
        this.currentTime = 0;
        this.cursorVelocity = { vx: 0, vy: 0 };

        if (this.signalProcessor) this.signalProcessor.reset();
        if (this.clusterTracker) this.clusterTracker.reset();
        if (this.globalMap) this.globalMap.reset();

        // Reset main view
        this.visualizer.updateMainView(null);
    }

    /**
     * Handle incoming message
     */
    handleMessage(data) {
        try {
            const msg = JSON.parse(data);

            if (msg.type === 'init') {
                this.handleInit(msg);
            } else if (msg.type === 'sample_batch') {
                this.handleSampleBatch(msg);
            }
        } catch (err) {
            console.error('Error parsing message:', err);
        }
    }

    /**
     * Handle init message
     */
    handleInit(msg) {
        console.log('Received init:', msg);

        this.channelsCoords = msg.channels_coords;
        this.gridSize = msg.grid_size || 32;
        this.fs = msg.fs || 500;

        // Reconfigure processor
        this.signalProcessor = new SignalProcessor({
            fs: this.fs,
            nChannels: this.channelsCoords.length,
            gridSize: this.gridSize,
            lowcut: 70,
            highcut: 150,
            alpha: 0.2,
            spatialSigma: 1.5
        });
    }

    /**
     * Handle sample batch
     */
    handleSampleBatch(msg) {
        const neuralData = msg.neural_data;
        const startTimeS = msg.start_time_s || 0;
        const sampleCount = msg.sample_count || neuralData.length;
        const fs = msg.fs || this.fs;
        const dt = 1.0 / fs;

        // Extract cursor velocity if present
        if (msg.cursor_data && msg.cursor_data.length > 0) {
            let totalVx = 0, totalVy = 0;
            for (const cursor of msg.cursor_data) {
                totalVx += cursor.vx || 0;
                totalVy += cursor.vy || 0;
            }
            this.cursorVelocity = {
                vx: totalVx / msg.cursor_data.length,
                vy: totalVy / msg.cursor_data.length
            };
        }

        // Buffer samples
        for (let i = 0; i < sampleCount; i++) {
            const sampleTime = startTimeS + i * dt;
            this.sampleBuffer.push(neuralData[i]);
            this.timeBuffer.push(sampleTime);
        }

        // Process frame if ready
        const now = performance.now();
        if (now - this.lastFrameTime >= this.frameInterval) {
            this.processFrame();
            this.lastFrameTime = now;
        }
    }

    /**
     * Process a frame
     */
    processFrame() {
        if (this.sampleBuffer.length === 0) return;

        this.currentTime = this.timeBuffer[this.timeBuffer.length - 1];

        const batch = this.sampleBuffer.slice();
        this.sampleBuffer = [];
        this.timeBuffer = [];

        // 1. Signal processing
        const processed = this.signalProcessor.processBatch(batch);

        // 2. Compute cluster center (with memory for off hotspots)
        const clusterCenter = this.clusterTracker.computeClusterCenter(processed.normalized);

        // 3. Infer array movement
        const movement = this.clusterTracker.inferArrayMovement();

        // 4. Get guidance direction
        const guidance = this.clusterTracker.getGuidanceDirection();

        // 5. Update global map (now with cluster entities for better anchoring)
        this.globalMap.update(
            processed.normalized,
            movement,
            this.clusterTracker.peakPositions,
            this.clusterTracker.getClusters()  // Pass cluster entities
        );

        // 6. Update main view
        if (guidance) {
            this.visualizer.updateMainView({
                direction: guidance,
                isOnTarget: guidance.isOnTarget,
                distance: guidance.distance,
                distanceMm: guidance.distanceMm
            });
        } else {
            this.visualizer.updateMainView(null);
        }

        // 7. Render global map
        this.visualizer.renderGlobalMap(this.globalMap);

        // 8. Render debug views
        this.visualizer.renderLiveHeatmap(
            processed.normalized,
            clusterCenter,
            this.clusterTracker.peakPositions
        );
        this.visualizer.renderClustersHeatmap(this.clusterTracker.getMemoryGrid());

        // 9. Update debug metrics
        this.visualizer.updateDebugMetrics({
            center: clusterCenter,
            confidence: guidance ? 1 - guidance.magnitude : 0,
            distance: guidance?.distance,
            vx: this.cursorVelocity.vx,
            vy: this.cursorVelocity.vy,
            activeHotspots: clusterCenter?.activePeaks || 0,
            trackedHotspots: this.clusterTracker.peakPositions.length,
            clusterCount: this.clusterTracker.getClusters().length,
            totalIntensity: clusterCenter?.totalIntensity || 0,
            connected: this.isConnected,
            movement: movement.confidence > 0.1 ?
                `${movement.dx.toFixed(2)}, ${movement.dy.toFixed(2)}` : 'stable'
        });

        // 10. Update FPS
        this.updateFps();

        // 11. Update status
        this.visualizer.updateStatus({
            time: this.currentTime,
            fps: this.currentFps
        });
    }

    /**
     * Update FPS counter
     */
    updateFps() {
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsUpdate >= 1000) {
            this.currentFps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
    }
}

// Initialize on DOM ready
let app = null;

document.addEventListener('DOMContentLoaded', () => {
    app = new ArrayPlacementApp();
    app.init();
});

// Export for debugging
if (typeof window !== 'undefined') {
    window.ArrayPlacementApp = ArrayPlacementApp;
    window.app = app;
}
