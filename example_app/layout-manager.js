/**
 * Layout Manager Module
 *
 * Provides flexible layout options for the surgeon UI:
 * - Tabs (single view)
 * - Horizontal split (2 views side by side)
 * - Vertical split (2 views stacked)
 * - 2x2 Grid (4 views)
 * - 1+2 Layout (main + 2 smaller)
 * - Triple horizontal (3 views side by side)
 *
 * Each slot can display any panel type.
 */

/**
 * Available layout configurations
 */
const LAYOUTS = {
    tabs: {
        name: 'Tabs',
        icon: '▣',
        slots: 1,
        css: 'layout-tabs',
        slotConfig: [{ id: 'main', default: 'guide' }]
    },
    hsplit: {
        name: 'Side by Side',
        icon: '◫',
        slots: 2,
        css: 'layout-hsplit',
        slotConfig: [
            { id: 'left', default: 'guide' },
            { id: 'right', default: 'global-map' }
        ]
    },
    vsplit: {
        name: 'Stacked',
        icon: '⬒',
        slots: 2,
        css: 'layout-vsplit',
        slotConfig: [
            { id: 'top', default: 'guide' },
            { id: 'bottom', default: 'global-map' }
        ]
    },
    grid: {
        name: '2x2 Grid',
        icon: '⊞',
        slots: 4,
        css: 'layout-grid',
        slotConfig: [
            { id: 'tl', default: 'guide' },
            { id: 'tr', default: 'global-map' },
            { id: 'bl', default: 'live' },
            { id: 'br', default: 'memory' }
        ]
    },
    mainSide: {
        name: 'Main + Side',
        icon: '◧',
        slots: 3,
        css: 'layout-main-side',
        slotConfig: [
            { id: 'main', default: 'guide' },
            { id: 'side-top', default: 'global-map' },
            { id: 'side-bottom', default: 'metrics' }
        ]
    },
    triple: {
        name: 'Triple',
        icon: '⫿',
        slots: 3,
        css: 'layout-triple',
        slotConfig: [
            { id: 'left', default: 'guide' },
            { id: 'center', default: 'global-map' },
            { id: 'right', default: 'live' }
        ]
    }
};

/**
 * Available panel types
 */
const PANEL_TYPES = {
    guide: {
        name: 'Direction Guide',
        description: 'Main surgeon view with direction arrow'
    },
    'global-map': {
        name: 'Global Map',
        description: 'Accumulated activity map'
    },
    live: {
        name: 'Live Activity',
        description: 'Real-time 32x32 heatmap'
    },
    memory: {
        name: 'Cluster Memory',
        description: 'Tracked hotspots with decay'
    },
    metrics: {
        name: 'Metrics',
        description: 'Numerical debug data'
    }
};

/**
 * Layout Manager Class
 */
class LayoutManager {
    constructor(containerId = 'layout-container') {
        this.containerId = containerId;
        this.container = null;
        this.currentLayout = 'tabs';
        this.panels = {};  // slotId -> Panel instance
        this.assignments = {};  // slotId -> panelType
        this.tabMode = true;  // For tabs layout, track which tab is active
        this.activeTab = 'guide';

        // Callbacks for rendering
        this.renderCallbacks = {};
    }

    /**
     * Initialize the layout manager
     */
    init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            console.error('Layout container not found:', this.containerId);
            return;
        }

        // Load saved layout from localStorage
        this.loadSavedLayout();

        // Build initial layout
        this.buildLayout();

        // Setup layout selector
        this.setupLayoutSelector();
    }

    /**
     * Load saved layout preferences
     */
    loadSavedLayout() {
        try {
            const saved = localStorage.getItem('arrayPlacementLayout');
            if (saved) {
                const data = JSON.parse(saved);
                this.currentLayout = data.layout || 'tabs';
                this.assignments = data.assignments || {};
                this.activeTab = data.activeTab || 'guide';
            }
        } catch (e) {
            console.warn('Could not load saved layout:', e);
        }
    }

    /**
     * Save layout preferences
     */
    saveLayout() {
        try {
            localStorage.setItem('arrayPlacementLayout', JSON.stringify({
                layout: this.currentLayout,
                assignments: this.assignments,
                activeTab: this.activeTab
            }));
        } catch (e) {
            console.warn('Could not save layout:', e);
        }
    }

    /**
     * Build the layout container and panels
     */
    buildLayout() {
        const layout = LAYOUTS[this.currentLayout];
        if (!layout) {
            console.error('Unknown layout:', this.currentLayout);
            return;
        }

        // Clear existing content
        this.container.innerHTML = '';
        this.container.className = `layout-container ${layout.css}`;
        this.panels = {};

        // Create slots
        for (const slotConfig of layout.slotConfig) {
            const slot = this.createSlot(slotConfig, layout);
            this.container.appendChild(slot);
        }

        // For tabs layout, show only active tab
        if (this.currentLayout === 'tabs') {
            this.updateTabVisibility();
        }
    }

    /**
     * Create a panel slot
     */
    createSlot(slotConfig, layout) {
        const slot = document.createElement('div');
        slot.className = 'panel-slot';
        slot.dataset.slotId = slotConfig.id;

        // Get assigned panel type or default
        const panelType = this.assignments[slotConfig.id] || slotConfig.default;
        this.assignments[slotConfig.id] = panelType;

        // Create panel header (with selector)
        const header = document.createElement('div');
        header.className = 'panel-header';

        // View selector dropdown
        const selector = document.createElement('select');
        selector.className = 'view-selector';
        for (const [type, info] of Object.entries(PANEL_TYPES)) {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = info.name;
            option.selected = type === panelType;
            selector.appendChild(option);
        }

        selector.addEventListener('change', (e) => {
            this.changePanel(slotConfig.id, e.target.value);
        });

        header.appendChild(selector);

        // Create panel content
        const content = document.createElement('div');
        content.className = 'panel-content';
        content.id = `panel-${slotConfig.id}`;

        // Build panel based on type
        this.buildPanelContent(content, panelType, slotConfig.id);

        slot.appendChild(header);
        slot.appendChild(content);

        // Store reference
        this.panels[slotConfig.id] = {
            slot,
            content,
            type: panelType,
            selector
        };

        return slot;
    }

    /**
     * Build panel content based on type
     */
    buildPanelContent(container, panelType, slotId) {
        container.innerHTML = '';
        container.className = `panel-content panel-${panelType}`;

        switch (panelType) {
            case 'guide':
                this.buildGuidePanel(container, slotId);
                break;
            case 'global-map':
                this.buildGlobalMapPanel(container, slotId);
                break;
            case 'live':
                this.buildLivePanel(container, slotId);
                break;
            case 'memory':
                this.buildMemoryPanel(container, slotId);
                break;
            case 'metrics':
                this.buildMetricsPanel(container, slotId);
                break;
        }
    }

    /**
     * Build guide panel (direction arrow)
     */
    buildGuidePanel(container, slotId) {
        container.innerHTML = `
            <div class="guide-panel-inner">
                <div class="arrow-container">
                    <svg viewBox="0 0 200 200" class="direction-arrow">
                        <circle cx="100" cy="100" r="95" class="arrow-bg" />
                        <circle cx="100" cy="100" r="20" class="target-zone" data-target="target-zone-${slotId}" />
                        <g class="arrow-group" data-target="arrow-group-${slotId}">
                            <line x1="100" y1="100" x2="100" y2="40" class="arrow-shaft" data-target="arrow-shaft-${slotId}" />
                            <polygon class="arrow-head" points="100,30 85,55 115,55" data-target="arrow-head-${slotId}" />
                        </g>
                        <g class="check-group" data-target="check-group-${slotId}">
                            <circle cx="100" cy="100" r="45" class="check-bg" />
                            <polyline points="75,100 92,117 125,84" class="check-mark" />
                        </g>
                    </svg>
                </div>
                <div class="instruction-area">
                    <div class="instruction-text" data-target="instruction-text-${slotId}">CONNECT</div>
                    <div class="instruction-detail" data-target="instruction-detail-${slotId}">Press Connect below</div>
                </div>
                <div class="distance-bar-container">
                    <div class="distance-bar">
                        <div class="distance-fill" data-target="distance-fill-${slotId}"></div>
                        <div class="distance-marker" data-target="distance-marker-${slotId}"></div>
                    </div>
                    <div class="distance-labels">
                        <span>FAR</span>
                        <span>ON TARGET</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Build global map panel
     */
    buildGlobalMapPanel(container, slotId) {
        container.innerHTML = `
            <div class="map-panel-inner">
                <canvas class="global-map-canvas" data-target="global-map-${slotId}" width="400" height="400"></canvas>
                <div class="map-info">
                    <div class="map-info-item">
                        <span class="map-label">Position</span>
                        <span class="map-value" data-target="map-pos-${slotId}">--, --</span>
                    </div>
                    <div class="map-info-item">
                        <span class="map-label">Best Cluster</span>
                        <span class="map-value" data-target="map-best-${slotId}">--</span>
                    </div>
                    <div class="map-info-item">
                        <span class="map-label">Explored</span>
                        <span class="map-value" data-target="map-explored-${slotId}">0%</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Build live activity panel
     */
    buildLivePanel(container, slotId) {
        container.innerHTML = `
            <div class="heatmap-panel-inner">
                <canvas class="heatmap-canvas" data-target="heatmap-live-${slotId}" width="320" height="320"></canvas>
            </div>
        `;
    }

    /**
     * Build memory panel
     */
    buildMemoryPanel(container, slotId) {
        container.innerHTML = `
            <div class="heatmap-panel-inner">
                <canvas class="heatmap-canvas" data-target="heatmap-memory-${slotId}" width="320" height="320"></canvas>
            </div>
        `;
    }

    /**
     * Build metrics panel
     */
    buildMetricsPanel(container, slotId) {
        container.innerHTML = `
            <div class="metrics-panel-inner">
                <div class="metrics-grid">
                    <div class="metric">
                        <span class="metric-label">Cluster Center</span>
                        <span class="metric-value" data-target="metric-center-${slotId}">--, --</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value" data-target="metric-confidence-${slotId}">--%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Distance</span>
                        <span class="metric-value" data-target="metric-distance-${slotId}">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Cursor Vx</span>
                        <span class="metric-value" data-target="metric-vx-${slotId}">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Cursor Vy</span>
                        <span class="metric-value" data-target="metric-vy-${slotId}">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Hotspots</span>
                        <span class="metric-value" data-target="metric-active-${slotId}">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Tracked Total</span>
                        <span class="metric-value" data-target="metric-tracked-${slotId}">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Movement</span>
                        <span class="metric-value" data-target="metric-movement-${slotId}">--</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Change panel type in a slot
     */
    changePanel(slotId, newType) {
        const panel = this.panels[slotId];
        if (!panel) return;

        this.assignments[slotId] = newType;
        panel.type = newType;

        this.buildPanelContent(panel.content, newType, slotId);
        this.saveLayout();

        // Trigger re-render
        if (this.onPanelChange) {
            this.onPanelChange(slotId, newType);
        }
    }

    /**
     * Change layout type
     */
    setLayout(layoutType) {
        if (!LAYOUTS[layoutType]) {
            console.error('Unknown layout:', layoutType);
            return;
        }

        this.currentLayout = layoutType;
        this.tabMode = layoutType === 'tabs';

        // Reset assignments to defaults for new layout
        const layout = LAYOUTS[layoutType];
        this.assignments = {};
        for (const slotConfig of layout.slotConfig) {
            this.assignments[slotConfig.id] = slotConfig.default;
        }

        this.buildLayout();
        this.saveLayout();

        // Update selector button states
        this.updateLayoutSelectorState();

        // Trigger re-render
        if (this.onLayoutChange) {
            this.onLayoutChange(layoutType);
        }
    }

    /**
     * Setup layout selector buttons
     */
    setupLayoutSelector() {
        const selector = document.getElementById('layout-selector');
        if (!selector) return;

        selector.innerHTML = '';

        for (const [type, layout] of Object.entries(LAYOUTS)) {
            const btn = document.createElement('button');
            btn.className = 'layout-btn';
            btn.dataset.layout = type;
            btn.title = layout.name;
            btn.innerHTML = `<span class="layout-icon">${layout.icon}</span>`;

            if (type === this.currentLayout) {
                btn.classList.add('active');
            }

            btn.addEventListener('click', () => {
                this.setLayout(type);
            });

            selector.appendChild(btn);
        }
    }

    /**
     * Update layout selector button states
     */
    updateLayoutSelectorState() {
        const buttons = document.querySelectorAll('.layout-btn');
        buttons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.layout === this.currentLayout);
        });
    }

    /**
     * For tabs layout: switch active tab
     */
    switchTab(panelType) {
        if (this.currentLayout !== 'tabs') return;

        this.activeTab = panelType;
        this.assignments['main'] = panelType;

        const panel = this.panels['main'];
        if (panel) {
            panel.selector.value = panelType;
            this.buildPanelContent(panel.content, panelType, 'main');
        }

        this.saveLayout();

        // Update tab buttons
        this.updateTabButtons();
    }

    /**
     * Update tab button states (for tabs layout)
     */
    updateTabButtons() {
        const buttons = document.querySelectorAll('.tab-btn');
        buttons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === this.activeTab);
        });
    }

    /**
     * Update tab visibility (for tabs layout)
     */
    updateTabVisibility() {
        // In tabs mode, we just have one slot that changes content
        // This is handled by switchTab/changePanel
    }

    /**
     * Get all panels of a specific type (for rendering)
     */
    getPanelsOfType(panelType) {
        const result = [];
        for (const [slotId, panel] of Object.entries(this.panels)) {
            if (panel.type === panelType) {
                result.push({
                    slotId,
                    content: panel.content,
                    canvas: panel.content.querySelector('canvas')
                });
            }
        }
        return result;
    }

    /**
     * Get element by data-target attribute
     */
    getElement(target) {
        return this.container.querySelector(`[data-target="${target}"]`);
    }

    /**
     * Get all elements matching a target prefix
     */
    getElementsWithPrefix(prefix) {
        return this.container.querySelectorAll(`[data-target^="${prefix}"]`);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LayoutManager, LAYOUTS, PANEL_TYPES };
}
