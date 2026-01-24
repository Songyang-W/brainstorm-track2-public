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
    },
    freeform: {
        name: 'Freeform',
        icon: '⧉',
        slots: 0,  // Dynamic - user adds windows
        css: 'layout-freeform',
        slotConfig: []  // Windows are added dynamically
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
 * Now with resizable panels
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

        // Resize state
        this.isResizing = false;
        this.resizeDirection = null;  // 'horizontal' or 'vertical'
        this.resizeStartPos = null;
        this.resizeSizes = {};  // Stores custom sizes for layouts

        // Freeform layout state
        this.freeformArea = null;

        // Bind resize handlers
        this.onResizeStart = this.onResizeStart.bind(this);
        this.onResizeMove = this.onResizeMove.bind(this);
        this.onResizeEnd = this.onResizeEnd.bind(this);
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
                this.resizeSizes = data.resizeSizes || {};
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
                activeTab: this.activeTab,
                resizeSizes: this.resizeSizes
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

        // Apply saved resize sizes if available
        this.applySavedSizes();

        // Create slots and handles based on layout type
        const slotConfigs = layout.slotConfig;

        if (this.currentLayout === 'tabs') {
            // Tabs: just one slot, no handles
            for (const slotConfig of slotConfigs) {
                const slot = this.createSlot(slotConfig, layout);
                this.container.appendChild(slot);
            }
            this.updateTabVisibility();
        } else if (layout.css === 'layout-hsplit') {
            // Horizontal split: panel, handle, panel
            this.container.appendChild(this.createSlot(slotConfigs[0], layout));
            this.container.appendChild(this.createResizeHandle('horizontal', 0));
            this.container.appendChild(this.createSlot(slotConfigs[1], layout));
        } else if (layout.css === 'layout-vsplit') {
            // Vertical split: panel, handle, panel
            this.container.appendChild(this.createSlot(slotConfigs[0], layout));
            this.container.appendChild(this.createResizeHandle('vertical', 0));
            this.container.appendChild(this.createSlot(slotConfigs[1], layout));
        } else if (layout.css === 'layout-grid') {
            // 2x2 Grid: 4 panels + 1 vertical handle + 1 horizontal handle
            this.container.appendChild(this.createSlot(slotConfigs[0], layout));  // TL
            this.container.appendChild(this.createSlot(slotConfigs[1], layout));  // TR
            this.container.appendChild(this.createSlot(slotConfigs[2], layout));  // BL
            this.container.appendChild(this.createSlot(slotConfigs[3], layout));  // BR
            this.container.appendChild(this.createResizeHandle('horizontal', 0)); // Vertical divider
            this.container.appendChild(this.createResizeHandle('vertical', 0));   // Horizontal divider
        } else if (layout.css === 'layout-main-side') {
            // Main + Side: main panel, vertical handle, 2 side panels with horizontal handle
            this.container.appendChild(this.createSlot(slotConfigs[0], layout));  // Main
            this.container.appendChild(this.createSlot(slotConfigs[1], layout));  // Side top
            this.container.appendChild(this.createSlot(slotConfigs[2], layout));  // Side bottom
            this.container.appendChild(this.createResizeHandle('horizontal', 0)); // Vertical between main and side
            this.container.appendChild(this.createResizeHandle('vertical', 0));   // Horizontal between side panels
        } else if (layout.css === 'layout-triple') {
            // Triple: panel, handle, panel, handle, panel
            this.container.appendChild(this.createSlot(slotConfigs[0], layout));
            this.container.appendChild(this.createResizeHandle('horizontal', 0));
            this.container.appendChild(this.createSlot(slotConfigs[1], layout));
            this.container.appendChild(this.createResizeHandle('horizontal', 1));
            this.container.appendChild(this.createSlot(slotConfigs[2], layout));
        } else if (layout.css === 'layout-freeform') {
            // Freeform: add toolbar and create windows from saved state
            this.buildFreeformLayout();
        }

        // Setup global resize listeners
        document.addEventListener('mousemove', this.onResizeMove);
        document.addEventListener('mouseup', this.onResizeEnd);
        document.addEventListener('touchmove', this.onResizeMove, { passive: false });
        document.addEventListener('touchend', this.onResizeEnd);
    }

    /**
     * Build freeform layout with draggable windows
     */
    buildFreeformLayout() {
        // Create toolbar for adding windows
        const toolbar = document.createElement('div');
        toolbar.className = 'freeform-toolbar';
        toolbar.innerHTML = `
            <span class="freeform-toolbar-label">Add Window:</span>
            <button class="freeform-add-btn" data-type="guide">Direction</button>
            <button class="freeform-add-btn" data-type="global-map">Map</button>
            <button class="freeform-add-btn" data-type="live">Live</button>
            <button class="freeform-add-btn" data-type="memory">Memory</button>
            <button class="freeform-add-btn" data-type="metrics">Metrics</button>
        `;
        this.container.appendChild(toolbar);

        // Add click handlers
        toolbar.querySelectorAll('.freeform-add-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.addFreeformWindow(btn.dataset.type);
            });
        });

        // Create windows area
        const windowsArea = document.createElement('div');
        windowsArea.className = 'freeform-windows-area';
        windowsArea.id = 'freeform-windows-area';
        this.container.appendChild(windowsArea);
        this.freeformArea = windowsArea;

        // Load saved windows or create defaults
        const savedWindows = this.resizeSizes.freeformWindows || [];
        if (savedWindows.length === 0) {
            // Default windows
            this.addFreeformWindow('guide', { x: 20, y: 20, width: 400, height: 350 });
            this.addFreeformWindow('global-map', { x: 440, y: 20, width: 350, height: 350 });
        } else {
            for (const win of savedWindows) {
                this.addFreeformWindow(win.type, win);
            }
        }
    }

    /**
     * Add a freeform window
     */
    addFreeformWindow(panelType, position = null) {
        if (!this.freeformArea) return;

        const windowId = `freeform-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;

        // Default position if not specified
        const pos = position || {
            x: 50 + Object.keys(this.panels).length * 30,
            y: 50 + Object.keys(this.panels).length * 30,
            width: 320,
            height: 280
        };

        // Create window element
        const win = document.createElement('div');
        win.className = 'freeform-window';
        win.id = windowId;
        win.style.left = `${pos.x}px`;
        win.style.top = `${pos.y}px`;
        win.style.width = `${pos.width}px`;
        win.style.height = `${pos.height}px`;

        // Window header (draggable)
        const header = document.createElement('div');
        header.className = 'freeform-window-header';

        const title = document.createElement('span');
        title.className = 'freeform-window-title';
        title.textContent = PANEL_TYPES[panelType]?.name || panelType;

        const closeBtn = document.createElement('button');
        closeBtn.className = 'freeform-window-close';
        closeBtn.innerHTML = '×';
        closeBtn.addEventListener('click', () => {
            this.removeFreeformWindow(windowId);
        });

        header.appendChild(title);
        header.appendChild(closeBtn);

        // Window content
        const content = document.createElement('div');
        content.className = 'panel-content';
        content.id = `panel-${windowId}`;
        this.buildPanelContent(content, panelType, windowId);

        // Resize handle
        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'freeform-resize-handle';

        win.appendChild(header);
        win.appendChild(content);
        win.appendChild(resizeHandle);

        this.freeformArea.appendChild(win);

        // Store reference
        this.panels[windowId] = {
            slot: win,
            content,
            type: panelType,
            position: pos
        };

        // Setup drag and resize
        this.setupFreeformWindowDrag(win, header, windowId);
        this.setupFreeformWindowResize(win, resizeHandle, windowId);

        // Bring to front on click
        win.addEventListener('mousedown', () => {
            this.bringWindowToFront(win);
        });

        this.saveFreeformWindows();

        // Notify of panel change
        if (this.onPanelChange) {
            this.onPanelChange(windowId, panelType);
        }
    }

    /**
     * Remove a freeform window
     */
    removeFreeformWindow(windowId) {
        const panel = this.panels[windowId];
        if (!panel) return;

        panel.slot.remove();
        delete this.panels[windowId];
        this.saveFreeformWindows();
    }

    /**
     * Setup drag for freeform window
     */
    setupFreeformWindowDrag(win, header, windowId) {
        let isDragging = false;
        let startX, startY, initialX, initialY;

        const onMouseDown = (e) => {
            if (e.target.closest('.freeform-window-close')) return;
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            initialX = win.offsetLeft;
            initialY = win.offsetTop;
            this.bringWindowToFront(win);
            document.body.style.userSelect = 'none';
        };

        const onMouseMove = (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            win.style.left = `${Math.max(0, initialX + dx)}px`;
            win.style.top = `${Math.max(0, initialY + dy)}px`;
        };

        const onMouseUp = () => {
            if (isDragging) {
                isDragging = false;
                document.body.style.userSelect = '';
                this.updateFreeformWindowPosition(windowId, win);
            }
        };

        header.addEventListener('mousedown', onMouseDown);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    }

    /**
     * Setup resize for freeform window
     */
    setupFreeformWindowResize(win, handle, windowId) {
        let isResizing = false;
        let startX, startY, startWidth, startHeight;

        const onMouseDown = (e) => {
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            startWidth = win.offsetWidth;
            startHeight = win.offsetHeight;
            document.body.style.userSelect = 'none';
            e.stopPropagation();
        };

        const onMouseMove = (e) => {
            if (!isResizing) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            win.style.width = `${Math.max(200, startWidth + dx)}px`;
            win.style.height = `${Math.max(150, startHeight + dy)}px`;
        };

        const onMouseUp = () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.userSelect = '';
                this.updateFreeformWindowPosition(windowId, win);
                if (this.onPanelChange) {
                    this.onPanelChange();
                }
            }
        };

        handle.addEventListener('mousedown', onMouseDown);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    }

    /**
     * Bring window to front
     */
    bringWindowToFront(win) {
        const windows = this.freeformArea?.querySelectorAll('.freeform-window') || [];
        let maxZ = 10;
        windows.forEach(w => {
            const z = parseInt(w.style.zIndex || 10);
            if (z > maxZ) maxZ = z;
        });
        win.style.zIndex = maxZ + 1;
    }

    /**
     * Update freeform window position in storage
     */
    updateFreeformWindowPosition(windowId, win) {
        const panel = this.panels[windowId];
        if (panel) {
            panel.position = {
                x: win.offsetLeft,
                y: win.offsetTop,
                width: win.offsetWidth,
                height: win.offsetHeight
            };
            this.saveFreeformWindows();
        }
    }

    /**
     * Save freeform windows to storage
     */
    saveFreeformWindows() {
        if (this.currentLayout !== 'freeform') return;

        const windows = [];
        for (const [id, panel] of Object.entries(this.panels)) {
            if (id.startsWith('freeform-')) {
                windows.push({
                    type: panel.type,
                    ...panel.position
                });
            }
        }
        this.resizeSizes.freeformWindows = windows;
        this.saveLayout();
    }

    /**
     * Create a resize handle
     * @param {string} direction - 'horizontal' (drag left/right) or 'vertical' (drag up/down)
     * @param {number} index - Index for multiple handles of same direction
     */
    createResizeHandle(direction, index) {
        const handle = document.createElement('div');
        handle.className = 'resize-handle';
        handle.dataset.index = index;
        handle.dataset.direction = direction;

        if (direction === 'horizontal') {
            handle.classList.add('resize-handle-vertical');  // Vertical bar for horizontal dragging
        } else {
            handle.classList.add('resize-handle-horizontal');  // Horizontal bar for vertical dragging
        }

        handle.addEventListener('mousedown', this.onResizeStart);
        handle.addEventListener('touchstart', this.onResizeStart, { passive: false });

        return handle;
    }

    /**
     * Handle resize start
     */
    onResizeStart(e) {
        e.preventDefault();
        this.isResizing = true;

        const handle = e.target;
        this.resizeDirection = handle.dataset.direction;
        this.resizeIndex = parseInt(handle.dataset.index);

        const touch = e.touches ? e.touches[0] : e;
        this.resizeStartPos = {
            x: touch.clientX,
            y: touch.clientY
        };

        // Get current container dimensions
        this.containerRect = this.container.getBoundingClientRect();

        // Add resizing class for cursor feedback
        document.body.classList.add('resizing');
        document.body.classList.add(`resize-${this.resizeDirection}`);
        handle.classList.add('active');
        this.activeHandle = handle;
    }

    /**
     * Handle resize move
     */
    onResizeMove(e) {
        if (!this.isResizing) return;
        e.preventDefault();

        const touch = e.touches ? e.touches[0] : e;
        const deltaX = touch.clientX - this.resizeStartPos.x;
        const deltaY = touch.clientY - this.resizeStartPos.y;

        const layout = LAYOUTS[this.currentLayout];

        if (this.resizeDirection === 'horizontal') {
            // Adjust column widths
            const totalWidth = this.containerRect.width;
            const deltaPercent = (deltaX / totalWidth) * 100;

            this.adjustHorizontalSizes(layout, deltaPercent);
        } else if (this.resizeDirection === 'vertical') {
            // Adjust row heights
            const totalHeight = this.containerRect.height;
            const deltaPercent = (deltaY / totalHeight) * 100;

            this.adjustVerticalSizes(layout, deltaPercent);
        }

        // Update start position for continuous dragging
        this.resizeStartPos = {
            x: touch.clientX,
            y: touch.clientY
        };
    }

    /**
     * Adjust horizontal (column) sizes
     */
    adjustHorizontalSizes(layout, deltaPercent) {
        const key = `${this.currentLayout}_cols`;
        if (!this.resizeSizes[key]) {
            // Initialize with equal sizes
            if (layout.css === 'layout-hsplit') {
                this.resizeSizes[key] = [50, 50];
            } else if (layout.css === 'layout-triple') {
                this.resizeSizes[key] = [33.33, 33.33, 33.34];
            } else if (layout.css === 'layout-grid') {
                this.resizeSizes[key] = [50, 50];
            } else if (layout.css === 'layout-main-side') {
                this.resizeSizes[key] = [66.67, 33.33];
            }
        }

        const sizes = this.resizeSizes[key];
        const idx = this.resizeIndex;

        // Adjust sizes (min 15%, max 85%)
        if (idx < sizes.length - 1) {
            const newLeft = Math.max(15, Math.min(85, sizes[idx] + deltaPercent));
            const diff = newLeft - sizes[idx];
            sizes[idx] = newLeft;
            sizes[idx + 1] = Math.max(15, sizes[idx + 1] - diff);
        }

        this.applyColumnSizes(sizes);
        this.saveLayout();
    }

    /**
     * Adjust vertical (row) sizes
     */
    adjustVerticalSizes(layout, deltaPercent) {
        const key = `${this.currentLayout}_rows`;
        if (!this.resizeSizes[key]) {
            // Initialize with equal sizes
            if (layout.css === 'layout-vsplit') {
                this.resizeSizes[key] = [50, 50];
            } else if (layout.css === 'layout-grid') {
                this.resizeSizes[key] = [50, 50];
            } else if (layout.css === 'layout-main-side') {
                this.resizeSizes[key] = [50, 50];
            }
        }

        const sizes = this.resizeSizes[key];
        const idx = this.resizeDirection === 'vertical' ? 0 : this.resizeIndex;

        // Adjust sizes (min 15%, max 85%)
        if (idx < sizes.length - 1) {
            const newTop = Math.max(15, Math.min(85, sizes[idx] + deltaPercent));
            const diff = newTop - sizes[idx];
            sizes[idx] = newTop;
            sizes[idx + 1] = Math.max(15, sizes[idx + 1] - diff);
        }

        this.applyRowSizes(sizes);
        this.saveLayout();
    }

    /**
     * Apply column sizes to CSS grid
     */
    applyColumnSizes(sizes) {
        const template = sizes.map(s => `${s}%`).join(' 6px ');  // 6px for handle
        this.container.style.gridTemplateColumns = template;
    }

    /**
     * Apply row sizes to CSS grid
     */
    applyRowSizes(sizes) {
        const template = sizes.map(s => `${s}%`).join(' 6px ');  // 6px for handle
        this.container.style.gridTemplateRows = template;
    }

    /**
     * Apply saved sizes on layout build
     */
    applySavedSizes() {
        // Clear any inline styles first (important when switching layouts!)
        this.container.style.gridTemplateColumns = '';
        this.container.style.gridTemplateRows = '';

        const colKey = `${this.currentLayout}_cols`;
        const rowKey = `${this.currentLayout}_rows`;

        if (this.resizeSizes[colKey]) {
            this.applyColumnSizes(this.resizeSizes[colKey]);
        }
        if (this.resizeSizes[rowKey]) {
            this.applyRowSizes(this.resizeSizes[rowKey]);
        }
    }

    /**
     * Handle resize end
     */
    onResizeEnd() {
        if (!this.isResizing) return;

        this.isResizing = false;
        document.body.classList.remove('resizing');
        document.body.classList.remove('resize-horizontal');
        document.body.classList.remove('resize-vertical');

        if (this.activeHandle) {
            this.activeHandle.classList.remove('active');
            this.activeHandle = null;
        }

        this.resizeDirection = null;
        this.resizeStartPos = null;

        // Trigger re-render for canvases
        if (this.onPanelChange) {
            this.onPanelChange();
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
     * Build metrics panel - Enhanced with graphs and color coding
     */
    buildMetricsPanel(container, slotId) {
        container.innerHTML = `
            <div class="metrics-panel-inner">
                <!-- Status indicators at top -->
                <div class="metrics-status-bar">
                    <div class="status-indicator" data-target="status-connection-${slotId}">
                        <span class="status-dot-sm"></span>
                        <span class="status-label">Signal</span>
                    </div>
                    <div class="status-indicator" data-target="status-tracking-${slotId}">
                        <span class="status-dot-sm"></span>
                        <span class="status-label">Tracking</span>
                    </div>
                    <div class="status-indicator" data-target="status-target-${slotId}">
                        <span class="status-dot-sm"></span>
                        <span class="status-label">On Target</span>
                    </div>
                </div>

                <!-- Main metrics grid -->
                <div class="metrics-grid">
                    <div class="metric metric-primary">
                        <span class="metric-label">Distance to Center</span>
                        <span class="metric-value metric-value-lg" data-target="metric-distance-${slotId}">--</span>
                        <span class="metric-unit">grid units</span>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" data-target="metric-distance-bar-${slotId}"></div>
                        </div>
                    </div>
                    <div class="metric metric-primary">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value metric-value-lg" data-target="metric-confidence-${slotId}">--%</span>
                        <div class="metric-bar metric-bar-green">
                            <div class="metric-bar-fill" data-target="metric-confidence-bar-${slotId}"></div>
                        </div>
                    </div>
                </div>

                <!-- Activity graph -->
                <div class="metrics-graph-section">
                    <span class="metrics-section-title">Signal Intensity</span>
                    <canvas class="metrics-graph" data-target="metrics-graph-intensity-${slotId}" width="300" height="60"></canvas>
                </div>

                <!-- Position graph -->
                <div class="metrics-graph-section">
                    <span class="metrics-section-title">Position (X/Y)</span>
                    <canvas class="metrics-graph" data-target="metrics-graph-position-${slotId}" width="300" height="60"></canvas>
                </div>

                <!-- Secondary metrics -->
                <div class="metrics-grid metrics-grid-compact">
                    <div class="metric metric-compact">
                        <span class="metric-label">Center</span>
                        <span class="metric-value" data-target="metric-center-${slotId}">--, --</span>
                    </div>
                    <div class="metric metric-compact">
                        <span class="metric-label">Hotspots</span>
                        <span class="metric-value">
                            <span class="metric-highlight" data-target="metric-active-${slotId}">--</span>
                            <span class="metric-dim">/ </span>
                            <span data-target="metric-tracked-${slotId}">--</span>
                        </span>
                    </div>
                    <div class="metric metric-compact">
                        <span class="metric-label">Cursor Vx</span>
                        <span class="metric-value" data-target="metric-vx-${slotId}">--</span>
                    </div>
                    <div class="metric metric-compact">
                        <span class="metric-label">Cursor Vy</span>
                        <span class="metric-value" data-target="metric-vy-${slotId}">--</span>
                    </div>
                    <div class="metric metric-compact">
                        <span class="metric-label">Clusters</span>
                        <span class="metric-value" data-target="metric-clusters-${slotId}">--</span>
                    </div>
                    <div class="metric metric-compact">
                        <span class="metric-label">Movement</span>
                        <span class="metric-value" data-target="metric-movement-${slotId}">--</span>
                    </div>
                </div>

                <!-- Peak intensity -->
                <div class="metrics-intensity-row">
                    <span class="metric-label">Peak Intensity</span>
                    <div class="intensity-bar-container">
                        <div class="intensity-bar" data-target="metric-intensity-bar-${slotId}"></div>
                    </div>
                    <span class="metric-value" data-target="metric-intensity-${slotId}">--</span>
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
