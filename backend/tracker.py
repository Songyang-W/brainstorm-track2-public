"""
Evidence accumulation and hotspot tracking for BCI guidance.

Implements:
- Persistent belief map that builds over time
- Spatial smoothing and clustering
- Centroid detection and guidance generation
- Global brain mapping with motion estimation
"""

from typing import Any

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label


class EvidenceTracker:
    """Accumulate evidence of neural activity over time."""

    def __init__(
        self,
        grid_size: int = 32,
        accumulation_alpha: float = 0.05,
        activity_threshold: float = 0.4,
        decay_rate: float = 0.995,
    ):
        """
        Initialize evidence tracker.

        Args:
            grid_size: Size of the electrode grid (32x32)
            accumulation_alpha: Rate of evidence accumulation
            activity_threshold: Minimum activity to accumulate
            decay_rate: How fast evidence decays over time
        """
        self.grid_size = grid_size
        self.accumulation_alpha = accumulation_alpha
        self.activity_threshold = activity_threshold
        self.decay_rate = decay_rate

        # Belief map: accumulated evidence of activity
        self.belief_map = np.zeros((grid_size, grid_size))

        # Instantaneous activity for display
        self.current_activity = np.zeros((grid_size, grid_size))

    def update(
        self, normalized_power: np.ndarray, bad_channels: np.ndarray
    ) -> np.ndarray:
        """
        Update belief map with new evidence.

        Args:
            normalized_power: Shape (n_channels,), values in [0, 1]
            bad_channels: Boolean mask of bad channels

        Returns:
            Updated belief map, shape (grid_size, grid_size)
        """
        # Reshape to grid
        activity = normalized_power.reshape(self.grid_size, self.grid_size)
        bad_mask = bad_channels.reshape(self.grid_size, self.grid_size)

        # Store current activity for display
        self.current_activity = activity.copy()

        # Apply decay to existing belief
        self.belief_map *= self.decay_rate

        # Only accumulate where activity exceeds threshold
        active = activity > self.activity_threshold

        # Zero out bad channels
        active = active & ~bad_mask

        # Accumulate evidence
        self.belief_map = (
            1 - self.accumulation_alpha
        ) * self.belief_map + self.accumulation_alpha * activity * active

        # Zero out bad channels in belief map
        self.belief_map[bad_mask] = 0

        return self.belief_map.copy()

    def reset(self):
        """Reset belief map."""
        self.belief_map = np.zeros((self.grid_size, self.grid_size))
        self.current_activity = np.zeros((self.grid_size, self.grid_size))


class BlendedEvidenceTracker:
    """
    Blend current belief_map with persistent_evidence for robust tracking.

    Combines short-term responsiveness with historical stability to prevent
    tracking loss during temporary signal dips.
    """

    def __init__(
        self,
        grid_size: int = 32,
        current_weight: float = 0.6,
        historical_weight: float = 0.4,
        adaptive_threshold: float = 0.5,
        adaptive_historical_weight: float = 0.7,
    ):
        """
        Initialize blended evidence tracker.

        Args:
            grid_size: Size of electrode grid
            current_weight: Weight for current belief_map (primary)
            historical_weight: Weight for persistent_evidence
            adaptive_threshold: When historical > this, increase its weight
            adaptive_historical_weight: Historical weight when above threshold
        """
        self.grid_size = grid_size
        self.current_weight = current_weight
        self.historical_weight = historical_weight
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_historical_weight = adaptive_historical_weight

    def blend(
        self,
        belief_map: np.ndarray,
        persistent_evidence: np.ndarray | None,
    ) -> np.ndarray:
        """
        Blend current belief_map with persistent_evidence.

        Args:
            belief_map: Current short-term evidence (32x32)
            persistent_evidence: Long-term historical evidence (32x32) or None

        Returns:
            Blended evidence map (32x32)
        """
        if persistent_evidence is None:
            return belief_map.copy()

        # Reshape if needed
        belief = belief_map.reshape(self.grid_size, self.grid_size)
        persistent = persistent_evidence.reshape(self.grid_size, self.grid_size)

        # Check if historical evidence is strong (indicates confirmed region)
        max_historical = np.max(persistent)

        if max_historical > self.adaptive_threshold:
            # Strong historical signal - increase historical weight for stability
            curr_w = 1.0 - self.adaptive_historical_weight
            hist_w = self.adaptive_historical_weight
        else:
            # Normal blending
            curr_w = self.current_weight
            hist_w = self.historical_weight

        # Normalize weights
        total = curr_w + hist_w
        curr_w /= total
        hist_w /= total

        # Blend the maps
        blended = curr_w * belief + hist_w * persistent

        return blended


class HotspotDetector:
    """Detect hotspot clusters and their centroids."""

    def __init__(
        self,
        grid_size: int = 32,
        spatial_sigma: float = 1.5,
        cluster_threshold: float = 0.3,
        min_cluster_size: int = 4,
        hysteresis_distance: float = 2.0,
        smoothing_alpha: float = 0.3,
    ):
        """
        Initialize hotspot detector.

        Args:
            grid_size: Size of the grid
            spatial_sigma: Gaussian smoothing sigma
            cluster_threshold: Minimum value for hotspot detection
            min_cluster_size: Minimum pixels in a valid cluster
            hysteresis_distance: Grid units before allowing immediate jump
            smoothing_alpha: Smoothing factor for small movements
        """
        self.grid_size = grid_size
        self.spatial_sigma = spatial_sigma
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.hysteresis_distance = hysteresis_distance
        self.smoothing_alpha = smoothing_alpha

        # Track last centroid for hysteresis
        self.last_centroid: tuple[float, float] | None = None

    def detect(
        self, belief_map: np.ndarray, bad_channels: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Detect hotspots in belief map.

        Args:
            belief_map: Shape (grid_size, grid_size)
            bad_channels: Optional boolean mask of bad channels

        Returns:
            Dictionary with:
                - smoothed: Smoothed belief map
                - clusters: Labeled cluster map
                - num_clusters: Number of clusters found
                - centroid: (row, col) of weighted centroid
                - cluster_info: List of cluster details
        """
        # Apply spatial smoothing
        smoothed = gaussian_filter(belief_map, sigma=self.spatial_sigma)

        # Zero out bad channels if provided
        if bad_channels is not None:
            bad_mask = bad_channels.reshape(self.grid_size, self.grid_size)
            smoothed[bad_mask] = 0

        # Threshold to find clusters
        binary = smoothed > self.cluster_threshold

        # Label connected components
        labeled, num_features = label(binary)

        # Filter small clusters
        cluster_info = []
        valid_clusters = np.zeros_like(labeled)

        for i in range(1, num_features + 1):
            cluster_mask = labeled == i
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= self.min_cluster_size:
                # Compute cluster centroid
                cluster_com = center_of_mass(smoothed, labeled, i)
                cluster_max = np.max(smoothed[cluster_mask])

                cluster_info.append(
                    {
                        "label": len(cluster_info) + 1,
                        "size": int(cluster_size),
                        "centroid": (float(cluster_com[0]), float(cluster_com[1])),
                        "max_value": float(cluster_max),
                    }
                )

                valid_clusters[cluster_mask] = len(cluster_info)

        # Compute overall weighted centroid
        raw_centroid = None
        if np.sum(smoothed > self.cluster_threshold) > 0:
            # Weight by activity level
            weights = smoothed * (smoothed > self.cluster_threshold)
            total_weight = np.sum(weights)
            if total_weight > 0:
                rows, cols = np.indices(smoothed.shape)
                centroid_row = np.sum(rows * weights) / total_weight
                centroid_col = np.sum(cols * weights) / total_weight
                raw_centroid = (float(centroid_row), float(centroid_col))

        # Apply centroid hysteresis
        centroid = self._apply_centroid_hysteresis(raw_centroid)

        return {
            "smoothed": smoothed,
            "clusters": valid_clusters,
            "num_clusters": len(cluster_info),
            "centroid": centroid,
            "cluster_info": cluster_info,
        }

    def _apply_centroid_hysteresis(
        self, new_centroid: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """
        Apply hysteresis to prevent centroid jitter.

        Large jumps (>hysteresis_distance) are allowed immediately.
        Small movements are smoothed with exponential averaging.

        Args:
            new_centroid: Raw detected centroid or None

        Returns:
            Smoothed centroid with hysteresis applied
        """
        if new_centroid is None:
            # No detection - keep last known position for a bit
            # (don't immediately lose tracking)
            return self.last_centroid

        if self.last_centroid is None:
            # First detection - use directly
            self.last_centroid = new_centroid
            return new_centroid

        # Calculate distance from last centroid
        dr = new_centroid[0] - self.last_centroid[0]
        dc = new_centroid[1] - self.last_centroid[1]
        distance = np.sqrt(dr * dr + dc * dc)

        if distance > self.hysteresis_distance:
            # Large jump - allow immediately (real movement)
            self.last_centroid = new_centroid
            return new_centroid
        else:
            # Small movement - smooth toward new position
            smoothed_row = (
                self.smoothing_alpha * new_centroid[0]
                + (1 - self.smoothing_alpha) * self.last_centroid[0]
            )
            smoothed_col = (
                self.smoothing_alpha * new_centroid[1]
                + (1 - self.smoothing_alpha) * self.last_centroid[1]
            )
            self.last_centroid = (smoothed_row, smoothed_col)
            return self.last_centroid


class GuidanceGenerator:
    """Generate surgical guidance based on hotspot location."""

    def __init__(self, grid_size: int = 32, center_tolerance: float = 2.0):
        """
        Initialize guidance generator.

        Args:
            grid_size: Size of the grid
            center_tolerance: How close to center counts as "centered"
        """
        self.grid_size = grid_size
        self.grid_center = (
            grid_size / 2 - 0.5,
            grid_size / 2 - 0.5,
        )  # 0-indexed center
        self.center_tolerance = center_tolerance

        # Smoothed guidance to prevent jitter
        self.smoothed_offset_row = 0.0
        self.smoothed_offset_col = 0.0
        self.guidance_alpha = 0.2

    def generate(
        self, centroid: tuple[float, float] | None, confidence: float
    ) -> dict[str, Any]:
        """
        Generate guidance instruction.

        Args:
            centroid: (row, col) of hotspot centroid, or None
            confidence: Confidence level (0-1)

        Returns:
            Dictionary with:
                - direction: Text direction ("MOVE LEFT", "CENTERED", etc.)
                - arrow: Arrow direction ("left", "up-right", etc.)
                - offset: (row_offset, col_offset) from center
                - distance: Distance from center in grid units
                - is_centered: Whether hotspot is centered
                - confidence: Confidence level
        """
        if centroid is None:
            return {
                "direction": "SEARCHING...",
                "arrow": None,
                "offset": (0, 0),
                "distance": 0,
                "is_centered": False,
                "confidence": 0,
            }

        # Calculate offset from center
        offset_row = centroid[0] - self.grid_center[0]
        offset_col = centroid[1] - self.grid_center[1]

        # Smooth the offset to prevent jitter
        self.smoothed_offset_row = (
            self.guidance_alpha * offset_row
            + (1 - self.guidance_alpha) * self.smoothed_offset_row
        )
        self.smoothed_offset_col = (
            self.guidance_alpha * offset_col
            + (1 - self.guidance_alpha) * self.smoothed_offset_col
        )

        # Use smoothed offsets
        offset_row = self.smoothed_offset_row
        offset_col = self.smoothed_offset_col

        # Calculate distance
        distance = np.sqrt(offset_row**2 + offset_col**2)

        # Check if centered
        is_centered = distance < self.center_tolerance

        if is_centered:
            return {
                "direction": "CENTERED",
                "arrow": "center",
                "offset": (float(offset_row), float(offset_col)),
                "distance": float(distance),
                "is_centered": True,
                "confidence": float(confidence),
            }

        # Generate direction text and arrow
        directions = []
        arrow_parts = []

        # Vertical component (row increases downward in grid)
        if offset_row < -self.center_tolerance / 2:
            directions.append("UP")
            arrow_parts.append("up")
        elif offset_row > self.center_tolerance / 2:
            directions.append("DOWN")
            arrow_parts.append("down")

        # Horizontal component (col increases rightward)
        if offset_col < -self.center_tolerance / 2:
            directions.append("LEFT")
            arrow_parts.append("left")
        elif offset_col > self.center_tolerance / 2:
            directions.append("RIGHT")
            arrow_parts.append("right")

        direction_text = "MOVE " + "-".join(directions) if directions else "HOLD"
        arrow = "-".join(arrow_parts) if arrow_parts else None

        return {
            "direction": direction_text,
            "arrow": arrow,
            "offset": (float(offset_row), float(offset_col)),
            "distance": float(distance),
            "is_centered": False,
            "confidence": float(confidence),
        }


class BCITracker:
    """Complete tracking system combining all components."""

    def __init__(
        self,
        grid_size: int = 32,
        accumulation_alpha: float = 0.05,
        activity_threshold: float = 0.4,
        decay_rate: float = 0.995,
        spatial_sigma: float = 1.5,
        cluster_threshold: float = 0.3,
        center_tolerance: float = 2.0,
        blend_current_weight: float = 0.6,
        blend_historical_weight: float = 0.4,
    ):
        """
        Initialize complete tracker.

        Args:
            grid_size: Size of electrode grid
            accumulation_alpha: Evidence accumulation rate
            activity_threshold: Minimum activity to accumulate
            decay_rate: Evidence decay rate
            spatial_sigma: Spatial smoothing sigma
            cluster_threshold: Hotspot detection threshold
            center_tolerance: Distance for "centered" state
            blend_current_weight: Weight for current belief_map in blending
            blend_historical_weight: Weight for persistent_evidence in blending
        """
        self.grid_size = grid_size

        self.evidence_tracker = EvidenceTracker(
            grid_size=grid_size,
            accumulation_alpha=accumulation_alpha,
            activity_threshold=activity_threshold,
            decay_rate=decay_rate,
        )

        self.blended_tracker = BlendedEvidenceTracker(
            grid_size=grid_size,
            current_weight=blend_current_weight,
            historical_weight=blend_historical_weight,
        )

        self.hotspot_detector = HotspotDetector(
            grid_size=grid_size,
            spatial_sigma=spatial_sigma,
            cluster_threshold=cluster_threshold,
        )

        self.guidance_generator = GuidanceGenerator(
            grid_size=grid_size,
            center_tolerance=center_tolerance,
        )

    def update(
        self,
        normalized_power: np.ndarray,
        bad_channels: np.ndarray,
        persistent_evidence: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Process new data and generate guidance.

        Args:
            normalized_power: Shape (n_channels,), values in [0, 1]
            bad_channels: Boolean mask of bad channels
            persistent_evidence: Optional long-term evidence from GlobalMapper

        Returns:
            Complete tracking result with all data for rendering
        """
        # Update belief map (short-term evidence)
        belief_map = self.evidence_tracker.update(normalized_power, bad_channels)

        # Blend current belief_map with persistent_evidence for robust tracking
        blended_map = self.blended_tracker.blend(belief_map, persistent_evidence)

        # Get current activity grid for direct detection
        current_activity = normalized_power.reshape(self.grid_size, self.grid_size)

        # Detect hotspots using current activity (what's actually displayed)
        # This ensures the centroid marker matches visible activity
        hotspot_result = self.hotspot_detector.detect(current_activity, bad_channels)

        # Calculate confidence based on hotspot strength
        if (
            hotspot_result["centroid"] is not None
            and hotspot_result["num_clusters"] > 0
        ):
            max_value = np.max(hotspot_result["smoothed"])
            confidence = min(1.0, max_value / 0.5)  # Scale to 0-1
        else:
            confidence = 0.0

        # Generate guidance
        guidance = self.guidance_generator.generate(
            hotspot_result["centroid"], confidence
        )

        return {
            "belief_map": belief_map.tolist(),
            "blended_map": blended_map.tolist(),
            "smoothed_map": hotspot_result["smoothed"].tolist(),
            "current_activity": self.evidence_tracker.current_activity.tolist(),
            "clusters": hotspot_result["clusters"].tolist(),
            "num_clusters": hotspot_result["num_clusters"],
            "centroid": hotspot_result["centroid"],
            "cluster_info": hotspot_result["cluster_info"],
            "guidance": guidance,
            "bad_channels": bad_channels.reshape(
                self.grid_size, self.grid_size
            ).tolist(),
        }

    def reset(self):
        """Reset tracker state."""
        self.evidence_tracker.reset()
        self.guidance_generator.smoothed_offset_row = 0.0
        self.guidance_generator.smoothed_offset_col = 0.0
        self.hotspot_detector.last_centroid = None


# ============================================================================
# Global Brain Mapping System
# ============================================================================


class HotspotTracker:
    """
    Track hotspot positions over time to estimate array motion.

    Key insight: The 4 velocity-tuned regions (Vx+, Vx-, Vy+, Vy-) are at
    fixed positions on the brain. If their apparent positions in our view
    shift, the array has moved.
    """

    def __init__(
        self,
        grid_size: int = 32,
        n_hotspots: int = 4,
        position_smoothing: float = 0.1,
        detection_threshold: float = 0.3,
        min_confidence: float = 0.2,
    ):
        self.grid_size = grid_size
        self.n_hotspots = n_hotspots
        self.position_smoothing = position_smoothing
        self.detection_threshold = detection_threshold
        self.min_confidence = min_confidence

        # Smoothed hotspot positions (row, col) - start at center
        self.hotspot_positions: list[tuple[float, float]] = []
        self.hotspot_confidences: list[float] = []

        # Reference centroid (where we first detected hotspots)
        self.reference_centroid: tuple[float, float] | None = None

        # Accumulated position offset
        self.position_offset = [0.0, 0.0]

        # History for stability detection
        self.centroid_history: list[tuple[float, float]] = []
        self.history_max = 50

    def update(
        self, activity: np.ndarray, bad_channels: np.ndarray
    ) -> tuple[float, float]:
        """
        Update hotspot tracking and return estimated motion.

        Args:
            activity: Current activity grid (32x32), normalized [0,1]
            bad_channels: Boolean mask of bad channels

        Returns:
            (dx, dy) motion since last frame
        """
        grid = activity.reshape(self.grid_size, self.grid_size)
        bad_mask = bad_channels.reshape(self.grid_size, self.grid_size)

        # Mask bad channels
        masked = grid.copy()
        masked[bad_mask] = 0

        # Apply smoothing to find stable peaks
        smoothed = gaussian_filter(masked, sigma=1.5)

        # Find hotspots using peak detection
        hotspots = self._find_hotspots(smoothed, bad_mask)

        if len(hotspots) < 2:
            # Not enough hotspots to track
            return (0.0, 0.0)

        # Compute centroid of detected hotspots
        centroid = self._compute_centroid(hotspots)

        # Initialize reference if needed
        if self.reference_centroid is None:
            self.reference_centroid = centroid
            self.centroid_history.append(centroid)
            return (0.0, 0.0)

        # Compute motion as shift from smoothed centroid
        # Use EMA to smooth the centroid tracking
        if self.centroid_history:
            prev_centroid = self.centroid_history[-1]
            smooth_centroid = (
                self.position_smoothing * centroid[0]
                + (1 - self.position_smoothing) * prev_centroid[0],
                self.position_smoothing * centroid[1]
                + (1 - self.position_smoothing) * prev_centroid[1],
            )
        else:
            smooth_centroid = centroid

        self.centroid_history.append(smooth_centroid)
        if len(self.centroid_history) > self.history_max:
            self.centroid_history.pop(0)

        # Motion is shift from reference
        dy = smooth_centroid[0] - self.reference_centroid[0]
        dx = smooth_centroid[1] - self.reference_centroid[1]

        # Update accumulated offset
        self.position_offset[0] = dy
        self.position_offset[1] = dx

        # Return incremental motion (difference from last frame)
        if len(self.centroid_history) >= 2:
            prev = self.centroid_history[-2]
            curr = self.centroid_history[-1]
            return (curr[1] - prev[1], curr[0] - prev[0])

        return (0.0, 0.0)

    def _find_hotspots(self, smoothed: np.ndarray, bad_mask: np.ndarray) -> list[dict]:
        """Find hotspot peaks in smoothed activity."""
        hotspots = []

        # Threshold to find candidate regions
        threshold = self.detection_threshold * np.max(smoothed)
        binary = smoothed > threshold

        # Find connected components
        labeled, num_features = label(binary)

        for i in range(1, num_features + 1):
            mask = labeled == i
            size = np.sum(mask)

            if size < 4:  # Minimum size
                continue

            # Find peak within this region
            region_vals = smoothed.copy()
            region_vals[~mask] = 0
            peak_idx = np.unravel_index(np.argmax(region_vals), region_vals.shape)

            # Skip if peak is on bad channel
            if bad_mask[peak_idx]:
                continue

            peak_val = smoothed[peak_idx]
            confidence = min(1.0, peak_val / 0.5)

            if confidence >= self.min_confidence:
                hotspots.append(
                    {
                        "position": (float(peak_idx[0]), float(peak_idx[1])),
                        "confidence": confidence,
                        "size": int(size),
                    }
                )

        # Sort by confidence and return top N
        hotspots.sort(key=lambda x: x["confidence"], reverse=True)
        return hotspots[: self.n_hotspots]

    def _compute_centroid(self, hotspots: list[dict]) -> tuple[float, float]:
        """Compute weighted centroid of hotspots."""
        if not hotspots:
            return (self.grid_size / 2, self.grid_size / 2)

        total_weight = sum(h["confidence"] for h in hotspots)
        if total_weight == 0:
            return (self.grid_size / 2, self.grid_size / 2)

        row = sum(h["position"][0] * h["confidence"] for h in hotspots) / total_weight
        col = sum(h["position"][1] * h["confidence"] for h in hotspots) / total_weight

        return (row, col)

    def get_position_offset(self) -> tuple[float, float]:
        """Get accumulated position offset from start."""
        return tuple(self.position_offset)

    def get_detected_hotspots(self) -> list[dict]:
        """Get currently tracked hotspots."""
        return self.hotspot_positions.copy() if self.hotspot_positions else []

    def reset(self):
        """Reset tracker state."""
        self.hotspot_positions = []
        self.hotspot_confidences = []
        self.reference_centroid = None
        self.position_offset = [0.0, 0.0]
        self.centroid_history = []


class GlobalBrainMap:
    """
    Maintains a global map of the brain in fixed coordinates.

    The array observes only a 32x32 window of the brain at any time.
    This class tracks the array position and accumulates evidence
    in brain-fixed coordinates.
    """

    def __init__(
        self,
        global_size: int = 96,
        array_size: int = 32,
        evidence_alpha: float = 0.1,
        confidence_alpha: float = 0.05,
    ):
        """
        Initialize global brain map.

        Args:
            global_size: Size of global map (pixels)
            array_size: Size of array view (32x32)
            evidence_alpha: Rate of evidence integration
            confidence_alpha: Rate of confidence update
        """
        self.global_size = global_size
        self.array_size = array_size
        self.evidence_alpha = evidence_alpha
        self.confidence_alpha = confidence_alpha

        # Array position in global coordinates (center of global map initially)
        # This is the TOP-LEFT corner of the array view
        self.array_position = [
            (global_size - array_size) // 2,
            (global_size - array_size) // 2,
        ]

        # Global maps in BRAIN coordinates
        self.evidence_map = np.zeros((global_size, global_size))
        self.confidence_map = np.zeros((global_size, global_size))
        self.visit_count = np.zeros((global_size, global_size))
        self.peak_evidence = np.zeros((global_size, global_size))

        # Track when each cell was last observed
        self.last_observation = np.full((global_size, global_size), -1.0)
        self.current_time = 0.0

    def update_position(self, dx: float, dy: float):
        """
        Update array position based on estimated motion.

        Args:
            dx: Horizontal motion detected (sub-pixel)
            dy: Vertical motion detected (sub-pixel)
        """
        # Accumulate motion directly (amplified for visibility)
        self.array_position[0] += dy * 2.0
        self.array_position[1] += dx * 2.0

        # Clamp to valid range
        max_pos = self.global_size - self.array_size
        self.array_position[0] = np.clip(self.array_position[0], 0, max_pos)
        self.array_position[1] = np.clip(self.array_position[1], 0, max_pos)

    def set_position_from_ground_truth(
        self, gt_center_row: float, gt_center_col: float
    ):
        """
        Set array position directly from ground truth (dev mode only).

        The ground truth center is in array coordinates (1-32).
        We map this to global coordinates.
        """
        # GT center (16,16) = array at global center (32,32)
        # GT center (8,8) = array shifted up-left
        # Offset from nominal center (16,16) tells us array displacement
        offset_row = gt_center_row - 16.0
        offset_col = gt_center_col - 16.0

        # Map to global position (center is 32,32)
        self.array_position[0] = 32.0 - offset_row
        self.array_position[1] = 32.0 - offset_col

        # Clamp
        max_pos = self.global_size - self.array_size
        self.array_position[0] = np.clip(self.array_position[0], 0, max_pos)
        self.array_position[1] = np.clip(self.array_position[1], 0, max_pos)

    def integrate_observation(
        self,
        array_activity: np.ndarray,
        bad_channels: np.ndarray,
        time_s: float = 0.0,
    ):
        """
        Integrate current observation into global map.

        Args:
            array_activity: Current activity (32x32), normalized [0, 1]
            bad_channels: Boolean mask of bad channels
            time_s: Current timestamp
        """
        self.current_time = time_s

        # Get array bounds in global coordinates
        r = int(round(self.array_position[0]))
        c = int(round(self.array_position[1]))

        # Ensure within bounds
        r = max(0, min(r, self.global_size - self.array_size))
        c = max(0, min(c, self.global_size - self.array_size))

        # Reshape inputs
        activity = array_activity.reshape(self.array_size, self.array_size)
        bad_mask = bad_channels.reshape(self.array_size, self.array_size)
        valid_mask = ~bad_mask

        # Extract current global region
        global_region = self.evidence_map[
            r : r + self.array_size, c : c + self.array_size
        ]
        conf_region = self.confidence_map[
            r : r + self.array_size, c : c + self.array_size
        ]
        visit_region = self.visit_count[
            r : r + self.array_size, c : c + self.array_size
        ]
        peak_region = self.peak_evidence[
            r : r + self.array_size, c : c + self.array_size
        ]
        last_obs_region = self.last_observation[
            r : r + self.array_size, c : c + self.array_size
        ]

        # Update evidence (EMA blend)
        global_region[valid_mask] = (
            self.evidence_alpha * activity[valid_mask]
            + (1 - self.evidence_alpha) * global_region[valid_mask]
        )

        # Update confidence (increases with observations)
        conf_region[valid_mask] = (
            self.confidence_alpha
            + (1 - self.confidence_alpha) * conf_region[valid_mask]
        )

        # Update visit count
        visit_region[valid_mask] += 1

        # Update peak evidence (never decays)
        peak_region[valid_mask] = np.maximum(
            peak_region[valid_mask], activity[valid_mask]
        )

        # Update last observation time
        last_obs_region[valid_mask] = time_s

    def get_array_bounds(self) -> tuple[int, int, int, int]:
        """Get current array bounds in global coordinates (r1, c1, r2, c2)."""
        r = int(round(self.array_position[0]))
        c = int(round(self.array_position[1]))
        return (r, c, r + self.array_size, c + self.array_size)

    def get_downsampled_map(self, target_size: int = 48) -> dict[str, np.ndarray]:
        """
        Get downsampled global maps for WebSocket transmission.

        Args:
            target_size: Target size for downsampled maps

        Returns:
            Dictionary with downsampled evidence, confidence, visit_count
        """
        factor = self.global_size // target_size

        # Simple block averaging for downsampling
        def downsample(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(target_size, factor, target_size, factor).mean(
                axis=(1, 3)
            )

        return {
            "evidence": downsample(self.evidence_map),
            "confidence": downsample(self.confidence_map),
            "visit_count": downsample(self.visit_count),
            "peak_evidence": downsample(self.peak_evidence),
        }

    def reset(self):
        """Reset global map."""
        self.array_position = [
            (self.global_size - self.array_size) // 2,
            (self.global_size - self.array_size) // 2,
        ]
        self.evidence_map = np.zeros((self.global_size, self.global_size))
        self.confidence_map = np.zeros((self.global_size, self.global_size))
        self.visit_count = np.zeros((self.global_size, self.global_size))
        self.peak_evidence = np.zeros((self.global_size, self.global_size))
        self.last_observation = np.full((self.global_size, self.global_size), -1.0)


class PersistentEvidenceTracker:
    """
    Track evidence with asymmetric rise/decay for intermittent activation.

    Motor cortex regions are velocity-tuned (Vx+, Vx-, Vy+, Vy-) and only
    activate when cursor moves in their direction. This tracker uses:
    - Fast rise when activity detected
    - Very slow decay when quiet
    - Almost no decay for confirmed hotspots
    """

    def __init__(
        self,
        grid_size: int = 32,
        rise_alpha: float = 0.15,
        decay_alpha: float = 0.001,
        confirmed_threshold: float = 0.4,
        confirmed_decay: float = 0.9999,
    ):
        """
        Initialize persistent evidence tracker.

        Args:
            grid_size: Size of grid
            rise_alpha: Rate of rise when activity > evidence
            decay_alpha: Rate of decay when activity < evidence
            confirmed_threshold: Evidence level to consider "confirmed"
            confirmed_decay: Decay rate for confirmed regions (very slow)
        """
        self.grid_size = grid_size
        self.rise_alpha = rise_alpha
        self.decay_alpha = decay_alpha
        self.confirmed_threshold = confirmed_threshold
        self.confirmed_decay = confirmed_decay

        self.evidence = np.zeros((grid_size, grid_size))

    def update(self, observation: np.ndarray, bad_channels: np.ndarray) -> np.ndarray:
        """
        Update evidence with asymmetric dynamics.

        Args:
            observation: Current normalized activity (32x32)
            bad_channels: Boolean mask of bad channels

        Returns:
            Updated evidence map
        """
        obs = observation.reshape(self.grid_size, self.grid_size)
        bad_mask = bad_channels.reshape(self.grid_size, self.grid_size)

        # Rising: observation > evidence -> rise quickly
        rising = obs > self.evidence
        self.evidence[rising] = (
            self.rise_alpha * obs[rising]
            + (1 - self.rise_alpha) * self.evidence[rising]
        )

        # Falling: observation < evidence -> decay slowly
        falling = ~rising

        # Confirmed regions (high evidence) decay even slower
        confirmed = self.evidence > self.confirmed_threshold
        confirmed_falling = confirmed & falling
        normal_falling = ~confirmed & falling

        # Normal decay
        self.evidence[normal_falling] = (
            self.decay_alpha * obs[normal_falling]
            + (1 - self.decay_alpha) * self.evidence[normal_falling]
        )

        # Confirmed decay (almost no decay)
        self.evidence[confirmed_falling] *= self.confirmed_decay

        # Zero out bad channels
        self.evidence[bad_mask] = 0

        return self.evidence.copy()

    def get_confirmed_regions(self) -> np.ndarray:
        """Get boolean mask of confirmed high-evidence regions."""
        return self.evidence > self.confirmed_threshold

    def reset(self):
        """Reset evidence."""
        self.evidence = np.zeros((self.grid_size, self.grid_size))


class ExplorationTracker:
    """
    Track which regions of the brain have been explored.

    Helps guide the surgeon to unexplored areas.
    """

    def __init__(
        self,
        global_size: int = 96,
        array_size: int = 32,
        visit_threshold: int = 50,
    ):
        """
        Initialize exploration tracker.

        Args:
            global_size: Size of global map
            array_size: Size of array view
            visit_threshold: Visits needed to consider "explored"
        """
        self.global_size = global_size
        self.array_size = array_size
        self.visit_threshold = visit_threshold

    def get_exploration_coverage(self, visit_count: np.ndarray) -> float:
        """
        Calculate fraction of brain that has been explored.

        Args:
            visit_count: Global visit count map

        Returns:
            Fraction explored [0, 1]
        """
        explored = visit_count >= self.visit_threshold
        return float(np.mean(explored))

    def get_unexplored_direction(
        self, visit_count: np.ndarray, array_position: tuple[float, float]
    ) -> str | None:
        """
        Suggest direction to unexplored regions.

        Args:
            visit_count: Global visit count map
            array_position: Current array position (row, col)

        Returns:
            Direction suggestion (e.g., "right-down") or None
        """
        r, c = int(array_position[0]), int(array_position[1])

        # Check exploration in each direction
        directions = []

        # Check above
        if r > 0:
            above_region = visit_count[max(0, r - 16) : r, :]
            if np.mean(above_region < self.visit_threshold) > 0.5:
                directions.append("up")

        # Check below
        if r < self.global_size - self.array_size:
            below_region = visit_count[
                r + self.array_size : r + self.array_size + 16, :
            ]
            if np.mean(below_region < self.visit_threshold) > 0.5:
                directions.append("down")

        # Check left
        if c > 0:
            left_region = visit_count[:, max(0, c - 16) : c]
            if np.mean(left_region < self.visit_threshold) > 0.5:
                directions.append("left")

        # Check right
        if c < self.global_size - self.array_size:
            right_region = visit_count[
                :, c + self.array_size : c + self.array_size + 16
            ]
            if np.mean(right_region < self.visit_threshold) > 0.5:
                directions.append("right")

        if not directions:
            return None

        return "-".join(directions)


class GlobalMapper:
    """
    Integrate all global mapping components.

    Combines motion estimation, global brain map, persistent evidence,
    and exploration tracking into a unified system.
    """

    def __init__(
        self,
        grid_size: int = 32,
        global_size: int = 96,
    ):
        """
        Initialize global mapper.

        Args:
            grid_size: Array size (32x32)
            global_size: Global brain map size
        """
        self.grid_size = grid_size
        self.global_size = global_size

        # Components
        self.hotspot_tracker = HotspotTracker(grid_size=grid_size)
        self.global_map = GlobalBrainMap(global_size=global_size, array_size=grid_size)
        self.persistent_evidence = PersistentEvidenceTracker(grid_size=grid_size)
        self.exploration_tracker = ExplorationTracker(
            global_size=global_size, array_size=grid_size
        )

        # Detected hotspots in global coordinates
        self.global_hotspots: list[dict] = []

    def update(
        self,
        normalized_power: np.ndarray,
        bad_channels: np.ndarray,
        time_s: float = 0.0,
    ) -> dict[str, Any]:
        """
        Update global mapping with new observation.

        Args:
            normalized_power: Normalized power (n_channels,), [0, 1]
            bad_channels: Boolean mask of bad channels
            time_s: Current timestamp

        Returns:
            Global mapping data for WebSocket message
        """
        # Reshape to grid
        activity = normalized_power.reshape(self.grid_size, self.grid_size)
        bad_mask = bad_channels.reshape(self.grid_size, self.grid_size)

        # 1. Position tracking - use hotspot drift detection
        # Always run hotspot tracker to detect motion (no ground truth cheating)
        dx, dy = self.hotspot_tracker.update(activity, bad_mask)
        self.global_map.update_position(dx, dy)

        # 3. Integrate observation into global map
        self.global_map.integrate_observation(activity, bad_mask, time_s)

        # 4. Update persistent evidence (for array view)
        persistent_evidence = self.persistent_evidence.update(activity, bad_mask)

        # 5. Detect hotspots in global map
        self._update_global_hotspots()

        # 6. Calculate exploration coverage
        coverage = self.exploration_tracker.get_exploration_coverage(
            self.global_map.visit_count
        )

        # 7. Get exploration suggestion
        exploration_suggestion = self.exploration_tracker.get_unexplored_direction(
            self.global_map.visit_count, tuple(self.global_map.array_position)
        )

        # 8. Get downsampled maps for transmission
        downsampled = self.global_map.get_downsampled_map(target_size=48)

        # Build result - ensure all numpy types are converted to Python types
        return {
            "array_position": [
                float(self.global_map.array_position[0]),
                float(self.global_map.array_position[1]),
            ],
            "array_bounds": [int(x) for x in self.global_map.get_array_bounds()],
            "cumulative_motion": [
                float(x) for x in self.hotspot_tracker.get_position_offset()
            ],
            "global_evidence": downsampled["evidence"].tolist(),
            "global_confidence": downsampled["confidence"].tolist(),
            "global_peak_evidence": downsampled["peak_evidence"].tolist(),
            "exploration_coverage": float(coverage),
            "exploration_suggestion": exploration_suggestion,
            "persistent_evidence": persistent_evidence.tolist(),
            "hotspots": self.global_hotspots,
        }

    def _update_global_hotspots(self, threshold: float = 0.3):
        """Detect hotspots in global evidence map."""
        evidence = self.global_map.evidence_map

        # Apply spatial smoothing
        smoothed = gaussian_filter(evidence, sigma=2.0)

        # Threshold and find connected components
        binary = smoothed > threshold
        labeled, num_features = label(binary)

        hotspots = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            size = np.sum(mask)

            if size < 16:  # Minimum cluster size
                continue

            # Find centroid
            com = center_of_mass(smoothed, labeled, i)
            max_val = np.max(smoothed[mask])

            hotspots.append(
                {
                    "global_position": [float(com[0]), float(com[1])],
                    "size": int(size),
                    "confidence": float(min(1.0, max_val / 0.5)),
                }
            )

        # Sort by confidence
        hotspots.sort(key=lambda x: x["confidence"], reverse=True)
        self.global_hotspots = hotspots[:8]  # Limit to top 8

    def reset(self):
        """Reset all components."""
        self.hotspot_tracker.reset()
        self.global_map.reset()
        self.persistent_evidence.reset()
        self.global_hotspots = []
