"""
Core tracking utilities for the Compass backend.

This module focuses on interpretable, noise-robust tracking primitives used in
real-time. Legacy experimentation code has been removed for clarity.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def extract_peak_observations(
    grid: np.ndarray,
    n_peaks: int = 2,
    smooth_sigma: float = 1.0,
    suppress_radius: int = 6,
    com_radius: int = 2,
    min_abs: float = 0.0,
) -> list[tuple[float, float, float]]:
    """
    Extract top-N peak locations from a 32x32 activity grid.

    min_abs filters out weak peaks after normalization.
    """
    g = ndimage.gaussian_filter(grid, sigma=smooth_sigma)
    work = g.copy()

    peaks: list[tuple[float, float, float]] = []
    for _ in range(n_peaks):
        idx = int(np.argmax(work))
        peak_val = float(work.flat[idx])
        if not np.isfinite(peak_val) or peak_val <= 0 or peak_val < float(min_abs):
            break

        r0, c0 = np.unravel_index(idx, work.shape)

        r1 = max(0, r0 - com_radius)
        r2 = min(work.shape[0], r0 + com_radius + 1)
        c1 = max(0, c0 - com_radius)
        c2 = min(work.shape[1], c0 + com_radius + 1)
        patch = g[r1:r2, c1:c2]
        pr, pc = np.indices(patch.shape)
        total = float(patch.sum())
        if total > 0:
            rr = (pr * patch).sum() / total + r1
            cc = (pc * patch).sum() / total + c1
        else:
            rr, cc = float(r0), float(c0)

        peaks.append((float(rr), float(cc), peak_val))

        # Suppress a neighborhood so we don't pick the same spot again.
        sr1 = max(0, r0 - suppress_radius)
        sr2 = min(work.shape[0], r0 + suppress_radius + 1)
        sc1 = max(0, c0 - suppress_radius)
        sc2 = min(work.shape[1], c0 + suppress_radius + 1)
        work[sr1:sr2, sc1:sc2] = float("-inf")

    return peaks


class PersistentSpotClusterTracker:
    """
    Track hotspot components across time with explicit memory.

    Estimates frame-to-frame drift from observed peaks, and propagates that drift
    to maintain stable tracks even when peaks drop out.
    """

    def __init__(
        self,
        max_tracks: int = 10,
        ema_alpha: float = 0.4,
        max_match_dist: float = 7.5,
        strength_gain: float = 0.3,
        strength_decay: float = 0.98,
        max_age: int = 240,
    ):
        self.max_tracks = int(max_tracks)
        self.ema_alpha = float(ema_alpha)
        self.max_match_dist = float(max_match_dist)
        self.strength_gain = float(strength_gain)
        self.strength_decay = float(strength_decay)
        self.max_age = int(max_age)
        self._tracks: list[dict[str, object]] = []

    def reset(self) -> None:
        self._tracks = []

    def _prune(self) -> None:
        kept = []
        for tr in self._tracks:
            if tr["age"] <= self.max_age and tr["strength"] > 1e-3:
                kept.append(tr)
        self._tracks = kept[: self.max_tracks]

    def update(self, observations: list[tuple[float, float, float]]):
        """
        Update tracker with observed (row, col, value) peaks.

        Returns:
            (center_row, center_col, tracks_positions)
        """
        obs = [np.array([r, c], dtype=float) for (r, c, _v) in observations]

        # Age/decay all tracks.
        for tr in self._tracks:
            tr["age"] += 1
            tr["strength"] *= self.strength_decay

        if not self._tracks:
            for p in obs[: self.max_tracks]:
                self._tracks.append({"pos": p, "strength": 0.8, "age": 0})
            return self.center()

        if not obs:
            self._prune()
            return self.center()

        # Assignment: tracks <-> observations.
        t_count = len(self._tracks)
        o_count = len(obs)
        cost = np.zeros((t_count, o_count), dtype=float)
        for i, tr in enumerate(self._tracks):
            for j, p in enumerate(obs):
                cost[i, j] = np.linalg.norm(p - tr["pos"])

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        unmatched_obs = set(range(o_count))
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= self.max_match_dist:
                matches.append((i, j))
                unmatched_obs.discard(j)

        # Estimate drift from matched pairs.
        if matches:
            deltas = [obs[j] - self._tracks[i]["pos"] for (i, j) in matches]
            delta = np.mean(deltas, axis=0)
        else:
            delta = np.array([0.0, 0.0], dtype=float)

        if np.any(delta):
            for tr in self._tracks:
                tr["pos"] = tr["pos"] + delta

        # Correct matched tracks.
        for i, j in matches:
            tr = self._tracks[i]
            tr["pos"] = (1 - self.ema_alpha) * tr["pos"] + self.ema_alpha * obs[j]
            tr["strength"] = min(1.0, tr["strength"] + self.strength_gain)
            tr["age"] = 0

        # Spawn new tracks.
        for j in list(unmatched_obs):
            if len(self._tracks) >= self.max_tracks:
                break
            self._tracks.append({"pos": obs[j], "strength": 0.6, "age": 0})

        self._prune()
        return self.center()

    def center(self):
        if not self._tracks:
            return None, None, []
        weights = np.array([tr["strength"] for tr in self._tracks], dtype=float)
        positions = np.stack([tr["pos"] for tr in self._tracks], axis=0)
        wsum = float(weights.sum())
        if wsum <= 0:
            center = positions.mean(axis=0)
        else:
            center = (positions * weights[:, None]).sum(axis=0) / wsum
        return float(center[0]), float(center[1]), [tuple(p.tolist()) for p in positions]

    def track_states(self) -> list[tuple[float, float, float, int]]:
        """Return tracked positions with strength and age for interpretability."""
        out = []
        for tr in self._tracks:
            r, c = float(tr["pos"][0]), float(tr["pos"][1])
            out.append((r, c, float(tr["strength"]), int(tr["age"])))
        return out


class InterpretableClusterTracker:
    """
    Interpretable tracker using peaks + persistent memory.

    Returns a stable center, confidence score, and long-memory map of anchors.
    """

    def __init__(
        self,
        grid_size: int = 32,
        peak_n: int = 6,
        peak_smooth_sigma: float = 1.0,
        peak_suppress_radius: int = 7,
        peak_com_radius: int = 2,
        peak_min_abs: float = 0.15,
        max_tracks: int = 10,
        ema_alpha: float = 0.4,
        max_match_dist: float = 7.5,
        strength_gain: float = 0.3,
        strength_decay: float = 0.98,
        max_age: int = 240,
        memory_half_life_s: float = 45.0,
        memory_sigma: float = 1.0,
        memory_update_conf: float = 0.25,
        memory_center_percentile: float = 85.0,
        memory_top_k: int = 4,
        memory_gain: float = 0.25,
        age_tau_updates: float = 120.0,
        conf_strength_weight: float = 0.7,
        conf_count_weight: float = 0.3,
    ):
        self.grid_size = int(grid_size)
        self.mid = (self.grid_size - 1) / 2.0
        self.peak_n = int(peak_n)
        self.peak_smooth_sigma = float(peak_smooth_sigma)
        self.peak_suppress_radius = int(peak_suppress_radius)
        self.peak_com_radius = int(peak_com_radius)
        self.peak_min_abs = float(peak_min_abs)
        self.tracker = PersistentSpotClusterTracker(
            max_tracks=max_tracks,
            ema_alpha=ema_alpha,
            max_match_dist=max_match_dist,
            strength_gain=strength_gain,
            strength_decay=strength_decay,
            max_age=max_age,
        )
        self.memory_half_life_s = float(memory_half_life_s)
        self.memory_sigma = float(memory_sigma)
        self.memory_update_conf = float(memory_update_conf)
        self.memory_center_percentile = float(memory_center_percentile)
        self.memory_top_k = int(memory_top_k)
        self.memory_gain = float(memory_gain)
        self.age_tau_updates = float(age_tau_updates)
        self.conf_strength_weight = float(conf_strength_weight)
        self.conf_count_weight = float(conf_count_weight)
        self.memory_map: np.ndarray | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self.memory_map = None

    def _age_weight(self, age: int) -> float:
        return float(np.exp(-float(age) / max(1.0, self.age_tau_updates)))

    def _confidence(self, track_states: list[tuple[float, float, float, int]]) -> float:
        if not track_states:
            return 0.0
        weighted = [float(s) * self._age_weight(age) for (_r, _c, s, age) in track_states]
        weighted.sort(reverse=True)
        strength_conf = float(np.mean(weighted[:2]))
        count_conf = min(1.0, len(weighted) / 2.0)
        return float(
            np.clip(
                self.conf_strength_weight * strength_conf + self.conf_count_weight * count_conf,
                0.0,
                1.0,
            )
        )

    def _update_memory(
        self,
        track_states: list[tuple[float, float, float, int]],
        center_rc: tuple[float, float] | None,
        dt_s: float,
        conf: float,
    ) -> np.ndarray:
        if self.memory_map is None:
            self.memory_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        decay = 0.5 ** (dt_s / max(1e-3, self.memory_half_life_s))
        self.memory_map *= decay

        if not track_states or conf < self.memory_update_conf or center_rc is None:
            return self.memory_map

        upd = np.zeros_like(self.memory_map)
        center_r, center_c = center_rc
        track_states = sorted(track_states, key=lambda x: float(x[2]), reverse=True)
        for r, c, s, age in track_states[: self.memory_top_k]:
            rr = int(round(float(r) - float(center_r) + self.mid))
            cc = int(round(float(c) - float(center_c) + self.mid))
            if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                val = float(s) * self._age_weight(int(age))
                if val > upd[rr, cc]:
                    upd[rr, cc] = val

        if float(upd.max()) > 0:
            if self.memory_sigma > 0:
                upd = ndimage.gaussian_filter(upd, sigma=self.memory_sigma)
            self.memory_map = np.clip(self.memory_map + self.memory_gain * upd, 0.0, 1.0)
        return self.memory_map

    def memory_center(self) -> tuple[float | None, float | None]:
        if self.memory_map is None:
            return None, None
        mem = self.memory_map
        mx = float(mem.max())
        if mx <= 0:
            return None, None
        thr = float(np.percentile(mem, self.memory_center_percentile))
        mask = mem >= thr
        total = float((mem * mask).sum())
        if total <= 0:
            return None, None
        rows, cols = np.indices(mem.shape)
        r = float((rows * mem * mask).sum() / total)
        c = float((cols * mem * mask).sum() / total)
        return r, c

    def update(
        self, grid: np.ndarray, dt_s: float = 0.05
    ) -> tuple[
        float | None,
        float | None,
        float,
        np.ndarray | None,
        list[tuple[float, float, float]],
        list[tuple[float, float, float, int]],
    ]:
        g = np.maximum(grid.astype(float), 0.0)
        scale = float(np.percentile(g, 99.5)) + 1e-6
        g_norm = np.clip(g / scale, 0.0, 1.0)
        observations = extract_peak_observations(
            g_norm,
            n_peaks=self.peak_n,
            smooth_sigma=self.peak_smooth_sigma,
            suppress_radius=self.peak_suppress_radius,
            com_radius=self.peak_com_radius,
            min_abs=self.peak_min_abs,
        )
        cr, cc, _tracks = self.tracker.update(observations)
        track_states = self.tracker.track_states()
        conf = self._confidence(track_states)
        center_rc = (float(cr), float(cc)) if cr is not None and cc is not None else None
        memory = self._update_memory(track_states, center_rc, dt_s, conf)
        return cr, cc, conf, memory, observations, track_states


class SimpleKalmanFilter2D:
    """
    Simple 2D Kalman filter with constant-velocity model.

    State: [row, col, vel_row, vel_col]
    """

    def __init__(
        self,
        process_noise: float = 0.15,
        measurement_noise: float = 2.0,
        initial_pos: tuple[float, float] = (15.5, 15.5),
    ):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0

        self.Q = np.eye(4, dtype=float)
        self.Q[0, 0] = process_noise * 0.5
        self.Q[1, 1] = process_noise * 0.5
        self.Q[2, 2] = process_noise
        self.Q[3, 3] = process_noise

        self.R = np.eye(2, dtype=float) * measurement_noise
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self._initialized = False

    def reset(self, pos: tuple[float, float] | None = None):
        if pos is not None:
            self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        else:
            self.x = np.array([15.5, 15.5, 0.0, 0.0], dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0
        self._initialized = False

    def predict(self, dt: float = 0.05):
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, measurement: np.ndarray, confidence: float = 1.0):
        if not self._initialized:
            self.x[0] = measurement[0]
            self.x[1] = measurement[1]
            self._initialized = True
            return

        R_adj = self.R / max(0.1, confidence)
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R_adj
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    @property
    def position(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))
