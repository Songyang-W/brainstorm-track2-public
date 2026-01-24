#!/usr/bin/env python3
"""
Offline evaluation helper for CompassProcessor against ground_truth (dev only).

This is NOT used in the competition runtime; it's for iterating locally to ensure:
- estimated center tracks ground-truth cluster center over time
- guidance vector (move_row/move_col) points the correct way
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import typer

from scripts.compass_backend import CompassProcessor
from scripts.hotspot_tracker import compute_ground_truth_center, _get_gt_time_array

app = typer.Typer(help="Evaluate CompassProcessor against ground_truth (dev only).")


@dataclass
class Metrics:
    rmse: float
    mae: float
    cos_mean: float
    cos_p25: float
    cos_p50: float
    cos_p75: float
    n: int


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _metrics(err: np.ndarray, cos: np.ndarray) -> Metrics:
    d = np.linalg.norm(err, axis=1)
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(d))
    cs = cos[np.isfinite(cos)]
    if cs.size == 0:
        return Metrics(rmse=rmse, mae=mae, cos_mean=float("nan"), cos_p25=float("nan"), cos_p50=float("nan"), cos_p75=float("nan"), n=int(err.shape[0]))
    return Metrics(
        rmse=rmse,
        mae=mae,
        cos_mean=float(np.mean(cs)),
        cos_p25=float(np.percentile(cs, 25)),
        cos_p50=float(np.percentile(cs, 50)),
        cos_p75=float(np.percentile(cs, 75)),
        n=int(err.shape[0]),
    )


@app.command()
def main(
    dataset: str = typer.Argument("super_easy", help="Dataset folder under data/"),
    seconds: float = typer.Option(30.0, help="Seconds to evaluate from start"),
    batch_size: int = typer.Option(10, help="Samples per batch (matches stream messages)"),
    ema_tau_s: float = typer.Option(0.20, help="EMA time constant (seconds)"),
    spatial_sigma: float = typer.Option(1.0, help="Spatial smoothing sigma"),
) -> None:
    data_path = f"data/{dataset}"
    df = pd.read_parquet(f"{data_path}/track2_data.parquet")
    gt = pd.read_parquet(f"{data_path}/ground_truth.parquet")

    fs = 500.0
    data = df.values.astype(np.float32)
    n = min(int(seconds * fs), data.shape[0])
    data = data[:n]

    proc = CompassProcessor(fs=fs, n_channels=data.shape[1], grid_size=32, ema_tau_s=ema_tau_s, spatial_sigma=spatial_sigma)

    gt_t = _get_gt_time_array(gt)
    mid = (32 - 1) / 2.0

    est = []
    truth = []
    cos = []

    for i in range(0, len(data) - batch_size + 1, batch_size):
        t_s = (i + batch_size - 1) / fs
        frame = proc.process_batch(data[i : i + batch_size], t_s)
        if frame is None:
            continue

        gt_idx = int(np.abs(gt_t - t_s).argmin())
        tr, tc, _ = compute_ground_truth_center(gt, gt_idx)
        if tr is None or tc is None:
            continue

        er = float(frame["center_row"]) - float(tr)
        ec = float(frame["center_col"]) - float(tc)

        # Compare guidance direction: should match move = -(center - mid).
        mv_true = np.array([-(float(tr) - mid), -(float(tc) - mid)], dtype=float)
        mv_est = np.array([-(float(frame["center_row"]) - mid), -(float(frame["center_col"]) - mid)], dtype=float)
        cos.append(_cos(mv_est, mv_true))

        est.append([float(frame["center_row"]), float(frame["center_col"])])
        truth.append([float(tr), float(tc)])

    est = np.asarray(est, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = est - truth
    cos = np.asarray(cos, dtype=float)

    m = _metrics(err, cos)
    typer.echo(f"dataset={dataset} seconds={seconds:.1f} n={m.n}")
    typer.echo(f"center_rmse={m.rmse:.3f}  center_mae={m.mae:.3f}")
    typer.echo(f"move_cos(mean/p25/p50/p75)={m.cos_mean:.3f}/{m.cos_p25:.3f}/{m.cos_p50:.3f}/{m.cos_p75:.3f}")


if __name__ == "__main__":
    app()
