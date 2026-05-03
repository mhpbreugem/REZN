#!/usr/bin/env python3
"""
Aggregate per-tau checkpoints into fig4A_g100_tau_sweep.json and write
pgfplots coordinates for the gamma=1.0 curve.

Reads:    results/full_ree/task3_g100_t*_mp50.json
Writes:   results/full_ree/fig4A_g100_tau_sweep.json
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from solve_posterior_v3 import weighted_1mR2

CKPT_GLOB = "results/full_ree/task3_g100_t*_mp50.json"
OUT_PATH = "results/full_ree/fig4A_g100_tau_sweep.json"


def load_one(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    u_grid = np.array([float(s) for s in d["u_grid"]])
    p_grids = np.array([[float(x) for x in row] for row in d["p_grid"]])
    mu = np.array([[float(x) for x in row] for row in d["mu_strings"]])
    one_mR2, slope, n = weighted_1mR2(u_grid, p_grids, mu, d["tau"], d["gamma"])
    return {
        "tau": float(d["tau"]),
        "1-R2": float(one_mR2),
        "slope": float(slope),
        "n_triples": int(n),
        "F_max": float(d["F_max"]),
        "F_med": float(d["F_med"]),
        "checkpoint": os.path.basename(path),
    }


def main():
    files = sorted(glob.glob(CKPT_GLOB))
    if not files:
        print(f"No checkpoints matched {CKPT_GLOB}")
        return
    points = []
    for path in files:
        info = load_one(path)
        points.append(info)
        print(f"tau={info['tau']:6.2f}  1-R^2={info['1-R2']:.6f}  "
              f"slope={info['slope']:.6f}  ||F||inf={info['F_max']:.2e}")

    points.sort(key=lambda d: d["tau"])

    payload = {
        "gamma": 1.0,
        "params": {"G": 20, "UMAX": 5.0, "trim": 0.0, "dps": 50,
                   "weighting": "ex-ante 0.5*(f0^3+f1^3)"},
        "points": points,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_PATH} ({len(points)} points)")


if __name__ == "__main__":
    main()
