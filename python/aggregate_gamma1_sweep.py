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

OUT_PATH = "results/full_ree/fig4A_g100_tau_sweep.json"


def load_one(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    u_grid = np.array([float(s) for s in d["u_grid"]])
    p_grids = np.array([[float(x) for x in row] for row in d["p_grid"]])
    mu = np.array([[float(x) for x in row] for row in d["mu_strings"]])
    one_mR2, slope, n = weighted_1mR2(u_grid, p_grids, mu, d["tau"], d["gamma"])
    out = {
        "tau": float(d["tau"]),
        "1-R2": float(one_mR2),
        "slope": float(slope),
        "n_triples": int(n),
        "F_max": float(d["F_max"]),
        "F_med": float(d["F_med"]),
        "dps": int(d.get("dps", 50)),
        "checkpoint": os.path.basename(path),
    }
    if "F_active" in d:
        out["F_active"] = float(d["F_active"])
    return out


def best_checkpoint_for_tau(indir: str, tau_token: str) -> str:
    """Prefer the highest-precision checkpoint available for a given tau."""
    candidates = [f for f in os.listdir(indir)
                  if f.startswith(f"task3_g100_{tau_token}") and f.endswith(".json")
                  and "_mp" in f]
    if not candidates:
        return None
    # Sort by dps descending: '_mp100' before '_mp50'
    def dps_of(name):
        m = re.search(r"_mp(\d+)\.json$", name)
        return int(m.group(1)) if m else 0
    candidates.sort(key=lambda n: -dps_of(n))
    return os.path.join(indir, candidates[0])


def main():
    indir = "results/full_ree"
    tau_tokens = sorted({re.match(r"task3_g100_(t\d+)_mp\d+\.json", f).group(1)
                         for f in os.listdir(indir)
                         if re.match(r"task3_g100_t\d+_mp\d+\.json", f)})
    if not tau_tokens:
        print("No checkpoints found")
        return
    points = []
    for token in tau_tokens:
        path = best_checkpoint_for_tau(indir, token)
        if path is None:
            continue
        info = load_one(path)
        points.append(info)
        active_str = (f"  ||F||act={info['F_active']:.2e}"
                      if "F_active" in info else "")
        print(f"tau={info['tau']:6.2f}  1-R^2={info['1-R2']:.6f}  "
              f"slope={info['slope']:.6f}  ||F||inf={info['F_max']:.2e}"
              f"{active_str}  dps={info['dps']}")

    points.sort(key=lambda d: d["tau"])

    payload = {
        "gamma": 1.0,
        "params": {"G": 20, "UMAX": 5.0, "trim": 0.0,
                   "weighting": "ex-ante 0.5*(f0^3+f1^3)",
                   "highest_dps_per_point": True},
        "points": points,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_PATH} ({len(points)} points)")


if __name__ == "__main__":
    main()
