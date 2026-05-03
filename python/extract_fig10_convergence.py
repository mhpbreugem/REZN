#!/usr/bin/env python3
"""Extract Fig 10 convergence path from the G=20 mp300 seed checkpoint.

Reads the iteration history recorded in
results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json and writes
||F||_inf vs iteration index as a pgfplots fragment.
"""

import json
from pathlib import Path

SEED = Path("results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json")
OUT = Path("results/full_ree/fig10_convergence_pgfplots.tex")


def main():
    with SEED.open() as f:
        d = json.load(f)
    history = d["history"]
    rows = []
    for h in history:
        it = int(h["iter"])
        Fmax = float(h["F_max"])
        rows.append((it, Fmax))

    coords = "".join(f"({it},{Fmax:.6e})" for it, Fmax in rows)

    lines = [
        f"% Fig 10 convergence path: ||F||_inf vs iteration",
        f"% Source: {SEED.name}",
        f"% G={d['G']} UMAX={d['UMAX']} gamma={d['gamma']} tau={d['tau']} dps={d['dps']}",
        f"% final F_max = {d['F_max'][:24]}... ({rows[-1][1]:.3e})",
        f"% n_iter = {len(rows)}",
        "",
        f"\\addplot coordinates {{{coords}}};",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT} ({len(rows)} iterations, final ||F||_inf={rows[-1][1]:.3e})")
    for it, Fmax in rows:
        print(f"  iter {it:2d}: ||F||_inf = {Fmax:.3e}")


if __name__ == "__main__":
    main()
