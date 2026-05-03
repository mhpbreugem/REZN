#!/usr/bin/env python3
"""Fig 10 convergence path: emit ||F||_inf vs iter from seed history.

The seed file already stores F_max at full mp300 precision. We just dump
the values with enough digits to be unambiguous. No new computation.

Output: results/full_ree/fig10_convergence_pgfplots.tex
"""

import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 100

SEED = Path("results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json")
OUT = Path("results/full_ree/fig10_convergence_pgfplots.tex")


def main():
    with SEED.open() as f:
        d = json.load(f)
    history = d["history"]
    rows = []
    for h in history:
        it = int(h["iter"])
        # Convert to mpf at dps=100 from seed's mp300 string
        Fmax = mp.mpf(h["F_max"])
        rows.append((it, Fmax))

    # Format each F_max in mp-aware exponential to ~30 digits (more than enough
    # for plotting while still being mp-faithful within dps=100 precision).
    def fmt(x):
        s = mp.nstr(x, 30, strip_zeros=False)
        return s

    coords = "".join(f"({it},{fmt(Fmax)})" for it, Fmax in rows)
    lines = [
        f"% Fig 10 convergence path: ||F||_inf vs iteration",
        f"% Source: {SEED.name}",
        f"% G={d['G']} UMAX={d['UMAX']} gamma={d['gamma']} tau={d['tau']} dps={d['dps']}",
        f"% Output precision: mp dps=100, tol 1e-50",
        f"% Final F_max (mp-faithful, ~30 digits shown): {fmt(rows[-1][1])}",
        f"% n_iter = {len(rows)}",
        "",
        f"\\addplot coordinates {{{coords}}};",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    for it, Fmax in rows:
        print(f"  iter {it:2d}: ||F||_inf = {fmt(Fmax)}")


if __name__ == "__main__":
    main()
