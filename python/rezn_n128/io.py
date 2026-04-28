"""Pickle save/load with metadata."""
from __future__ import annotations
import datetime
import pickle
import numpy as np


SCHEMA_VERSION = "rezn_n128/1"


def save(path, P_f128, *, taus, gammas, Ws, G, umax, Finf, one_minus_R2,
         label="", history=None, diagnostics=None, extra=None):
    """Pickle a result dict to disk. P_f128 is the f128 tensor;
    a f64 cast is included for compatibility with downstream tools."""
    record = {
        "schema": SCHEMA_VERSION,
        "P": np.asarray(P_f128).astype(np.float64),
        "P_f128": np.asarray(P_f128, dtype=np.float128),
        "taus": np.asarray(taus, dtype=np.float64),
        "gammas": np.asarray(gammas, dtype=np.float64),
        "Ws": np.asarray(Ws, dtype=np.float64),
        "G": int(G),
        "umax": float(umax),
        "Finf": float(Finf),
        "1-R²": float(one_minus_R2),
        "kernel": "rezn_n128 strict f128 (linear-interp contour)",
        "label": label,
        "history": list(history) if history is not None else [],
        "diagnostics": diagnostics,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        record.update(extra)
    with open(path, "wb") as f:
        pickle.dump(record, f)
    return record


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
