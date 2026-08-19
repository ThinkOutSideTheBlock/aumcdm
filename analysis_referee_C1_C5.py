#!/usr/bin/env python3
"""Referee C1–C3, C5-prep from existing summary.csv files."""
import numpy as np
import pandas as pd
from pathlib import Path

T = 500
W_STAR = np.array([0.2375, 0.7125, 0.05, 0.0])  # make_w_star(3.0)

PATHS = {
    "def_mis": "results_headline/summary.csv",
    "def_al": "results_aligned/summary.csv",
    "repl_mis": "results_repl_misaligned/summary.csv",
    "repl_al": "results_repl_aligned/summary.csv",
}


def load(name):
    p = Path(PATHS[name])
    if not p.exists():
        # try alternates
        alts = list(Path(".").glob(f"**/{p.name}"))
        raise FileNotFoundError(f"Missing {p}. Found: {alts[:10]}")
    return pd.read_csv(p)


def paired_p1(df):
    skip = df[df.arm == "B4_eg_skip"].set_index("seed")
    full = df[df.arm == "B4_full"].set_index("seed")
    common = skip.index.intersection(full.index)
    d = skip.loc[common, "w2_drift"] - full.loc[common, "w2_drift"]
    return d


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(x.std(ddof=1))


print("=" * 72)
print("C1: UPDATE-COUNT NULL")
print("=" * 72)
print("Approx n_updates ≈ T * (1 - abstain_rate)  [terminal act rounds only]")
print("drift_per_update ≈ w2_drift / n_updates")
print()

for label, key in [
    ("Defining misaligned", "def_mis"),
    ("Defining aligned", "def_al"),
    ("Repl misaligned", "repl_mis"),
    ("Repl aligned", "repl_al"),
]:
    df = load(key)
    rows = []
    for arm in ["B4_eg_skip", "B4_full"]:
        sub = df[df.arm == arm]
        ab = sub["abstain_rate"].mean()
        drift = sub["w2_drift"].mean()
        n_up = T * (1 - ab)
        dpu = drift / max(n_up, 1e-9)
        rows.append((arm, drift, ab, n_up, dpu))
    (a1, d1, ab1, n1, p1), (a2, d2, ab2, n2, p2) = rows
    ratio_drift = d1 / d2 if d2 != 0 else np.nan
    ratio_upd = n1 / n2
    print(f"{label}")
    print(
        f"  skip: drift={d1:+.4f}  abstain={ab1:.3f}  n_up≈{n1:.1f}  drift/up={p1:+.6f}")
    print(
        f"  full: drift={d2:+.4f}  abstain={ab2:.3f}  n_up≈{n2:.1f}  drift/up={p2:+.6f}")
    print(
        f"  ratio drift skip/full={ratio_drift:.3f}  ratio updates={ratio_upd:.3f}")
    print(f"  P1' raw={d1-d2:+.4f}  P1'_per_update={p1-p2:+.6f}")
    print()

print("=" * 72)
print("C2: DISTANCE TO w* (aligned cells) — uses w2 only if full w absent")
print("=" * 72)
for label, key in [("Defining aligned", "def_al"), ("Repl aligned", "repl_al")]:
    df = load(key)
    for arm in ["B4_eg_skip", "B4_full", "B3_static"]:
        sub = df[df.arm == arm]
        # w2 error only (full vector better if you log it)
        err = (sub["w2_final"] - W_STAR[1]).abs()
        print(f"  {label} {arm}: mean |w2_T - w*_2|={err.mean():.4f}  "
              f"mean w2_final={sub['w2_final'].mean():.4f}")
    print()

print("=" * 72)
print("C3: POOL 40 SEEDS")
print("=" * 72)
for tag, k1, k2 in [
    ("Misaligned def+repl", "def_mis", "repl_mis"),
    ("Aligned def+repl", "def_al", "repl_al"),
]:
    d1, d2 = paired_p1(load(k1)), paired_p1(load(k2))
    # reindex to avoid seed collision: offset repl
    d2 = d2.copy()
    d2.index = d2.index + 1000
    d = pd.concat([d1, d2])
    m, lo, hi, sd = boot_ci(d.values)
    print(
        f"{tag}: n={len(d)}  P1'={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  seed_SD={sd:.4f}")
print()

print("=" * 72)
print("C5 proxy: catastrophe & harm (not full CVaR composition)")
print("=" * 72)
for label, key in [("def_mis", "def_mis"), ("repl_mis", "repl_mis")]:
    df = load(key)
    for arm in ["B3_static", "B4_eg_skip", "B4_full"]:
        sub = df[df.arm == arm]
        print(f"  {label} {arm}: cat={sub['catastrophic_rate'].mean():.4f}  "
              f"harm={sub['harm_rate'].mean():.4f}  abs={sub['abstain_rate'].mean():.3f}  "
              f"CVaR={sub['cvar10'].mean():.4f}")
    print()
