#!/usr/bin/env python3
"""R2 per-update paired tests + R4 four-coordinate drift."""
import numpy as np
import pandas as pd
from pathlib import Path

W_STAR = np.array([0.2375, 0.7125, 0.05, 0.0])


def perm_pvalue(d, n_perm=100_000, seed=0):
    d = np.asarray(d, float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    # exact sign-flip permutation for paired differences
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.abs(null) >= abs(obs)).mean()
    return float(obs), float(d.std(ddof=1)), float(p)


def load(path):
    return pd.read_csv(path)


def cell_report(path, label):
    df = load(path)
    skip = df[df.arm == "B4_eg_skip"].set_index("seed")
    full = df[df.arm == "B4_full"].set_index("seed")
    common = skip.index.intersection(full.index)
    # prefer logged counts
    if "n_updates" in skip.columns:
        n_s = skip.loc[common, "n_updates"].astype(float)
        n_f = full.loc[common, "n_updates"].astype(float)
    else:
        T = 500
        n_s = T * (1 - skip.loc[common, "abstain_rate"])
        n_f = pd.Series(T, index=common, dtype=float)
        print(f"  WARNING {label}: using inferred n_updates")
    dpu_s = skip.loc[common, "w2_drift"] / n_s.replace(0, np.nan)
    dpu_f = full.loc[common, "w2_drift"] / n_f.replace(0, np.nan)
    delta = (dpu_s - dpu_f).dropna()
    m, sd, p = perm_pvalue(delta.values)
    print(f"{label}: mean Δ(dpu)={m:+.3e}  SD={sd:.3e}  perm-p={p:.4f}  n={len(delta)}")
    # cumulative for reference
    pc = (skip.loc[common, "w2_drift"] - full.loc[common, "w2_drift"])
    print(f"         mean P1'_cum={pc.mean():+.4f}  SD={pc.std(ddof=1):.4f}")


print("=== R2 per-update paired Δ ===")
# edit paths to your counted runs
for lab, p in [
    ("A mis", "results_cnt_mis/summary.csv"),
    ("A al", "results_cnt_al/summary.csv"),
    ("B mis", "results_cnt_mis_b/summary.csv"),
    ("B al", "results_cnt_al_b/summary.csv"),
]:
    if Path(p).exists():
        cell_report(p, lab)
    else:
        print(f"missing {p}")

print("\n=== R4 four-coordinate drift (misspecified skip) ===")
for lab, p in [("A mis", "results_cnt_mis/summary.csv"), ("B mis", "results_cnt_mis_b/summary.csv")]:
    if not Path(p).exists():
        continue
    df = load(p)
    # if only w2 in summary, use w_traj last rows — fallback message
    if not set(["w1_final", "w2_final", "w3_final", "w4_final"]).issubset(df.columns):
        print(f"{lab}: need w1_final..w4_final in summary (log full w_final)")
        continue
    sub = df[df.arm == "B4_eg_skip"]
    w0 = np.array([0.45, 0.40, 0.10, 0.05])
    for j, name in enumerate(["w1", "w2", "w3", "w4"]):
        drift = sub[f"w{j+1}_final"].mean() - w0[j]
        print(f"  {lab} skip mean drift {name}={drift:+.4f}")
