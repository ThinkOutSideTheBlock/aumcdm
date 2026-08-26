#!/usr/bin/env python3
"""Revision-3 tables from existing results_cnt_* and results_logs_*."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")
LOGS = {
    ("A", "mis", "B3_static"): ROOT / "results_logs_mis/episodes_A_misspec_B3_static.csv",
    ("A", "mis", "B4_eg_skip"): ROOT / "results_logs_mis/episodes_A_misspec_B4_eg_skip.csv",
    ("A", "al", "B3_static"): ROOT / "results_logs_al/episodes_A_aligned_B3_static.csv",
    ("A", "al", "B4_eg_skip"): ROOT / "results_logs_al/episodes_A_aligned_B4_eg_skip.csv",
    ("B", "mis", "B3_static"): ROOT / "results_logs_mis_b/episodes_B_misspec_B3_static.csv",
    ("B", "mis", "B4_eg_skip"): ROOT / "results_logs_mis_b/episodes_B_misspec_B4_eg_skip.csv",
    ("B", "al", "B3_static"): ROOT / "results_logs_al_b/episodes_B_aligned_B3_static.csv",
    ("B", "al", "B4_eg_skip"): ROOT / "results_logs_al_b/episodes_B_aligned_B4_eg_skip.csv",
}
CNT = {
    ("A", "mis"): ROOT / "results_cnt_mis/summary.csv",
    ("A", "al"): ROOT / "results_cnt_al/summary.csv",
    ("B", "mis"): ROOT / "results_cnt_mis_b/summary.csv",
    ("B", "al"): ROOT / "results_cnt_al_b/summary.csv",
}


def load_ep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    return df


def perm_p(delta: np.ndarray, n: int = 100_000, seed: int = 0) -> tuple[float, float]:
    d = np.asarray(delta, float)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    return float((np.abs(null) >= abs(obs)).mean()), obs


def boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def composition_table() -> pd.DataFrame:
    rows = []
    for (seed_set, init, arm), path in LOGS.items():
        df = load_ep(path)
        for seed, g in df.groupby("seed_id"):
            term = g["terminal_action"].astype(str)
            succ = g["success"]
            is_act = term.str.startswith("act_")
            is_abs = (g["abstain_flag"] == 1) | (term == "abstain")
            n_s = int((is_act & (succ == 1)).sum())
            n_f = int((is_act & (succ == 0)).sum())
            n_a = int(is_abs.sum())
            rows.append({
                "seed_set": seed_set, "init": init, "arm": arm, "seed": seed,
                "n_success": n_s, "n_failure": n_f, "n_abstain": n_a,
                "mean_R": float(g["reward"].mean()),
                "mean_R_fail": float(g.loc[is_act & (succ == 0), "reward"].mean()) if n_f else np.nan,
            })
    out = pd.DataFrame(rows)
    summary = (out.groupby(["seed_set", "init", "arm"], as_index=False)
               .agg(n_success=("n_success", "mean"),
                    n_failure=("n_failure", "mean"),
                    n_abstain=("n_abstain", "mean"),
                    mean_R=("mean_R", "mean"),
                    mean_R_fail=("mean_R_fail", "mean")))
    return out, summary


def failure_by_tool() -> pd.DataFrame:
    rows = []
    for (seed_set, init, arm), path in LOGS.items():
        df = load_ep(path)
        term = df["terminal_action"].astype(str)
        succ = df["success"]
        fail = df.loc[term.str.startswith("act_") & (succ == 0)].copy()
        fail["tool"] = pd.to_numeric(fail["terminal_action"].str.replace("act_", "", regex=False),
                                     errors="coerce")
        vc = fail["tool"].value_counts().sort_index()
        for tool, cnt in vc.items():
            rows.append({"seed_set": seed_set, "init": init, "arm": arm,
                         "tool": int(tool), "n_fail": int(cnt)})
    return pd.DataFrame(rows)


def static_mis_vs_al() -> None:
    print("\n=== B3: static misspec − static aligned CVaR ===")
    for ss in ("A", "B"):
        mis = pd.read_csv(CNT[(ss, "mis")])
        al = pd.read_csv(CNT[(ss, "al")])
        m = mis[mis.arm == "B3_static"].set_index("seed").sort_index()
        a = al[al.arm == "B3_static"].set_index("seed").sort_index()
        common = m.index.intersection(a.index)
        d = (m.loc[common, "cvar10"] - a.loc[common, "cvar10"]).values
        p, obs = perm_p(d)
        lo, hi = boot_ci(d)
        print(f"  set {ss}: Δ={obs:+.4f}  CI[{lo:.4f},{hi:.4f}]  p={p:.6f}")


def family_skip_static() -> None:
    print("\n=== Family: skip − static (12) ===")
    for ss, init in [("A", "mis"), ("B", "mis"), ("A", "al"), ("B", "al")]:
        df = pd.read_csv(CNT[(ss, init)])
        s = df[df.arm == "B3_static"].set_index("seed")
        k = df[df.arm == "B4_eg_skip"].set_index("seed")
        common = s.index.intersection(k.index)
        for metric in ("cvar10", "mean_R", "catastrophic_rate"):
            d = (k.loc[common, metric] - s.loc[common, metric]).values
            p, obs = perm_p(d)
            print(f"  {ss}/{init} {metric}: Δ={obs:+.5f} p={p:.5f}")


def main() -> None:
    _, summary = composition_table()
    print("=== B1 composition (mean per seed) ===")
    print(summary.to_string(index=False))
    summary.to_csv("revision3_composition.csv", index=False)

    tools = failure_by_tool()
    print("\n=== Failure counts by tool (all episodes, pooled seeds) ===")
    piv = tools.pivot_table(index=["seed_set", "init", "arm"], columns="tool",
                            values="n_fail", fill_value=0)
    print(piv.to_string())
    piv.to_csv("revision3_fail_by_tool.csv")

    static_mis_vs_al()
    family_skip_static()
    print("\nwrote revision3_composition.csv, revision3_fail_by_tool.csv")


if __name__ == "__main__":
    main()
