#!/usr/bin/env python3
"""
Production ablations for referee M3 / M4.

1) η-interpolation of static controllers: w(η) = (1-η) w0 + η w*
2) Scale-invariant threshold: tau *= ||w||_2 at decision time

Does not alter RNG streams of the main counted runs
(scale_invariant=False, tau0=0.65 remains the default path).

Writes:
  ablation_out/ablation_eta.csv
  ablation_out/ablation_scale.csv
  ablation_out/figures/eta_curve.png
  ablation_out/figures/scale_compare_misspec.png
  ablation_out/figures/scale_compare_aligned.png
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aumcdm.envs.tool_select import make_w_star
from aumcdm.engine.baselines import make_suite
from runner import run_arm, DEFAULT_W0

# --------------------------------------------------------------------------- #
# Single source of truth for the scale-invariant base threshold.
# --------------------------------------------------------------------------- #
TAU0_SI: float = 1.05          # <-- change only this constant after calibration


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _mean_row(rows: list[dict], keys: list[str]) -> dict[str, float]:
    out = {}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in rows]))
    return out


def _boot_ci(vals: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(vals, float)
    if len(x) < 2:
        m = float(x.mean()) if len(x) else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _tau_use(tau0: float, scale_invariant: bool) -> float:
    return float(TAU0_SI) if scale_invariant else float(tau0)


# --------------------------------------------------------------------------- #
# η-interpolation (static only)
# --------------------------------------------------------------------------- #

def run_eta_sweep(
    etas: list[float],
    T: int,
    seeds: int,
    rho: float,
    tau0: float,
    alpha: float,
    seed_base: int,
    scale_invariant: bool = False,
) -> pd.DataFrame:
    w_star = make_w_star(rho)
    records = []

    for eta in etas:
        w0 = (1.0 - eta) * DEFAULT_W0 + eta * w_star
        w0 = w0 / w0.sum()

        tau_use = _tau_use(tau0, scale_invariant)
        suite = make_suite(
            w0, alpha=alpha, eps=0.10, tau0=tau_use, kappa=1.0, eta=0.0,
            scale_invariant=bool(scale_invariant),
        )
        static_arm = next(a for a in suite if a.name == "B3_static")

        if hasattr(static_arm, "cfg") and hasattr(static_arm.cfg, "scale_invariant"):
            static_arm.cfg.scale_invariant = bool(scale_invariant)
        elif hasattr(static_arm, "cfg"):
            try:
                static_arm.cfg = replace(
                    static_arm.cfg, scale_invariant=bool(scale_invariant))
            except Exception:
                setattr(static_arm.cfg, "scale_invariant",
                        bool(scale_invariant))

        seed_rows = []
        for seed in range(seeds):
            r = run_arm(static_arm, T=T, seed=seed, w_star=w_star,
                        seed_base=seed_base, log_episodes=False)
            seed_rows.append(r)

        means = _mean_row(
            seed_rows, ["mean_R", "cvar10", "abstain_rate", "catastrophic_rate", "w2_final"])
        cvar_lo, cvar_hi = _boot_ci([r["cvar10"] for r in seed_rows])
        abs_lo, abs_hi = _boot_ci([r["abstain_rate"] for r in seed_rows])

        w_norm = float(np.linalg.norm(w0))
        tau_eff = float(tau_use * (w_norm + 1e-12)
                        ) if scale_invariant else float(tau_use)

        records.append({
            "eta": eta,
            "scale_invariant": bool(scale_invariant),
            "mean_R": means["mean_R"],
            "cvar10": means["cvar10"],
            "cvar10_lo": cvar_lo,
            "cvar10_hi": cvar_hi,
            "abstain_rate": means["abstain_rate"],
            "abstain_lo": abs_lo,
            "abstain_hi": abs_hi,
            "catastrophic_rate": means["catastrophic_rate"],
            "w2_final": means["w2_final"],
            "L1_to_wstar": float(np.abs(w0 - w_star).sum()),
            "tau_use": tau_use,
            "w_norm": w_norm,
            "tau_eff": tau_eff,
        })
        print(
            f"eta={eta:.2f}  scale={scale_invariant}  CVaR={means['cvar10']:.4f}  abs={means['abstain_rate']:.3f}  L1={records[-1]['L1_to_wstar']:.3f}  ||w||={w_norm:.4f}  tau_eff={tau_eff:.4f}")

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Adaptive skip under fixed vs scale-invariant threshold
# --------------------------------------------------------------------------- #

def run_adaptive_compare(
    T: int,
    seeds: int,
    rho: float,
    tau0: float,
    alpha: float,
    seed_base: int,
    aligned: bool,
) -> pd.DataFrame:
    w_star = make_w_star(rho)
    w0 = w_star.copy() if aligned else DEFAULT_W0.copy()
    records = []

    for scale_inv in (False, True):
        tau_use = _tau_use(tau0, scale_inv)
        suite = make_suite(w0, alpha=alpha, eps=0.10, tau0=tau_use,
                           kappa=1.0, eta=0.0, scale_invariant=bool(scale_inv))
        skip = next(a for a in suite if a.name == "B4_eg_skip")
        static = next(a for a in suite if a.name == "B3_static")

        for arm in (static, skip):
            seed_rows = []
            for seed in range(seeds):
                r = run_arm(arm, T=T, seed=seed, w_star=w_star,
                            seed_base=seed_base, log_episodes=False)
                seed_rows.append(r)

            means = _mean_row(seed_rows, [
                              "mean_R", "cvar10", "abstain_rate", "catastrophic_rate", "w2_drift", "w2_final"])
            cvar_lo, cvar_hi = _boot_ci([r["cvar10"] for r in seed_rows])
            records.append({
                "arm": arm.name,
                "scale_invariant": bool(scale_inv),
                "aligned": bool(aligned),
                **means,
                "cvar10_lo": cvar_lo,
                "cvar10_hi": cvar_hi,
                "tau_use": tau_use,
            })
            print(
                f"{arm.name:12s} scale={scale_inv} aligned={aligned}  CVaR={means['cvar10']:.4f} abs={means['abstain_rate']:.3f} w2_drift={means.get('w2_drift', float('nan')):+.4f}")

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Plotting functions
# --------------------------------------------------------------------------- #

def plot_eta(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for scale, g in df.groupby("scale_invariant"):
        g = g.sort_values("eta")
        label = "scale-inv" if scale else "fixed τ₀"
        ax.plot(g["eta"], g["cvar10"], "o-", label=label)
        ax.fill_between(g["eta"], g["cvar10_lo"], g["cvar10_hi"], alpha=0.18)
    ax.set_xlabel("η (interpolation weight)")
    ax.set_ylabel("CVaR@10")
    ax.set_title("CVaR@10 vs η-interpolation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def plot_scale(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for scale, g in df.groupby("scale_invariant"):
        g = g.sort_values("arm")
        label = "scale-inv" if scale else "fixed τ₀"
        # Plot points with error bars
        ax.errorbar(g["arm"], g["cvar10"],
                    yerr=[g["cvar10"] - g["cvar10_lo"],
                          g["cvar10_hi"] - g["cvar10"]],
                    fmt="o-", label=label, capsize=4)
    ax.set_xlabel("Arm")
    ax.set_ylabel("CVaR@10")
    aligned_status = "Aligned" if df["aligned"].iloc[0] else "Misspecified"
    ax.set_title(f"Scale-invariant vs Fixed τ₀ ({aligned_status})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def plot_coverage(df: pd.DataFrame, path: str) -> None:
    """
    CVaR@10 vs realised abstention rate using the full span of
    η-path + adaptive points (fixed vs scale-invariant).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for scale, g in df.groupby("scale_invariant"):
        g = g.sort_values("abstain_rate")
        label = "scale-inv" if scale else "fixed τ₀"
        ax.plot(g["abstain_rate"], g["cvar10"],
                "o-", label=label, markersize=7)
        if "cvar10_lo" in g.columns and "cvar10_hi" in g.columns:
            ax.fill_between(
                g["abstain_rate"], g["cvar10_lo"], g["cvar10_hi"], alpha=0.18
            )

    ax.set_xlabel("Abstain rate (fraction of episodes)")
    ax.set_ylabel("CVaR@10")
    ax.set_title("CVaR@10 vs realised abstention rate (exploratory)")
    ax.legend(loc="best")
    ax.set_xlim(0.12, 0.26)          # full paper range
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--rho", type=float, default=3.0)
    p.add_argument("--tau0", type=float, default=0.65)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed_base", type=int, default=10_000)
    p.add_argument("--out", type=str, default="ablation_out")
    p.add_argument("--etas", type=str, default="0,0.25,0.5,0.75,1.0")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    etas = [float(x) for x in args.etas.split(",")]

    print(f"TAU0_SI = {TAU0_SI}  (scale-invariant base threshold)")

    # --- M4: η curve under fixed τ0 ---
    print("=== η-interpolation (fixed τ0) ===")
    df_eta_fixed = run_eta_sweep(etas, args.T, args.seeds, args.rho,
                                 args.tau0, args.alpha, args.seed_base, scale_invariant=False)
    print("=== η-interpolation (scale-invariant τ) ===")
    df_eta_scale = run_eta_sweep(etas, args.T, args.seeds, args.rho,
                                 args.tau0, args.alpha, args.seed_base, scale_invariant=True)
    df_eta = pd.concat([df_eta_fixed, df_eta_scale], ignore_index=True)
    eta_path = os.path.join(args.out, "ablation_eta.csv")
    df_eta.to_csv(eta_path, index=False)
    print(f"wrote {eta_path}")
    plot_eta(df_eta, os.path.join(args.out, "figures", "eta_curve.png"))

    # --- M3: adaptive skip fixed vs scale-invariant ---
    print("=== Adaptive compare (misspec) ===")
    df_mis = run_adaptive_compare(
        args.T, args.seeds, args.rho, args.tau0, args.alpha, args.seed_base, aligned=False)
    print("=== Adaptive compare (aligned) ===")
    df_al = run_adaptive_compare(
        args.T, args.seeds, args.rho, args.tau0, args.alpha, args.seed_base, aligned=True)
    df_ad = pd.concat([df_mis, df_al], ignore_index=True)
    ad_path = os.path.join(args.out, "ablation_scale.csv")
    df_ad.to_csv(ad_path, index=False)
    print(f"wrote {ad_path}")
    plot_scale(df_ad[df_ad["aligned"] == False], os.path.join(
        args.out, "figures", "scale_compare_misspec.png"))
    plot_scale(df_ad[df_ad["aligned"] == True], os.path.join(
        args.out, "figures", "scale_compare_aligned.png"))

    # --- CVaR vs realised coverage (full span = η-path + adaptive) ---
    print("=== CVaR vs realised coverage (exploratory, full span) ===")
    # η-path already has the wide coverage range; adaptive adds the skip points
    df_cov = pd.concat(
        [
            df_eta[["scale_invariant", "abstain_rate",
                    "cvar10", "cvar10_lo", "cvar10_hi"]],
            df_ad[["scale_invariant", "abstain_rate",
                   "cvar10", "cvar10_lo", "cvar10_hi"]],
        ],
        ignore_index=True,
    )
    # drop exact duplicates that appear in both tables
    df_cov = df_cov.drop_duplicates(
        subset=["scale_invariant", "abstain_rate", "cvar10"]
    ).sort_values(["scale_invariant", "abstain_rate"])
    plot_coverage(
        df_cov,
        os.path.join(args.out, "figures", "cvar_vs_coverage.png"),
    )


if __name__ == "__main__":
    main()
