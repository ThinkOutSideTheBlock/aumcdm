#!/usr/bin/env python3
"""
Smoke calibration of TAU0_SI so that the scale-invariant abstain rate
matches the fixed-rule abstain rate under the default misspecified w0.

Usage (fast):
  python calibrate_tau_si.py --T 50 --seeds 5

Usage (more reliable):
  python calibrate_tau_si.py --T 100 --seeds 10

Prints a table of (tau0_si, abstain_rate) and recommends the value
closest to the fixed-rule target.  Then copy that number into
ablation.py → TAU0_SI.
"""
from __future__ import annotations

import argparse
import numpy as np

from aumcdm.envs.tool_select import make_w_star
from aumcdm.engine.baselines import make_suite
from runner import run_arm, DEFAULT_W0


def measure_abstain(tau0: float, scale_invariant: bool,
                    T: int, seeds: int, rho: float,
                    alpha: float, seed_base: int) -> float:
    w_star = make_w_star(rho)
    w0 = DEFAULT_W0.copy()
    suite = make_suite(
        w0, alpha=alpha, eps=0.10, tau0=tau0, kappa=1.0, eta=0.0,
        scale_invariant=bool(scale_invariant),
    )
    arm = next(a for a in suite if a.name == "B3_static")
    rates = []
    for seed in range(seeds):
        r = run_arm(arm, T=T, seed=seed, w_star=w_star,
                    seed_base=seed_base, log_episodes=False)
        rates.append(r["abstain_rate"])
    return float(np.mean(rates))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--rho", type=float, default=3.0)
    p.add_argument("--tau0", type=float, default=0.65)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed_base", type=int, default=10_000)
    p.add_argument("--grid", type=str,
                   default="0.8,0.9,1.0,1.05,1.1,1.15,1.2,1.3,1.4,1.5")
    args = p.parse_args()

    # Fixed-rule target
    target = measure_abstain(
        args.tau0, False, args.T, args.seeds, args.rho,
        args.alpha, args.seed_base)
    print(f"Fixed τ0={args.tau0}  abstain rate = {target:.4f}  "
          f"(T={args.T}, seeds={args.seeds})")
    print("-" * 48)

    grid = [float(x) for x in args.grid.split(",")]
    best_tau, best_diff = None, 1e9
    for tau_si in grid:
        rate = measure_abstain(
            tau_si, True, args.T, args.seeds, args.rho,
            args.alpha, args.seed_base)
        diff = abs(rate - target)
        mark = "  <-- closest" if diff < best_diff else ""
        if diff < best_diff:
            best_diff = diff
            best_tau = tau_si
        print(f"  tau0_si={tau_si:.3f}  abs={rate:.4f}  |Δ|={diff:.4f}{mark}")

    print("-" * 48)
    print(f"RECOMMENDED  TAU0_SI = {best_tau}")
    print("Copy that value into ablation.py (top-level constant) and re-run:")
    print("  python ablation.py --T 500 --seeds 20 --tau0 0.65")


if __name__ == "__main__":
    main()