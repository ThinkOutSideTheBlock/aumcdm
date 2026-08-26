#!/usr/bin/env python3
"""
Table M1 — bottom-decile (worst 50 / T=500) composition.

Re-runs B3_static and B4_eg_skip with episode-level logging, writes
episodes_*.csv, then reports pure-failure / abstain / catastrophic counts
in the bottom 50 episodes per seed.

Does NOT alter the main counted MVP runs (different seed_base offset is fine;
this is an analysis-only path).

Usage:
  python bottom_decile.py --T 500 --seeds 20
  python bottom_decile.py --T 500 --seeds 20 --from-csv   # if CSVs already exist
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

from aumcdm.envs.tool_select import ToolSelectEnv, make_w_star
from aumcdm.engine.baselines import make_suite
from aumcdm.engine.decision import decide, risk_adjust
from runner import DEFAULT_W0, MAX_STEPS, select

# --------------------------------------------------------------------------- #
# Episode with full fields for bottom-decile analysis
# --------------------------------------------------------------------------- #


def run_episode_logged(env, arm, adapter, rng) -> dict[str, Any]:
    env.reset()
    v_oracle = env.oracle_value()
    last_aux = None
    last_info = None

    for _ in range(MAX_STEPS):
        w = adapter.snapshot()
        a, aux = select(env, arm, w, rng)
        last_aux = aux
        terminal, phi, info = env.step(a)
        last_info = info
        if terminal:
            R = env.reward(phi)
            acted = (info["kind"] == "act")
            adapter.update(
                qt=aux["qt"], u_exec=aux["u_exec"], R=R,
                propensity=aux["propensity"], terminal_acted=acted,
            )
            kind = info["kind"]
            success = bool(info.get("success", False))
            return {
                "reward": float(R),
                "regret": float(v_oracle - R),
                "success": int(success),
                "abstain_flag": int(kind == "abstain"),
                "catastrophic": int(bool(info.get("catastrophic", False))),
                "terminal_action": (
                    f"act_{info['k']}" if kind == "act"
                    else ("abstain" if kind == "abstain" else kind)
                ),
                "harm": float(info.get("harm", 0.0)),
                "censored": bool(aux.get("censored", False)),
                "forced": bool(aux.get("forced", False)),
            }

    # step-cap fallback
    phi = env.phi_terminal(env.ABSTAIN)
    R = env.reward(phi)
    return {
        "reward": float(R),
        "regret": float(v_oracle - R),
        "success": 0,
        "abstain_flag": 1,
        "catastrophic": 0,
        "terminal_action": "abstain",
        "harm": 0.0,
        "censored": bool(last_aux["censored"]) if last_aux else True,
        "forced": False,
    }


def collect_episodes(
    arm_name: str,
    T: int,
    seeds: int,
    rho: float,
    tau0: float,
    alpha: float,
    seed_base: int,
) -> pd.DataFrame:
    w_star = make_w_star(rho)
    w0 = DEFAULT_W0.copy()
    suite = make_suite(
        w0, alpha=alpha, eps=0.10, tau0=tau0, kappa=1.0, eta=0.0,
        scale_invariant=False,
    )
    arm = next(a for a in suite if a.name == arm_name)

    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed_base + seed)
        env = ToolSelectEnv(w_star, rng)
        adapter = arm.make_adapter()
        for ep in range(T):
            rec = run_episode_logged(env, arm, adapter, rng)
            rec["seed_id"] = seed
            rec["episode"] = ep
            rec["arm"] = arm_name
            rows.append(rec)
        print(f"  {arm_name}  seed={seed:02d}  done")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bottom-50 (bottom 10% of T=500) composition — your original logic
# --------------------------------------------------------------------------- #

def bottom50_stats(df: pd.DataFrame, k: int = 50) -> dict:
    """df: one arm, all seeds. Returns mean counts in the worst-k episodes."""
    rows = []
    for seed, g in df.groupby("seed_id"):
        bot = g.sort_values("reward").head(k)
        # pure failure = acted and did not succeed
        failures = int(
            ((bot["success"] == 0) &
             bot["terminal_action"].str.startswith("act_")).sum()
        )
        abstains = int((bot["abstain_flag"] == 1).sum())
        cats = int(bot["catastrophic"].sum())
        rows.append({
            "seed_id": seed,
            "failures": failures,
            "abstains": abstains,
            "cats": cats,
            "n_bot": len(bot),
        })
    out = pd.DataFrame(rows)
    return {
        "failures_mean": float(out.failures.mean()),
        "failures_std": float(out.failures.std(ddof=1)) if len(out) > 1 else 0.0,
        "abstains_mean": float(out.abstains.mean()),
        "abstains_std": float(out.abstains.std(ddof=1)) if len(out) > 1 else 0.0,
        "cats_mean": float(out.cats.mean()),
        "cats_std": float(out.cats.std(ddof=1)) if len(out) > 1 else 0.0,
        "n_seeds": len(out),
        "k": k,
    }


def print_markdown_table(stats: dict[str, dict]) -> None:
    print()
    print("| Arm | Pure failures (mean ± std) | Abstains (mean ± std) | Catastrophic (mean ± std) | n seeds |")
    print("|-----|----------------------------|-----------------------|---------------------------|---------|")
    for arm, s in stats.items():
        print(
            f"| {arm} "
            f"| {s['failures_mean']:.1f} ± {s['failures_std']:.1f} "
            f"| {s['abstains_mean']:.1f} ± {s['abstains_std']:.1f} "
            f"| {s['cats_mean']:.1f} ± {s['cats_std']:.1f} "
            f"| {s['n_seeds']} |"
        )
    print()
    # share of bottom-50 that are pure failures
    for arm, s in stats.items():
        share = 100.0 * s["failures_mean"] / s["k"]
        print(f"  {arm}: pure-failure share of bottom-{s['k']} = {share:.1f}%")


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--rho", type=float, default=3.0)
    p.add_argument("--tau0", type=float, default=0.65)
    p.add_argument("--alpha", type=float, default=0.05)
    # offset from main runs
    p.add_argument("--seed_base", type=int, default=50_000)
    p.add_argument("--out", type=str, default="bottom_decile_out")
    p.add_argument("--from-csv", action="store_true",
                   help="Skip re-run; load episodes_*.csv from --out")
    p.add_argument("--k", type=int, default=50,
                   help="Worst-k episodes per seed (default 50 = bottom 10%% of T=500)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    arms = ["B3_static", "B4_eg_skip"]
    stats = {}

    for arm_name in arms:
        path = os.path.join(args.out, f"episodes_{arm_name}.csv")
        if args.from_csv and os.path.exists(path):
            print(f"Loading {path}")
            df = pd.read_csv(path)
        else:
            print(f"Collecting episodes for {arm_name} ...")
            df = collect_episodes(
                arm_name, args.T, args.seeds, args.rho,
                args.tau0, args.alpha, args.seed_base,
            )
            df.to_csv(path, index=False)
            print(f"  wrote {path}  ({len(df)} rows)")

        stats[arm_name] = bottom50_stats(df, k=args.k)

    print_markdown_table(stats)

    # machine-readable summary
    summary_path = os.path.join(args.out, "bottom50_summary.csv")
    rows = [{"arm": a, **s} for a, s in stats.items()]
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
