"""Day-1 driver.

    python runner.py --T 50 --seeds 3          # Gate-1 smoke run
    python runner.py --T 500 --seeds 20        # MVP result
    python runner.py --T 500 --seeds 20 --log_episodes --out results_logs_mis
"""
import argparse
import csv
import json
import os
from typing import Optional

import numpy as np

from aumcdm.envs.tool_select import ToolSelectEnv, make_w_star, M
from aumcdm.engine.decision import decide, risk_adjust
from aumcdm.engine.baselines import make_suite

DEFAULT_W0 = np.array([0.45, 0.40, 0.10, 0.05])
MAX_STEPS = ToolSelectEnv.BUDGET + 2
C_MAX = 0.30  # matches spend_eff definition in the paper


# --------------------------------------------------------------------------- #
# Episode logging (optional; never touches RNG)
# --------------------------------------------------------------------------- #
def _spend_eff(spend: float) -> float:
    return float(max(0.0, 1.0 - float(spend) / C_MAX))


def _write_episodes_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def _random_action(env, w, cfg, rng):
    mask = env.feasible_mask()
    idx = np.flatnonzero(mask)
    a = int(rng.choice(idx))
    Q, S = env.score()
    Qt = risk_adjust(Q, S, cfg.kappa)
    aux = {"qt": Qt[a].copy(), "u_exec": float(Qt[a] @ w), "u_top": np.nan,
           "a_top": a, "tau": float(cfg.tau0), "censored": False,
           "forced": False, "propensity": 1.0 / len(idx),
           "Qt": Qt}
    return a, aux


def _oracle_action(env, w, cfg):
    a, _ = env.oracle_action()
    Q, S = env.score()
    Qt = risk_adjust(Q, S, cfg.kappa)
    aux = {"qt": Qt[a].copy(), "u_exec": float(Qt[a] @ w), "u_top": np.nan,
           "a_top": a, "tau": float(cfg.tau0), "censored": False,
           "forced": False, "propensity": 1.0,
           "Qt": Qt}
    return int(a), aux


def select(env, arm, w, rng):
    if arm.policy == "random":
        return _random_action(env, w, arm.cfg, rng)
    if arm.policy == "oracle":
        return _oracle_action(env, w, arm.cfg)
    return decide(env, w, arm.cfg, rng=rng,
                  allow_abstain=arm.allow_abstain, allow_info=arm.allow_info)


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
def run_episode(env, arm, adapter, rng):
    """One episode. Returns metrics + per-decision censor/force counts.

    Extra keys for logging (no new randomness):
      kind, success (0/1 or None), tool (int or None)
    """
    env.reset()
    v_oracle = env.oracle_value()
    last_aux = None
    n_cens = 0
    n_forced = 0
    kind_hist = {"act": 0, "abstain": 0, "gather": 0, "ask": 0, "other": 0}

    for _ in range(MAX_STEPS):
        w = adapter.snapshot()
        a, aux = select(env, arm, w, rng)
        last_aux = aux

        if aux.get("censored", False):
            n_cens += 1
            if aux.get("forced", False):
                n_forced += 1

        terminal, phi, info = env.step(a)
        if terminal:
            kind = info["kind"]
            kname = kind if kind in kind_hist else "other"
            kind_hist[kname] += 1
            acted = (kind == "act")

            # success / tool for logging (read-only from env outcome)
            success = info.get("success", None)
            if success is None and acted:
                try:
                    success = int(float(phi[0]) >= 0.5)
                except Exception:
                    success = None
            if success is not None:
                success = int(success)
            tool = info.get("tool", None)
            if tool is None and acted and a < env.K:
                tool = int(a)

            # ---- B4_full: on abstain, reveal counterfactual of a_top and update ----
            if (getattr(arm, "full_feedback", False)
                    and kind == "abstain"
                    and aux.get("a_top") is not None
                    and aux["a_top"] < env.K):
                k = int(aux["a_top"])
                success_cf = bool(rng.random() < env.ep.theta[k])
                phi_cf = env.phi_terminal(k, success_cf)
                R_cf = env.reward(phi_cf)
                qt_cf = aux["Qt"][k] if "Qt" in aux else aux["qt"]
                u_cf = float(aux["u_top"])
                adapter.update(qt=qt_cf, u_exec=u_cf, R=R_cf,
                               propensity=1.0, terminal_acted=True)
                R = env.reward(phi)
            else:
                R = env.reward(phi)
                adapter.update(qt=aux["qt"], u_exec=aux["u_exec"], R=R,
                               propensity=aux["propensity"],
                               terminal_acted=acted)

            return {
                "R": R,
                "regret": float(v_oracle - R),
                "abstained": kind == "abstain",
                "harm": float(info.get("harm", 0.0) or 0.0),
                "catastrophic": bool(info.get("catastrophic", False)),
                "censored": bool(aux["censored"]),
                "forced": bool(aux["forced"]),
                "n_cens": n_cens,
                "n_forced": n_forced,
                "spend": float(env.ep.spend),
                "steps": env.ep.steps,
                "kind_hist": kind_hist,
                "kind": kind,
                "success": success if acted else None,
                "tool": tool if acted else None,
                "u_top": aux.get("u_top", None),
                "a_top": aux.get("a_top", None),
            }

    # step cap: reservation outcome, no learning signal
    kind_hist["abstain"] += 1
    phi = env.phi_terminal(env.ABSTAIN)
    R = env.reward(phi)
    return {
        "R": R, "regret": float(v_oracle - R), "abstained": True,
        "harm": 0.0, "catastrophic": False,
        "censored": bool(last_aux["censored"]) if last_aux else True,
        "forced": False, "n_cens": n_cens, "n_forced": n_forced,
        "spend": float(env.ep.spend), "steps": env.ep.steps,
        "kind_hist": kind_hist,
        "kind": "abstain",
        "success": None,
        "tool": None,
        "u_top": last_aux.get("u_top") if last_aux else None,
        "a_top": last_aux.get("a_top") if last_aux else None,
    }


# --------------------------------------------------------------------------- #
# One arm x one seed
# --------------------------------------------------------------------------- #
def cvar(xs, q=0.10):
    xs = np.sort(np.asarray(xs, float))
    n = max(1, int(np.ceil(q * len(xs))))
    return float(xs[:n].mean())


def slope(y):
    y = np.asarray(y, float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float((x @ (y - y.mean())) / (x @ x))


def run_arm(arm, T, seed, w_star, seed_base=10_000, log_every=10,
            log_episodes: bool = False, seed_set: str = "A", init: str = "misspec"):
    rng = np.random.default_rng(seed_base + seed)
    env = ToolSelectEnv(w_star, rng)
    adapter = arm.make_adapter()

    eps_hist, w_hist = [], []
    episode_rows: list[dict] = []

    for t in range(T):
        w_before = adapter.snapshot().copy()
        n_before = int(getattr(adapter, "n_updates", 0))

        e = run_episode(env, arm, adapter, rng)
        eps_hist.append(e)

        w_after = adapter.snapshot().copy()
        n_this = int(getattr(adapter, "n_updates", 0)) - n_before

        if log_episodes:
            kind = e.get("kind", "other")
            if kind == "act" and e.get("tool") is not None:
                term_str = f"act_{int(e['tool'])}"
            elif kind in ("abstain", "gather", "ask"):
                term_str = kind
            else:
                term_str = str(kind)

            success = e.get("success", None)
            harm = e.get("harm", None)
            if term_str == "abstain" or not term_str.startswith("act_"):
                success = None
            cat = int(bool(e.get("catastrophic", False)))

            spend = float(e.get("spend", 0.0))
            u_top = e.get("u_top", None)
            try:
                u_top_f = float(u_top) if u_top is not None and np.isfinite(
                    u_top) else ""
            except Exception:
                u_top_f = ""

            episode_rows.append({
                "seed_set": seed_set,
                "seed_id": int(seed),
                "init": init,
                "arm": arm.name,
                "episode": int(t),
                "terminal_action": term_str,
                "success": "" if success is None else int(success),
                "harm": "" if harm is None else float(harm),
                "catastrophic": cat,
                "reward": float(e["R"]),
                "spend": spend,
                "spend_eff": _spend_eff(spend),
                "abstain_flag": int(term_str == "abstain"),
                "n_updates": int(n_this),
                "w2_before": float(w_before[1]),
                "w2_after": float(w_after[1]),
                "u_max": u_top_f,
                "a_top": "" if e.get("a_top") is None else int(e["a_top"]),
            })

        if t % log_every == 0:
            w_hist.append(adapter.snapshot())
    w_hist.append(adapter.snapshot())

    R = [e["R"] for e in eps_hist]
    half = max(1, T // 2)
    w_hist = np.array(w_hist)
    w0n = arm.w0 / arm.w0.sum()

    kh = {"act": 0, "abstain": 0, "gather": 0, "ask": 0, "other": 0}
    for e in eps_hist:
        for k, v in e.get("kind_hist", {}).items():
            kh[k] = kh.get(k, 0) + int(v)

    out = {
        "arm": arm.name, "seed": seed,
        "mean_R": float(np.mean(R)),
        "cvar10": cvar(R, 0.10),
        "mean_regret": float(np.mean([e["regret"] for e in eps_hist])),
        "abstain_rate": float(np.mean([e["abstained"] for e in eps_hist])),
        "abstain_first": float(np.mean([e["abstained"] for e in eps_hist[:half]])),
        "abstain_last": float(np.mean([e["abstained"] for e in eps_hist[half:]])),
        "catastrophic_rate": float(np.mean([e["catastrophic"] for e in eps_hist])),
        "cat_first": float(np.mean([e["catastrophic"] for e in eps_hist[:half]])),
        "cat_last": float(np.mean([e["catastrophic"] for e in eps_hist[half:]])),
        "harm_rate": float(np.mean([e["harm"] > 0 for e in eps_hist])),
        "mean_spend": float(np.mean([e["spend"] for e in eps_hist])),
        "forced_rate": float(np.mean([e["forced"] for e in eps_hist])),
        "cens_decisions": float(np.mean([e["n_cens"] for e in eps_hist])),
        "forced_decisions": float(np.mean([e["n_forced"] for e in eps_hist])),
        "force_given_cens": float(
            np.sum([e["n_forced"] for e in eps_hist])
            / max(1, np.sum([e["n_cens"] for e in eps_hist]))
        ),
        "w2_init": float(w0n[1]),
        "w2_final": float(w_hist[-1, 1]),
        "w2_drift": float(w_hist[-1, 1] - w0n[1]),
        "w2_slope": slope(w_hist[:, 1]),
        "n_updates": int(getattr(adapter, "n_updates", 0)),
        "w1_final": float(w_hist[-1, 0]),
        "w3_final": float(w_hist[-1, 2]),
        "w4_final": float(w_hist[-1, 3]),
        "n_term_act": int(kh["act"]),
        "n_term_abstain": int(kh["abstain"]),
        "n_term_gather": int(kh["gather"]),
        "n_term_ask": int(kh["ask"]),
        "n_term_other": int(kh["other"]),
        "w_final": w_hist[-1].copy(),
        "w_hist": w_hist,
    }
    if log_episodes:
        out["_episode_rows"] = episode_rows
    return out


# --------------------------------------------------------------------------- #
# Expected drift (fixed)
# --------------------------------------------------------------------------- #
def expected_drift(w0: np.ndarray, phi_means: np.ndarray) -> np.ndarray:
    """Closed-form expected relative drift under multiplicative EG with fixed phi_means."""
    w = w0.copy()
    for _ in range(100):  # converges in <100 steps
        phi = phi_means
        denom = np.dot(w, np.exp(phi))
        w = w * np.exp(phi) / denom
    return w - w0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(rows, args, w_star, w0):
    names = list(dict.fromkeys(r["arm"] for r in rows))
    by = {n: [r for r in rows if r["arm"] == n] for n in names}
    def g(n, k): return float(np.mean([r[k] for r in by[n]]))

    print()
    print("=" * 104)
    print(f"  AU-MCDM Gate-1   T={args.T}  seeds={args.seeds}  rho={args.rho}  "
          f"alpha={args.alpha}  eps={args.eps}  tau0={args.tau0}")
    print(f"  w* = {np.round(w_star, 3)}      w0 = {np.round(w0, 3)}")
    print("=" * 104)
    hdr = (f"{'arm':<14}{'meanR':>8}{'CVaR10':>8}{'regret':>8}{'cat%':>7}"
           f"{'cat1st':>8}{'cat2nd':>8}{'abs1st':>8}{'abs2nd':>8}"
           f"{'w2_end':>8}{'w2_drift':>9}{'spend':>7}")
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        print(f"{n:<14}{g(n, 'mean_R'):8.3f}{g(n, 'cvar10'):8.3f}"
              f"{g(n, 'mean_regret'):8.3f}{100*g(n, 'catastrophic_rate'):7.2f}"
              f"{100*g(n, 'cat_first'):8.2f}{100*g(n, 'cat_last'):8.2f}"
              f"{100*g(n, 'abstain_first'):8.2f}{100*g(n, 'abstain_last'):8.2f}"
              f"{g(n, 'w2_final'):8.3f}{g(n, 'w2_drift'):+9.3f}"
              f"{g(n, 'mean_spend'):7.3f}")
    print("-" * len(hdr))

    print("\nGATE-1 SANITY")
    ab3 = g("B3_static", "abstain_rate")
    checks = [
        ("abstain region non-degenerate (B3)",
         0.02 < ab3 < 0.98, f"{ab3:.3f}"),
        ("oracle beats random",
         g("ORACLE", "mean_R") > g("B0_random", "mean_R"),
         f"{g('ORACLE', 'mean_R'):.3f} > {g('B0_random', 'mean_R'):.3f}"),
        ("greedy never abstains",
         g("B1_greedy", "abstain_rate") == 0.0,
         f"{g('B1_greedy', 'abstain_rate'):.3f}"),
        ("greedy is more catastrophic than static AU-MCDM",
         g("B1_greedy", "catastrophic_rate") > g(
             "B3_static", "catastrophic_rate"),
         f"{100*g('B1_greedy', 'catastrophic_rate'):.2f}% vs "
         f"{100*g('B3_static', 'catastrophic_rate'):.2f}%"),
        ("static arms did not drift",
         abs(g("B3_static", "w2_drift")) < 1e-12, f"{g('B3_static', 'w2_drift'):.2e}"),
        ("B5 actually forced exploration",
         g("B5_eg_ips", "forced_rate") > 0.0, f"{g('B5_eg_ips', 'forced_rate'):.4f}"),
        ("B5 force|cens ≈ eps (instrument)",
         abs(g("B5_eg_ips", "force_given_cens") - args.eps) < 0.05,
         f"{g('B5_eg_ips', 'force_given_cens'):.4f} vs eps={args.eps}"),
    ]
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<48} {detail}")

    print("\nDIRECTIONAL READOUT  (predictions, not assertions)")
    for pid, label, vals in [
        ("P1", "B4 safety-weight drift < 0",
         [r["w2_drift"] for r in by["B4_eg_skip"]]),
        ("P2", "B4 catastrophic rate rises (last - first)",
         [r["cat_last"] - r["cat_first"] for r in by["B4_eg_skip"]]),
        ("P3", "B4 - B3 tail risk (CVaR10 diff, expect < 0)",
         [a["cvar10"] - b["cvar10"]
          for a, b in zip(by["B4_eg_skip"], by["B3_static"])]),
        ("P4", "B5 - B4 safety-weight drift (expect > 0)",
         [a["w2_drift"] - b["w2_drift"]
          for a, b in zip(by["B5_eg_ips"], by["B4_eg_skip"])]),
    ]:
        m = float(np.mean(vals))
        lo, hi = boot_ci(vals)
        print(f"  {pid}  {label:<46} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    if args.seeds < 10:
        print("\n  NOTE: seeds < 10. CIs are indicative only; do not read as evidence.")
    print()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--rho", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--tau0", type=float, default=0.55)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--aligned", action="store_true",
                   help="Set w0 = w* (aligned-start control)")
    p.add_argument("--seed_base", type=int, default=10_000,
                   help="RNG offset; use 20000 for held-out replication")
    p.add_argument("--log_episodes", action="store_true",
                   help="Write episode-level CSVs (no effect on RNG)")
    p.add_argument("--seed_set", type=str, default=None,
                   help="Label for logs: A or B (default: A if seed_base<20000 else B)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    w_star = make_w_star(args.rho)
    w0 = w_star.copy() if args.aligned else DEFAULT_W0.copy()
    init = "aligned" if args.aligned else "misspec"
    seed_set = args.seed_set or ("B" if args.seed_base >= 20_000 else "A")

    rows, traj = [], []
    episode_bags: dict[str, list] = {}

    for seed in range(args.seeds):
        for arm in make_suite(w0, alpha=args.alpha, eps=args.eps,
                              tau0=args.tau0, kappa=args.kappa, eta=args.eta):
            r = run_arm(arm, T=args.T, seed=seed, w_star=w_star,
                        seed_base=args.seed_base,
                        log_episodes=args.log_episodes,
                        seed_set=seed_set, init=init)
            if args.log_episodes and "_episode_rows" in r:
                episode_bags.setdefault(arm.name, []).extend(
                    r.pop("_episode_rows"))
            for i, w in enumerate(r["w_hist"]):
                traj.append({"seed": seed, "arm": arm.name, "t": i * 10,
                             "w1": w[0], "w2": w[1], "w3": w[2], "w4": w[3]})
            rows.append(r)

    scalar = [k for k, v in rows[0].items() if not isinstance(v, np.ndarray)]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=scalar)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in scalar})
    with open(os.path.join(args.out, "w_traj.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        wr.writeheader()
        wr.writerows(traj)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args) | {"w_star": w_star.tolist(), "w0": w0.tolist(),
                                "init": init, "seed_set": seed_set},
                  f, indent=2)

    if args.log_episodes:
        for arm_name, erows in episode_bags.items():
            path = os.path.join(
                args.out, f"episodes_{seed_set}_{init}_{arm_name}.csv")
            _write_episodes_csv(path, erows)
            print(f"wrote {path}  ({len(erows)} rows)")

    report(rows, args, w_star, w0)
    print(f"wrote {args.out}/summary.csv, {args.out}/w_traj.csv, "
          f"{args.out}/config.json")


if __name__ == "__main__":
    main()
