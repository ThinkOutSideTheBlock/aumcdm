"""Deterministic AU-MCDM decision rule.

    Utility      U(a) = (q(a) - kappa * sigma(a)) . w  -  lambda * c(a)
    Threshold    tau  = tau0 + eta * (sigma(a_top) . w)
    Rule         abstain iff  max_{a != Abstain} U(a) < tau

Free functions only. No controller object.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DecisionConfig:
    kappa: float = 1.0        # risk aversion on epistemic dispersion
    # explicit cost price (criterion 2 already prices it)
    lam: float = 0.0
    tau0: float = 0.55        # base reservation utility
    eta: float = 0.0          # uncertainty inflation of the threshold
    eps_force: float = 0.0    # forced-exploration rate inside the abstain region


# ---------------------------------------------------------------------------
def project_simplex(v):
    """Euclidean projection onto the probability simplex (Duchi et al., 2008)."""
    v = np.asarray(v, float)
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    idx = np.nonzero(u * np.arange(1, n + 1) > (css - 1.0))[0]
    rho = idx[-1] if idx.size else 0
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def risk_adjust(Q, S, kappa):
    return np.asarray(Q, float) - float(kappa) * np.asarray(S, float)


def utilities(Qt, cost, w, lam):
    return np.asarray(Qt, float) @ np.asarray(w, float) \
        - float(lam) * np.asarray(cost, float)


# ---------------------------------------------------------------------------
def decide(env, w, cfg, rng=None, allow_abstain=True, allow_info=True):
    """Return (action, aux).

    aux keys
    --------
    U           masked utility vector
    Qt          risk-adjusted score matrix
    qt          risk-adjusted feature row of the EXECUTED action
    u_exec      utility of the executed action
    u_top       max utility over non-abstain feasible actions
    a_top       argmax action over non-abstain feasible actions
    tau         realised reservation threshold
    censored    True iff the RULE said abstain (independent of forcing)
    forced      True iff forced exploration overrode the abstention
    propensity  behaviour-policy probability of the executed action
    """
    w = np.asarray(w, float)
    Q, S = env.score()
    cost = env.cost_vector()

    mask = env.feasible_mask().copy()
    if not allow_info:
        mask[env.K:env.ASK + 1] = False

    Qt = risk_adjust(Q, S, cfg.kappa)
    U = utilities(Qt, cost, w, cfg.lam)
    U = np.where(mask, U, -np.inf)

    non_abstain = mask.copy()
    non_abstain[env.ABSTAIN] = False

    if non_abstain.any():
        a_top = int(np.argmax(np.where(non_abstain, U, -np.inf)))
        u_top = float(U[a_top])
        tau = float(cfg.tau0 + cfg.eta * float(S[a_top] @ w))
        censored = bool(u_top < tau)
    else:
        a_top, u_top, tau, censored = env.ABSTAIN, - \
            np.inf, float(cfg.tau0), True

    if not allow_abstain and non_abstain.any():
        censored = False

    forced = False
    if censored and non_abstain.any() and cfg.eps_force > 0.0 and rng is not None:
        forced = bool(rng.random() < cfg.eps_force)

    if censored and not forced:
        a = env.ABSTAIN
        prop = (1.0 - cfg.eps_force) if non_abstain.any() else 1.0
    else:
        a = a_top
        prop = cfg.eps_force if censored else 1.0

    aux = {"U": U, "Qt": Qt, "qt": Qt[a].copy(),
           "u_exec": float(U[a]), "u_top": u_top, "a_top": int(a_top),
           "tau": tau, "censored": bool(censored), "forced": bool(forced),
           "propensity": float(prop)}
    return int(a), aux
