"""Arm registry. Minimal Day-1 scope: B0, B1, B3, B4, B5, ORACLE."""
from dataclasses import dataclass, field
import numpy as np

from .decision import DecisionConfig
from .adapt import build_adapter


@dataclass
class Arm:
    name: str
    w0: np.ndarray
    cfg: DecisionConfig
    adapter_kind: str = "static"     # static | eg | ips
    policy: str = "mcdm"             # mcdm | random | oracle
    allow_abstain: bool = True
    allow_info: bool = True
    alpha: float = 0.05
    eps: float = 0.10
    full_feedback: bool = False      # B4_full: counterfactual update on abstain

    def make_adapter(self):
        return build_adapter(self.adapter_kind, self.w0,
                             alpha=self.alpha, eps=self.eps)


def make_suite(w0, alpha=0.05, eps=0.10, tau0=0.55, kappa=1.0, eta=0.0,
               lam=0.0, scale_invariant: bool = False):
    """The locked Day-1 arm family.

    scale_invariant: if True, DecisionConfig scales tau by ||w||_2 at decision time.
    Default False preserves all existing counted runs bit-for-bit.
    """
    w0 = np.asarray(w0, float)
    base = dict(kappa=kappa, lam=lam, tau0=tau0, eta=eta,
                scale_invariant=bool(scale_invariant))

    return [
        # B0: uniform random over feasible actions. Floor.
        Arm("B0_random", w0, DecisionConfig(**base),
            "static", policy="random"),

        # B1: greedy expected task value. No risk term, no abstention, no info.
        Arm("B1_greedy", np.array([1.0, 0.0, 0.0, 0.0]),
            DecisionConfig(kappa=0.0, lam=lam, tau0=-1e9, eta=0.0,
                           scale_invariant=False),
            "static", allow_abstain=False, allow_info=False),

        # B3: static AU-MCDM. Full rule, no learning. Key reference arm.
        Arm("B3_static", w0, DecisionConfig(**base), "static"),

        # B4: adaptive AU-MCDM, skip-on-abstain.
        Arm("B4_eg_skip", w0, DecisionConfig(**base), "eg", alpha=alpha),

        # B4_full: same rule + adapter, counterfactual outcome on abstain.
        Arm("B4_full", w0, DecisionConfig(**base), "eg", alpha=alpha,
            full_feedback=True),

        # B5: forced exploration + clipped IPS.
        Arm("B5_eg_ips", w0, DecisionConfig(eps_force=eps, **base),
            "ips", alpha=alpha, eps=eps),

        # ORACLE: true theta, true w*. Normaliser only.
        Arm("ORACLE", w0, DecisionConfig(**base), "static", policy="oracle"),
    ]


def arm_names(w0=None):
    w0 = np.ones(4) / 4 if w0 is None else w0
    return [a.name for a in make_suite(w0)]
