"""Outcome-driven weight adaptation under censored feedback.

Every adapter exposes the same call:

    w = adapter.update(qt=..., u_exec=..., R=..., propensity=...,
                       terminal_acted=...)

`terminal_acted` is False whenever no criterion outcome was observed
(abstention for skip, or a non-terminal step). That single flag is the
censoring channel.

B4_full passes terminal_acted=True on the counterfactual abstain branch
so the weight update counts; beliefs are still not updated in the env.
"""
import numpy as np


class StaticAdapter:
    """No adaptation. Reference arm for 'is adaptation harmful?'"""
    name = "static"

    def __init__(self, w0, **kw):
        w0 = np.asarray(w0, float).copy()
        self.w = w0 / w0.sum()
        self.n_updates = 0

    def update(self, **kw):
        return self.w

    def snapshot(self):
        return self.w.copy()


class EGAdapter:
    """Exponentiated gradient, skip-on-abstain. The censored learner (B4)."""
    name = "eg_skip"

    def __init__(self, w0, alpha=0.05, clip=5.0, w_floor=1e-8, **kw):
        w0 = np.asarray(w0, float).copy()
        self.w = w0 / w0.sum()
        self.alpha = float(alpha)
        self.clip = float(clip)
        self.w_floor = float(w_floor)
        self.n_updates = 0

    def gradient(self, qt, u_exec, R):
        adv = float(R) - float(u_exec)
        return np.clip(adv * np.asarray(qt, float), -self.clip, self.clip)

    def _apply(self, g):
        z = self.alpha * np.asarray(g, float)
        z = z - z.max()
        self.w = self.w * np.exp(z)
        self.w = np.maximum(self.w, self.w_floor)
        self.w = self.w / self.w.sum()
        self.n_updates += 1
        return self.w

    def update(self, qt=None, u_exec=0.0, R=0.0, propensity=1.0,
               terminal_acted=False, **kw):
        if not terminal_acted:
            return self.w
        return self._apply(self.gradient(qt, u_exec, R))

    def snapshot(self):
        return self.w.copy()


class IPSEGAdapter(EGAdapter):
    """EG on clipped-IPS gradients under forced exploration (B5)."""
    name = "eg_ips"

    def __init__(self, w0, alpha=0.05, clip=5.0, eps=0.10, **kw):
        super().__init__(w0, alpha=alpha, clip=clip)
        self.eps = float(eps)
        self.Mc = 1.0 / max(self.eps, 1e-9)
        # n_updates already 0 from super

    def update(self, qt=None, u_exec=0.0, R=0.0, propensity=1.0,
               terminal_acted=False, **kw):
        if not terminal_acted:
            return self.w
        iw = min(1.0 / max(float(propensity), 1e-9), self.Mc)
        return self._apply(iw * self.gradient(qt, u_exec, R))


ADAPTERS = {"static": StaticAdapter, "eg": EGAdapter, "ips": IPSEGAdapter}


def build_adapter(kind, w0, alpha=0.05, eps=0.10):
    if kind not in ADAPTERS:
        raise ValueError("unknown adapter kind: %r" % (kind,))
    return ADAPTERS[kind](w0, alpha=alpha, eps=eps)
