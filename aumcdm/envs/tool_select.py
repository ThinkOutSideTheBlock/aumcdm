"""E1: tool selection under harm asymmetry.

Actions
-------
    0 .. K-1        Act(k)      terminal, executes tool k
    K .. 2K-1       Gather(k)   one Bernoulli observation of tool k
    2K              Ask         near-perfect information on the leading tool
    2K+1            Abstain     terminal, reservation outcome

Criteria (M = 4), all oriented "higher is better", all in [0, 1]
    0  task value        expected success
    1  safety            1 - harm_k * P(failure)
    2  cost efficiency   1 - spend / C_MAX
    3  epistemic gain    normalised expected variance reduction
"""
import numpy as np

# -----------------------------------------------------------------------------
# Locked constants
# -----------------------------------------------------------------------------
K = 6                       # tools
M = 4                       # criteria
BUDGET = 6                  # max non-terminal steps
N_INIT = 3                  # initial observations per available tool
P_AVAIL = 0.5               # per-tool availability probability

PRIORS = np.array([[18., 2.],       # tool 0: reliable, low harm
                   [14., 6.],       # tool 1: good, low harm
                   [8., 8.],        # tool 2: uncertain, moderate harm
                   [3., 3.],        # tool 3: uncertain, high harm
                   [1., 1.],        # tool 4: unknown, high harm
                   [2., 6.]])       # tool 5: poor, extreme harm

HARM = np.array([0.10, 0.15, 0.40, 0.90, 0.60, 0.95])

C_G = 0.02                  # cost of one Gather
C_A = 0.15                  # cost of one Ask
C_MAX = 0.30                # spend cap (normaliser for criterion 2)

CATASTROPHIC = 0.50         # harm >= this counts as a catastrophic outcome


# -----------------------------------------------------------------------------
# Preference vectors
# -----------------------------------------------------------------------------
def make_w_star(rho, w3=0.05, w4=0.0):
    """True criterion weights with harm asymmetry rho = w2 / w1.

    Returns a length-M vector on the simplex with w[1] / w[0] == rho exactly.
    """
    rho = float(rho)
    rest = 1.0 - float(w3) - float(w4)
    if rest <= 0.0:
        raise ValueError("w3 + w4 must be < 1")
    w1 = rest / (1.0 + rho)
    return np.array([w1, rho * w1, float(w3), float(w4)])


# Alias kept because both names appear in the design notes.
w_star_from_rho = make_w_star


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
class Episode:
    __slots__ = ("theta", "avail", "ab", "spend", "steps", "asks")

    def __init__(self, theta, avail, ab):
        self.theta = theta
        self.avail = avail
        self.ab = ab
        self.spend = 0.0
        self.steps = 0
        self.asks = 0


class ToolSelectEnv:
    K = K
    M = M
    N_ACTIONS = 2 * K + 2
    ASK = 2 * K
    ABSTAIN = 2 * K + 1
    BUDGET = BUDGET
    C_MAX = C_MAX

    def __init__(self, w_star, rng):
        self.w_star = np.asarray(w_star, float)
        if self.w_star.shape != (M,):
            raise ValueError("w_star must have length %d" % M)
        self.rng = rng
        self.ep = None
        self._kstar = 0

    # ------------------------------------------------------------------ reset
    def reset(self):
        theta = self.rng.beta(PRIORS[:, 0], PRIORS[:, 1])

        avail = self.rng.random(K) < P_AVAIL
        if not avail.any():
            avail[int(self.rng.integers(K))] = True

        ab = PRIORS.copy()
        for k in range(K):
            if avail[k]:
                s = int(self.rng.binomial(N_INIT, theta[k]))
                ab[k, 0] += s
                ab[k, 1] += N_INIT - s

        self.ep = Episode(theta=theta, avail=avail, ab=ab)
        self._kstar = 0
        return self.ep

    # -------------------------------------------------------------- beliefs
    def mu_sigma(self):
        a, b = self.ep.ab[:, 0], self.ep.ab[:, 1]
        n = a + b
        mu = a / n
        sd = np.sqrt(mu * (1.0 - mu) / (n + 1.0))
        return mu, sd

    # --------------------------------------------------------------- scores
    def score(self):
        """Return (Q, S) with shape (N_ACTIONS, M), Q in [0,1], S >= 0."""
        mu, sd = self.mu_sigma()
        n = self.ep.ab.sum(axis=1)
        sp = self.ep.spend

        Q = np.zeros((self.N_ACTIONS, M))
        S = np.zeros((self.N_ACTIONS, M))

        def eff(extra): return max(0.0, 1.0 - (sp + extra) / C_MAX)

        # --- Act(k): terminal ------------------------------------------------
        Q[:K, 0] = mu
        Q[:K, 1] = 1.0 - HARM * (1.0 - mu)
        Q[:K, 2] = eff(0.0)
        Q[:K, 3] = 0.0
        S[:K, 0] = sd
        S[:K, 1] = HARM * sd

        # --- Gather(k): one observation --------------------------------------
        var_now = mu * (1.0 - mu) / (n + 1.0)
        var_next = mu * (1.0 - mu) / (n + 2.0)
        Q[K:2 * K, 0] = 0.0
        Q[K:2 * K, 1] = 1.0
        Q[K:2 * K, 2] = eff(C_G)
        Q[K:2 * K, 3] = np.clip((var_now - var_next) / 0.25, 0.0, 1.0)

        # --- Ask: near-perfect information on the leading available tool -----
        score_rank = np.where(self.ep.avail, mu - HARM * (1.0 - mu), -np.inf)
        kstar = int(np.argmax(score_rank))
        self._kstar = kstar
        Q[self.ASK, 0] = 0.0
        Q[self.ASK, 1] = 1.0
        Q[self.ASK, 2] = eff(C_A)
        Q[self.ASK, 3] = float(np.clip(var_now[kstar] / 0.25, 0.0, 1.0))

        # --- Abstain: the reservation outcome --------------------------------
        Q[self.ABSTAIN, 0] = 0.0
        Q[self.ABSTAIN, 1] = 1.0
        Q[self.ABSTAIN, 2] = eff(0.0)
        Q[self.ABSTAIN, 3] = 0.0

        return np.clip(Q, 0.0, 1.0), np.clip(S, 0.0, 1.0)

    # ---------------------------------------------------------------- costs
    def cost_vector(self):
        c = np.zeros(self.N_ACTIONS)
        c[K:2 * K] = C_G
        c[self.ASK] = C_A
        return c

    # ---------------------------------------------------------- feasibility
    def feasible_mask(self):
        m = np.zeros(self.N_ACTIONS, dtype=bool)
        ep = self.ep
        m[:K] = ep.avail
        room_g = (ep.steps < BUDGET) and (ep.spend + C_G <= C_MAX + 1e-12)
        room_a = (ep.steps < BUDGET) and (ep.spend + C_A <= C_MAX + 1e-12)
        m[K:2 * K] = ep.avail & room_g
        m[self.ASK] = room_a and bool(ep.avail.any())
        m[self.ABSTAIN] = True
        return m

    def is_terminal_action(self, a):
        a = int(a)
        return (a < K) or (a == self.ABSTAIN)

    # -------------------------------------------------- realised criteria
    def phi_terminal(self, a, success=False):
        sp_eff = max(0.0, 1.0 - self.ep.spend / C_MAX)
        a = int(a)
        if a == self.ABSTAIN:
            return np.array([0.0, 1.0, sp_eff, 0.0])
        s = float(bool(success))
        return np.array([s, 1.0 - HARM[a] * (1.0 - s), sp_eff, 0.0])

    # ----------------------------------------------------------- transition
    def step(self, a):
        """Return (terminal, phi, info). phi is None on non-terminal steps."""
        a = int(a)
        ep = self.ep
        ep.steps += 1

        if a < K:                                           # Act(k)
            success = bool(self.rng.random() < ep.theta[a])
            phi = self.phi_terminal(a, success)
            harm = float(HARM[a] * (1.0 - success))
            return True, phi, {"kind": "act", "k": a, "success": success,
                               "harm": harm,
                               "catastrophic": bool(harm >= CATASTROPHIC)}

        if a < 2 * K:                                       # Gather(k)
            k = a - K
            y = bool(self.rng.random() < ep.theta[k])
            ep.ab[k, 0 if y else 1] += 1.0
            ep.spend += C_G
            return False, None, {"kind": "gather", "k": k, "obs": y}

        if a == self.ASK:                                   # Ask
            k = self._kstar
            ep.ab[k] = np.array([1.0 + 1e4 * ep.theta[k],
                                 1.0 + 1e4 * (1.0 - ep.theta[k])])
            ep.spend += C_A
            ep.asks += 1
            return False, None, {"kind": "ask", "k": k}

        phi = self.phi_terminal(self.ABSTAIN)               # Abstain
        return True, phi, {"kind": "abstain", "k": -1, "success": False,
                           "harm": 0.0, "catastrophic": False}

    # --------------------------------------------------------------- reward
    def reward(self, phi):
        return float(self.w_star @ np.asarray(phi, float))

    # --------------------------------------------------------------- oracle
    def oracle_action(self):
        """Best terminal action under TRUE theta and TRUE w*. Never available
        to any learning arm; used only as a normaliser."""
        th = self.ep.theta
        sp_eff = max(0.0, 1.0 - self.ep.spend / C_MAX)
        val = (self.w_star[0] * th
               + self.w_star[1] * (1.0 - HARM * (1.0 - th))
               + self.w_star[2] * sp_eff)
        val = np.where(self.ep.avail, val, -np.inf)
        k = int(np.argmax(val))
        v_abstain = self.w_star[1] * 1.0 + self.w_star[2] * sp_eff
        return (k, float(val[k])) if val[k] >= v_abstain else (self.ABSTAIN,
                                                               float(v_abstain))

    def oracle_value(self):
        return self.oracle_action()[1]
