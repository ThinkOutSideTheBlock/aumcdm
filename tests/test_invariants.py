"""Structural invariants. These must hold for the measured effect to mean
anything; none of them assert the paper's predictions (P1-P5), which are
falsifiable claims measured by runner.py, not properties of the code."""
import numpy as np
import pytest

from aumcdm.envs.tool_select import (ToolSelectEnv, make_w_star, w_star_from_rho,
                                     PRIORS, HARM, K, M, BUDGET, C_G, C_A, C_MAX)
from aumcdm.engine.decision import (DecisionConfig, decide, project_simplex,
                                    risk_adjust, utilities)
from aumcdm.engine.adapt import (StaticAdapter, EGAdapter, IPSEGAdapter,
                                 build_adapter)
from aumcdm.engine.baselines import make_suite, arm_names

W_STAR = make_w_star(3.0)
W0 = np.array([0.45, 0.40, 0.10, 0.05])


def fresh(seed=0):
    env = ToolSelectEnv(W_STAR, np.random.default_rng(seed))
    env.reset()
    return env


# =========================================================================== #
# Preference vectors
# =========================================================================== #
@pytest.mark.parametrize("rho", [1.0, 2.0, 3.0, 8.0, 20.0])
def test_w_star_on_simplex_with_exact_ratio(rho):
    w = make_w_star(rho)
    assert w.shape == (M,)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert w[1] / w[0] == pytest.approx(rho, rel=1e-12)


def test_alias_is_the_same_function():
    assert w_star_from_rho is make_w_star
    assert np.allclose(w_star_from_rho(4.0), make_w_star(4.0))


# =========================================================================== #
# Environment structure
# =========================================================================== #
def test_score_shapes_and_bounds():
    for s in range(10):
        Q, S = fresh(s).score()
        assert Q.shape == (2 * K + 2, M) and S.shape == (2 * K + 2, M)
        assert np.all(Q >= 0.0) and np.all(Q <= 1.0)
        assert np.all(S >= 0.0) and np.all(S <= 1.0)
        assert np.all(np.isfinite(Q)) and np.all(np.isfinite(S))


def test_cost_criterion_has_zero_dispersion():
    """Spend is observed, not inferred: sigma on criterion 2 must vanish."""
    for s in range(5):
        _, S = fresh(s).score()
        assert np.allclose(S[:, 2], 0.0)


def test_abstain_row_is_the_reservation_outcome():
    env = fresh(0)
    Q, S = env.score()
    i = env.ABSTAIN
    assert Q[i, 0] == 0.0 and Q[i, 3] == 0.0        # no value, no information
    assert Q[i, 1] == 1.0                           # perfectly safe
    assert Q[i, 2] == pytest.approx(1.0)            # nothing spent yet
    assert np.allclose(S[i], 0.0)                   # and known with certainty


def test_abstain_is_always_feasible():
    env = fresh(1)
    for _ in range(BUDGET + 3):
        assert env.feasible_mask()[env.ABSTAIN]
        env.ep.steps += 1
        env.ep.spend = C_MAX


def test_at_least_one_tool_available():
    for s in range(50):
        assert fresh(s).ep.avail.any()


def test_feasible_acts_are_exactly_available_tools():
    for s in range(20):
        env = fresh(s)
        m = env.feasible_mask()
        assert np.array_equal(m[:K], env.ep.avail)
        # budget unspent at t=0
        assert np.array_equal(m[K:2 * K], env.ep.avail)


def test_episodes_are_heterogeneous():
    """If every episode looked identical the abstain rate would be 0 or 1 and
    the whole censoring story would be vacuous."""
    tops = []
    for s in range(40):
        env = fresh(s)
        Q, S = env.score()
        U = (Q - S) @ W0
        U = np.where(env.feasible_mask(), U, -np.inf)
        U[env.ABSTAIN] = -np.inf
        tops.append(float(U.max()))
    assert np.std(tops) > 0.02, "episodes too homogeneous to induce censoring"


def test_safety_score_decreasing_in_harm_at_equal_beliefs():
    env = fresh(0)
    env.ep.ab[:] = np.array([5.0, 5.0])            # equalise all beliefs
    Q, _ = env.score()
    order = np.argsort(HARM)
    assert np.all(np.diff(Q[:K, 1][order]) <= 1e-12)


def test_gather_adds_exactly_one_observation():
    env = fresh(3)
    k = int(np.flatnonzero(env.ep.avail)[0])
    n0 = env.ep.ab.sum(axis=1).copy()
    terminal, phi, info = env.step(K + k)
    n1 = env.ep.ab.sum(axis=1)
    assert not terminal and phi is None and info["kind"] == "gather"
    assert n1[k] == pytest.approx(n0[k] + 1.0)
    assert np.allclose(np.delete(n1, k), np.delete(n0, k))


def test_epistemic_gain_is_nonnegative_and_active():
    env = fresh(4)
    Q, _ = env.score()
    gain = Q[K:2 * K, 3]
    assert np.all(gain >= 0.0)
    assert gain.max() > 0.0


def test_ask_collapses_uncertainty_on_its_target():
    env = fresh(5)
    env.score()                                     # sets _kstar
    k = env._kstar
    _, sd0 = env.mu_sigma()
    env.step(env.ASK)
    _, sd1 = env.mu_sigma()
    assert sd1[k] < 0.01
    assert sd1[k] < sd0[k]


def test_ask_is_more_expensive_than_gather():
    env = fresh(0)
    c = env.cost_vector()
    assert c[env.ASK] > c[K] > 0.0
    assert c[:K].sum() == 0.0 and c[env.ABSTAIN] == 0.0


def test_budget_exhaustion_blocks_information_actions():
    env = fresh(6)
    for _ in range(BUDGET):
        k = int(np.flatnonzero(env.ep.avail)[0])
        env.step(K + k)
    m = env.feasible_mask()
    assert not m[K:2 * K].any()
    assert not m[env.ASK]
    assert np.array_equal(m[:K], env.ep.avail)
    assert m[env.ABSTAIN]


def test_spend_is_monotone_and_bounded():
    env = fresh(7)
    k = int(np.flatnonzero(env.ep.avail)[0])
    prev = 0.0
    for _ in range(BUDGET):
        env.step(K + k)
        assert env.ep.spend > prev
        prev = env.ep.spend
    assert env.ep.spend <= C_MAX + 1e-9


def test_cost_efficiency_criterion_falls_with_spend():
    env = fresh(8)
    Q0, _ = env.score()
    k = int(np.flatnonzero(env.ep.avail)[0])
    env.step(K + k)
    Q1, _ = env.score()
    assert Q1[0, 2] < Q0[0, 2]


def test_terminal_flags():
    env = fresh(9)
    assert env.is_terminal_action(0)
    assert env.is_terminal_action(env.ABSTAIN)
    assert not env.is_terminal_action(K)
    assert not env.is_terminal_action(env.ASK)


def test_abstain_yields_zero_harm_and_positive_reward():
    env = fresh(10)
    terminal, phi, info = env.step(env.ABSTAIN)
    assert terminal and info["kind"] == "abstain"
    assert info["harm"] == 0.0 and not info["catastrophic"]
    assert env.reward(phi) > 0.0                    # safety credit is real


def test_act_reward_matches_realised_criteria():
    env = fresh(11)
    k = int(np.flatnonzero(env.ep.avail)[0])
    terminal, phi, info = env.step(k)
    assert terminal and info["kind"] == "act"
    assert phi == pytest.approx(env.phi_terminal(k, info["success"]))
    assert env.reward(phi) == pytest.approx(float(W_STAR @ phi))


def test_harm_only_on_failure():
    env = fresh(12)
    env.ep.theta[:] = 1.0                            # never fails
    k = int(np.flatnonzero(env.ep.avail)[0])
    _, _, info = env.step(k)
    assert info["success"] and info["harm"] == 0.0


# =========================================================================== #
# Oracle
# =========================================================================== #
def test_oracle_returns_a_feasible_terminal_action():
    for s in range(30):
        env = fresh(s)
        a, v = env.oracle_action()
        assert env.is_terminal_action(a)
        assert env.feasible_mask()[a]
        assert np.isfinite(v)


def test_oracle_dominates_every_terminal_choice_in_expectation():
    for s in range(20):
        env = fresh(s)
        a_star, v_star = env.oracle_action()
        th, sp = env.ep.theta, 1.0
        for k in np.flatnonzero(env.ep.avail):
            v = (W_STAR[0] * th[k]
                 + W_STAR[1] * (1.0 - HARM[k] * (1.0 - th[k]))
                 + W_STAR[2] * sp)
            assert v <= v_star + 1e-9
        assert env.oracle_value() == pytest.approx(v_star)


# =========================================================================== #
# Decision rule
# =========================================================================== #
def test_decide_respects_the_feasibility_mask():
    for s in range(20):
        env = fresh(s)
        a, _ = decide(env, W0, DecisionConfig(tau0=-1e9),
                      rng=np.random.default_rng(s))
        assert env.feasible_mask()[a]


def test_allow_info_false_blocks_information_actions():
    for s in range(20):
        env = fresh(s)
        a, _ = decide(env, W0, DecisionConfig(tau0=-1e9), allow_info=False)
        assert a < K or a == env.ABSTAIN


def test_high_threshold_forces_abstention():
    env = fresh(0)
    a, aux = decide(env, W0, DecisionConfig(tau0=10.0))
    assert a == env.ABSTAIN and aux["censored"]


def test_low_threshold_forbids_abstention():
    env = fresh(0)
    a, aux = decide(env, W0, DecisionConfig(tau0=-10.0))
    assert a != env.ABSTAIN and not aux["censored"]


def test_allow_abstain_false_never_abstains():
    for s in range(20):
        env = fresh(s)
        a, aux = decide(env, W0, DecisionConfig(
            tau0=10.0), allow_abstain=False)
        assert a != env.ABSTAIN and not aux["censored"]


@pytest.mark.parametrize("seed", range(8))
def test_u_top_is_nonincreasing_in_kappa(seed):
    """S >= 0 and w >= 0, so risk aversion can only lower the leading score.
    Abstention is therefore monotone in kappa."""
    env = fresh(seed)
    tops = []
    for kap in [0.0, 0.5, 1.0, 2.0, 4.0]:
        _, aux = decide(env, W0, DecisionConfig(kappa=kap, tau0=-1e9))
        tops.append(aux["u_top"])
    assert np.all(np.diff(tops) <= 1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_censoring_is_monotone_in_tau(seed):
    env = fresh(seed)
    prev_censored = True
    for tau in [5.0, 1.0, 0.8, 0.6, 0.0, -5.0]:
        _, aux = decide(env, W0, DecisionConfig(tau0=tau))
        assert not (aux["censored"] and not prev_censored)   # never re-censors
        prev_censored = aux["censored"]
    assert not prev_censored


def test_safety_weight_widens_the_safe_minus_risky_gap():
    """Raising w2 must increase U(safe) - U(risky) when both tools are feasible.

    Root cause of earlier failure: availability sampling can make the chosen
    safe/risky tools infeasible, so their U entries are -inf and the gap is
    undefined. Force full availability so the comparison is on-support.
    """
    env = fresh(0)
    # both tools must be feasible
    env.ep.avail[:] = True
    env.ep.ab[:] = PRIORS.copy()                     # deterministic beliefs
    safe, risky = int(np.argmin(HARM)), int(np.argmax(HARM[:4]))
    assert env.ep.avail[safe] and env.ep.avail[risky]
    _, a1 = decide(env, np.array([0.60, 0.20, 0.10, 0.10]), DecisionConfig())
    _, a2 = decide(env, np.array([0.20, 0.60, 0.10, 0.10]), DecisionConfig())
    assert np.isfinite(a1["U"][safe]) and np.isfinite(a1["U"][risky])
    assert np.isfinite(a2["U"][safe]) and np.isfinite(a2["U"][risky])
    assert (a2["U"][safe] - a2["U"][risky]) > (a1["U"][safe] - a1["U"][risky])


def test_utility_is_linear_in_w():
    env = fresh(2)
    Q, S = env.score()
    Qt = risk_adjust(Q, S, 1.0)
    c = env.cost_vector()
    wa, wb = np.array([.7, .1, .1, .1]), np.array([.1, .7, .1, .1])
    lo = 0.3
    mix = lo * wa + (1 - lo) * wb
    assert np.allclose(utilities(Qt, c, mix, 0.5),
                       lo * utilities(Qt, c, wa, 0.5)
                       + (1 - lo) * utilities(Qt, c, wb, 0.5))


def test_qt_row_matches_executed_action():
    for s in range(15):
        env = fresh(s)
        a, aux = decide(env, W0, DecisionConfig(tau0=0.5))
        assert np.allclose(aux["qt"], aux["Qt"][a])


def test_eta_inflates_the_threshold():
    env = fresh(0)
    _, a0 = decide(env, W0, DecisionConfig(eta=0.0, tau0=0.5))
    _, a1 = decide(env, W0, DecisionConfig(eta=2.0, tau0=0.5))
    assert a1["tau"] >= a0["tau"]


# =========================================================================== #
# Lemma 1 -- overlap failure is exact, not approximate
# =========================================================================== #
def test_lemma1_zero_propensity_for_action_under_censoring():
    """A deterministic rule assigns probability EXACTLY zero to acting inside
    the abstain region. This is the formal core of Proposition 2."""
    cfg = DecisionConfig(tau0=0.75, eps_force=0.0)
    n_cens = n_acted = 0
    for s in range(400):
        env = fresh(s)
        a, aux = decide(env, W0, cfg, rng=np.random.default_rng(s))
        if aux["censored"]:
            n_cens += 1
            n_acted += int(a != env.ABSTAIN)
    assert n_cens > 20, "threshold does not exercise the censored region"
    assert n_acted == 0


def test_both_regimes_occur_at_the_locked_threshold():
    cfg = DecisionConfig(tau0=0.55)
    flags = [decide(fresh(s), W0, cfg)[1]["censored"] for s in range(200)]
    assert 0 < sum(flags) < 200, "tau0 gives a degenerate abstain rate"


def test_forcing_restores_positivity():
    eps = 0.25
    cfg = DecisionConfig(tau0=0.75, eps_force=eps)
    props, n_cens, n_forced = [], 0, 0
    for s in range(800):
        env = fresh(s % 80)
        a, aux = decide(env, W0, cfg, rng=np.random.default_rng(1000 + s))
        props.append(aux["propensity"])
        if aux["censored"]:
            n_cens += 1
            n_forced += int(aux["forced"])
    assert min(props) >= eps - 1e-12                 # positivity everywhere
    assert n_forced > 0
    assert abs(n_forced / n_cens - eps) < 0.08


def test_forcing_does_not_change_uncensored_behaviour():
    for s in range(40):
        a0, x0 = decide(fresh(s), W0, DecisionConfig(tau0=0.55, eps_force=0.0),
                        rng=np.random.default_rng(s))
        a1, x1 = decide(fresh(s), W0, DecisionConfig(tau0=0.55, eps_force=0.3),
                        rng=np.random.default_rng(s))
        if not x0["censored"]:
            assert a0 == a1 and x1["propensity"] == 1.0


# =========================================================================== #
# Lemma 2 -- skip-on-abstain is biased, clipped IPS is not
# =========================================================================== #
def _gradient_streams(n=60000, eps=0.2, seed=0):
    """Deterministic, endogenous censoring. Returns (full, skip, ips) means."""
    rng = np.random.default_rng(seed)
    theta = np.array([0.4, 0.6])
    full = np.zeros(2)
    skip = np.zeros(2)
    ips = np.zeros(2)
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0, size=2)
        g = (theta @ x + 0.1 * rng.normal()) * x
        full += g
        if (theta @ x) >= 0.0:                       # uncensored
            skip += g
            ips += g
        else:                                        # censored region
            if rng.random() < eps:
                ips += g / eps
    return full / n, skip / n, ips / n


def test_ips_is_unbiased_and_skip_is_not():
    full, skip, ips = _gradient_streams()
    assert np.allclose(full, ips, atol=0.03), (full, ips)
    # bias = p * E[g | censored]
    assert np.linalg.norm(full - skip) > 0.05
    assert np.linalg.norm(full - skip) > np.linalg.norm(full - ips)


def test_ips_clip_never_binds_at_matched_eps():
    ad = IPSEGAdapter(W0, alpha=0.05, eps=0.10)
    assert ad.Mc == pytest.approx(10.0)
    assert min(1.0 / 0.10, ad.Mc) == pytest.approx(10.0)


# =========================================================================== #
# Adapters
# =========================================================================== #
@pytest.mark.parametrize("kind", ["static", "eg", "ips"])
def test_weights_stay_on_the_simplex(kind):
    ad = build_adapter(kind, W0, alpha=0.2, eps=0.1)
    rng = np.random.default_rng(0)
    for _ in range(500):
        w = ad.update(qt=rng.uniform(-1, 1, M), u_exec=rng.normal(),
                      R=rng.normal(), propensity=0.1, terminal_acted=True)
        assert np.all(w >= -1e-12)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.isfinite(w))


@pytest.mark.parametrize("kind", ["static", "eg", "ips"])
def test_no_update_without_an_observed_outcome(kind):
    ad = build_adapter(kind, W0)
    before = ad.snapshot()
    for _ in range(10):
        ad.update(qt=np.ones(M), u_exec=0.0, R=5.0, propensity=0.1,
                  terminal_acted=False)
    assert np.allclose(ad.snapshot(), before)


def test_static_adapter_never_moves():
    ad = StaticAdapter(W0)
    ad.update(qt=np.ones(M), u_exec=0.0, R=99.0, terminal_acted=True)
    assert np.allclose(ad.snapshot(), W0 / W0.sum())


def test_positive_advantage_raises_the_matching_weight():
    ad = EGAdapter(np.full(M, 0.25), alpha=0.5)
    ad.update(qt=np.array([1.0, 0.0, 0.0, 0.0]), u_exec=0.0, R=1.0,
              terminal_acted=True)
    w = ad.snapshot()
    assert w[0] > 0.25 and w[1] < 0.25


def test_negative_advantage_lowers_the_matching_weight():
    ad = EGAdapter(np.full(M, 0.25), alpha=0.5)
    ad.update(qt=np.array([1.0, 0.0, 0.0, 0.0]), u_exec=1.0, R=0.0,
              terminal_acted=True)
    assert ad.snapshot()[0] < 0.25


def test_ips_equals_eg_when_propensity_is_one():
    eg, ips = EGAdapter(W0, alpha=0.1), IPSEGAdapter(W0, alpha=0.1, eps=0.1)
    rng = np.random.default_rng(3)
    for _ in range(50):
        kw = dict(qt=rng.uniform(-1, 1, M), u_exec=rng.normal(),
                  R=rng.normal(), terminal_acted=True)
        eg.update(propensity=1.0, **kw)
        ips.update(propensity=1.0, **kw)
    assert np.allclose(eg.snapshot(), ips.snapshot())


def test_ips_amplifies_rare_events():
    eg, ips = EGAdapter(np.full(M, .25), alpha=.1), \
        IPSEGAdapter(np.full(M, .25), alpha=.1, eps=.1)
    kw = dict(qt=np.array([1., 0., 0., 0.]), u_exec=0.0, R=1.0,
              terminal_acted=True)
    eg.update(propensity=0.1, **kw)
    ips.update(propensity=0.1, **kw)
    assert ips.snapshot()[0] > eg.snapshot()[0]


def test_gradient_clipping_bounds_the_step():
    ad = EGAdapter(W0, alpha=0.1, clip=5.0)
    g = ad.gradient(np.array([100.0, 0.0, 0.0, 0.0]), 0.0, 1e6)
    assert np.all(np.abs(g) <= 5.0 + 1e-12)


def test_unknown_adapter_raises():
    with pytest.raises(ValueError):
        build_adapter("nope", W0)


def test_project_simplex_is_idempotent():
    for v in [np.array([0.7, -0.3, 0.9, 0.2]), np.array([-1., -2., -3., -4.]),
              np.array([0.25, 0.25, 0.25, 0.25])]:
        p = project_simplex(v)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p >= -1e-12)
        assert np.allclose(project_simplex(p), p)


# =========================================================================== #
# Arm registry
# =========================================================================== #
def test_suite_is_the_locked_day1_scope():
    names = arm_names(W0)
    assert names == ["B0_random", "B1_greedy", "B3_static",
                     "B4_eg_skip", "B4_full", "B5_eg_ips", "ORACLE"]


def test_arm_configurations_match_their_roles():
    arms = {a.name: a for a in make_suite(W0, alpha=0.05, eps=0.10)}
    assert arms["B1_greedy"].cfg.kappa == 0.0
    assert not arms["B1_greedy"].allow_abstain
    assert not arms["B1_greedy"].allow_info
    assert arms["B3_static"].adapter_kind == "static"
    assert arms["B4_eg_skip"].adapter_kind == "eg"
    # no overlap: the point
    assert arms["B4_eg_skip"].cfg.eps_force == 0.0
    assert arms["B5_eg_ips"].adapter_kind == "ips"
    assert arms["B5_eg_ips"].cfg.eps_force == pytest.approx(0.10)
    # B4 and B5 must be identical wherever they share a hyperparameter
    assert arms["B4_eg_skip"].alpha == arms["B5_eg_ips"].alpha
    assert arms["B4_eg_skip"].cfg.tau0 == arms["B5_eg_ips"].cfg.tau0
    assert arms["B4_eg_skip"].cfg.kappa == arms["B5_eg_ips"].cfg.kappa


def test_every_arm_builds_an_adapter():
    for a in make_suite(W0):
        ad = a.make_adapter()
        assert ad.snapshot().shape == (M,)
        assert ad.snapshot().sum() == pytest.approx(1.0)


# =========================================================================== #
# End-to-end plumbing and reproducibility
# =========================================================================== #
def test_every_arm_runs_and_terminates():
    from runner import run_episode
    for arm in make_suite(W0):
        rng = np.random.default_rng(0)
        env = ToolSelectEnv(W_STAR, rng)
        ad = arm.make_adapter()
        for _ in range(20):
            rec = run_episode(env, arm, ad, rng)
            assert rec["steps"] <= BUDGET + 1
            assert np.isfinite(rec["R"])
            assert rec["harm"] >= 0.0


def test_greedy_arm_never_abstains():
    from runner import run_episode
    arm = {a.name: a for a in make_suite(W0)}["B1_greedy"]
    rng = np.random.default_rng(1)
    env = ToolSelectEnv(W_STAR, rng)
    ad = arm.make_adapter()
    assert not any(run_episode(env, arm, ad, rng)[
                   "abstained"] for _ in range(40))


def test_static_arm_weights_are_invariant_end_to_end():
    from runner import run_episode
    arm = {a.name: a for a in make_suite(W0)}["B3_static"]
    rng = np.random.default_rng(2)
    env = ToolSelectEnv(W_STAR, rng)
    ad = arm.make_adapter()
    for _ in range(60):
        run_episode(env, arm, ad, rng)
    assert np.allclose(ad.snapshot(), W0 / W0.sum())


def test_adaptive_arm_weights_actually_move():
    """Guards against a silent no-op adapter faking a null result."""
    from runner import run_episode
    arm = {a.name: a for a in make_suite(W0, alpha=0.1)}["B4_eg_skip"]
    rng = np.random.default_rng(3)
    env = ToolSelectEnv(W_STAR, rng)
    ad = arm.make_adapter()
    for _ in range(100):
        run_episode(env, arm, ad, rng)
    assert not np.allclose(ad.snapshot(), W0 / W0.sum(), atol=1e-4)


def test_same_seed_same_trajectory():
    from runner import run_episode
    out = []
    for _ in range(2):
        rng = np.random.default_rng(7)
        env = ToolSelectEnv(W_STAR, rng)
        arm = {a.name: a for a in make_suite(W0)}["B4_eg_skip"]
        ad = arm.make_adapter()
        out.append([run_episode(env, arm, ad, rng)["R"] for _ in range(40)])
    assert out[0] == out[1]


def test_different_seeds_differ():
    from runner import run_episode
    out = []
    for s in (1, 2):
        rng = np.random.default_rng(s)
        env = ToolSelectEnv(W_STAR, rng)
        arm = {a.name: a for a in make_suite(W0)}["B4_eg_skip"]
        ad = arm.make_adapter()
        out.append([run_episode(env, arm, ad, rng)["R"] for _ in range(40)])
    assert out[0] != out[1]
