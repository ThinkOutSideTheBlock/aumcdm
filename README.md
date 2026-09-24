```markdown
# AU-MCDM: Adaptive Uncertainty-Aware Multi-Criteria Decision Making under Censored Feedback

**Rising Safety Weights Are Not Reassuring: Feature-Scale Artifacts under Abstention-Censored Multiplicative Preference Updates**

Code companion to the paper. This repository contains the environment, controllers, analysis plan, and scripts used for the experiments. Large result trees are **not** included; regenerate them with the commands below or obtain archived data via the paper’s data statement (Zenodo/OSF when available).

---

## What this is

Stationary multi-criteria tool-selection environment (K=6 tools, M=4 criteria) with:

- Abstention under a utility threshold τ₀
- Skip-on-abstain and counterfactual EG preference updates
- Primary finding: absolute decay of the safety weight under censored feedback is **falsified** (160/160 adaptive seed-runs show positive w₂ drift)
- Empirical account: under multiplicative EG with positive mean advantage, relative drift is ordered by mean feature magnitude (feature-scale artifact)
- Exploratory scale-invariant threshold τ(w) = τ₀^SI ‖w‖₂

Scope is **one synthetic environment, one updater family, one primary threshold**. We do not claim transfer to RLHF or deployed systems.

---

## Repository layout

```
aumcdm-project/
├── ANALYSIS_PLAN.md          # Documented analysis plan (not pre-registered)
├── aumcdm/                   # Package: env, decision, baselines, adapt
│   ├── envs/tool_select.py   # ToolSelectEnv (canonical safety encoding)
│   └── engine/               # DecisionConfig, EG adapters, suite
├── runner.py                 # Main experiment runner (Gate-1, full arms)
├── ablation.py               # η-path + scale-invariant ablations + figures
├── calibrate_tau_si.py       # Calibrate TAU0_SI to match fixed-rule abstain rate
├── bottom_decile.py          # Bottom-50 composition from episode logs
├── analysis_*.py             # Post-hoc tables / permutation tests
├── logging_utils.py
├── conftest.py
├── tests/
├── requirements.txt
└── README.md
```

Ignored by design (do not commit): `results_*/`, `ablation_out/`, `bottom_decile_out/`, generated CSVs, `__pycache__/`, `.venv/`.

---

## Setup

```bash
git clone <this-repo-url>
cd aumcdm-project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.10 recommended. Dependencies: numpy, pandas, matplotlib, and whatever is listed in `requirements.txt`.

---

## Quick sanity (Gate-1)

```bash
python runner.py --T 50 --seeds 3 --tau0 0.65 --out results_smoke_canon
```

Expect (approximate):

- B3_static abstain ~0.15–0.25 (not 1.0)
- B4_eg_skip w₂ drift **> 0**
- Greedy never abstains; static arms do not drift

Canonical safety encoding must be active in `aumcdm/envs/tool_select.py`  
(`Q[:,1] = 1 - HARM * (1 - mu)`, not the low-mean flip).

---

## Primary counted runs (paper tables)

```bash
# Example shape used in the paper (T=500, 20 seeds)
python runner.py --T 500 --seeds 20 --tau0 0.65 --out results_cnt_mis
# Aligned / seed-set B variants as described in ANALYSIS_PLAN.md
```

Canonical data dirs named in the paper (when regenerated or restored from archive):

- `results_cnt_*` — outcomes and weights
- `results_term_*` — terminal census
- `results_logs_*` — episode-level logs (bottom-decile, failure-by-tool)

---

## Scale-invariant threshold (exploratory)

1. Calibrate `TAU0_SI` so SI abstain rate matches fixed-rule ~0.22 under misspec w₀:

```bash
python calibrate_tau_si.py --T 100 --seeds 10
# Then set TAU0_SI in ablation.py (locked value in paper: 1.05)
```

2. Production ablation (η-path + fixed vs SI + figures):

```bash
python ablation.py --T 500 --seeds 20 --tau0 0.65 --out ablation_out
```

Writes:

- `ablation_out/ablation_eta.csv`
- `ablation_out/ablation_scale.csv`
- `ablation_out/figures/eta_curve.png`
- `ablation_out/figures/scale_compare_misspec.png`
- `ablation_out/figures/scale_compare_aligned.png`
- `ablation_out/figures/cvar_vs_coverage.png`

---

## Bottom-decile composition

```bash
python bottom_decile.py --T 500 --seeds 20 --tau0 0.65
# Or from existing episode CSVs:
# python bottom_decile.py --from-csv --out <folder_with_episodes_*.csv>
```

---

## Analysis plan and statistics

See **`ANALYSIS_PLAN.md`**.

- Family of 14 contrasts (skip−static × cells + static misspec−aligned CVaR), Holm at 0.05/14
- Plan file was committed **after** the primary runs (documented, not pre-registered)
- Exact sign-flip p-values used where all 20 seed differences share a sign

Reproduce tables with `analysis_revision3.py` and related scripts once `results_logs_*` / `results_cnt_*` are present.

---

## Hyperparameters (paper defaults)

| Symbol        | Value | Role                                      |
|---------------|-------|-------------------------------------------|
| τ₀            | 0.65  | Fixed abstention threshold                |
| τ₀^SI         | 1.05  | Scale-invariant base (calibrated)         |
| ρ             | 3.0   | w* harm asymmetry                         |
| α_EG          | 0.05  | EG step size                              |
| κ             | 1.0   | Risk adjustment on U                      |
| T             | 500   | Episodes per seed                         |
| seeds         | 20    | Per cell; sets A/B via seed_base          |
| Clip c        | 5     | Vestigial (\|Aqⱼ\| ≤ 1)                     |

---

## Citation

If you use this code, please cite the paper (preprint / venue version when available) and this repository.

```bibtex
@misc{aumcdm2026,
  title  = {Rising Safety Weights Are Not Reassuring: Feature-Scale Artifacts under Abstention-Censored Multiplicative Preference Updates},
  author = {Sajjad Khoshakhlagh},
  year   = {2026},
  note   = {Code:https://github.com/ThinkOutSideTheBlock/aumcdm. Analysis plan: ANALYSIS_PLAN.md}
}
```

---

## License

MIT License (see LICENSE file).

---

## Reproducibility notes

- Primary T=500 streams are defined by `seed` + `seed_base` and the default path `scale_invariant=False`, τ₀=0.65.
- Do not enable the low-mean safety flip in `tool_select.py` for paper replication; that path zeros safety features after clipping and collapses abstention behavior.
- Large `results_*` directories are intentionally omitted from git; archive them externally if you need bit-for-bit table checks.

---

## Contact

Issues and questions: open a GitHub issue on this repository.
```