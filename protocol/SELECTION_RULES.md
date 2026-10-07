# Selection Rules — Research Concept Alignment

**Status**: `PROPOSED_LOCK` (D8 + Step2 risk/causal extension)  
**Updated**: 2026-08-03  
**Parent**: [`RESEARCH.md`](RESEARCH.md), [`DECISIONS.md`](DECISIONS.md)  
**Code**: [`code/causal_epi.py`](code/causal_epi.py), [`code/run_sim_to_real.py`](code/run_sim_to_real.py)

---

## Diagram (how to read the study)

```
                    ┌──────────── F1 Risk ────────────┐  ┌──────── F2 Siegel Recovery ────────┐
                    │ case–control; Real Control fixed │  │ case-only; steroid / HBOT → recover │
┌──────────────┐    ├──────────────────────────────────┤  ├────────────────────────────────────┤
│ Step 1       │    │ Clinical constraint gate         │  │ Clinical constraint gate           │
│ DATA         │───▶│ Privacy gate (DCR)               │  │ Privacy gate (DCR)                 │
│ (intrinsic)  │    │ Fidelity rank (KS, corr, AUC)    │  │ Fidelity rank (KS, corr, AUC)      │
│              │    │ → select Step1-winner            │  │ → select Step1-winner              │
└──────────────┘    └────────────────┬─────────────────┘  └─────────────────┬──────────────────┘
                                     │                                      │
┌──────────────┐                     ▼                                      ▼
│ Step 2       │    │ (1) OR concordance A/B/C         │  │ (1) OR concordance A/B/C           │
│ EPIDEMIOLOGY │───▶│ (2) Risk RR/RD concordance       │  │ (2) Risk RR/RD + causal ATE        │
│ + CAUSAL     │    │ (3) E-value / positivity          │  │ (3) g-comp & IPW; E-value          │
│ (extrinsic)  │    │ Primary: OR ∧ risk blocks, B∧C   │  │ Primary: OR ∧ risk/causal, B∧C     │
└──────────────┘    └──────────────────────────────────┘  └────────────────────────────────────┘
```

**Hard separation**
- Step1 **must not** use OR / RR / RD / E-value / TSTR for generator selection.
- Step2 **must not** re-rank generators by KS / AUC / DCR.
- F1 and F2 share the **same Step1 rule book**; Step2 estimands differ by design (below).

---

## Step 1 — Data differences (per family)

*(unchanged from prior D8 — clinical → privacy → fidelity)*

### 1A. Clinical constraint gate (pass/fail)

| Check | F1 | F2 |
|-------|----|----|
| Age ∈ [0,120] integer | ✓ | ✓ |
| Sex ∈ {0,1} | ✓ | ✓ |
| Binary exposures ∈ {0,1} | hear/htn/dm/steroid_pre | steroid_post, hbot_post |
| PTA ∈ [-10, 120] | — | ✓ |
| PTA on **1.25 dB** 4-freq grid | — | ✓ |
| Siegel ↔ recovered consistent | — | ✓ |

Pass iff max violation rate ≤ **0.02**.

### 1B. Privacy gate
`DCR(synth) ≥ 0.9 × DCR_rr` when `DCR_rr > 0`; else descriptive only.

### 1C. Fidelity rank among passers
`|AUC−0.5|` → KS → corr → `step1_selected`.

---

## Step 2 — Epidemiology + causal risk (per family)

Evaluated on **Step1-selected** (primary) and all `step1_eligible` (sensitivity).

### Setups

| Setup | F1 | F2 |
|-------|----|----|
| A | Real Control × Real Case | Real Case |
| B | Real Control × Synth Case | Synth Case |
| C | Real Control × Real+Synth Case | Real+Synth Case |

### Block A — Association (OR)

For each pre-registered exposure/intervention:

- sign concordance ≥ 80%  
- median \|ΔlogOR\| ≤ **τ_OR = 0.20**  
- CI overlap ≥ 80%  

`or_overcome` = Setup **B ∧ C** both pass.

### Block B — Risk measures (RR / RD)

| Measure | Definition | F1 | F2 |
|---------|------------|----|----|
| Crude risk | P(Y=1\|X) | descriptive in analytic table | primary descriptive |
| RR | risk1 / risk0 | associative contrast† | causal target scale |
| RD | risk1 − risk0 | associative contrast† | causal target scale |

† F1 is case–control: **population RR/RD are not identified** without sampling fractions.  
We still report RR/RD on the **analytic matched table** as risk-scale association contrasts for Sim-to-Real concordance, and state this limitation explicitly.

Risk concordance (Setup B/C vs A), using **g-computation ATE** as preferred scale:

- sign(RR) and sign(RD) concordance ≥ 80% each  
- median \|ΔlogRR\| ≤ **τ_RR = 0.20**  
- median \|ΔRD\| ≤ **τ_RD = 0.10**  

`risk_causal_overcome` = Setup **B ∧ C** both pass.

### Block C — Causal estimands (identification + estimators)

**Target estimand (F2 primary causal claim)**  
Average Treatment Effect on the **risk scale**:

- ATE_RD = E[Y(1) − Y(0)]  
- ATE_RR = E[Y(1)] / E[Y(0)]  

for interventions steroid / HBOT on Siegel recovery.

**Identification assumptions (stated, not proven by data alone)**

1. **Consistency** — observed outcome under observed treatment = potential outcome  
2. **Positivity** — within age/sex strata, both treated and untreated exist (report empty-cell count)  
3. **Conditional exchangeability** — no unmeasured confounding given age, sex  

**Estimators (report all; concordance uses g-comp)**

| Estimator | Role |
|-----------|------|
| Outcome regression **g-computation** | Primary ATE_RD / ATE_RR |
| **IPW** (stabilized, truncated) | Co-primary robustness |
| Adjusted **OR** (logit) | Association bridge to Block A |
| **E-value** (VanderWeele–Ding) | Sensitivity to unmeasured confounding |

**What “causal evidence” means here**  
This is **not** randomized proof. Step2 requires that synthetic Case analyses **reproduce the same causal estimands and sensitivity profile** as Real Case under the same assumptions — i.e. Sim-to-Real validity of the causal *analysis*, plus E-value/positivity diagnostics. Absolute causation remains assumption-dependent.

### Primary Step2 success (family-level)

```
primary_overcome = or_overcome ∧ risk_causal_overcome
```

Both blocks must pass on Setup **B and C**.

**Forbidden in Step2**
- re-picking generator by KS/AUC/DCR after seeing effects  
- clinical violation rate as overcome criterion  

TSTR AUROC = secondary descriptive only.

---

## F1 vs F2 in Step2 (why both blocks)

| | F1 Risk | F2 Recovery |
|--|---------|-------------|
| Design | Case–control | Case-only cohort-like |
| Design-native association | **OR** | OR + **RR/RD** |
| Causal ATE on risk scale | Limited (sampling) | **Supported** under assumptions |
| Concordance primary | OR ∧ analytic RR/RD | OR ∧ g-comp ATE RR/RD |
| Causal sensitivity | E-value on RR contrast | E-value on ATE_RR + IPW check |

---

## Reporting skeleton

1. Step1 — F1 / F2 (data)  
2. Step2 — F1: OR forest + RR/RD table + E-value + B∧C  
3. Step2 — F2: OR forest + g-comp/IPW ATE + E-value + positivity + B∧C  
4. Supplement: eligible-generator sensitivity; assumption checklist  

---

## Co-primary reporting (added 2026-08-04)

- `step1_selected`: fidelity/privacy/clinical only (unchanged).
- `step2_co_primary`: among **Step1-eligible** generators with `primary_overcome=True`, re-pick by the **same Step1 fidelity rank** (`|AUC-0.5|`, KS, corr).
- This does **not** override Step1 selection and does **not** optimize on OR magnitude.
- F1 missing exposures: **preserve NaN** (no NaN→0); OR/RR fit uses complete cases (`dropna`).

---

## Reporting hierarchy (LOCKED 2026-08-12) — no pipeline re-run

Pipeline flags (`adopted_generator`, `primary_overcome` on Dummy / Cell-preserving) stay as computed.  
**Manuscript claims use the labels below.**

```
S (shared PTA-trackable Case)
        │
        ├─ Step1  (filter)   Is the spine allowable as data?
        │     F1: Dummy · Marginal · Gaussian · CTGAN eligible → rank-1 Marginal
        │     F2: Dummy fails DCR; Cell-preserving is rank-1
        │
        └─ Step2  (test)     Do allowable spines reproduce inference?
              F1: Dummy · Marginal Pass / Gaussian · CTGAN Fail
              F2: Cell-preserving only Pass  ← all non-trivial generators Fail
```

| Role | Generators | How to write |
|------|------------|--------------|
| **Positive control** | Dummy (row bootstrap); Cell-preserving (locks outcome×exposure cells) | “Concordance is expected if the contingency structure is copied.” Not a utility success. |
| **Scientific primary (non-trivial)** | Marginal, Gaussian, CTGAN (`*_strat` → SI) | F1 **Marginal Pass**. F2 **all Fail**. |
| **On-disk package** | F1 `dummy`; F2 `cell_preserving` | Legacy adoption rule. Disclose; do not equate with scientific primary. |

**Primary manuscript sentence**  
Under a shared PTA-trackable Case definition, non-trivial Marginal synthesis reproduced pre-index case–control ORs (F1). Non-trivial generators did not reproduce Siegel recovery ORs (F2); only the cell-preserving positive control passed.

---

## v7 (D9) — do not confuse with v6 reporting

Next experiment: [`V7_PLAN.md`](V7_PLAN.md) (**D9a**).  
Same Step1→Step2 **per method**, **one spine** for F1+F2, Dummy/Cell-preserving **and Y-peeking AC** out of winner pool, F2 **OR ∧ ATE** primary.  
Causal DGP: generate I+PTA; `recovered := Siegel`. First v7 (`ystrat_copula` winner) is archived, not the causal claim.  
v6 claims above stay until a new reporting lock is written.

---
