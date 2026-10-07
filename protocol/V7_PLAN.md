# Sim-to-Real v7 Plan — inference- and causality-preserving Case twins

**Status**: `CLOSED` (submission lock v7.2 sequential) — first v7 run archived in `results/sim_to_real_v7/`; causal re-run → `results/sim_to_real_v7_1/`; scientific package → `results/sim_to_real_v7_2/`. New experiments start at [`V8_PLAN.md`](V8_PLAN.md). Do not rewrite this plan.  
**Parent**: [`DECISIONS.md`](DECISIONS.md) D9, [`SELECTION_RULES.md`](SELECTION_RULES.md), [`RESEARCH.md`](RESEARCH.md)  
**Does not replace v6 reporting lock**: v6 manuscript still claims F1 Marginal Pass / F2 non-trivial Fail; Dummy & Cell-preserving = positive controls.  
**Does not re-run v6.** This document freezes the **next experiment**.

---

## 0. Why v7

Patient twins should not only look like data (Step1). They should **preserve epidemiologic inference and causal estimands** (Step2).

v6 showed:

| | Non-trivial | Positive control |
|--|-------------|------------------|
| **F1** risk OR | Marginal **Pass** | Dummy Pass (row copy) |
| **F2** recovery OR / ATE | **All Fail** | Cell-preserving Pass (locks Y×I cells) |

v7 question: **Can a non-trivial method (including structure ensembles) preserve F2 inference/causality without copying the contingency table — on one shared spine with F1?**

---

## 1. Locked evaluation skeleton (unchanged shape)

Same PTA-trackable **S** (v6 cohort; do not change index/windows unless a separate amendment).

```
S (same PTA-trackable Case)
        │
        ├─ Step1  (filter)   “Is this spine allowable as data?”
        │     clinical ≤2% · DCR privacy · fidelity rank
        │     OR / RR / ATE peek forbidden
        │
        └─ Step2  (test)     “Does inference / causality reproduce?”
              F1: case–control OR (primary) + analytic RR/RD (secondary)
              F2: Siegel recovery OR ∧ g-comp ATE (both primary)
              Setup B ∧ C vs A; τ as in D5 / SELECTION_RULES
```

**Run this skeleton independently for every candidate method M.**  
Step1 does not pick the scientific winner. Step2 does not re-rank by KS/AUC.

---

## 2. One twin = one method

| Lock | Rule |
|------|------|
| Spine | **One M** generates the Case spine used for **both F1 and F2** |
| Control | Never synthesized |
| Forbidden | F1-best M₁ + F2-best M₂ as “the twin” |
| Package | 11-table OMOP only for the **scientific winner** (and optional +control packages) |

If no non-trivial M passes F1∧F2, report that failure. Do not silently split spines.

---

## 3. Roles (winner pool)

| Role | Members | Use |
|------|---------|-----|
| **Positive control (not eligible to win)** | Dummy (row bootstrap); Cell-preserving (full Y×I cell lock) | Upper/lower bound. Expected Pass. Not utility. |
| **Non-trivial solo** | Marginal; Gaussian (current MVN+0.5 cut); CTGAN | Baseline search set |
| **Structure ensemble (primary v7 novelty)** | See §4 C | Attempt to keep joints/ATE without full cell lock |
| **Sensitivity ensemble** | Seed bagging; patient mixture | Report; not default winner |

**Scientific winner pool = Step1-eligible ∩ (non-trivial solo ∪ structure ensemble ∪ sensitivity ensemble).**  
Dummy and Cell-preserving are **never** the scientific winner.

---

## 4. What “ensemble” means (locked)

| Type | Definition | v7 priority |
|------|------------|-------------|
| **A. Seed ensemble** | Same algorithm, ≥5 seeds, pool or median spine metrics | Sensitivity (esp. CTGAN) |
| **B. Patient mixture** | Row-mix e.g. 50% Marginal + 50% Gaussian into one cohort | Sensitivity |
| **C. Structure ensemble** | Keep margins; **target associations / ATE** without locking the full Y×I table | **Primary novelty** |

Type C is the method that matches the research goal. A/B alone will not fix F2’s structural Fail.

---

## 5. Candidate methods M (implementation order)

### 5.1 Positive controls (always run, never win)

| ID | Method | Note |
|----|--------|------|
| PC1 | Dummy | F2 may fail DCR (as in v6); still report |
| PC2 | Cell-preserving | Full Y×{steroid, HBOT} cell lock |

### 5.2 Non-trivial solo (carry from v6)

| ID | Method | Expected (from v6) |
|----|--------|--------------------|
| N1 | Marginal | F1 Pass / F2 Fail |
| N2 | Gaussian MVN + 0.5 snap | F1 Fail / F2 near-miss on C only |
| N3 | CTGAN | F1 Fail / F2 Fail |

`*_strat` that sample **I independent of Y within Y-strata** stay **SI only** (v6: little new information).

### 5.3 Structure ensembles — **must implement** (C)

| ID | Method | Intent |
|----|--------|--------|
| C1 | **Gaussian copula** (continuous + binary separated; no naive 0.5 cut on all) | Keep rank correlations; fix N2’s binary distortion |
| C2 | **IPF / 2-way association target** | Fix 1-way margins (Marginal) + match 2-way OR/counts for pre-registered pairs only — **not** full multi-way cell lock |
| C3 | **Y-strat + Gaussian-within** | Stratify on recovered; **preserve within-stratum covariance including I** (unlike v6 `*_strat`) |
| C4 | **Partial-cell** | Lock **Y margin only** or **one pre-registered 2-way** (e.g. recovered×steroid) but not the full Y×I₁×I₂ table |

C2/C4 must be weaker than PC2. If an M locks the same cells as Cell-preserving, relabel it as PC2, not a winner.

### 5.4 Sensitivity ensembles (A/B)

| ID | Method |
|----|--------|
| S1 | CTGAN seed ensemble (≥5 seeds) |
| S2 | Mixture Marginal ⊕ Gaussian (50/50 rows, then clinical snap) |
| S3 | Mixture C1 ⊕ N1 (optional if C1 exists) |

---

## 6. Step2 estimands (v7)

| Family | Primary | Secondary | Note |
|--------|---------|-----------|------|
| **F1** | Adjusted **OR** concordance (B∧C, τ_OR=0.20, sign≥80%, CI overlap≥80%) | Analytic-table RR/RD; E-value | Population RR **not identified** (case–control). Do not promote RR to F1 primary. |
| **F2** | **OR ∧ g-comp ATE** (RR and RD) on steroid & HBOT → recovered; B∧C | IPW ATE, positivity, E-value, 4-class Siegel | Matches “inference **and** causality” |

F2 `risk_causal_overcome` uses the same τ as SAP: τ_logRR=0.20, τ_RD=0.10, sign≥80%.  
Setup **B is the binding Sim-to-Real test**; C remains required for Overcome but is interpreted as dilution toward A.

---

## 7. Winner rule (scientific; locked)

Among **winner-pool** methods that **pass Step1**:

1. **F1 Overcome ∧ F2 Overcome** (OR∧ATE on B∧C) → if several, minimize  
   `0.5 · mean_F1|ΔlogOR| + 0.5 · mean_F2(|ΔlogOR| + |ΔlogRR| + |ΔRD|)/3`
2. Else **F2 Overcome** (F1 may already be solvable by Marginal; F2 is the v7 target)
3. Else **F1 Overcome** only
4. Else no scientific winner: report best non-trivial drift + PC bounds

Tie-break: fewer structure constraints (C1/C3 before C2/C4), then Step1 fidelity (`|AUC−0.5|` → KS → corr) — fidelity only after Step2, never instead of it.

**Explicitly forbidden as winner:** Dummy, Cell-preserving, any M that is equivalent to full Y×I cell lock.

This is a **pre-registered methodology comparison**, not Step1 peeking. Step1 still cannot use OR/ATE. The winner rule uses Step2 because v7’s object of inference **is** utility/causality.

---

## 8. What v7 will claim (hygiene)

**May claim**

- Non-trivial M (possibly C1–C4) preserved F1 and/or F2 estimands on shared S, if true.
- Structure ensemble without full cell lock beat solo Marginal/Gaussian/CTGAN on F2, if true.
- If all non-trivial M fail F2: **association-preserving twinning needs explicit joint structure**; copying cells (PC2) is not a solution.

**Must not claim**

- Dummy / Cell-preserving utility success.
- Two different M’s as one digital twin.
- F2 ATE = RCT treatment effect.
- 5,311 Case file replacement.

---

## 9. Outputs

| Artifact | Path (planned) |
|----------|----------------|
| Plan (this file) | `V7_PLAN.md` |
| Code | `code/run_sim_to_real_v7.py` (+ ensemble helpers, e.g. `code/synth_ensemble.py`) |
| Results | `results/sim_to_real_v7/` (`summary.json`, `REPORT.md`, `cohort/`, `deliverable/`) |
| Manuscript export | `code/export_manuscript_v7.py` → `results/sim_to_real_v7/manuscript/` |

Reuse v6: `cohort_v5.py` PTA-index, `evaluate_v5.py` / `causal_epi.py` gates, `clinical_constraints.py`, same S construction. Do not rebuild the 1,653 cohort unless QA forces it.

---

## 10. Implementation order

1. Freeze this plan (done).  
2. Implement C1 → C3 → C2 → C4; wire N1–N3 + PC1–PC2 + S1–S2.  
3. `run_sim_to_real_v7.py`: for each M, Step1 then Step2 on **shared spine**.  
4. Apply §7 winner rule; write `REPORT.md` with PC vs non-trivial tables.  
5. Only then: deliverable 11-table package for scientific winner (if any).

npj v6 reporting can proceed in parallel; v7 is a new experiment, not a silent rewrite of v6 claims.

---

## 11. Lock record

| Date | Item |
|------|------|
| 2026-08-12 | D9 / `V7_PLAN.md` locked: per-M Step1→Step2; Dummy & Cell-preserving out of winner pool; single spine; F2 OR∧ATE primary; structure ensembles C1–C4 as novelty |
| 2026-08-12 | **D9a**: first v7 Y-targeted C2/C3/C4 passed F2 by peeking at `recovered` (weaker cell lock). Relabel those as **association controls (AC, cannot win)**. Causal structure ensembles generate **I + PTA only**; `recovered := Siegel(PTA)`. See §12. |
| 2026-08-12 | **D9b**: add C5 **sequential cascade** X→I\|X→PTA\|I (structure ensemble; Y=Siegel). Run → `results/sim_to_real_v7_2/`. |
| 2026-08-12 | **Bugfix**: within-cell **independent** column draws broke Y lock (Siegel overwrite 5–14%); switched to **joint row** draws. `ipf_pta` mapped `pta_fu` only (gain sign broken) → joint `(pta_base,pta_fu)|I`. Re-eval v7.2. |

Amendment requires an explicit new decision (do not silently change winner pool or re-include Dummy/Cell-preserving as utility winners).

---

## 12. D9a amendment — causality-oriented DGP (2026-08-12)

**Why.** First v7 winner `ystrat_copula` (and `ipf_2way`, `partial_cell`) preserved F2 **by conditioning on Y or targeting I|Y**. That copies association structure, not a treatment mechanism. Meanwhile C1 copula sampled `recovered` then Siegel **overwrote** it, destroying I–Y. That does not match the goal: twins that keep **inference and causality**, not just table similarity.

**Locked DGP**

- Generators (except Dummy / Cell-preserving / AC) **do not sample `recovered`**.
- After clinical snap: **`recovered := Siegel(pta_base, pta_fu)`**.
- F2 estimands I → recovered then flow through **I → PTA → Siegel**.

**Roles (winner pool)**

| Role | Members | Eligible to win? |
|------|---------|------------------|
| PC | Dummy; Cell-preserving (full Y×I lock) | **No** |
| AC (association / Y-peek) | `ystrat_copula`; `ipf_2way` (I\|Y); `partial_cell` (Y×steroid) | **No** — SI / bound only |
| Causal C (primary novelty) | C1 `copula`; **C5 `sequential` (X→I\|X→PTA\|I)**; C3 `istrata_copula`; C2 `ipf_pta_given_i`; C4 `partial_pta_steroid` | Yes if Step1 |
| N / S | Marginal, Gaussian, CTGAN; seed/mixture | Yes if Step1 |

Step1 / Step2 estimands / τ / single spine / Dummy&PC never-win: **unchanged**.  
F2 secondary (report only): crude PTA-gain contrast I → (pta_base − pta_fu), not required for Overcome.

**May claim after D9a:** a non-Y-peeking M preserved F1 and/or F2, if true.  
**Must not claim:** AC/PC as utility; that first-v7 `ystrat_copula` is the causal twin.
