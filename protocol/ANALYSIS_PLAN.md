# Statistical Analysis Plan (SAP)  
## Case-side Sim-to-Real — F1 Risk + F2 Siegel Recovery

**Version**: 2026-08-04c (v5)  
**Decisions**: [`DECISIONS.md`](DECISIONS.md)  
**Rulebook**: [`SYNTHETIC_RULEBOOK.md`](SYNTHETIC_RULEBOOK.md)  
**Codebook**: [`codebook/`](codebook/)

---

## 0. Principles

1. Case **11-table package** synthesized only; Control never synthesized.  
2. F1 = case–control; F2 = **case-only Siegel recovery**.  
3. F2 interventions = **steroid**, **HBOT**.  
4. Overcome = Setup **B ∧ C**, τ=0.20; primary = OR ∧ risk/causal.  
5. No NC. ρ* operational = N_synth=N_train + grid.  
6. Generators = all; on-disk deliverable = legacy Step2-adopted package. **Scientific reporting (2026-08-12):** Dummy / Cell-preserving = positive controls; primary claim = F1 Marginal Pass / F2 non-trivial Fail.  
7. Windows: history **365d**, follow-up **365d**; index = **first complete AC 4PTA**.  
8. F1 and F2 share the **same PTA-trackable synthesis set S**.

---

## 1. Cohort / index

Case file membership; index = **first complete AC 4PTA** (0.5/1/2/4 kHz; worse ear if both); OP front-year; F1 1:1 match; events sliced to ±365d; **non-trackable (no FU) excluded from S**.

### 1.1 Audiometry (defines index and F2 outcome)
- Baseline PTA = PTA on the index date (first complete 4PTA).  
- Follow-up PTA: last complete 4PTA on the **same ear** after baseline within **index+365d**.  
- Missing follow-up → **exclude from S**.

---

## 2. Endpoints and features

### 2.1 F1 Risk
E1 diagnosis / E2 steroid pre-index / E3 PTA pre-index; adjusters age, sex.  
Model: case~Ej + age + sex among matched pairs/cohort.

### 2.2 F2 Recovery (USER)

**Siegel PTA**: mean of 0.5, 1, 2, 4 kHz (dB HL), affected ear.  
Gain = PTA_baseline − PTA_followup.

| Class | Rule |
|-------|------|
| CR | final PTA ≤ 25 |
| PR | gain > 15 and final PTA in [25, 45] |
| SI | gain > 15 and final PTA > 45 |
| NI | gain < 15 or final PTA ≥ 75 |

**Primary outcome**: `Y = 1` if CR or PR, else 0.  
**Interventions**:
- I1 steroid (post-index, any systemic/IT per codebook)  
- I2 HBOT (post-index procedure/observation concepts)

**Primary F2 models** (separate):  
`logit P(Y=1) = β0 + β_I I_j + β_age age + β_sex sex`, j ∈ {steroid, HBOT}.

Secondary: 4-level Siegel ordinal / multinomial; steroid×HBOT combination flag.

PTA is **not** an F2 “exposure”; it defines Y.

---

## 3. Sim-to-Real setups

### F1
A/B/C with Real Control × Case variants (as before).

### F2 (case-only)
| Setup | Data for recovery model |
|-------|-------------------------|
| A | Real Case with Siegel Y + I1/I2 |
| B | Synthetic Case only |
| C | Real Train Case ∪ Synthetic Case |

Overcome on I1 and I2 (K=2): sign≥80%, median\|ΔlogOR\|≤0.20, CI overlap≥80% for **both** B and C.

---

## 4. Generators / Step1 / Step2

Canonical selection: [`SELECTION_RULES.md`](SELECTION_RULES.md) (D8).

### Step1 (data) — identical rule book for F1 and F2
1. Apply clinical hard-snap; **gate** if violation rate > 0.02.  
2. **Privacy gate**: DCR ≥ 0.9 × real–real DCR when real–real DCR > 0.  
3. **Select** among passers by `|AUC−0.5|`, then KS, then corr distance.  
4. Do **not** use association drift / OR / TSTR for selection.

### Step2 (epidemiology + causal risk) — per family
- **Block A**: Setup A/B/C **OR** concordance (τ_OR=0.20).  
- **Block B**: **RR/RD** concordance via g-computation ATE (τ_logRR=0.20, τ_RD=0.10).  
- **Block C**: causal package — g-comp + IPW ATE, positivity, **E-value** (sensitivity to unmeasured confounding).  
- Primary success = `or_overcome ∧ risk_causal_overcome` on B∧C.  
- F1 caveat: case–control → population RR not identified without sampling fractions; analytic-table RR/RD used for concordance only.  
- F2: case-only recovery supports risk-scale ATE under consistency / positivity / conditional exchangeability.  
- Sensitivity: all Step1-eligible generators; τ grid.

F2 table includes Y_siegel_binary, siegel_class, I_steroid, I_hbot, PTA_base, PTA_fu, age, sex.  
Synthesis must still be able to support F2 OR concordance, but that is judged **only in Step2**.

---

## 5. Codebook files

| File | Content |
|------|---------|
| `exposures_f1_risk.yaml` | F1 E1–E3 |
| `exposures_f2_care.yaml` | Siegel rules + I1 steroid + I2 HBOT + PTA concepts |
| `cohort_rules.yaml` | index, OP, matching |
| `constraints/ssnhl_rules.yaml` | temporal + PTA order (baseline before follow-up) |

---

## 6. Deliverables

1. Attrition (incl. F2 PTA-complete)  
2. Siegel class distribution Real vs Synth  
3. F1 OR forest A/B/C  
4. **F2 main**: steroid & HBOT → recovered OR forest + B∧C badge  
5. ρ grid; τ sens; generator-all Pareto  

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Controls lack Siegel | F2 case-only by design |
| Index=any Rx vs steroid start | Report; sens index=first steroid |
| PTA concept sparse in OMOP | Measurement concept map blocker for F2 |
| Siegel NI cut ≥75 vs 76 | Fixed ≥75 in DECISIONS |

---

## 8. Version

| Date | Change |
|------|--------|
| 2026-07-30e | D2–D7 |
| 2026-07-30f | F2 Siegel + steroid/HBOT; case-only recovery models |
| 2026-08-03 | D8 Step1 data-only / Step2 OR-only; see SELECTION_RULES.md |
