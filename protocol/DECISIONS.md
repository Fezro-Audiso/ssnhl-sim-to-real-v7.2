# Decisions

**Status**: `USER_LOCKED` (D1–D7 + F2 clinical amendment)  
**Updated**: 2026-07-30  

합의 유지:
- Case만 합성 / Real Control 고정 (F1 case–control용)
- Primary endpoint 철학 = effect concordance (OR 등)
- Eckardt F/U/P = Methods 준수 (novelty 아님)

---

## D1 → USER LOCKED: **C** (+ F2 임상 초점 명확화)

| Family | Question | Window | 분석 단위 |
|--------|----------|--------|-----------|
| **F1 Risk** | 진단 전·사전 요인 | Index **이전** 전구간 | Case vs Control (1:1) |
| **F2 Care / Recovery** | **돌발성난청 회복** | Index **이후** 전구간 | **Case-only** (회복은 Case에만 정의) |

**F2 임상 목적 (USER, 2026-07-30 추가)**: Siegel 기준 회복 평가.  
**중재**: **스테로이드** + **고압산소치료(HBOT)**.

Family 분리. F1에 post-index 치료/회복 변수 투입 금지.

---

## D5 → USER LOCKED

- Primary success (family별) = Setup **B and C** 모두 overcome
- τ = **0.20** (민감도 0.10 / 0.30)
- sign ≥ 80%, CI overlap ≥ 80%

### Family별 추정량

| Family | Concordance 대상 |
|--------|------------------|
| F1 | Case–control OR (사전 노출) |
| F2 | Case-only: 중재(스테로이드, HBOT) → **Siegel 회복** OR (또는 ordinal model; primary = binary recovery — 아래) |

### ρ* / NC

| Item | Rule |
|------|------|
| NC | Primary 미포함 |
| C ρ* | `N_synth = N_real_train` 운영점 + **ρ grid 병기** |

---

## D2 / D3 → USER AMENDED (2026-08-04)

이전 LOCK(전구간 lookback / F2 전구간)을 아래으로 개정:

| Item | New rule |
|------|----------|
| Index | **첫 complete AC 4PTA** (진단일 아님) |
| History / lookback | **index 기준 이전 1년** |
| Follow-up | **이후 1년**; FU PTA = 같은 귀 last complete 4PTA ≤365d |
| Synthetic scope | Case **구성 테이블 전부** 동형 합성; 원본과 **같은 테이블 규모** |
| Cohort | **PTA-trackable**만; F1·F2 **동일 S** |
| PTA missing / no FU | 합성 학습·생성 **제외** |
| Steroid/HBOT missing | **결측률 유지** (용량·기간·경로 등 원본 필드 구조) |
| Deliverable files | F2: Step2 통과분만 보존 / F1: Step1-eligible 중 Step2 성능 최고 (통과 불필요) |

상세: [`SYNTHETIC_RULEBOOK.md`](SYNTHETIC_RULEBOOK.md) v2026-08-04b.

---

## D4 → USER LOCKED (F2 개정)

### F1 Risk 노출 (유지)

| ID | Domain |
|----|--------|
| E1 | 돌발성 난청 진단 (index 전) |
| E2 | Steroid (index 전) |
| E3 | PTA 존재/요약 (index 전) |

### F2 Recovery — 개정 (USER)

| Role | Item | Definition |
|------|------|------------|
| **Outcome** | Siegel recovery | 아래 Siegel 4분류; primary binary = **CR+PR** vs **SI+NI** |
| **Intervention I1** | Steroid | index 이후 전구간 steroid 노출 (전신/고실 구분 codebook) |
| **PTA** | 측정 원천 | Official AC 4PTA (CDM map): RT 900000029/030/031/033, LT 900000042/043/044/046; worse ear at baseline |
| **Intervention I2** | HBOT | procedure_concept_id **4087611** |

K_F2 interventions = **2** (steroid, HBOT).  
PTA는 endpoint 구성 요소.

#### Siegel’s criteria (표준 4분류 — SAP 고정)

PTA = 0.5, 1, 2, 4 kHz 평균 (dB HL), 환측.  
Hearing gain = baseline PTA − follow-up PTA.

| Class | Label | Rule (문헌 상용) |
|-------|-------|------------------|
| I | Complete recovery (CR) | Final PTA ≤ 25 dB |
| II | Partial recovery (PR) | Gain > 15 dB **and** final PTA 25–45 dB |
| III | Slight improvement (SI) | Gain > 15 dB **and** final PTA > 45 dB |
| IV | No improvement (NI) | Gain < 15 dB **or** final PTA ≥ 75 dB (문헌 표기 75/76 dB 혼재 → **≥75 dB**로 고정) |

Primary F2 endpoint: `recovered = 1{CR or PR}`.  
Secondary: 4-class ordinal / multinomial concordance.

Baseline PTA: index 전후 허용 창 codebook (draft: index−7d ~ index+3d 최근접).  
Follow-up PTA: F2 전구간 내 **마지막** PTA (민감도: index+30/90d 최근접).

---

## D6 → USER LOCKED

기존 OMOP → person-level analytic table (F1 행: case–control; F2 행: case + Siegel + interventions).

---

## D7 → USER LOCKED: **전부**

CTGAN, TVAE, Dummy, Marginal, DP 가능분, seeds≥5.

---

## D8 → PROPOSED LOCK: Step1 / Step2 selection boundary

Canonical detail: [`SELECTION_RULES.md`](SELECTION_RULES.md)

| Stage | Allowed for gate / selection / success | Forbidden |
|-------|----------------------------------------|-----------|
| **Step1 (data)** | clinical ≤2%; DCR privacy; rank \|AUC−0.5\|→KS→corr | OR, RR, RD, E-value, TSTR, overcome |
| **Step2 (epi+causal)** | OR concordance; **RR/RD** concordance; g-comp/IPW ATE; E-value; positivity; primary = OR ∧ risk/causal, B∧C | re-rank by KS/AUC/DCR |

F1 and F2 share the **same Step1 rule book**; Step2 causal interpretation differs (F1 case–control OR-native; F2 risk-scale ATE).

---

## D9 → LOCKED (2026-08-12): v7 methodology search

Canonical: [`V7_PLAN.md`](V7_PLAN.md)

| Item | Lock |
|------|------|
| Goal | Non-trivial Case twin that preserves **inference + causal estimands**, not only Step1 fidelity |
| Skeleton | Same S; **each method M** runs Step1 (filter) → Step2 (test) |
| Twin | **One M** for F1 **and** F2 (no split spines) |
| Winner pool | Non-trivial solo + structure ensembles + sensitivity ensembles. **Dummy & Cell-preserving never win** |
| F2 Step2 primary | **OR ∧ g-comp ATE** (B∧C) |
| F1 Step2 primary | OR (RR/RD secondary; case–control RR not identified) |
| Novelty | **D9a:** causal structure = I→PTA→Siegel (copula / I-strat / pta_fu\|I). Y-targeted C2/C3/C4 from first v7 = **AC, cannot win** |
| v6 | Reporting lock unchanged; v7 is a new run, not a silent rewrite |
| First v7 | Archived `results/sim_to_real_v7/` (Y-peek winner `ystrat_copula`). Not the causal claim. |

---

## Version

| Date | Change |
|------|--------|
| 2026-07-30 | D1–D7 lock |
| 2026-07-30 | F2 = Siegel recovery; interventions = steroid + HBOT |
| 2026-08-03 | D8 draft: Step1 data-only / Step2 OR-only selection |
| 2026-08-03 | Step2 extended: RR/RD + causal ATE (g-comp/IPW) + E-value |
| 2026-08-12 | Reporting lock: Dummy/Cell-preserving = positive controls; scientific primary = F1 Marginal Pass / F2 non-trivial Fail (no pipeline re-run). See `SELECTION_RULES.md` |
| 2026-08-12 | **D9 v7 plan lock**: [`V7_PLAN.md`](V7_PLAN.md) — per-M Step1→Step2; single spine; Dummy/Cell-preserving out of winner pool; F2 OR∧ATE primary; structure ensembles |
| 2026-08-12 | **D9a**: Y-peeking methods = association controls (cannot win); causal generators use I+PTA only, `recovered:=Siegel`. Re-run → `results/sim_to_real_v7_1/` |
| 2026-08-12 | **D9b**: C5 sequential cascade X→I→PTA→Siegel → `results/sim_to_real_v7_2/` |
