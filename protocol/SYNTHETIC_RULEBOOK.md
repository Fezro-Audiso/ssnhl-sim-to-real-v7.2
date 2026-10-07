# 가상 돌발성난청(SSNHL) 환자셋 — 생성 규칙 & 임상 Rule Book

**Version**: 2026-08-04e (v6 PTA-index)  
**Status**: `DESIGN_ADOPTED` — implemented by `code/run_sim_to_real_v6.py`  
**결과**: `results/sim_to_real_v6/`

---

## 1. 정의

| 항목 | 내용 |
|------|------|
| **무엇인가** | SSNHL **Case**를 학습해 만든 **합성 Case(가상 돌발성난청 환자셋)** |
| **Control** | **합성하지 않음** (F1만 Real Control 고정) |
| **합성 범위** | Case 폴더 **11개 OMOP 테이블 전부** 동형 합성 |
| **규모 정합** | 합성 대상 코호트 \(S_{\mathrm{train}}\)과 **PERSON 수·테이블별 총 행 수 일치** |
| **Index** | **첫 complete AC 4PTA** (0.5/1/2/4 kHz; 양측이면 worse ear) |
| **코호트** | **PTA-trackable** (base+같은 귀 FU ≤365d); F1·F2 **동일 S** |
| **시간창** | History = index−365d, Follow-up = index+365d |

### 구성 테이블 (11)

PERSON, OBSERVATION_PERIOD, CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE, MEASUREMENT, OBSERVATION, DEVICE_EXPOSURE, DEATH, VISIT_OCCURRENCE, VISIT_DETAIL.

---

## 2. 생성 규칙

### 2.1 알고리즘 (v6)

1. 원본 Case 프로파일 (dtype·범위·결측·행 수)
2. Index = first complete 4PTA → ±365d 슬라이스 → **PTA-trackable만 \(S\)** (F1=F2)
3. Person spine 생성: dummy / marginal / gaussian / ctgan (+ F2 strat, cell_preserving)
4. Clinical hard-snap
5. Step1: clinical ≤2% + privacy + fidelity → `step1_selected`
6. Step2: OR ∧ risk/causal → primary / co-primary
7. 채택 generator만 NN-donor 번들로 **11테이블 패키지** 생성 + 행 수 정확 맞춤
8. Steroid/HBOT 관련 필드 **결측률 보존**
9. **Deliverable에 채택 패키지만 보존** (중간 생성기는 workdir scratch)

### 2.2 생성기

dummy, marginal, gaussian, ctgan, `*_strat`, `cell_preserving` (F2).

### 2.3 파일 보관

`deliverable/F1|F2/`에 최종 1개만. 비채택 CSV는 최종 보존하지 않음.

- **F2**: Step2 통과 채택(selected 또는 co-primary)만 보존. 미통과 시 `FAIL_NO_ADOPTED`
- **F1**: 통과 여부와 무관하게 Step1-eligible 중 **Step2 성능 최고** 생성기 보존  
  (primary_overcome 우선 → mean \|ΔlogOR\| → risk drift → fidelity)

**Reporting (2026-08-12, no re-run):** Dummy / Cell-preserving = **양성 대조**. 과학적 primary = F1 **Marginal Pass**, F2 **비사소 Fail**. 디스크 패키지는 위 채택 규칙 산출물 그대로.

---

## 3. 임상 Rule Book

- Age [0,120], sex {0,1}; 원본 형태 프로파일 적용
- PTA −10~120, 5 dB → 4PTA 1.25 dB; **PTA 결측 Case는 합성 제외**
- Steroid (quantity, days_supply, route 등) / HBOT: **결측률 유지**
- Siegel CR/PR/SI/NI, recovered=CR|PR, 합성 후 PTA로 재계산
- Clinical gate 위반률 ≤ 2%
- History/FU = 각 1년

---

## 4. F1 vs F2

| | F1 | F2 |
|--|----|----|
| 초점 | 사전 노출 (1y lookback) | PTA·Siegel·steroid/HBOT (1y FU) |
| Control | Real 고정 | Case-only |
| Deliverable | 11테이블 Case 패키지 1식 | 동일 |

---

## 5. 구현 상태 (v5)

| 요구 | 상태 |
|------|------|
| 11테이블 동형 + 행 수 정합 | `synth_package.py` |
| ±365d, SSNHL dx index | `cohort_v5.py` |
| PTA-incomplete 제외 | `cohort_v5.py` |
| 치료 결측률 보존 | `synth_package.py` |
| Step2 채택만 deliverable | `deliver.py` |
| Step1/2 + co-primary | `evaluate_v5.py` |

---

## Version

| Date | Change |
|------|--------|
| 2026-08-04e | v6: index=first complete 4PTA; F1=F2 same PTA-trackable S |
| 2026-08-04d | F1 deliverable = best Step2 performer (pass not required); F2 still pass-only |
| 2026-08-04c | v5 design adopted + implementation modules |
