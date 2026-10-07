"""
Sim-to-Real v7.2 (D9b): shared spine + sequential cascade X->I->PTA->Siegel.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from causal_epi import fit_risk_bundle, risk_concordance
from deliver import _jsonable, adopt_and_deliver
from evaluate_v5 import (
    clinical_violation_rate,
    concordance,
    corr_distance,
    dcr_median,
    distinguish_auc,
    fit_or,
    ks_mean,
)
from synth_ensemble import (
    ASSOC_CONTROLS,
    CONSTRAINT_RANK,
    POSITIVE_CONTROLS,
    SHARED_COLS,
    WINNER_POOL,
    generate_v7_methods,
    synth_sequential,
    _snap_and_id,
    GEN_COLS,
)

warnings.filterwarnings("ignore", category=FutureWarning)

V6 = config.ROOT / "results" / "sim_to_real_v6"
V71 = config.ROOT / "results" / "sim_to_real_v7_1"
OUT = config.ROOT / "results" / "sim_to_real_v7_2"
F1_FEATURES = ["age_at_index", "sex", "hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"]
F1_EXP = ["hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"]
F2_FEATURES = ["age_at_index", "sex", "steroid_post", "hbot_post", "pta_base", "pta_fu", "recovered"]
ADJUST = ["age_at_index", "sex"]


def _copy_v6_cohort(out: Path) -> dict:
    """Reuse v6 S / analytic tables / donors; do not rebuild the 1,653 cohort."""
    cohort_dst = out / "cohort"
    cohort_dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "S_train.csv",
        "S_hold.csv",
        "synthesis_set_S.csv",
        "f1_analytic.csv",
        "f2_analytic.csv",
        "matched_pairs.csv",
        "attrition.json",
    ):
        src = V6 / "cohort" / name
        if src.exists():
            shutil.copy2(src, cohort_dst / name)
    prof_dst = out / "profile"
    prof_dst.mkdir(parents=True, exist_ok=True)
    for name in ("table_profile.json", "target_row_counts_Strain.json", "table_row_counts.csv"):
        src = V6 / "profile" / name
        if src.exists():
            shutil.copy2(src, prof_dst / name)
    attrition = json.loads((V6 / "cohort" / "attrition.json").read_text(encoding="utf-8"))
    targets = json.loads((V6 / "profile" / "target_row_counts_Strain.json").read_text(encoding="utf-8"))
    profile = json.loads((V6 / "profile" / "table_profile.json").read_text(encoding="utf-8"))
    donor_dir = V6 / "cohort" / "donor_events"
    return {
        "attrition": attrition,
        "targets": targets,
        "profile": profile,
        "donor_dir": donor_dir,
        "S_train": pd.read_csv(cohort_dst / "S_train.csv"),
        "S_hold": pd.read_csv(cohort_dst / "S_hold.csv"),
        "f1": pd.read_csv(cohort_dst / "f1_analytic.csv"),
        "f2": pd.read_csv(cohort_dst / "f2_analytic.csv"),
    }


def _step1_shared(train: pd.DataFrame, hold: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> dict:
    viol_f1 = clinical_violation_rate(syn, "F1")
    viol_f2 = clinical_violation_rate(syn, "F2")
    viol = max(viol_f1, viol_f2)
    rr_dcr = dcr_median(train, hold, cols)
    dcr = dcr_median(train, syn, cols)
    privacy_ok = (dcr >= 0.9 * rr_dcr) if rr_dcr > 0 else True
    clinical_ok = viol <= config.CLINICAL_MAX_VIOL
    return {
        "clinical_violation_rate": viol,
        "clinical_violation_F1": viol_f1,
        "clinical_violation_F2": viol_f2,
        "ks": ks_mean(train, syn, cols),
        "corr_dist": corr_distance(train, syn, cols),
        "distinguish_auc": distinguish_auc(train, syn, cols, config.SEED),
        "dcr": dcr,
        "real_real_dcr": rr_dcr,
        "real_real_ks": ks_mean(train, hold, cols),
        "step1_clinical_pass": bool(clinical_ok),
        "step1_privacy_pass": bool(privacy_ok),
        "step1_eligible": bool(clinical_ok and privacy_ok),
    }


def _assemble(case_part: pd.DataFrame, ycol: str, control_df: pd.DataFrame | None) -> pd.DataFrame:
    cp = case_part.copy()
    if control_df is None:
        return cp
    if ycol not in cp.columns:
        cp[ycol] = 1
    ctrl = control_df.copy()
    if ycol not in ctrl.columns:
        ctrl[ycol] = 0
    return pd.concat([cp, ctrl], ignore_index=True)


def _step2_family(
    family: str,
    real_cases: pd.DataFrame,
    syn_cases: pd.DataFrame,
    train_cases: pd.DataFrame,
    ycol: str,
    exposures: list[str],
    adjust: list[str],
    control_df: pd.DataFrame | None,
) -> dict:
    syn = syn_cases.copy()
    if family == "F1":
        syn[ycol] = 1
    setup_a = _assemble(real_cases, ycol, control_df)
    setup_b = _assemble(syn, ycol, control_df)
    n_tr = len(train_cases)
    hybrid = pd.concat(
        [train_cases.copy(), syn.sample(n=min(n_tr, len(syn)), random_state=config.SEED)],
        ignore_index=True,
    )
    if family == "F1":
        hybrid[ycol] = 1
    setup_c = _assemble(hybrid, ycol, control_df)

    ora = {e: fit_or(setup_a, ycol, e, adjust) for e in exposures}
    orb = {e: fit_or(setup_b, ycol, e, adjust) for e in exposures}
    orc = {e: fit_or(setup_c, ycol, e, adjust) for e in exposures}
    risk_a = {e: fit_risk_bundle(setup_a, ycol, e, adjust, family) for e in exposures}
    rb = risk_concordance(risk_a, {e: fit_risk_bundle(setup_b, ycol, e, adjust, family) for e in exposures}, exposures)
    rc = risk_concordance(risk_a, {e: fit_risk_bundle(setup_c, ycol, e, adjust, family) for e in exposures}, exposures)
    b = concordance(ora, orb, exposures, config.TAU)
    c = concordance(ora, orc, exposures, config.TAU)
    or_ok = bool(b["overcome"] and c["overcome"])
    risk_ok = bool(rb["risk_overcome"] and rc["risk_overcome"])
    if family == "F1":
        overcome = or_ok  # OR primary only
    else:
        overcome = bool(or_ok and risk_ok)  # F2: OR ∧ ATE
    gain = {}
    if family == "F2" and "pta_base" in setup_a.columns and "pta_fu" in setup_a.columns:
        for e in exposures:
            ga = _gain_contrast(setup_a, e)
            gb = _gain_contrast(setup_b, e)
            gc = _gain_contrast(setup_c, e)
            gain[e] = {
                "A": ga,
                "B": gb,
                "C": gc,
                "abs_dB": abs(gb - ga) if ga == ga and gb == gb else None,
                "abs_dC": abs(gc - ga) if ga == ga and gc == gc else None,
            }
    dB = [v["abs_dB"] for v in gain.values() if v.get("abs_dB") is not None]
    dC = [v["abs_dC"] for v in gain.values() if v.get("abs_dC") is not None]
    return {
        "OR_A": ora,
        "OR_B": orb,
        "OR_C": orc,
        "B_vs_A": b,
        "C_vs_A": c,
        "RISK_B_vs_A": rb,
        "RISK_C_vs_A": rc,
        "or_overcome": or_ok,
        "risk_causal_overcome": risk_ok,
        "overcome": overcome,
        "B_median_abs_dlogOR": b.get("median_abs_dlogOR"),
        "C_median_abs_dlogOR": c.get("median_abs_dlogOR"),
        "B_median_abs_dlogRR": rb.get("median_abs_dlogRR"),
        "C_median_abs_dlogRR": rc.get("median_abs_dlogRR"),
        "B_median_abs_dRD": rb.get("median_abs_dRD"),
        "C_median_abs_dRD": rc.get("median_abs_dRD"),
        "pta_gain_contrast": gain,
        "B_median_abs_dGain": float(np.median(dB)) if dB else None,
        "C_median_abs_dGain": float(np.median(dC)) if dC else None,
    }


def _gain_contrast(df: pd.DataFrame, icol: str) -> float:
    g = pd.to_numeric(df["pta_base"], errors="coerce") - pd.to_numeric(df["pta_fu"], errors="coerce")
    i = pd.to_numeric(df[icol], errors="coerce")
    g1, g0 = g[i == 1].mean(), g[i == 0].mean()
    if g1 != g1 or g0 != g0:
        return float("nan")
    return float(g1 - g0)


def _finite(x, default=999.0) -> float:
    try:
        v = float(x)
        return default if v != v or np.isinf(v) else v
    except Exception:
        return default


def _drift_score(f1: dict, f2: dict) -> float:
    f1_or = (_finite(f1.get("B_median_abs_dlogOR")) + _finite(f1.get("C_median_abs_dlogOR"))) / 2.0
    f2_or = (_finite(f2.get("B_median_abs_dlogOR")) + _finite(f2.get("C_median_abs_dlogOR"))) / 2.0
    f2_rr = (_finite(f2.get("B_median_abs_dlogRR")) + _finite(f2.get("C_median_abs_dlogRR"))) / 2.0
    f2_rd = (_finite(f2.get("B_median_abs_dRD")) + _finite(f2.get("C_median_abs_dRD"))) / 2.0
    return 0.5 * f1_or + 0.5 * (f2_or + f2_rr + f2_rd) / 3.0


def pick_winner(rows: list[dict], step1_map: dict) -> tuple[str | None, str]:
    """§7 + D9a: Dummy / Cell-preserving / Y-peek AC never win."""
    pool = [
        r
        for r in rows
        if r["method"] in WINNER_POOL
        and r["step1_eligible"]
        and not r["positive_control"]
        and not r.get("association_control")
    ]
    if not pool:
        return None, "No Step1-eligible non-trivial method."

    def key(r):
        s1 = step1_map[r["method"]]
        return (
            0 if (r["F1_overcome"] and r["F2_overcome"]) else 1 if r["F2_overcome"] else 2 if r["F1_overcome"] else 3,
            _finite(r["drift_score"]),
            CONSTRAINT_RANK.get(r["method"], 9),
            abs(float(s1["distinguish_auc"]) - 0.5),
            float(s1["ks"]),
            float(s1["corr_dist"]),
            r["method"],
        )

    ranked = sorted(pool, key=key)
    best = ranked[0]
    tier = (
        "F1∧F2 Overcome"
        if best["F1_overcome"] and best["F2_overcome"]
        else "F2 Overcome only"
        if best["F2_overcome"]
        else "F1 Overcome only"
        if best["F1_overcome"]
        else "no Overcome (least drift among non-trivial Step1-eligible)"
    )
    if not best["F1_overcome"] and not best["F2_overcome"]:
        return None, f"No scientific winner ({tier}). Best non-trivial drift: {best['method']}."
    return best["method"], f"Winner={best['method']} ({tier})."


def _role(name: str, is_pc: bool, is_ac: bool = False) -> str:
    if is_pc:
        return "PC"
    if is_ac:
        return "AC"
    if name in ("copula", "sequential", "istrata_copula", "ipf_pta_given_i", "partial_pta_steroid"):
        return "C"
    if "mix" in name or "ensemble" in name:
        return "S"
    return "N"


def write_v7_report(summary: dict, out: Path) -> None:
    att = summary["attrition"]
    win = summary["winner"]
    detail = summary.get("detail", {})
    lines = [
        "# Sim-to-Real v7.2 Results (D9b sequential cascade)",
        "",
        "Shared PTA-trackable S (reused from v6); **one spine per method** for F1 and F2.",
        "Step1 = data filter (clinical / DCR / fidelity). Step2 = inference/causality test.",
        "F1 primary = OR Overcome (B∧C). F2 primary = OR ∧ g-comp ATE (B∧C).",
        "Causal DGP: generate I+PTA; **recovered := Siegel**. Dummy / Cell-preserving / Y-peek AC cannot win.",
        "**C5 sequential** = structure cascade X → I|X → PTA|I (ensemble of stages, not seed bagging).",
        "Prior: v7 archive Y-peek; v7.1 ipf_pta_given_i. This run adds sequential.",
        "",
        "## Attrition (v6 cohort)",
        f"- n_synthesis_set_S: **{att.get('n_synthesis_set_S')}**",
        f"- n_S_train / n_S_hold: **{att.get('n_S_train')}** / **{att.get('n_S_hold')}**",
        f"- index: **{att.get('index_rule')}**",
        "",
        "## Scientific winner",
        f"- **{win.get('method')}**",
        f"- Rule: {win.get('reason')}",
        "",
        "## Methods",
        "",
        "| Method | Role | Step1 | viol | DCR | |AUC-0.5| | F1 OR | F2 OR | F2 ATE | F1 Ov | F2 Ov | dGain | Drift |",
        "|--------|------|-------|------|-----|----------|-------|-------|--------|-------|-------|-------|-------|",
    ]
    for r in summary["methods"]:
        s1 = detail.get(r["method"], {}).get("step1", {})
        viol = s1.get("clinical_violation_rate")
        dcr = s1.get("dcr")
        auc = s1.get("distinguish_auc")
        auc_d = abs(float(auc) - 0.5) if auc is not None else None
        drift = r.get("drift_score")
        f2d = detail.get(r["method"], {}).get("F2", {})
        dgain = f2d.get("B_median_abs_dGain")
        lines.append(
            "| `{m}` | {role} | {s1e} | {viol} | {dcr} | {aucd} | {f1or} | {f2or} | {f2ate} | {f1ov} | {f2ov} | {dgain} | {drift} |".format(
                m=r["method"],
                role=_role(r["method"], r["positive_control"], r.get("association_control", False)),
                s1e=r["step1_eligible"],
                viol=f"{viol:.4f}" if viol is not None else "-",
                dcr=f"{dcr:.4f}" if dcr is not None else "-",
                aucd=f"{auc_d:.4f}" if auc_d is not None else "-",
                f1or=r["F1_or_overcome"],
                f2or=r["F2_or_overcome"],
                f2ate=r["F2_risk_overcome"],
                f1ov=r["F1_overcome"],
                f2ov=r["F2_overcome"],
                dgain=f"{dgain:.3f}" if dgain is not None else "-",
                drift=f"{drift:.4f}" if drift is not None else "-",
            )
        )
    lines += [
        "",
        "Role: PC=positive control, AC=Y-peek association control (cannot win),",
        "C=causal structure (I+PTA, Y=Siegel; includes sequential cascade), N=solo, S=sensitivity.",
        "dGain = median |Delta PTA-gain contrast| Setup B vs A (secondary, not Overcome).",
        "",
        "## Claim hygiene",
        "- Do not claim Dummy / Cell-preserving / Y-peek AC as utility success.",
        "- Do not split F1/F2 into two different twins.",
        "- F2 ATE is observational (age/sex conditional), not RCT efficacy.",
        "- v7.2 does not rewrite v6 claims (F1 Marginal Pass / F2 non-trivial Fail).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Sim-to-Real v7.2 (D9b, sequential cascade) ===")
    print("Loading v6 cohort (no rebuild)...")
    assets = _copy_v6_cohort(OUT)
    S_train = assets["S_train"]
    S_hold = assets["S_hold"]
    f1 = assets["f1"]
    f2 = assets["f2"]

    train = S_train.copy()
    hold = S_hold.copy()
    missing = [c for c in SHARED_COLS if c not in train.columns]
    if missing:
        raise SystemExit(f"S_train missing shared cols: {missing}")

    pid_s = pd.concat([train, hold])["person_id"].astype(str)
    pid_tr = train["person_id"].astype(str)
    f1_cases = f1[f1["case"] == 1].copy()
    f1_ctrl = f1[f1["case"] == 0].copy()
    f1_real = f1_cases[f1_cases["person_id"].astype(str).isin(pid_s)].copy()
    f1_train_cases = f1_cases[f1_cases["person_id"].astype(str).isin(pid_tr)].copy()
    f2_real = f2[f2["person_id"].astype(str).isin(pid_s)].copy()
    f2_train_cases = f2[f2["person_id"].astype(str).isin(pid_tr)].copy()
    f2_exp = ["steroid_post"]
    if f2_real["hbot_post"].nunique() > 1:
        f2_exp.append("hbot_post")
    print(f"F1 real/train cases: {len(f1_real)}/{len(f1_train_cases)}; F2: {len(f2_real)}/{len(f2_train_cases)}")

    spine_dir = OUT / "spines"
    reuse = "--reuse-spines" in sys.argv
    v71_spines = V71 / "spines"
    spine_dir.mkdir(parents=True, exist_ok=True)

    if reuse and (spine_dir / "marginal.csv").exists():
        print("Reusing v7.2 spines...")
        gens = {p.stem: pd.read_csv(p) for p in spine_dir.glob("*.csv")}
    elif reuse and v71_spines.exists() and (v71_spines / "marginal.csv").exists():
        print("Reusing v7.1 spines + generating sequential...")
        gens = {p.stem: pd.read_csv(p) for p in v71_spines.glob("*.csv")}
        for name, syn in gens.items():
            syn.to_csv(spine_dir / f"{name}.csv", index=False)
    else:
        print("Generating all spines (incl. sequential)...")
        gens = generate_v7_methods(train, SHARED_COLS)
        for name, syn in gens.items():
            syn.to_csv(spine_dir / f"{name}.csv", index=False)

    if "sequential" not in gens:
        print("Generating sequential cascade X->I->PTA...")
        gcols = [c for c in GEN_COLS if c in train.columns]
        rng = np.random.default_rng(config.SEED)
        gens["sequential"] = _snap_and_id(train, synth_sequential(train, len(train), gcols, rng), "seq", SHARED_COLS)
        gens["sequential"].to_csv(spine_dir / "sequential.csv", index=False)

    print("Methods:", list(gens.keys()), flush=True)

    method_rows = []
    detail = {}
    for name, syn in gens.items():
        print(f"  Evaluating {name}...")
        s1 = _step1_shared(train, hold, syn, SHARED_COLS)
        syn_f1 = syn[F1_FEATURES].copy()
        syn_f2 = syn[[c for c in F2_FEATURES if c in syn.columns]].copy()
        st1 = _step2_family(
            "F1",
            f1_real[F1_FEATURES + ["case"]],
            syn_f1,
            f1_train_cases[F1_FEATURES + ["case"]],
            "case",
            F1_EXP,
            ADJUST,
            f1_ctrl,
        )
        st2 = _step2_family(
            "F2",
            f2_real[F2_FEATURES],
            syn_f2,
            f2_train_cases[F2_FEATURES],
            "recovered",
            f2_exp,
            ADJUST,
            None,
        )
        row = {
            "method": name,
            "positive_control": name in POSITIVE_CONTROLS,
            "association_control": name in ASSOC_CONTROLS,
            "in_winner_pool": name in WINNER_POOL,
            "step1_eligible": s1["step1_eligible"],
            "F1_or_overcome": st1["or_overcome"],
            "F2_or_overcome": st2["or_overcome"],
            "F2_risk_overcome": st2["risk_causal_overcome"],
            "F1_overcome": st1["overcome"],
            "F2_overcome": st2["overcome"],
            "drift_score": _drift_score(st1, st2),
        }
        method_rows.append(row)
        detail[name] = {"step1": s1, "F1": st1, "F2": st2, "n_synth": len(syn)}
        print(
            f"    Step1={s1['step1_eligible']} F1={st1['overcome']} F2={st2['overcome']} "
            f"(OR {st2['or_overcome']} ATE {st2['risk_causal_overcome']}) drift={row['drift_score']:.3f}"
        )

    winner, reason = pick_winner(method_rows, {m: detail[m]["step1"] for m in detail})
    print("Winner:", winner, "|", reason)
    pd.DataFrame(method_rows).to_csv(OUT / "methods.csv", index=False)

    summary = {
        "version": "v7.2",
        "plan": "V7_PLAN.md / D9b sequential",
        "attrition": assets["attrition"],
        "shared_cols": SHARED_COLS,
        "methods": method_rows,
        "detail": detail,
        "winner": {"method": winner, "reason": reason},
        "windows": {"lookback_days": config.LOOKBACK_DAYS, "followup_days": config.FOLLOWUP_DAYS},
        "index": "first_complete_4pta",
        "cohort_source": str(V6 / "cohort"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_v7_report(summary, OUT)

    if winner:
        fake_eval = {
            "step1_selected": winner,
            "step2_co_primary": winner,
            "adopted_generator": winner,
            "adoption_rule": reason,
            "step2": {
                "primary_overcome": bool(
                    next(r for r in method_rows if r["method"] == winner)["F1_overcome"]
                    and next(r for r in method_rows if r["method"] == winner)["F2_overcome"]
                ),
                "or_overcome": next(r for r in method_rows if r["method"] == winner)["F1_or_overcome"],
                "risk_causal_overcome": next(r for r in method_rows if r["method"] == winner)["F2_risk_overcome"],
            },
        }
        print("Building 11-table package for winner...")
        man = adopt_and_deliver(
            "shared",
            fake_eval,
            gens,
            assets["donor_dir"],
            assets["targets"],
            assets["profile"],
            OUT,
        )
        summary["deliverable"] = man
        (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print("No scientific winner - skipping deliverable package.")
        (OUT / "deliverable").mkdir(parents=True, exist_ok=True)
        (OUT / "deliverable" / "NO_WINNER.txt").write_text(reason + "\n", encoding="utf-8")

    print("Done ->", OUT)
    try:
        print((OUT / "REPORT.md").read_text(encoding="utf-8"))
    except UnicodeEncodeError:
        print("REPORT.md written.")


if __name__ == "__main__":
    main()
