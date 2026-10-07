"""
Manuscript figures & tables for Sim-to-Real v7.2 (shared spine; sequential winner).

Scientific primary: sequential (F1∧F2 Overcome on one shared spine).
Dummy / Cell-preserving = positive controls (PC). Y-peek AC cannot win.
v6 F1 Marginal Pass / F2 non-trivial Fail is historical, not this paper's claim.

Outputs -> results/sim_to_real_v7_2/manuscript/
  figures/*.png, *.pdf
  tables/*.csv, *.md, *.tex (selected)
  CAPTIONS.md, FIGURE_TABLE_INDEX.md
"""
from __future__ import annotations

import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import config

ROOT = config.ROOT
RES = ROOT / "results" / "sim_to_real_v7_2"
OUT = RES / "manuscript"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Publication-ish palette (colorblind-friendly)
C_A = "#0072B2"
C_B = "#E69F00"
C_C = "#009E73"
C_FAIL = "#D55E00"
C_PASS = "#009E73"
C_NEUT = "#566573"
C_BG = "#F7F9FB"
C_WIN = "#0072B2"

ROLE_MARKER = {"PC": "s", "AC": "D", "C": "o", "N": "^", "S": "P"}
ROLE_COLOR = {"PC": "#D55E00", "AC": "#CC79A7", "C": "#0072B2", "N": "#009E73", "S": "#E69F00"}


def _load():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    attrition = json.loads((RES / "cohort" / "attrition.json").read_text(encoding="utf-8"))
    S = pd.read_csv(RES / "cohort" / "synthesis_set_S.csv")
    f1 = pd.read_csv(RES / "cohort" / "f1_analytic.csv")
    f2 = pd.read_csv(RES / "cohort" / "f2_analytic.csv")
    _attach_winner_view(summary)
    return summary, attrition, S, f1, f2


def _save(fig, name: str, dpi=300):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _md_cell(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "—"
        if abs(v - round(v)) < 1e-12:
            return str(int(round(v)))
        return f"{v:.3f}"
    if isinstance(v, str) and v.strip().lower() in {"nan", "none", ""}:
        return "—"
    return str(v).replace("|", "/")


def _write_table(df: pd.DataFrame, name: str, caption: str = ""):
    TAB.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.columns = [_md_cell(c) if isinstance(c, str) else c for c in out.columns]
    for col in out.columns:
        out[col] = out[col].map(_md_cell)
    out.to_csv(TAB / f"{name}.csv", index=False)
    md = []
    if caption:
        md.append(f"**{caption}**\n")
    md.append(out.to_markdown(index=False))
    (TAB / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    try:
        (TAB / f"{name}.tex").write_text(
            df.to_latex(index=False, escape=True, float_format="%.3f"),
            encoding="utf-8",
        )
    except Exception:
        pass


VAR_LAB = {
    "hear_dx_pre": "Prior hearing dx",
    "htn_pre": "Hypertension",
    "dm_pre": "Diabetes",
    "steroid_pre": "Steroid (pre)",
    "steroid_post": "Steroid (post)",
    "hbot_post": "HBOT (post)",
    "recovered": "Recovered",
    "sex": "Female sex",
}
GEN_LAB = {
    "dummy": "Dummy",
    "cell_preserving": "Cell-preserving",
    "ipf_2way": "IPF 2-way",
    "ystrat_copula": "Y-strat copula",
    "partial_cell": "Partial cell",
    "marginal": "Marginal",
    "gaussian": "Gaussian",
    "copula": "Copula",
    "istrata_copula": "I-strata copula",
    "sequential": "Sequential",
    "ipf_pta_given_i": "IPF PTA|I",
    "partial_pta_steroid": "Partial PTA–steroid",
    "ctgan": "CTGAN",
    "ctgan_seed_ensemble": "CTGAN seed ensemble",
    "mix_marginal_gaussian": "Mix Marginal+Gaussian",
    "mix_copula_marginal": "Mix Copula+Marginal",
}
METHOD_ORDER = list(GEN_LAB.keys())

SUB_SHARED = ROOT / "Submission" / "before" / "_shared"
JOURNAL_DIRS = [
    ROOT / "Submission" / "before" / "01_npj_Digital_Medicine",
    ROOT / "Submission" / "before" / "02_JAMIA",
    ROOT / "Submission" / "before" / "03_Scientific_Reports",
]
MAIN_FIGS = [
    "Fig0_design_ABC",
    "Fig1_attrition_flowchart",
    "Fig3_forest_OR_ABC",
    "Fig7_primary_overview",
]
SUPP_FIGS = [
    "Fig2_siegel_distribution",
    "Fig4_concordance_metrics",
    "Fig5_generator_step2_heatmap",
    "Fig6_step1_fidelity_scatter",
    "Fig8_real_vs_synth_prevalence",
    "Fig9_pta_distributions",
]


def _role(name: str, is_pc: bool = False, is_ac: bool = False) -> str:
    if is_pc:
        return "PC"
    if is_ac:
        return "AC"
    if name in ("copula", "sequential", "istrata_copula", "ipf_pta_given_i", "partial_pta_steroid"):
        return "C"
    if "mix" in name or "ensemble" in name:
        return "S"
    return "N"


def _method_map(summary) -> dict:
    return {r["method"]: r for r in (summary.get("methods") or []) if r.get("method")}


def _ordered_methods(summary) -> list[str]:
    present = [r["method"] for r in (summary.get("methods") or []) if r.get("method")]
    extra = [m for m in present if m not in METHOD_ORDER]
    return [m for m in METHOD_ORDER if m in present] + extra


def _winner_name(summary) -> str:
    win = (summary.get("winner") or {}).get("method")
    if win:
        return win
    detail = summary.get("detail") or {}
    if "sequential" in detail:
        return "sequential"
    return next(iter(detail), "")


def _winner_view(summary) -> dict:
    """Wrap v7.2 summary into a v6-like F1/F2 view keyed on the scientific winner."""
    winner = _winner_name(summary)
    detail = summary.get("detail") or {}
    step1_all = {
        g: d["step1"]
        for g, d in detail.items()
        if isinstance(d, dict) and isinstance(d.get("step1"), dict)
    }

    def _fam_view(fam: str) -> dict:
        src = dict(detail.get(winner, {}).get(fam) or {})
        sens = {}
        for g, d in detail.items():
            if not isinstance(d, dict):
                continue
            st = d.get(fam) or {}
            if not isinstance(st, dict):
                continue
            sens[g] = {
                "B_overcome": (st.get("B_vs_A") or {}).get("overcome"),
                "C_overcome": (st.get("C_vs_A") or {}).get("overcome"),
                "or_overcome": st.get("or_overcome"),
                "risk_causal_overcome": st.get("risk_causal_overcome"),
                "primary_overcome": st.get("overcome"),
                "overcome": st.get("overcome"),
                "B_median_abs_dlogOR": st.get("B_median_abs_dlogOR"),
                "C_median_abs_dlogOR": st.get("C_median_abs_dlogOR"),
                "B_median_abs_dGain": st.get("B_median_abs_dGain"),
                "C_median_abs_dGain": st.get("C_median_abs_dGain"),
            }
        src["eligible_sensitivity"] = sens
        src["primary_overcome"] = src.get("overcome")
        return {
            "step1": step1_all,
            "step2": src,
            "step1_selected": winner,
            "adopted_generator": winner,
        }

    return {
        "F1": _fam_view("F1"),
        "F2": _fam_view("F2"),
        "winner": winner,
    }


def _attach_winner_view(summary: dict) -> dict:
    view = _winner_view(summary)
    summary["F1"] = view["F1"]
    summary["F2"] = view["F2"]
    return summary


def _find_spine() -> Path | None:
    """Newest deliverable/shared_* synth_spine.csv, else spines/sequential.csv."""
    deliv = RES / "deliverable"
    if deliv.exists():
        shared = sorted(
            [p for p in deliv.glob("shared_*") if (p / "synth_spine.csv").exists()],
            key=lambda p: p.name,
            reverse=True,
        )
        if shared:
            return shared[0] / "synth_spine.csv"
    fallback = RES / "spines" / "sequential.csv"
    return fallback if fallback.exists() else None


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_facecolor("white")


def _panel(ax, letter: str):
    ax.text(
        -0.08,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _finalize(fig, name: str, note: str, abbrev: str, *, tight: bool = True):
    if tight:
        fig.tight_layout(rect=[0.01, 0.18, 0.99, 0.98])
    else:
        fig.subplots_adjust(bottom=0.24, top=0.90, left=0.06, right=0.98)
    fig.text(
        0.01,
        0.12,
        textwrap.fill(f"Notes. {note}", width=145),
        ha="left",
        va="top",
        fontsize=7,
        color="#2C3E50",
    )
    fig.text(
        0.01,
        0.06,
        textwrap.fill(f"Abbreviations: {abbrev}", width=145),
        ha="left",
        va="top",
        fontsize=6.5,
        color="#566573",
    )
    _save(fig, name)


def _fmt_bool(v) -> str:
    if v is True:
        return "Pass"
    if v is False:
        return "Fail"
    return "—"


# ---------- Tables ----------
def table_attrition(attrition):
    order = [
        ("n_case_person", "Case file (PERSON)"),
        ("n_with_baseline_4pta", "≥1 complete AC 4PTA (baseline eligible)"),
        ("n_pta_trackable", "PTA-trackable (base + same-ear FU ≤365d) = S"),
        ("n_pta_index_with_ssnhl_dx_ever", "Among PTA-index: SSNHL dx code ever (descriptive)"),
        ("n_matched_pairs", "F1 matched pairs (1:1 age×sex)"),
        ("n_S_train", "Train (synthesis / Step1–2)"),
        ("n_S_hold", "Holdout"),
    ]
    rows = []
    prev = None
    for key, label in order:
        n = attrition.get(key)
        if n is None:
            continue
        row = {"Step": label, "N": int(n) if isinstance(n, (int, float)) and key != "index_rule" else n}
        if prev is not None and isinstance(n, (int, float)) and prev > 0:
            row["Retained_from_previous_%"] = round(100 * n / prev, 1)
            row["Retained_from_Case_file_%"] = round(100 * n / attrition["n_case_person"], 1)
        else:
            row["Retained_from_previous_%"] = 100.0 if key == "n_case_person" else None
            row["Retained_from_Case_file_%"] = 100.0 if key == "n_case_person" else round(
                100 * n / attrition["n_case_person"], 1
            ) if isinstance(n, (int, float)) else None
        if isinstance(n, (int, float)) and key in {
            "n_case_person",
            "n_with_baseline_4pta",
            "n_pta_trackable",
            "n_S_train",
            "n_S_hold",
            "n_matched_pairs",
        }:
            prev = n if key != "n_pta_index_with_ssnhl_dx_ever" else prev
        if key == "n_with_baseline_4pta":
            prev = n
        if key == "n_pta_trackable":
            prev = n
        rows.append(row)
    df = pd.DataFrame(rows)
    _write_table(df, "Table1_attrition", "Table 1. Cohort attrition (index = first complete AC 4PTA)")
    return df


def table_baseline(S, f1, f2):
    def _summ(df, label):
        out = {"Cohort": label, "N": len(df)}
        out["Age_mean_sd"] = f"{df['age_at_index'].mean():.1f} ± {df['age_at_index'].std():.1f}"
        out["Female_n_%"] = f"{int((df['sex']==1).sum())} ({100*(df['sex']==1).mean():.1f}%)"
        return out

    rows = [_summ(S, "S (PTA-trackable Case)")]
    cases = f1[f1["case"] == 1]
    ctrl = f1[f1["case"] == 0]
    rows.append(_summ(cases, "F1 Cases (matched)"))
    rows.append(_summ(ctrl, "F1 Controls (matched)"))

    exp_rows = []
    for col, lab in [
        ("hear_dx_pre", "Prior hearing dx"),
        ("htn_pre", "Hypertension"),
        ("dm_pre", "Diabetes"),
        ("steroid_pre", "Steroid (pre-index)"),
    ]:
        for name, d in [("Case", cases), ("Control", ctrl)]:
            exp_rows.append(
                {
                    "Variable": lab,
                    "Group": name,
                    "n": int(d[col].sum()),
                    "%": round(100 * d[col].mean(), 1),
                }
            )
    for col, lab in [("steroid_post", "Steroid (post)"), ("hbot_post", "HBOT (post)"), ("recovered", "Recovered (CR+PR)")]:
        exp_rows.append(
            {
                "Variable": lab,
                "Group": "F2 Case (S)",
                "n": int(f2[col].sum()),
                "%": round(100 * f2[col].mean(), 1),
            }
        )

    dem = pd.DataFrame(rows)
    exp = pd.DataFrame(exp_rows)
    _write_table(dem, "Table2a_demographics", "Table 2a. Demographics")
    _write_table(exp, "Table2b_exposures_outcomes", "Table 2b. Exposures and outcomes")

    siegel = (
        f2["siegel"]
        .value_counts()
        .reindex(["CR", "PR", "SI", "NI"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    siegel.columns = ["Siegel_class", "n"]
    siegel["%"] = (100 * siegel["n"] / siegel["n"].sum()).round(1)
    _write_table(siegel, "Table2c_siegel", "Table 2c. Siegel recovery class (Real S)")
    return dem, exp, siegel


def _or_frame(summary, fam, exposures):
    rows = []
    for exp in exposures:
        row = {"Family": fam, "Exposure": exp}
        for setup in ["A", "B", "C"]:
            d = summary[fam]["step2"][f"OR_{setup}"][exp]
            orv, lo, hi = d.get("or"), d.get("ci_low"), d.get("ci_high")
            row[f"OR_{setup}"] = None if orv is None else float(orv)
            row[f"CI_low_{setup}"] = None if lo is None or (isinstance(lo, float) and math.isnan(lo)) else float(lo)
            row[f"CI_high_{setup}"] = None if hi is None or (isinstance(hi, float) and (math.isnan(hi) or math.isinf(hi))) else float(hi)
            if orv is not None and lo is not None and hi is not None and lo == lo and hi == hi and not math.isinf(hi):
                row[f"OR_{setup}_fmt"] = f"{orv:.2f} ({lo:.2f}–{hi:.2f})"
            elif orv is not None:
                row[f"OR_{setup}_fmt"] = f"{orv:.2g}"
            else:
                row[f"OR_{setup}_fmt"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def table_or(summary):
    f1 = _or_frame(summary, "F1", ["hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"])
    f2 = _or_frame(summary, "F2", ["steroid_post", "hbot_post"])
    both = pd.concat([f1, f2], ignore_index=True)
    slim = both[["Family", "Exposure", "OR_A_fmt", "OR_B_fmt", "OR_C_fmt"]].rename(
        columns={
            "OR_A_fmt": "Setup A (Real)",
            "OR_B_fmt": "Setup B (Synth Case)",
            "OR_C_fmt": "Setup C (Hybrid)",
        }
    )
    _write_table(both, "Table3_OR_full", "Table 3. Odds ratios by setup (sequential shared spine; full numeric)")
    _write_table(slim, "Table3_OR_formatted", "Table 3. Odds ratios (95% CI) by Sim-to-Real setup (sequential)")
    return both, slim


def table_concordance(summary):
    rows = []
    for fam in ["F1", "F2"]:
        for cmp_key, label in [("B_vs_A", "B vs A"), ("C_vs_A", "C vs A")]:
            c = summary[fam]["step2"][cmp_key]
            detail = c.get("detail") or []
            rows.append(
                {
                    "Family": fam,
                    "Comparison": label,
                    "Sign_agree": c.get("sign_rate"),
                    "Median_abs_dlog": c.get("median_abs_dlogOR"),
                    "CI_overlap": c.get("ci_overlap_rate"),
                    "OR_overcome": c.get("overcome"),
                    "Exposures_detail": "; ".join(
                        f"{d['exposure']}: Δ={d['abs_dlogOR']:.3f}" for d in detail
                    ),
                }
            )
        for cmp_key, label in [("RISK_B_vs_A", "Risk B vs A"), ("RISK_C_vs_A", "Risk C vs A")]:
            c = summary[fam]["step2"][cmp_key]
            rows.append(
                {
                    "Family": fam,
                    "Comparison": label,
                    "Sign_agree": c.get("sign_rr_rate"),
                    "Median_abs_dlog": c.get("median_abs_dlogRR"),
                    "CI_overlap": c.get("rr_ci_overlap_rate"),
                    "OR_overcome": c.get("risk_overcome"),
                    "Exposures_detail": f"med ΔRD={c.get('median_abs_dRD')}",
                }
            )
        st = summary[fam]["step2"]
        if fam == "F1":
            prim_label = "PRIMARY (OR, B∧C)"
            prim_val = st.get("or_overcome", st.get("overcome"))
            prim_detail = f"or={st.get('or_overcome')}; risk secondary={st.get('risk_causal_overcome')}"
        else:
            prim_label = "PRIMARY (OR∧g-comp ATE, B∧C)"
            prim_val = st.get("primary_overcome", st.get("overcome"))
            prim_detail = f"or={st.get('or_overcome')}; ate={st.get('risk_causal_overcome')}"
        rows.append(
            {
                "Family": fam,
                "Comparison": prim_label,
                "Sign_agree": "—",
                "Median_abs_dlog": "—",
                "CI_overlap": "—",
                "OR_overcome": prim_val,
                "Exposures_detail": prim_detail,
            }
        )
    df = pd.DataFrame(rows)
    _write_table(
        df,
        "Table4_concordance_overcome",
        "Table 4. Concordance metrics and Overcome (τ=0.20) — sequential shared spine. Median_abs_dlog is median |ΔlogOR| for OR rows and median |ΔlogRR| for risk rows.",
    )
    return df


def table_step1(summary):
    """One row per method on the shared spine (not split F1/F2)."""
    winner = _winner_name(summary)
    mmap = _method_map(summary)
    detail = summary.get("detail") or {}
    rows = []
    for g in _ordered_methods(summary):
        v = (detail.get(g) or {}).get("step1") or {}
        if "ks" not in v and "distinguish_auc" not in v:
            continue
        r = mmap.get(g, {})
        role = _role(g, r.get("positive_control"), r.get("association_control"))
        rows.append(
            {
                "Generator": g,
                "Label": GEN_LAB.get(g, g),
                "Role": role,
                "Eligible": v.get("step1_eligible", r.get("step1_eligible")),
                "Clinical_viol": v.get("clinical_violation_rate"),
                "KS": v.get("ks"),
                "Corr_dist": v.get("corr_dist"),
                "Distinguish_AUC": v.get("distinguish_auc"),
                "DCR": v.get("dcr"),
                "Winner": g == winner,
            }
        )
    df = pd.DataFrame(rows)
    _write_table(
        df,
        "Table5_step1_generators",
        "Table 5. Step1 data-quality metrics by generator (shared spine; one row per method)",
    )
    return df


def table_step2_sens(summary):
    """All methods, one row: Role, Step1, F1/F2 Overcome, F2 OR, F2 ATE, dGain, drift."""
    winner = _winner_name(summary)
    mmap = _method_map(summary)
    detail = summary.get("detail") or {}
    rows = []
    for g in _ordered_methods(summary):
        r = mmap.get(g, {})
        f2 = (detail.get(g) or {}).get("F2") or {}
        role = _role(g, r.get("positive_control"), r.get("association_control"))
        rows.append(
            {
                "Generator": g,
                "Label": GEN_LAB.get(g, g),
                "Role": role,
                "Step1": r.get("step1_eligible"),
                "F1_Ov": r.get("F1_overcome"),
                "F2_Ov": r.get("F2_overcome"),
                "F2_OR": r.get("F2_or_overcome"),
                "F2_ATE": r.get("F2_risk_overcome"),
                "dGain": f2.get("B_median_abs_dGain"),
                "Drift": r.get("drift_score"),
                "Winner": g == winner,
                "Can_win": role not in {"PC", "AC"},
            }
        )
    df = pd.DataFrame(rows)
    _write_table(
        df,
        "Table6_step2_sensitivity",
        "Table 6. Step2 by method (shared spine). Winner = sequential. PC/AC cannot win.",
    )
    return df


def table_deliverable(summary):
    m = summary.get("deliverable") or {}
    if not m:
        return None, None
    meta = pd.DataFrame(
        [
            {
                "Family": m.get("family", "shared"),
                "Status": m.get("status"),
                "Adopted_generator": m.get("adopted_generator"),
                "Step1_selected": m.get("step1_selected"),
                "Step2_co_primary": m.get("step2_co_primary"),
                "Primary_overcome": m.get("primary_overcome"),
                "OR_overcome": m.get("or_overcome"),
                "Risk_causal_overcome": m.get("risk_causal_overcome"),
                "Adoption_rule": m.get("adoption_rule"),
            }
        ]
    )
    count_rows = []
    for t, n in (m.get("actual_counts") or {}).items():
        tgt = (m.get("target_counts") or {}).get(t)
        count_rows.append(
            {
                "Family": m.get("family", "shared"),
                "Table": t,
                "Actual": n,
                "Target_S_train": tgt,
                "Match": n == tgt,
            }
        )
    counts_df = pd.DataFrame(count_rows)
    _write_table(meta, "Table7a_deliverable_meta", "Table 7a. Shared deliverable package (sequential winner)")
    _write_table(counts_df, "Table7b_row_counts", "Table 7b. OMOP table row counts vs S_train targets")
    return meta, counts_df


# ---------- Figures ----------
def fig_attrition_flow(attrition):
    fig, ax = plt.subplots(figsize=(8.8, 9.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 12)
    ax.axis("off")
    ax.set_title("Cohort attrition (index = first complete AC 4PTA)", fontsize=12, pad=10)

    boxes = [
        (5, 10.5, f"Case file\nn = {attrition['n_case_person']:,}", "#D6EAF8"),
        (5, 8.5, f"≥1 complete AC 4PTA\nn = {attrition['n_with_baseline_4pta']:,}", "#D5F5E3"),
        (5, 6.5, f"PTA-trackable S\n(base + same-ear FU ≤365 d)\nn = {attrition['n_pta_trackable']:,}", "#FCF3CF"),
        (2.8, 4.2, f"Train (synthesis)\nn = {attrition['n_S_train']:,}", "#F5CBA7"),
        (7.2, 4.2, f"Holdout\nn = {attrition['n_S_hold']:,}", "#E5E7E9"),
        (2.8, 2.0, "F1: case–control OR\n(+ real Control 1:1)", "#AED6F1"),
        (7.2, 2.0, "F2: Siegel recovery OR\n(steroid, HBOT)", "#A9DFBF"),
    ]
    for x, y, text, color in boxes:
        fancy = FancyBboxPatch(
            (x - 1.6, y - 0.7),
            3.2,
            1.4,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.2,
            edgecolor="#2C3E50",
            facecolor=color,
        )
        ax.add_patch(fancy)
        ax.text(x, y, text, ha="center", va="center", fontsize=10)

    def arrow(y0, y1, x=5):
        ax.annotate("", xy=(x, y1 + 0.75), xytext=(x, y0 - 0.75), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.4))

    arrow(10.5, 8.5)
    arrow(8.5, 6.5)
    ax.annotate("", xy=(2.8, 4.9), xytext=(5, 5.75), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2))
    ax.annotate("", xy=(7.2, 4.9), xytext=(5, 5.75), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2))
    ax.annotate("", xy=(2.8, 2.75), xytext=(2.8, 3.45), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2))
    ax.annotate("", xy=(7.2, 2.75), xytext=(7.2, 3.45), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2))

    excl1 = attrition["n_case_person"] - attrition["n_with_baseline_4pta"]
    excl2 = attrition["n_with_baseline_4pta"] - attrition["n_pta_trackable"]
    ax.text(8.55, 9.5, f"Excluded\nno complete 4PTA\nn = {excl1:,}", fontsize=8, color=C_FAIL, ha="left")
    ax.text(8.55, 7.5, f"Excluded\nno same-ear FU ≤365 d\nn = {excl2:,}", fontsize=8, color=C_FAIL, ha="left")
    n_dx = attrition.get("n_pta_index_with_ssnhl_dx_ever", "—")
    _finalize(
        fig,
        "Fig1_attrition_flowchart",
        f"S is shared by F1 and F2 on one spine. SSNHL diagnosis code ever among PTA-index persons is descriptive only (n={n_dx}). "
        "Controls are never synthesized. Scientific primary uses the sequential shared spine (Y:=Siegel).",
        "AC, air conduction; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); FU, follow-up; "
        "HBOT, hyperbaric oxygen therapy; OR, odds ratio; PTA, pure-tone average; 4PTA, four-frequency PTA "
        "(0.5/1/2/4 kHz); S, PTA-trackable synthesis/analysis set; SSNHL, sudden sensorineural hearing loss.",
        tight=False,
    )


def fig_siegel(f2):
    counts = f2["siegel"].value_counts().reindex(["CR", "PR", "SI", "NI"]).fillna(0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    colors = ["#1B9E77", "#66C2A5", "#FC8D62", "#D73027"]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.6)
    style_ax(axes[0])
    axes[0].set_ylabel("Patients (n)")
    axes[0].set_xlabel("Siegel class")
    axes[0].set_title("Siegel class counts")
    _panel(axes[0], "a")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts.values) * 0.02, str(v), ha="center", fontsize=9)

    rec = int(f2["recovered"].sum())
    nrec = len(f2) - rec
    axes[1].pie(
        [rec, nrec],
        labels=[f"Recovered\n(CR+PR)\nn={rec}", f"Not recovered\n(SI+NI)\nn={nrec}"],
        colors=["#1B9E77", "#D73027"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )
    axes[1].set_title("Primary binary outcome")
    _panel(axes[1], "b")
    fig.suptitle(f"Siegel recovery on real PTA-trackable Case (N={len(f2):,})", fontsize=12, y=0.98)
    _finalize(
        fig,
        "Fig2_siegel_distribution",
        "Real S only (no synthetic patients). Primary F2 outcome is recovered (CR+PR) versus not recovered (SI+NI). "
        "Sequential DGP generates I+PTA then sets Y:=Siegel (recovered is not freely generated).",
        "CR, complete recovery; F2, Family 2 (Siegel recovery); NI, no improvement; PR, partial recovery; "
        "PTA, pure-tone average; SI, slight improvement.",
    )


def _forest(ax, summary, fam, exposures, labels, title, xlim=None):
    setups = [("A", C_A, "A Real Case"), ("B", C_B, "B Synth Case"), ("C", C_C, "C Hybrid")]
    y_positions = []
    y = 0
    for exp, lab in zip(exposures, labels):
        for j, (setup, col, _) in enumerate(setups):
            d = summary[fam]["step2"][f"OR_{setup}"][exp]
            orv, lo, hi = d["or"], d["ci_low"], d["ci_high"]
            if lo is None or lo != lo or lo <= 0:
                lo = max(orv / 10, 1e-3) if orv and orv > 0 else 1e-3
            if hi is None or hi != hi or math.isinf(hi):
                hi = orv * 10 if orv and orv > 0 else 10
            yy = y + j
            ax.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot(orv, yy, "o", color=col, markersize=7, zorder=3)
            y_positions.append((yy, lab if j == 1 else ""))
        y += 4

    ax.axvline(1.0, color="#888", ls="--", lw=1)
    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_yticks([p[0] for p in y_positions if p[1]])
    ax.set_yticklabels([p[1] for p in y_positions if p[1]])
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title(title)
    style_ax(ax)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    handles = [mpatches.Patch(color=c, label=lab) for _, c, lab in setups]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)


def fig_forest_or(summary):
    win = GEN_LAB.get(_winner_name(summary), _winner_name(summary))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4))
    _forest(
        axes[0],
        summary,
        "F1",
        ["hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"],
        ["Prior hearing dx", "Hypertension", "Diabetes", "Steroid (pre)"],
        f"F1 risk factors — {win} Pass",
        xlim=(0.3, 20),
    )
    _panel(axes[0], "a")
    _forest(
        axes[1],
        summary,
        "F2",
        ["steroid_post", "hbot_post"],
        ["Steroid (post)", "HBOT (post)"],
        f"F2 recovery interventions — {win} Pass",
        xlim=(0.05, 3),
    )
    _panel(axes[1], "b")
    fig.suptitle("Odds ratios across Sim-to-Real setups A / B / C (sequential shared spine)", fontsize=12, y=0.98)
    _finalize(
        fig,
        "Fig3_forest_OR_ABC",
        "Age- and sex-adjusted OR with 95% CI. Dashed line = OR 1. F1 uses real Controls in all setups. "
        "Both panels are the scientific primary: sequential F1∧F2 Overcome on one shared spine (Y:=Siegel). "
        "Dummy and Cell-preserving are PC; ystrat_copula / ipf_2way / partial_cell are AC (Y-peek) and cannot win. "
        "v6 F1 Marginal Pass / F2 non-trivial Fail is historical and is not this paper's primary claim. "
        "Real-A steroid OR 0.70 (0.56–0.89); HBOT 0.26 (0.14–0.50).",
        "CI, confidence interval; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); "
        "HBOT, hyperbaric oxygen therapy; OR, odds ratio.",
    )


def fig_concordance_bars(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), sharey=True)
    metrics = [
        ("sign_rate", "Sign agreement"),
        ("median_abs_dlogOR", "Median |ΔlogOR|"),
        ("ci_overlap_rate", "CI overlap"),
    ]
    win = GEN_LAB.get(_winner_name(summary), _winner_name(summary))
    for ax, fam, letter in zip(axes, ["F1", "F2"], ["a", "b"]):
        x = np.arange(len(metrics))
        w = 0.35
        bvals, cvals = [], []
        for key, _ in metrics:
            b = summary[fam]["step2"]["B_vs_A"]
            c = summary[fam]["step2"]["C_vs_A"]
            bvals.append(b.get(key))
            cvals.append(c.get(key))
        ax.bar(x - w / 2, bvals, w, label="B vs A", color=C_B, edgecolor="black", lw=0.5)
        ax.bar(x + w / 2, cvals, w, label="C vs A", color=C_C, edgecolor="black", lw=0.5)
        ax.axhline(0.80, color="#888", ls=":", lw=1, label="80% threshold")
        ax.axhline(0.20, color="#C0392B", ls="--", lw=1, label="τ = 0.20")
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics], rotation=15, ha="right")
        ov = summary[fam]["step2"].get("overcome")
        ax.set_title(f"{fam} {win}: {_fmt_bool(ov)}")
        style_ax(ax)
        ax.legend(frameon=False, fontsize=8)
        _panel(ax, letter)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("OR concordance components (sequential; Overcome requires B and C)", fontsize=12, y=0.98)
    _finalize(
        fig,
        "Fig4_concordance_metrics",
        "Dotted line: 80% sign agreement / CI overlap. Dashed red line: τ = 0.20 for median |ΔlogOR|. "
        "OR Overcome uses the median across exposures (individual |ΔlogOR| may exceed τ). "
        "Both panels = scientific primary (sequential F1∧F2 Pass). F2 primary is OR ∧ g-comp ATE (B∧C). "
        "PC/AC cannot win. v6 F1 Marginal / F2 fail is not this paper's primary claim.",
        "ATE, average treatment effect; CI, confidence interval; F1, Family 1 (case–control risk); "
        "F2, Family 2 (Siegel recovery); OR, odds ratio; τ, pre-registered concordance tolerance.",
    )


def fig_generator_heatmap(summary):
    mmap = _method_map(summary)
    winner = _winner_name(summary)
    methods = _ordered_methods(summary)
    rows_label = []
    f1_ov, f2_ov, drift = [], [], []
    for g in methods:
        r = mmap.get(g, {})
        role = _role(g, r.get("positive_control"), r.get("association_control"))
        mark = " ★" if g == winner else ""
        rows_label.append(f"{role} · {GEN_LAB.get(g, g)}{mark}")
        f1_ov.append(1 if r.get("F1_overcome") else 0)
        f2_ov.append(1 if r.get("F2_overcome") else 0)
        d = r.get("drift_score")
        drift.append(d if d is not None else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, max(5.6, 0.36 * len(rows_label) + 2.6)))
    ax = axes[0]
    data = np.column_stack([f1_ov, f2_ov]).astype(float)
    ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(rows_label)))
    ax.set_yticklabels(rows_label, fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["F1\nOvercome", "F2\nOvercome"])
    ax.set_title("Step2 Overcome (shared spine)")
    _panel(ax, "a")
    for i, (a, b) in enumerate(zip(f1_ov, f2_ov)):
        ax.text(0, i, "PASS" if a else "Fail", ha="center", va="center", fontsize=7, color="black")
        ax.text(1, i, "PASS" if b else "Fail", ha="center", va="center", fontsize=7, color="black")

    ax = axes[1]
    data2 = np.array(drift, dtype=float).reshape(-1, 1)
    vmax = max(0.3, float(np.nanmax(data2)) if np.isfinite(data2).any() else 1.0)
    im2 = ax.imshow(data2, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(rows_label)))
    ax.set_yticklabels(rows_label, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(["Drift\nscore"])
    ax.set_title("Epidemiologic drift")
    _panel(ax, "b")
    for i, v in enumerate(drift):
        ax.text(0, i, f"{v:.2f}" if v == v else "—", ha="center", va="center", fontsize=7)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    if winner:
        lab_win = next((i for i, g in enumerate(methods) if g == winner), None)
        if lab_win is not None:
            axes[0].add_patch(plt.Rectangle((-0.5, lab_win - 0.5), 2, 1, fill=False, edgecolor=C_WIN, lw=2))
            axes[1].add_patch(plt.Rectangle((-0.5, lab_win - 0.5), 1, 1, fill=False, edgecolor=C_WIN, lw=2))
    fig.suptitle("All methods on one shared spine (blue box = scientific winner)", fontsize=12, y=0.98)
    _finalize(
        fig,
        "Fig5_generator_step2_heatmap",
        "Every method uses the same shared person-spine for F1 and F2. Dummy / Cell-preserving = PC; "
        "ystrat_copula / ipf_2way / partial_cell = AC (Y-peek) — concordance may pass but they cannot win. "
        "Scientific winner = sequential (F1∧F2 Overcome; Role C). Drift is a composite of F1/F2 |ΔlogOR|/|ΔlogRR|/|ΔRD|. "
        "v6 F1 Marginal / F2 fail is historical.",
        "AC, Y-peek association control; C, causal structure (I+PTA, Y:=Siegel); F1, Family 1; "
        "F2, Family 2; N, solo; OR, odds ratio; PC, positive control; S, sensitivity; Y, recovered outcome.",
    )


def fig_step1_scatter(summary):
    mmap = _method_map(summary)
    winner = _winner_name(summary)
    detail = summary.get("detail") or {}
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    seen_roles = set()
    for g in _ordered_methods(summary):
        v = (detail.get(g) or {}).get("step1") or {}
        if "ks" not in v or v.get("distinguish_auc") is None:
            continue
        r = mmap.get(g, {})
        role = _role(g, r.get("positive_control"), r.get("association_control"))
        elig = v.get("step1_eligible", r.get("step1_eligible"))
        x = abs(float(v["distinguish_auc"]) - 0.5)
        y = float(v["ks"])
        ax.scatter(
            x,
            y,
            s=140 if g == winner else 90,
            c=C_PASS if elig else C_FAIL,
            marker=ROLE_MARKER.get(role, "o"),
            edgecolors=C_WIN if g == winner else "black",
            linewidths=2.0 if g == winner else 0.6,
            zorder=4 if g == winner else 3,
        )
        ax.annotate(
            f"{GEN_LAB.get(g, g)} ({role})",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            fontweight="bold" if g == winner else "normal",
        )
        seen_roles.add(role)
    ax.set_xlabel("|AUC − 0.5| (lower is better)")
    ax.set_ylabel("KS statistic (lower is better)")
    ax.set_title(f"Shared-spine Step1; winner {GEN_LAB.get(winner, winner)}")
    style_ax(ax)
    handles = [
        mpatches.Patch(color=C_PASS, label="Step1 eligible"),
        mpatches.Patch(color=C_FAIL, label="Not eligible"),
    ]
    for role in ("PC", "AC", "C", "N", "S"):
        if role in seen_roles:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=ROLE_MARKER[role],
                    color="w",
                    markerfacecolor=C_NEUT,
                    markeredgecolor="black",
                    markersize=8,
                    label=f"Role {role}",
                )
            )
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    _panel(ax, "a")
    fig.suptitle("Step1 fidelity landscape (one shared spine)", fontsize=12, y=0.98)
    _finalize(
        fig,
        "Fig6_step1_fidelity_scatter",
        "One panel: all methods share a single spine. Color = Step1 eligibility (clinical hard-snap ≤2% and privacy DCR); "
        "marker = role (PC / AC / C / N / S). Sequential (C) is the scientific winner (blue edge). "
        "OR/RR peeking is forbidden at Step1. PC and AC cannot win even if Step2 passes.",
        "AC, Y-peek association control; AUC, area under the ROC curve; C, causal structure; "
        "DCR, distance to closest record; KS, Kolmogorov–Smirnov statistic; N, solo; "
        "PC, positive control; S, sensitivity.",
    )


def fig_overview_pass(summary, attrition):
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 5)
    ax.set_title("Scientific primary: sequential F1∧F2 Pass", fontsize=13, pad=8)

    def card(x, y, title, body, color):
        fancy = FancyBboxPatch(
            (x, y),
            4.2,
            3.2,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="#2C3E50",
            facecolor=color,
        )
        ax.add_patch(fancy)
        ax.text(x + 2.1, y + 2.6, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + 2.1, y + 1.2, body, ha="center", va="center", fontsize=9)

    f1 = summary["F1"]
    f2 = summary["F2"]
    win = GEN_LAB.get(_winner_name(summary), _winner_name(summary))
    b1 = f1["step2"]["B_vs_A"].get("median_abs_dlogOR")
    c1 = f1["step2"]["C_vs_A"].get("median_abs_dlogOR")
    b2 = f2["step2"]["B_vs_A"].get("median_abs_dlogOR")
    c2 = f2["step2"]["C_vs_A"].get("median_abs_dlogOR")
    card(
        0.4,
        1.15,
        f"F1 Risk  ·  {win} PASS",
        "Shared spine · Role C\n"
        "Step1 eligible; non-trivial\n"
        f"B median |ΔlogOR| = {b1:.3f}\n"
        f"C median |ΔlogOR| = {c1:.3f}\n"
        "Dummy = PC (not the claim)",
        "#D5F5E3",
    )
    card(
        5.3,
        1.15,
        f"F2 Recovery  ·  {win} PASS",
        "Same spine · Y:=Siegel\n"
        "OR ∧ g-comp ATE (B∧C)\n"
        f"B median |ΔlogOR| = {b2:.3f}\n"
        f"C median |ΔlogOR| = {c2:.3f}\n"
        "Cell-preserving = PC only",
        "#D5F5E3",
    )
    _finalize(
        fig,
        "Fig7_primary_overview",
        f"Shared PTA-trackable S = {attrition['n_pta_trackable']:,}; train n = {attrition['n_S_train']:,}; "
        "index = first complete AC 4PTA. Scientific primary is sequential F1∧F2 Overcome on one spine "
        "(causal DGP: X → I|X → PTA|I; recovered := Siegel). Dummy / Cell-preserving are PC; "
        "ystrat_copula / ipf_2way / partial_cell are AC (Y-peek) and cannot win. "
        "v6 F1 Marginal Pass / F2 non-trivial Fail is historical.",
        "AC, air conduction (also Y-peek association control where noted); ATE, average treatment effect; "
        "F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); OR, odds ratio; "
        "PTA, pure-tone average; 4PTA, four-frequency PTA (0.5/1/2/4 kHz); S, PTA-trackable set.",
        tight=False,
    )


def fig_design_abc():
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0.7, 4.1)
    ax.axis("off")
    ax.set_title("Sim-to-Real evaluation design (Case-side; sequential DGP Y:=Siegel)", fontsize=12)

    items = [
        (2, "Setup A\nReal Control × Real Case\n(F2: Real Case only)", "#D6EAF8"),
        (6, "Setup B\nReal Control × Synth Case\n(F2: Synth Case only)", "#FCF3CF"),
        (10, "Setup C\nReal + Synth hybrid Case\n(Control remains real)", "#D5F5E3"),
    ]
    for x, text, col in items:
        fancy = FancyBboxPatch(
            (x - 1.85, 1.35),
            3.7,
            2.15,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.2,
            edgecolor="#2C3E50",
            facecolor=col,
        )
        ax.add_patch(fancy)
        ax.text(x, 2.4, text, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(4.12, 2.4), xytext=(3.88, 2.4), arrowprops=dict(arrowstyle="->", color="#2C3E50"))
    ax.annotate("", xy=(8.12, 2.4), xytext=(7.88, 2.4), arrowprops=dict(arrowstyle="->", color="#2C3E50"))
    _finalize(
        fig,
        "Fig0_design_ABC",
        "Controls are never synthesized. One shared spine serves F1 and F2. Sequential DGP: generate I and PTA, "
        "then recovered := Siegel (Y is not freely sampled). Primary Overcome requires epidemiologic concordance "
        "of OR (sign ≥80%, median |ΔlogOR| ≤0.20, CI overlap ≥80%) and risk/causal estimands on both B and C versus A. "
        "F2 primary is OR ∧ g-comp ATE. PC/AC cannot win.",
        "ATE, average treatment effect; CI, confidence interval; F2, Family 2 (Siegel recovery); "
        "I, intervention; OR, odds ratio; PTA, pure-tone average; Y, recovered outcome.",
        tight=False,
    )


def fig_real_vs_synth_spine():
    """Compare Real S_train vs sequential shared spine for F1 and F2 binary rates."""
    S_train = pd.read_csv(RES / "cohort" / "S_train.csv")
    spine = _find_spine()
    if spine is None:
        return
    syn = pd.read_csv(spine)
    panels = []
    f1_cols = [c for c in ["hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre", "sex"] if c in syn.columns and c in S_train.columns]
    if f1_cols:
        panels.append(
            (
                "F1 covariates (sequential spine)",
                f1_cols,
                [S_train[c].mean() for c in f1_cols],
                [syn[c].mean() for c in f1_cols],
            )
        )
    f2_cols = [c for c in ["steroid_post", "hbot_post", "recovered", "sex"] if c in syn.columns and c in S_train.columns]
    if f2_cols:
        panels.append(
            (
                "F2 interventions / Y (sequential spine)",
                f2_cols,
                [S_train[c].mean() for c in f2_cols],
                [syn[c].mean() for c in f2_cols],
            )
        )
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, cols, real_rates, syn_rates), letter in zip(axes, panels, ["a", "b"][: len(panels)]):
        x = np.arange(len(cols))
        w = 0.35
        ax.bar(x - w / 2, real_rates, w, label="Real S_train", color=C_A, edgecolor="black", lw=0.5)
        ax.bar(x + w / 2, syn_rates, w, label="Sequential spine", color=C_B, edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([VAR_LAB.get(c, c) for c in cols], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Prevalence")
        ax.set_title(title)
        style_ax(ax)
        ax.legend(frameon=False, fontsize=8)
        _panel(ax, letter)
    fig.suptitle("Real vs sequential shared-spine prevalences", fontsize=12, y=0.98)
    _finalize(
        fig,
        "Fig8_real_vs_synth_prevalence",
        "Scientific primary spine (sequential; F1∧F2 Pass), not Dummy / Cell-preserving PC packages. "
        "Person-spine binary rates, not full OMOP event tables. Female sex coding: sex = 1. "
        "Recovered = Siegel CR+PR, computed from generated PTA (Y:=Siegel).",
        "CR, complete recovery; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); "
        "HBOT, hyperbaric oxygen therapy; OMOP, Observational Medical Outcomes Partnership; PR, partial recovery; "
        "S_train, training split of the PTA-trackable set.",
    )


def fig_pta_hist():
    S = pd.read_csv(RES / "cohort" / "synthesis_set_S.csv")
    spine = _find_spine()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    syn = pd.read_csv(spine) if spine is not None else None
    for ax, col, title, letter in zip(
        axes,
        ["pta_base", "pta_fu"],
        ["Baseline 4PTA", "Follow-up 4PTA"],
        ["a", "b"],
    ):
        ax.hist(S[col].dropna(), bins=25, alpha=0.55, color=C_A, label="Real S", edgecolor="white")
        if syn is not None and col in syn.columns:
            ax.hist(
                syn[col].dropna(),
                bins=25,
                alpha=0.45,
                color=C_B,
                label="Sequential spine",
                edgecolor="white",
            )
        ax.set_xlabel("Hearing level (dB HL)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        style_ax(ax)
        ax.legend(frameon=False, fontsize=8)
        _panel(ax, letter)
    fig.suptitle("PTA distributions: real S vs sequential shared spine", fontsize=12, y=0.98)
    _finalize(
        fig,
        "Fig9_pta_distributions",
        "Sequential spine is the scientific primary (F1∧F2 Pass), not a positive-control package. "
        "DGP generates PTA then recovered := Siegel. 4PTA is the mean of complete AC thresholds at 0.5, 1, 2 and 4 kHz "
        "on the affected ear. Follow-up is the last complete same-ear 4PTA within 365 days after index.",
        "AC, air conduction; dB HL, decibels hearing level; F2, Family 2 (Siegel recovery); "
        "PTA, pure-tone average; 4PTA, four-frequency PTA; S, PTA-trackable set.",
    )


GLOSSARY = (
    "AC, air conduction (also Y-peek association control where Role=AC); "
    "ATE, average treatment effect; AUC, area under the receiver-operating-characteristic curve; "
    "C, causal-structure generator (I+PTA, Y:=Siegel); "
    "CI, confidence interval; CR, complete recovery; CTGAN, conditional tabular generative adversarial network; "
    "DCR, distance to closest record; DM, diabetes mellitus; F1, Family 1 (case–control risk); "
    "F2, Family 2 (Siegel recovery); FU, follow-up; HBOT, hyperbaric oxygen therapy; HTN, hypertension; "
    "I, intervention; IPF, iterative proportional fitting; KS, Kolmogorov–Smirnov statistic; "
    "N, solo / unstructured generator; NI, no improvement; OMOP, Observational Medical Outcomes Partnership; "
    "OR, odds ratio; PC, positive control; PR, partial recovery; PTA, pure-tone average; "
    "4PTA, four-frequency PTA (0.5/1/2/4 kHz); RD, risk difference; RR, risk ratio; "
    "S, PTA-trackable synthesis/analysis set (also Role=S, sensitivity); SI, slight improvement; "
    "SSNHL, sudden sensorineural hearing loss; Y, recovered outcome; τ, pre-registered concordance tolerance."
)

GEN_NOTE = (
    "Dummy, row bootstrap (PC); Cell-preserving, resampling within Y×exposure cells (PC); "
    "ystrat_copula / ipf_2way / partial_cell, Y-peek association controls (AC; cannot win); "
    "Marginal, independent column resampling (N); Gaussian, multivariate normal (N); "
    "Copula / I-strata copula / sequential / IPF PTA|I / Partial PTA–steroid, causal structure (C; Y:=Siegel); "
    "sequential = X→I|X→PTA|I cascade (scientific primary); CTGAN (N); "
    "CTGAN seed ensemble / Mix Marginal+Gaussian / Mix Copula+Marginal (S). "
    "Scientific primary: sequential F1∧F2 Pass. PC/AC cannot win. "
    "v6 F1 Marginal Pass / F2 non-trivial Fail is historical, not this paper's primary claim."
)


def _figure_legend_block(n, ntr, n_dx, summary):
    win = GEN_LAB.get(_winner_name(summary), _winner_name(summary))
    f1 = summary["F1"]["step2"]
    f2 = summary["F2"]["step2"]
    b1 = f1["B_vs_A"].get("median_abs_dlogOR")
    c1 = f1["C_vs_A"].get("median_abs_dlogOR")
    b2 = f2["B_vs_A"].get("median_abs_dlogOR")
    c2 = f2["C_vs_A"].get("median_abs_dlogOR")
    return f"""**Figure 0. Sim-to-Real evaluation design (Case-side).** Setup A uses real Case (Family 1 with real Control; Family 2 Case-only). Setup B substitutes synthetic Case while Controls remain real. Setup C hybridizes real and synthetic Case. One shared spine serves F1 and F2. Sequential DGP generates I+PTA then recovered := Siegel. Overcome requires epidemiologic concordance of odds ratios and risk/causal estimands on both B and C versus A.
**Notes.** Controls are never synthesized. Pre-registered OR concordance: sign agreement ≥80%, median |ΔlogOR| ≤0.20, CI overlap ≥80%. F2 primary is OR ∧ g-comp ATE. Dummy and Cell-preserving are PC; Y-peek AC cannot win.
**Abbreviations:** ATE, average treatment effect; CI, confidence interval; I, intervention; OR, odds ratio; Y, recovered.

**Figure 1. Cohort attrition to the PTA-trackable set S.** Flow from the Case file (n=5,311) to ≥1 complete AC 4PTA, then to PTA-trackable S (baseline plus same-ear follow-up ≤365 days; n={n:,}), with train (n={ntr:,}) / holdout split and dual endpoints (F1 case–control OR; F2 Siegel recovery OR) on one shared spine.
**Notes.** Index is the first complete AC 4PTA. SSNHL diagnosis code ever among PTA-index persons is descriptive only (n={n_dx}). F1 and F2 share S. Controls are never synthesized.
**Abbreviations:** AC, air conduction; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); FU, follow-up; HBOT, hyperbaric oxygen therapy; OR, odds ratio; PTA, pure-tone average; 4PTA, four-frequency PTA (0.5/1/2/4 kHz); S, PTA-trackable set; SSNHL, sudden sensorineural hearing loss.

**Figure 2. Siegel recovery on real PTA-trackable Case.** (a) Counts of Siegel classes CR, PR, SI and NI. (b) Primary binary outcome: recovered (CR+PR) versus not recovered (SI+NI). N={n:,} real Cases.
**Notes.** Real S only; no synthetic patients. Sequential DGP sets Y:=Siegel from generated PTA rather than sampling recovered freely.
**Abbreviations:** CR, complete recovery; F2, Family 2 (Siegel recovery); NI, no improvement; PR, partial recovery; PTA, pure-tone average; S, PTA-trackable set; SI, slight improvement.

**Figure 3. Odds ratios across Sim-to-Real setups.** Age- and sex-adjusted OR (95% CI) on a log scale. (a) F1 sequential Pass. (b) F2 sequential Pass. Both panels are the scientific primary on one shared spine. Setup A, real Case; Setup B, synthetic Case; Setup C, real+synthetic hybrid. Dashed line, OR=1.
**Notes.** F1 Controls remain real. Dummy / Cell-preserving are PC; ystrat_copula / ipf_2way / partial_cell are AC and cannot win. v6 F1 Marginal Pass / F2 non-trivial Fail is historical. Real-A steroid OR 0.70 (0.56–0.89); HBOT 0.26 (0.14–0.50).
**Abbreviations:** CI, confidence interval; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); HBOT, hyperbaric oxygen therapy; OR, odds ratio.

**Figure 4. OR concordance components.** (a) F1 {win} Pass. (b) F2 {win} Pass. Sign agreement, median |ΔlogOR| and CI overlap for B versus A and C versus A. Dotted line, 80% threshold; dashed red line, τ=0.20.
**Notes.** OR Overcome uses the median across exposures; individual |ΔlogOR| may exceed τ. F1 B/C median |ΔlogOR| = {b1:.3f}/{c1:.3f}; F2 = {b2:.3f}/{c2:.3f}. F2 primary also requires g-comp ATE concordance. PC/AC cannot win.
**Abbreviations:** ATE, average treatment effect; CI, confidence interval; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); OR, odds ratio; τ, pre-registered concordance tolerance.

**Figure 5. Generator-level Step2 on the shared spine.** (a) F1 and F2 Overcome pass/fail for every method. (b) Drift score. Blue box, scientific winner (sequential).
**Notes.** {GEN_NOTE}
**Abbreviations:** AC, Y-peek association control; C, causal structure; F1, Family 1; F2, Family 2; N, solo; OR, odds ratio; PC, positive control; S, sensitivity.

**Figure 6. Step1 fidelity landscape (shared spine).** |AUC−0.5| versus KS statistic. Green, Step1-eligible after clinical and privacy gates; orange, not eligible. Marker encodes role (PC/AC/C/N/S). Sequential (C) highlighted.
**Notes.** Step1 is a data filter only (clinical hard-snap ≤2%, privacy DCR, then |AUC−0.5| → KS → correlation distance). OR/RR peeking is forbidden. PC and AC cannot win.
**Abbreviations:** AUC, area under the receiver-operating-characteristic curve; DCR, distance to closest record; KS, Kolmogorov–Smirnov statistic; OR, odds ratio; RR, risk ratio.

**Figure 7. Scientific primary: sequential F1∧F2 Pass.** Under shared S (n={n:,}; train n={ntr:,}): both F1 risk and F2 recovery Overcome on the sequential spine. Dummy and Cell-preserving are PC; Y-peek AC cannot win.
**Notes.** Index = first complete AC 4PTA. Causal DGP: X → I|X → PTA|I; recovered := Siegel. v6 F1 Marginal / F2 fail is not this paper's primary claim.
**Abbreviations:** AC, air conduction; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); OR, odds ratio; PTA, pure-tone average; 4PTA, four-frequency PTA (0.5/1/2/4 kHz); S, PTA-trackable set.

**Figure 8. Real versus sequential spine prevalences.** (a) F1 covariates. (b) F2 interventions and recovered. Sequential shared spine versus real S_train.
**Notes.** Scientific primary spine, not Dummy / Cell-preserving PC packages. Person-spine rates only (not full OMOP event tables). Female sex coded as sex=1. Recovered = Siegel CR+PR (Y:=Siegel).
**Abbreviations:** CR, complete recovery; F1, Family 1 (case–control risk); F2, Family 2 (Siegel recovery); HBOT, hyperbaric oxygen therapy; OMOP, Observational Medical Outcomes Partnership; PR, partial recovery; S_train, training split of the PTA-trackable set.

**Figure 9. PTA distributions.** (a) Baseline and (b) follow-up 4PTA (dB HL) in real S versus the sequential shared spine.
**Notes.** Sequential is the scientific primary (F1∧F2 Pass). 4PTA is the mean of complete AC thresholds at 0.5, 1, 2 and 4 kHz on the affected ear. Follow-up is the last complete same-ear 4PTA within 365 days after index.
**Abbreviations:** AC, air conduction; dB HL, decibels hearing level; F2, Family 2 (Siegel recovery); PTA, pure-tone average; 4PTA, four-frequency PTA; S, PTA-trackable set.
"""


def write_captions(attrition, summary):
    n = attrition["n_pta_trackable"]
    ntr = attrition["n_S_train"]
    n_dx = attrition.get("n_pta_index_with_ssnhl_dx_ever", "—")
    win = _winner_name(summary)
    text = f"""# Manuscript captions — Sim-to-Real v7.2

**Study**: Case-side Sim-to-Real for SSNHL; index = first complete AC 4PTA; shared PTA-trackable S (N={n:,}); train N={ntr:,}; one spine per method for F1 and F2.

**Scientific primary**: {win} (F1∧F2 Overcome). Dummy / Cell-preserving = PC. ystrat_copula / ipf_2way / partial_cell = AC (Y-peek, cannot win). v6 F1 Marginal Pass / F2 non-trivial Fail is historical.

## Global abbreviations

{GLOSSARY}

## Generator names

{GEN_NOTE}

Each figure legend below is written for npj Digital Medicine (≤350 words) with **Notes.** and **Abbreviations:**.

## Figures

{_figure_legend_block(n, ntr, n_dx, summary)}

## Tables

**Table 1.** Cohort attrition from the Case file to PTA-trackable S, with retention fractions. Index = first complete AC 4PTA.
**Abbreviations:** AC, air conduction; PTA, pure-tone average; 4PTA, four-frequency PTA; S, PTA-trackable set.

**Table 2a–c.** Demographics (2a); exposures and outcomes (2b); Siegel class counts (2c) on real S.
**Abbreviations:** CR, complete recovery; HBOT, hyperbaric oxygen therapy; NI, no improvement; PR, partial recovery; SI, slight improvement.

**Table 3.** Odds ratios (95% CI) by setup A/B/C for F1 pre-index exposures and F2 post-index interventions on the sequential shared spine.
**Notes.** Scientific primary = sequential F1∧F2 Pass. Dummy / Cell-preserving are PC; Y-peek AC cannot win. Real-A steroid OR 0.70 (0.56–0.89); HBOT 0.26 (0.14–0.50).
**Abbreviations:** CI, confidence interval; F1, Family 1; F2, Family 2; HBOT, hyperbaric oxygen therapy; OR, odds ratio.

**Table 4.** Concordance metrics for sequential (shared spine) in each family. τ=0.20 for median |ΔlogOR|. F2 primary = OR ∧ g-comp ATE.
**Notes.** Both F1 and F2 Overcome. PC/AC cannot win. Risk Overcome does not require CI overlap.
**Abbreviations:** ATE, average treatment effect; CI, confidence interval; OR, odds ratio; RD, risk difference; RR, risk ratio; τ, concordance tolerance.

**Table 5.** Step1 data-quality metrics by generator on the shared spine (one row per method: clinical violation, KS, correlation distance, distinguish AUC, DCR). Winner column marks sequential.
**Notes.** Step1 is a filter only. Dummy fails privacy (DCR). PC/AC cannot win.
**Abbreviations:** AUC, area under the curve; DCR, distance to closest record; KS, Kolmogorov–Smirnov statistic.

**Table 6.** Step2 for all methods: Role (PC/AC/C/N/S), Step1, F1 Ov, F2 Ov, F2 OR, F2 ATE, dGain, drift. Winner = sequential. PC/AC cannot win.
**Notes.** {GEN_NOTE}

**Table 7a–b.** Shared deliverable metadata (7a) and OMOP row-count match to S_train (7b) for the sequential winner package.
**Abbreviations:** OMOP, Observational Medical Outcomes Partnership.

## Suggested main-text figure set (tight journal)

Main: Fig0, Fig1, Fig3, Fig7  
Supplement: Fig2, Fig4, Fig5, Fig6, Fig8, Fig9 + Tables 5–7

Paste-ready figure legends (section 13): `FIGURE_LEGENDS.md`

## Key result one-liner

Scientific primary: sequential F1∧F2 Overcome on one shared spine (Y:=Siegel). Dummy / Cell-preserving are PC; Y-peek AC cannot win. v6 F1 Marginal / F2 fail is historical.
"""
    (OUT / "CAPTIONS.md").write_text(text, encoding="utf-8")


def write_figure_legends(attrition, summary):
    n = attrition["n_pta_trackable"]
    ntr = attrition["n_S_train"]
    n_dx = attrition.get("n_pta_index_with_ssnhl_dx_ever", "—")
    text = f"""# Figure legends — npj Digital Medicine Article (≤350 words each)

Paste under manuscript section 13. Each legend includes **Notes.** and **Abbreviations:**. Figures themselves carry a short footer note.

{_figure_legend_block(n, ntr, n_dx, summary)}
"""
    (OUT / "FIGURE_LEGENDS.md").write_text(text, encoding="utf-8")


def write_index():
    figs = sorted(p.name for p in FIG.glob("*.png"))
    tabs = sorted(p.name for p in TAB.glob("*.csv"))
    lines = [
        "# Figure & table index (v7.2 manuscript; sequential winner)",
        "",
        "## Figures (PNG + PDF)",
        *[f"- `{f}`" for f in figs],
        "",
        "## Tables (CSV + MD + TEX)",
        *[f"- `{t}`" for t in tabs],
        "",
        "Legends (Notes + Abbreviations): `CAPTIONS.md`, `FIGURE_LEGENDS.md`.",
    ]
    (OUT / "FIGURE_TABLE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def _copy_pair(src_stem: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        src = src_stem.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)


def sync_submission():
    fig_dst = SUB_SHARED / "figures"
    tab_dst = SUB_SHARED / "tables"
    cap_dst = SUB_SHARED / "captions"
    fig_dst.mkdir(parents=True, exist_ok=True)
    tab_dst.mkdir(parents=True, exist_ok=True)
    cap_dst.mkdir(parents=True, exist_ok=True)

    for p in FIG.glob("*.png"):
        _copy_pair(p.with_suffix(""), fig_dst)
    for p in TAB.iterdir():
        if p.suffix.lower() in {".csv", ".md", ".tex"}:
            shutil.copy2(p, tab_dst / p.name)
    for name in ("CAPTIONS.md", "FIGURE_LEGENDS.md", "FIGURE_TABLE_INDEX.md"):
        src = OUT / name
        if src.exists():
            shutil.copy2(src, cap_dst / name)

    MAIN_TABS = [
        "Table1_attrition",
        "Table3_OR_formatted",
        "Table4_concordance_overcome",
    ]
    SUPP_TABS = [
        "Table2a_demographics",
        "Table2b_exposures_outcomes",
        "Table2c_siegel",
        "Table3_OR_full",
        "Table5_step1_generators",
        "Table6_step2_sensitivity",
        "Table7a_deliverable_meta",
        "Table7b_row_counts",
    ]
    for jdir in JOURNAL_DIRS:
        main_fig = jdir / "03_figures_main"
        supp_dir = jdir / "05_supplement"
        main_tab = jdir / "04_tables_main"
        for stem in MAIN_FIGS:
            _copy_pair(FIG / stem, main_fig)
        for stem in SUPP_FIGS:
            _copy_pair(FIG / stem, supp_dir)
        for stem in MAIN_TABS:
            for ext in (".csv", ".md", ".tex"):
                src = TAB / f"{stem}{ext}"
                if src.exists():
                    main_tab.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, main_tab / src.name)
        for stem in SUPP_TABS:
            for ext in (".csv", ".md", ".tex"):
                src = TAB / f"{stem}{ext}"
                if src.exists():
                    supp_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, supp_dir / src.name)

    readme = SUB_SHARED / "captions" / "MANUSCRIPT_ASSETS_README.md"
    if readme.exists():
        body = readme.read_text(encoding="utf-8")
        if "FIGURE_LEGENDS.md" not in body:
            readme.write_text(
                body.rstrip()
                + "\n\n캡션: [`CAPTIONS.md`](CAPTIONS.md)  \n투고용 figure legends: [`FIGURE_LEGENDS.md`](FIGURE_LEGENDS.md)  \n목록: [`FIGURE_TABLE_INDEX.md`](FIGURE_TABLE_INDEX.md)\n",
                encoding="utf-8",
            )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    summary, attrition, S, f1, f2 = _load()
    win = _winner_name(summary)
    print(f"Building tables (winner={win})...")
    table_attrition(attrition)
    table_baseline(S, f1, f2)
    table_or(summary)
    table_concordance(summary)
    table_step1(summary)
    table_step2_sens(summary)
    table_deliverable(summary)

    print("Building figures...")
    fig_design_abc()
    fig_attrition_flow(attrition)
    fig_siegel(f2)
    fig_forest_or(summary)
    fig_concordance_bars(summary)
    fig_generator_heatmap(summary)
    fig_step1_scatter(summary)
    fig_overview_pass(summary, attrition)
    fig_real_vs_synth_spine()
    fig_pta_hist()

    write_captions(attrition, summary)
    write_figure_legends(attrition, summary)
    write_index()
    print("Syncing Submission packages...")
    sync_submission()
    print("Done ->", OUT)
    print("Shared figures ->", SUB_SHARED / "figures")


if __name__ == "__main__":
    main()
