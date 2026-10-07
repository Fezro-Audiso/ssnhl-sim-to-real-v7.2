"""
Step2 epidemiology: association (OR) + risk (RR/RD) + causal estimands.
F1 (case-control): OR primary; RR/RD not identified without sampling fractions.
F2 (case-only recovery): RR/RD + g-computation / IPW under stated assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _prep(df: pd.DataFrame, ycol: str, xcol: str, adjust: list[str]) -> pd.DataFrame | None:
    use = df[[ycol, xcol] + adjust].copy()
    for c in [ycol, xcol] + adjust:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna()
    if len(use) < 20 or use[ycol].nunique() < 2 or use[xcol].nunique() < 2:
        return None
    use[ycol] = use[ycol].astype(int)
    use[xcol] = use[xcol].astype(int)
    return use


def crude_risk(df: pd.DataFrame, ycol: str, xcol: str) -> dict:
    """Unadjusted risk, RR, RD among exposed vs unexposed (cohort-style)."""
    use = df[[ycol, xcol]].copy()
    use[ycol] = pd.to_numeric(use[ycol], errors="coerce")
    use[xcol] = pd.to_numeric(use[xcol], errors="coerce")
    use = use.dropna()
    use[ycol] = use[ycol].astype(int)
    use[xcol] = use[xcol].astype(int)
    if use[xcol].nunique() < 2 or len(use) < 10:
        return _empty_risk()
    r1 = float(use.loc[use[xcol] == 1, ycol].mean())
    r0 = float(use.loc[use[xcol] == 0, ycol].mean())
    n1 = int((use[xcol] == 1).sum())
    n0 = int((use[xcol] == 0).sum())
    # Wald CI on log RR / RD
    eps = 1e-9
    rr = (r1 + eps) / (r0 + eps)
    rd = r1 - r0
    # SE log RR approx
    se_logrr = np.sqrt(max((1 - r1) / (r1 * n1 + eps) + (1 - r0) / (r0 * n0 + eps), 0))
    se_rd = np.sqrt(max(r1 * (1 - r1) / (n1 + eps) + r0 * (1 - r0) / (n0 + eps), 0))
    z = 1.96
    return {
        "risk1": r1,
        "risk0": r0,
        "rr": float(rr),
        "logrr": float(np.log(rr)),
        "rr_ci_low": float(np.exp(np.log(rr) - z * se_logrr)),
        "rr_ci_high": float(np.exp(np.log(rr) + z * se_logrr)),
        "rd": float(rd),
        "rd_ci_low": float(rd - z * se_rd),
        "rd_ci_high": float(rd + z * se_rd),
        "n": len(use),
        "n1": n1,
        "n0": n0,
    }


def _empty_risk():
    return {
        "risk1": np.nan,
        "risk0": np.nan,
        "rr": np.nan,
        "logrr": np.nan,
        "rr_ci_low": np.nan,
        "rr_ci_high": np.nan,
        "rd": np.nan,
        "rd_ci_low": np.nan,
        "rd_ci_high": np.nan,
        "n": 0,
        "n1": 0,
        "n0": 0,
    }


def gcomp_ate(df: pd.DataFrame, ycol: str, xcol: str, adjust: list[str]) -> dict:
    """
    Outcome-regression g-computation for ATE on risk scale (RD) and RR.
    Assumptions: consistency, positivity, conditional exchangeability given `adjust`.
    """
    use = _prep(df, ycol, xcol, adjust)
    if use is None:
        return {**_empty_risk(), "estimator": "gcomp", "ate_rd": np.nan, "ate_rr": np.nan}
    X = use[[xcol] + adjust].astype(float)
    y = use[ycol].astype(int)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    x1 = use[[xcol] + adjust].astype(float).copy()
    x0 = x1.copy()
    x1[xcol] = 1.0
    x0[xcol] = 0.0
    p1 = clf.predict_proba(x1)[:, 1]
    p0 = clf.predict_proba(x0)[:, 1]
    r1 = float(np.mean(p1))
    r0 = float(np.mean(p0))
    eps = 1e-9
    rr = (r1 + eps) / (r0 + eps)
    rd = r1 - r0
    # bootstrap SE (light)
    rng = np.random.default_rng(20260803)
    boots_rd, boots_rr = [], []
    n = len(use)
    for _ in range(80):
        idx = rng.integers(0, n, size=n)
        b = use.iloc[idx]
        if b[xcol].nunique() < 2 or b[ycol].nunique() < 2:
            continue
        try:
            cf = LogisticRegression(max_iter=1000)
            cf.fit(b[[xcol] + adjust].astype(float), b[ycol].astype(int))
            b1 = b[[xcol] + adjust].astype(float).copy()
            b0 = b1.copy()
            b1[xcol] = 1.0
            b0[xcol] = 0.0
            q1 = float(cf.predict_proba(b1)[:, 1].mean())
            q0 = float(cf.predict_proba(b0)[:, 1].mean())
            boots_rd.append(q1 - q0)
            boots_rr.append((q1 + eps) / (q0 + eps))
        except Exception:
            continue
    if boots_rd:
        rd_lo, rd_hi = float(np.percentile(boots_rd, 2.5)), float(np.percentile(boots_rd, 97.5))
        rr_lo, rr_hi = float(np.percentile(boots_rr, 2.5)), float(np.percentile(boots_rr, 97.5))
    else:
        rd_lo = rd_hi = rr_lo = rr_hi = np.nan
    return {
        "estimator": "gcomp",
        "risk1": r1,
        "risk0": r0,
        "ate_rd": float(rd),
        "ate_rr": float(rr),
        "logrr": float(np.log(rr)),
        "rd": float(rd),
        "rr": float(rr),
        "rd_ci_low": rd_lo,
        "rd_ci_high": rd_hi,
        "rr_ci_low": rr_lo,
        "rr_ci_high": rr_hi,
        "n": n,
        "n1": int((use[xcol] == 1).sum()),
        "n0": int((use[xcol] == 0).sum()),
    }


def ipw_ate(df: pd.DataFrame, ycol: str, xcol: str, adjust: list[str]) -> dict:
    """IPW ATE (RD) and RR; stabilized weights; truncated at 1st/99th pct."""
    use = _prep(df, ycol, xcol, adjust)
    if use is None:
        return {**_empty_risk(), "estimator": "ipw", "ate_rd": np.nan, "ate_rr": np.nan}
    # propensity
    Z = use[adjust].astype(float) if adjust else pd.DataFrame({"intercept": np.ones(len(use))})
    if adjust:
        ps_clf = LogisticRegression(max_iter=2000)
        ps_clf.fit(Z, use[xcol].astype(int))
        ps = ps_clf.predict_proba(Z)[:, 1]
    else:
        ps = np.full(len(use), use[xcol].mean())
    ps = np.clip(ps, 0.01, 0.99)
    p_x = float(use[xcol].mean())
    w = np.where(use[xcol] == 1, p_x / ps, (1 - p_x) / (1 - ps))
    lo, hi = np.percentile(w, [1, 99])
    w = np.clip(w, lo, hi)
    y = use[ycol].to_numpy(dtype=float)
    x = use[xcol].to_numpy(dtype=float)
    # Hajek estimator
    r1 = float(np.sum(w * y * x) / np.sum(w * x))
    r0 = float(np.sum(w * y * (1 - x)) / np.sum(w * (1 - x)))
    eps = 1e-9
    rr = (r1 + eps) / (r0 + eps)
    rd = r1 - r0
    return {
        "estimator": "ipw",
        "risk1": r1,
        "risk0": r0,
        "ate_rd": float(rd),
        "ate_rr": float(rr),
        "logrr": float(np.log(rr)),
        "rd": float(rd),
        "rr": float(rr),
        "rr_ci_low": np.nan,
        "rr_ci_high": np.nan,
        "rd_ci_low": np.nan,
        "rd_ci_high": np.nan,
        "n": len(use),
        "n1": int((use[xcol] == 1).sum()),
        "n0": int((use[xcol] == 0).sum()),
        "ess": float((np.sum(w) ** 2) / np.sum(w**2)),
    }


def evalue_rr(rr: float) -> dict:
    """
    E-value for a risk ratio (VanderWeele & Ding).
    For RR<1 use reciprocal. Observational sensitivity — not proof.
    """
    if rr != rr or rr <= 0:
        return {"evalue_point": np.nan, "rr_used": np.nan}
    r = rr if rr >= 1 else 1.0 / rr
    ev = r + np.sqrt(r * (r - 1))
    return {"evalue_point": float(ev), "rr_used": float(r)}


def positivity_summary(df: pd.DataFrame, xcol: str, adjust: list[str]) -> dict:
    use = df[[xcol] + adjust].copy()
    for c in [xcol] + adjust:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna()
    if adjust:
        # coarse strata: median-split continuous adjusters
        tmp = use.copy()
        for c in adjust:
            if tmp[c].nunique() > 4:
                tmp[c] = (tmp[c] >= tmp[c].median()).astype(int)
        g = tmp.groupby(adjust + [xcol]).size().unstack(fill_value=0)
        if g.shape[1] < 2:
            return {"min_stratum_n": int(len(use)), "empty_cells": 0, "n_strata": 0}
        empty = int(((g.iloc[:, 0] == 0) | (g.iloc[:, 1] == 0)).sum())
        return {
            "min_stratum_n": int(g.min().min()),
            "empty_cells": empty,
            "n_strata": int(len(g)),
        }
    return {
        "min_stratum_n": int(min((use[xcol] == 0).sum(), (use[xcol] == 1).sum())),
        "empty_cells": 0,
        "n_strata": 1,
    }


def fit_risk_bundle(df: pd.DataFrame, ycol: str, xcol: str, adjust: list[str], family: str) -> dict:
    """Family-aware risk / causal bundle for one exposure."""
    out = {
        "family": family,
        "exposure": xcol,
        "crude": crude_risk(df, ycol, xcol),
        "gcomp": gcomp_ate(df, ycol, xcol, adjust),
        "ipw": ipw_ate(df, ycol, xcol, adjust),
        "positivity": positivity_summary(df, xcol, adjust),
    }
    # Primary causal risk scale for concordance: g-comp ATE RR/RD
    rr = out["gcomp"].get("ate_rr", np.nan)
    out["evalue"] = evalue_rr(rr if rr == rr else out["crude"].get("rr", np.nan))
    if family == "F1":
        out["note"] = (
            "Case-control: RR/RD are not sampling-fraction–corrected; "
            "interpret as associative risk contrasts in the analytic table, "
            "causal claim rests on OR + E-value sensitivity under exchangeability."
        )
        out["primary_risk_measure"] = "associative_rr_rd_descriptive"
    else:
        out["note"] = (
            "Case-only recovery cohort: target estimand = ATE on risk scale "
            "(RD, RR) under consistency, positivity, conditional exchangeability given age/sex."
        )
        out["primary_risk_measure"] = "gcomp_ate"
    return out


def risk_concordance(risk_a: dict, risk_b: dict, exposures: list[str], tau_logrr: float = 0.20, tau_rd: float = 0.10) -> dict:
    """
    Concordance of causal/risk estimands across setups.
    Uses g-comp ATE RR (log scale) and RD.
    """
    signs_rr, d_logrr, signs_rd, d_rd, overlaps = [], [], [], [], []
    rows = []
    for e in exposures:
        a = risk_a[e]["gcomp"]
        b = risk_b[e]["gcomp"]
        if np.isnan(a.get("logrr", np.nan)) or np.isnan(b.get("logrr", np.nan)):
            rows.append({"exposure": e, "status": "na"})
            continue
        # RR sign relative to null (1): use logrr
        sign_rr = np.sign(a["logrr"]) == np.sign(b["logrr"]) or (abs(a["logrr"]) < 1e-12 and abs(b["logrr"]) < 1e-12)
        # RD sign
        sign_rd = np.sign(a["ate_rd"]) == np.sign(b["ate_rd"]) or (abs(a["ate_rd"]) < 1e-12 and abs(b["ate_rd"]) < 1e-12)
        dl = abs(a["logrr"] - b["logrr"])
        dr = abs(a["ate_rd"] - b["ate_rd"])
        # CI overlap on RR if available
        overlap = np.nan
        if not np.isnan(a.get("rr_ci_low", np.nan)) and not np.isnan(b.get("rr_ci_low", np.nan)):
            overlap = not (a["rr_ci_high"] < b["rr_ci_low"] or b["rr_ci_high"] < a["rr_ci_low"])
            overlaps.append(int(overlap))
        signs_rr.append(int(sign_rr))
        signs_rd.append(int(sign_rd))
        d_logrr.append(dl)
        d_rd.append(dr)
        rows.append(
            {
                "exposure": e,
                "RR_A": a["ate_rr"],
                "RR_cmp": b["ate_rr"],
                "RD_A": a["ate_rd"],
                "RD_cmp": b["ate_rd"],
                "sign_rr_ok": bool(sign_rr),
                "sign_rd_ok": bool(sign_rd),
                "abs_dlogRR": dl,
                "abs_dRD": dr,
                "rr_ci_overlap": overlap,
                "evalue_A": risk_a[e]["evalue"].get("evalue_point"),
                "evalue_cmp": risk_b[e]["evalue"].get("evalue_point"),
            }
        )
    med_logrr = float(np.median(d_logrr)) if d_logrr else np.nan
    med_rd = float(np.median(d_rd)) if d_rd else np.nan
    sign_rr_rate = float(np.mean(signs_rr)) if signs_rr else np.nan
    sign_rd_rate = float(np.mean(signs_rd)) if signs_rd else np.nan
    ov = float(np.mean(overlaps)) if overlaps else np.nan
    risk_overcome = (
        (sign_rr_rate >= 0.80)
        and (sign_rd_rate >= 0.80)
        and (med_logrr <= tau_logrr)
        and (med_rd <= tau_rd)
    )
    return {
        "sign_rr_rate": sign_rr_rate,
        "sign_rd_rate": sign_rd_rate,
        "median_abs_dlogRR": med_logrr,
        "median_abs_dRD": med_rd,
        "rr_ci_overlap_rate": ov,
        "risk_overcome": bool(risk_overcome),
        "tau_logrr": tau_logrr,
        "tau_rd": tau_rd,
        "detail": rows,
    }
