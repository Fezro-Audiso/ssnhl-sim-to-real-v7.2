"""Step1 / Step2 / co-primary evaluation for v5."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import stats

import causal_epi
import clinical_constraints as clinical
import config


def ks_mean(real, synth, cols):
    scores = []
    for c in cols:
        a = pd.to_numeric(real[c], errors="coerce").dropna()
        b = pd.to_numeric(synth[c], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        scores.append(stats.ks_2samp(a, b).statistic)
    return float(np.mean(scores)) if scores else np.nan


def corr_distance(real, synth, cols):
    r = real[cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0).to_numpy()
    s = synth[cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0).to_numpy()
    return float(np.mean(np.abs(r - s)))


def distinguish_auc(real, synth, cols, seed):
    r = real[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    s = synth[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = pd.concat([r, s], ignore_index=True)
    y = np.array([0] * len(r) + [1] * len(s))
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=6)
    from sklearn.model_selection import train_test_split

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return 0.5
    clf.fit(xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))


def dcr_median(real, synth, cols):
    r = StandardScaler().fit_transform(real[cols].apply(pd.to_numeric, errors="coerce").fillna(0))
    s = StandardScaler().fit(real[cols].apply(pd.to_numeric, errors="coerce").fillna(0)).transform(
        synth[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    nn = NearestNeighbors(n_neighbors=1).fit(r)
    dist, _ = nn.kneighbors(s)
    return float(np.median(dist))


def _logit_mle_wald(X_df, y):
    """Unpenalized logit MLE + Wald CI (statsmodels fallback if scipy/sm mismatch)."""
    X = np.column_stack([np.ones(len(X_df)), np.asarray(X_df, dtype=float)])
    yv = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(80):
        eta = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-12, None)
        H = X.T @ (X * w[:, None])
        g = X.T @ (yv - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(X @ beta, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-12, None)
    H = X.T @ (X * w[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    logor = float(beta[1])
    z = 1.96
    # clip to avoid overflow → inf CI that breaks concordance
    lo = float(np.exp(np.clip(logor - z * se[1], -20.0, 20.0)))
    hi = float(np.exp(np.clip(logor + z * se[1], -20.0, 20.0)))
    return {
        "or": float(np.exp(np.clip(logor, -20.0, 20.0))),
        "logor": logor,
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(yv)),
    }


def fit_or(df, ycol, xcol, adjust):
    use = df[[ycol, xcol] + adjust].dropna().copy()
    if use[ycol].nunique() < 2 or use[xcol].nunique() < 2:
        return {"or": np.nan, "logor": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": len(use)}
    X = use[[xcol] + adjust].astype(float)
    y = use[ycol].astype(int)
    try:
        import statsmodels.api as sm

        X2 = sm.add_constant(X, has_constant="add")
        res = sm.Logit(y, X2).fit(disp=False)
        logor = float(res.params[xcol])
        ci = res.conf_int().loc[xcol]
        return {
            "or": float(np.exp(logor)),
            "logor": logor,
            "ci_low": float(np.exp(ci[0])),
            "ci_high": float(np.exp(ci[1])),
            "n": len(use),
        }
    except Exception:
        try:
            return _logit_mle_wald(X, y)
        except Exception:
            clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
            clf.fit(X, y)
            coef = float(clf.coef_[0][0])
            return {"or": float(np.exp(coef)), "logor": coef, "ci_low": np.nan, "ci_high": np.nan, "n": len(use)}


def concordance(or_a, or_b, exposures, tau):
    signs, dlogs, overlaps, rows = [], [], [], []
    for e in exposures:
        a, b = or_a[e], or_b[e]
        if np.isnan(a["logor"]) or np.isnan(b["logor"]):
            rows.append({"exposure": e, "status": "na"})
            continue
        sign_ok = np.sign(a["logor"]) == np.sign(b["logor"]) or (a["logor"] == 0 and b["logor"] == 0)
        dlog = abs(a["logor"] - b["logor"])
        if not np.isnan(a["ci_low"]) and not np.isnan(b["ci_low"]):
            overlap = not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])
        else:
            overlap = np.nan
        signs.append(int(sign_ok))
        dlogs.append(dlog)
        if not np.isnan(overlap):
            overlaps.append(int(overlap))
        rows.append({"exposure": e, "sign_ok": sign_ok, "abs_dlogOR": dlog, "ci_overlap": overlap})
    sign_rate = float(np.mean(signs)) if signs else np.nan
    med_d = float(np.median(dlogs)) if dlogs else np.nan
    ov_rate = float(np.mean(overlaps)) if overlaps else np.nan
    overcome = (
        (sign_rate >= config.SIGN_THR)
        and (med_d <= tau)
        and (ov_rate >= config.CI_OVERLAP_THR if not np.isnan(ov_rate) else False)
    )
    return {
        "sign_rate": sign_rate,
        "median_abs_dlogOR": med_d,
        "ci_overlap_rate": ov_rate,
        "overcome": bool(overcome),
        "detail": rows,
    }


def clinical_violation_rate(df, family):
    rep = clinical.violation_report(df, family)
    keys = (
        ["age_out_of_range", "age_missing", "sex_not_binary", "hear_dx_pre_not_binary", "htn_pre_not_binary", "dm_pre_not_binary", "steroid_pre_not_binary"]
        if family == "F1"
        else [
            "age_out_of_range",
            "age_missing",
            "sex_not_binary",
            "steroid_post_not_binary",
            "hbot_post_not_binary",
            "pta_base_out_of_range",
            "pta_fu_out_of_range",
            "pta_base_off_grid_1p25",
            "pta_fu_off_grid_1p25",
            "siegel_recovered_inconsistent",
        ]
    )
    vals = [float(rep[k]) for k in keys if k in rep and rep[k] == rep[k]]
    return float(max(vals)) if vals else 0.0


def select_step1(step1, generators, family):
    rr_dcr = float(step1.get("real_real_dcr", 0.0) or 0.0)
    ranked, eligible = [], []
    for gname, syn in generators.items():
        s1 = step1[gname]
        viol = clinical_violation_rate(syn, family)
        s1["clinical_violation_rate"] = viol
        clinical_ok = viol <= config.CLINICAL_MAX_VIOL
        privacy_ok = (s1["dcr"] >= 0.9 * rr_dcr) if rr_dcr > 0 else True
        s1["step1_eligible"] = bool(clinical_ok and privacy_ok)
        if s1["step1_eligible"]:
            eligible.append(gname)
            ranked.append((abs(float(s1["distinguish_auc"]) - 0.5), float(s1["ks"]), float(s1["corr_dist"]), gname))
    if ranked:
        return sorted(ranked)[0][3], eligible
    fallback = [
        (abs(float(step1[g]["distinguish_auc"]) - 0.5), float(step1[g]["ks"]), float(step1[g]["corr_dist"]), g)
        for g in generators
    ]
    return sorted(fallback)[0][3], eligible


def evaluate_family(
    family: str,
    train: pd.DataFrame,
    hold: pd.DataFrame,
    generators: dict,
    feature_cols: list[str],
    ycol: str,
    exposures: list[str],
    adjust: list[str],
    control_df: pd.DataFrame | None,
):
    results = {"family": family, "n_train": len(train), "n_hold": len(hold), "step1": {}, "step2": {}}
    results["step1"]["real_real_ks"] = ks_mean(train, hold, feature_cols)
    results["step1"]["real_real_dcr"] = dcr_median(train, hold, feature_cols)

    gens = {}
    for gname, syn in generators.items():
        syn = syn.copy()
        if control_df is not None:
            syn[ycol] = 1
        gens[gname] = syn
        results["step1"][gname] = {
            "ks": ks_mean(train, syn, feature_cols),
            "corr_dist": corr_distance(train, syn, feature_cols),
            "distinguish_auc": distinguish_auc(train, syn, feature_cols, config.SEED),
            "dcr": dcr_median(train, syn, feature_cols),
        }

    best, eligible = select_step1(results["step1"], gens, family)
    results["step1_selected"] = best
    results["step1_eligible"] = eligible
    syn = gens[best]
    print(f"  [{family}] Step1 selected={best} eligible={eligible}")

    def assemble(case_part):
        cp = case_part.copy()
        if control_df is None:
            return cp
        if ycol not in cp.columns:
            cp[ycol] = 1
        ctrl = control_df.copy()
        if ycol not in ctrl.columns:
            ctrl[ycol] = 0
        return pd.concat([cp, ctrl], ignore_index=True)

    real_cases = pd.concat([train, hold], ignore_index=True)
    n = len(train)

    def or_map(df):
        return {e: fit_or(df, ycol, e, adjust) for e in exposures}

    def risk_map(df):
        return {e: causal_epi.fit_risk_bundle(df, ycol, e, adjust, family) for e in exposures}

    setup_a = assemble(real_cases)
    ora, risk_a = or_map(setup_a), risk_map(setup_a)
    results["step2"]["OR_A"] = ora
    results["step2"]["RISK_A"] = risk_a

    sens = {}
    for gname in eligible:
        gsyn = gens[gname]
        orb = or_map(assemble(gsyn))
        orc = or_map(
            assemble(pd.concat([train, gsyn.sample(n=min(n, len(gsyn)), random_state=config.SEED)], ignore_index=True))
        )
        rb = causal_epi.risk_concordance(risk_a, risk_map(assemble(gsyn)), exposures)
        rc = causal_epi.risk_concordance(
            risk_a,
            risk_map(
                assemble(
                    pd.concat([train, gsyn.sample(n=min(n, len(gsyn)), random_state=config.SEED)], ignore_index=True)
                )
            ),
            exposures,
        )
        b = concordance(ora, orb, exposures, config.TAU)
        c = concordance(ora, orc, exposures, config.TAU)
        or_pass = bool(b["overcome"] and c["overcome"])
        risk_pass = bool(rb["risk_overcome"] and rc["risk_overcome"])
        sens[gname] = {
            "B_overcome": b["overcome"],
            "C_overcome": c["overcome"],
            "or_overcome": or_pass,
            "risk_causal_overcome": risk_pass,
            "primary_overcome": bool(or_pass and risk_pass),
            "B_median_abs_dlogOR": b["median_abs_dlogOR"],
            "C_median_abs_dlogOR": c["median_abs_dlogOR"],
            "B_vs_A": b,
            "C_vs_A": c,
            "RISK_B_vs_A": rb,
            "RISK_C_vs_A": rc,
            "OR_B": orb,
            "OR_C": orc,
        }

    # Fill step2 for selected
    sel = sens.get(best)
    if sel is None:
        # evaluate selected even if not eligible
        gsyn = syn
        orb = or_map(assemble(gsyn))
        orc = or_map(
            assemble(pd.concat([train, gsyn.sample(n=min(n, len(gsyn)), random_state=config.SEED)], ignore_index=True))
        )
        b = concordance(ora, orb, exposures, config.TAU)
        c = concordance(ora, orc, exposures, config.TAU)
        rb = causal_epi.risk_concordance(risk_a, risk_map(assemble(gsyn)), exposures)
        rc = causal_epi.risk_concordance(
            risk_a,
            risk_map(
                assemble(
                    pd.concat([train, gsyn.sample(n=min(n, len(gsyn)), random_state=config.SEED)], ignore_index=True)
                )
            ),
            exposures,
        )
        or_pass = bool(b["overcome"] and c["overcome"])
        risk_pass = bool(rb["risk_overcome"] and rc["risk_overcome"])
        sel = {
            "primary_overcome": bool(or_pass and risk_pass),
            "or_overcome": or_pass,
            "risk_causal_overcome": risk_pass,
            "B_vs_A": b,
            "C_vs_A": c,
            "RISK_B_vs_A": rb,
            "RISK_C_vs_A": rc,
            "OR_B": orb,
            "OR_C": orc,
        }

    results["step2"]["OR_B"] = sel["OR_B"]
    results["step2"]["OR_C"] = sel["OR_C"]
    results["step2"]["B_vs_A"] = sel["B_vs_A"]
    results["step2"]["C_vs_A"] = sel["C_vs_A"]
    results["step2"]["RISK_B_vs_A"] = sel["RISK_B_vs_A"]
    results["step2"]["RISK_C_vs_A"] = sel["RISK_C_vs_A"]
    results["step2"]["or_overcome"] = sel["or_overcome"]
    results["step2"]["risk_causal_overcome"] = sel["risk_causal_overcome"]
    results["step2"]["primary_overcome"] = sel["primary_overcome"]
    results["step2"]["eligible_sensitivity"] = {
        g: {k: v for k, v in s.items() if k not in {"OR_B", "OR_C", "B_vs_A", "C_vs_A", "RISK_B_vs_A", "RISK_C_vs_A"}}
        for g, s in sens.items()
    }

    co_ranked = []
    for gname in eligible:
        if not sens.get(gname, {}).get("primary_overcome"):
            continue
        s1 = results["step1"][gname]
        co_ranked.append((abs(float(s1["distinguish_auc"]) - 0.5), float(s1["ks"]), float(s1["corr_dist"]), gname))
    if co_ranked:
        co_best = sorted(co_ranked)[0][3]
    elif results["step2"]["primary_overcome"]:
        co_best = best
    else:
        co_best = None
    results["step2_co_primary"] = co_best
    results["step2"]["co_primary_generator"] = co_best

    # Adopted for deliverable
    # F2: only Step2-pass (selected if pass, else co-primary)
    # F1: always keep the best Step2 performer among Step1-eligible (pass not required)
    if family == "F1":
        pool = eligible if eligible else list(gens.keys())
        scored = []
        for gname in pool:
            s = sens.get(gname)
            if s is None:
                continue
            b_or = s.get("B_median_abs_dlogOR")
            c_or = s.get("C_median_abs_dlogOR")
            if b_or is None:
                b_or = s.get("B_vs_A", {}).get("median_abs_dlogOR", 999.0)
            if c_or is None:
                c_or = s.get("C_vs_A", {}).get("median_abs_dlogOR", 999.0)
            rb = s.get("RISK_B_vs_A", {})
            b_rr = rb.get("median_abs_dlogRR", 999.0) if isinstance(rb, dict) else 999.0
            b_rd = rb.get("median_abs_dRD", 999.0) if isinstance(rb, dict) else 999.0
            # finite-safe
            def _f(x):
                try:
                    x = float(x)
                    return 999.0 if x != x else x
                except Exception:
                    return 999.0

            score = (
                0 if s.get("primary_overcome") else 1,  # pass first
                (_f(b_or) + _f(c_or)) / 2.0,
                _f(b_rr),
                _f(b_rd),
                abs(float(results["step1"][gname]["distinguish_auc"]) - 0.5),
                float(results["step1"][gname]["ks"]),
                gname,
            )
            scored.append(score)
        adopted = sorted(scored)[0][-1] if scored else best
        results["adoption_rule"] = (
            "F1: best Step2 performer among Step1-eligible "
            "(primary_overcome preferred; else min mean |ΔlogOR|, then risk drift, then fidelity). "
            "Pass not required."
        )
    else:
        if results["step2"]["primary_overcome"]:
            adopted = best
        elif co_best:
            adopted = co_best
        else:
            adopted = None
        results["adoption_rule"] = (
            "F2: Step2-adopted only (step1_selected if primary else co-primary; else none)"
        )

    results["adopted_generator"] = adopted
    print(f"  [{family}] co-primary={co_best} adopted={adopted} rule={results['adoption_rule'][:60]}...")
    return results, gens
