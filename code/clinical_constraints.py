"""Clinical measurement constraints for synthetic analytic tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Individual PTA thresholds are 5 dB HL; 4-frequency PTA is their mean → 1.25 dB grid.
PTA_MIN = -10.0
PTA_MAX = 120.0
PTA_STEP_4FREQ = 1.25  # mean of four 5 dB values
PTA_STEP_SINGLE = 5.0


def quantize(x: np.ndarray, step: float, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.clip(x, lo, hi)
    out = np.round(out / step) * step
    # floating dust
    out = np.round(out / step) * step
    return np.clip(out, lo, hi)


def siegel_class(pta_base: float, pta_fu: float) -> str:
    if pd.isna(pta_base) or pd.isna(pta_fu):
        return "NA"
    gain = pta_base - pta_fu
    final = pta_fu
    if final <= 25:
        return "CR"
    if gain > 15 and 25 <= final <= 45:
        return "PR"
    if gain > 15 and final > 45:
        return "SI"
    if gain < 15 or final >= 75:
        return "NI"
    if gain >= 15:
        return "SI"
    return "NI"


def snap_binary(s: pd.Series, fill_missing: bool = False) -> pd.Series:
    """Snap non-missing values to {0,1}. Missing kept as NaN unless fill_missing=True."""
    v = pd.to_numeric(s, errors="coerce")
    snapped = v.round().clip(0, 1)
    if fill_missing:
        return snapped.fillna(0).astype(int)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    mask = snapped.notna()
    out.loc[mask] = snapped.loc[mask].astype(int)
    return out


def apply_f1_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """F1: demographics + binary exposures. Preserve exposure NaN (no NaN→0)."""
    out = df.copy()
    if "age_at_index" in out.columns:
        age = pd.to_numeric(out["age_at_index"], errors="coerce")
        fixed = age.round().clip(0, 120)
        out["age_at_index"] = fixed  # keep NaN if missing
    if "sex" in out.columns:
        out["sex"] = snap_binary(out["sex"], fill_missing=True)
    if "case" in out.columns:
        out["case"] = snap_binary(out["case"], fill_missing=True)
    for c in ["hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"]:
        if c in out.columns:
            out[c] = snap_binary(out[c], fill_missing=False)
    return out


def apply_f2_constraints(df: pd.DataFrame, recompute_siegel: bool = True) -> pd.DataFrame:
    """
    F2: PTA clinical grid + binary treatments + Siegel consistency.
    PTA_4freq quantized to 1.25 dB within [-10, 120].
    Treatment NaN preserved (no NaN→0); recovered recomputed from PTA.
    """
    out = df.copy()
    if "age_at_index" in out.columns:
        age = pd.to_numeric(out["age_at_index"], errors="coerce")
        out["age_at_index"] = age.round().clip(0, 120)
    if "sex" in out.columns:
        out["sex"] = snap_binary(out["sex"], fill_missing=True)
    for c in ["steroid_post", "hbot_post"]:
        if c in out.columns:
            out[c] = snap_binary(out[c], fill_missing=False)

    for c in ["pta_base", "pta_fu"]:
        if c in out.columns:
            raw = pd.to_numeric(out[c], errors="coerce")
            fixed = quantize(raw.to_numpy(dtype=float), PTA_STEP_4FREQ, PTA_MIN, PTA_MAX)
            mask = raw.isna().to_numpy()
            fixed = fixed.astype(float)
            fixed[mask] = np.nan
            out[c] = fixed

    if recompute_siegel and "pta_base" in out.columns and "pta_fu" in out.columns:
        out["siegel"] = [siegel_class(b, f) for b, f in zip(out["pta_base"], out["pta_fu"])]
        out["recovered"] = out["siegel"].isin(["CR", "PR"]).astype(int)
    elif "recovered" in out.columns:
        out["recovered"] = snap_binary(out["recovered"], fill_missing=True)

    return out


def violation_report(df: pd.DataFrame, family: str) -> dict:
    """Share of rows violating clinical rules. Missing exposures are allowed (not violations)."""
    n = len(df)
    if n == 0:
        return {"n": 0}
    rep = {"n": n, "family": family}

    if "age_at_index" in df.columns:
        age = pd.to_numeric(df["age_at_index"], errors="coerce")
        # missing age is a violation; out-of-range among observed also
        rep["age_missing"] = float(age.isna().mean())
        obs = age.dropna()
        rep["age_out_of_range"] = float(((obs < 0) | (obs > 120)).mean()) if len(obs) else 0.0
        rep["age_non_integer"] = float((obs % 1 != 0).mean()) if len(obs) else 0.0

    if "sex" in df.columns:
        sex = pd.to_numeric(df["sex"], errors="coerce")
        rep["sex_not_binary"] = float((~sex.isin([0, 1])).mean())

    binaries = {
        "F1": ["hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"],
        "F2": ["steroid_post", "hbot_post", "recovered"],
    }.get(family, [])
    for c in binaries:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        # only non-missing invalid values count; missing allowed for exposures
        if c == "recovered":
            rep[f"{c}_not_binary"] = float((~v.isin([0, 1])).mean())
        else:
            obs = v.dropna()
            rep[f"{c}_not_binary"] = float((~obs.isin([0, 1])).mean()) if len(obs) else 0.0
            rep[f"{c}_missing_rate"] = float(v.isna().mean())

    if family == "F2":
        for c in ["pta_base", "pta_fu"]:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) == 0:
                rep[f"{c}_out_of_range"] = np.nan
                rep[f"{c}_off_grid_1p25"] = np.nan
                continue
            rep[f"{c}_out_of_range"] = float(((v < PTA_MIN) | (v > PTA_MAX)).mean())
            rem = np.abs(np.round(v / PTA_STEP_4FREQ) * PTA_STEP_4FREQ - v)
            rep[f"{c}_off_grid_1p25"] = float((rem > 1e-6).mean())
            rem5 = np.abs(np.round(v / PTA_STEP_SINGLE) * PTA_STEP_SINGLE - v)
            rep[f"{c}_off_grid_5"] = float((rem5 > 1e-6).mean())

        if {"pta_base", "pta_fu", "recovered"}.issubset(df.columns):
            true_rec = [
                siegel_class(b, f) in ("CR", "PR")
                for b, f in zip(
                    pd.to_numeric(df["pta_base"], errors="coerce"),
                    pd.to_numeric(df["pta_fu"], errors="coerce"),
                )
            ]
            pred = pd.to_numeric(df["recovered"], errors="coerce").fillna(-1).astype(int)
            rep["siegel_recovered_inconsistent"] = float(
                (pred.to_numpy() != np.array(true_rec).astype(int)).mean()
            )

    return rep


def apply_shared_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """F1 + F2 snaps on a joint twin spine; Siegel recomputed from PTA."""
    donor = df["_donor_person_id"] if "_donor_person_id" in df.columns else None
    pid = df["person_id"] if "person_id" in df.columns else None
    out = apply_f1_constraints(df)
    out = apply_f2_constraints(out, recompute_siegel=True)
    if donor is not None:
        out["_donor_person_id"] = donor.values
    if pid is not None:
        out["person_id"] = pid.values
    return out


def apply_constraints(df: pd.DataFrame, family: str) -> pd.DataFrame:
    donor = df["_donor_person_id"] if "_donor_person_id" in df.columns else None
    pid = df["person_id"] if "person_id" in df.columns else None
    if family == "F1":
        out = apply_f1_constraints(df)
    elif family == "F2":
        out = apply_f2_constraints(df, recompute_siegel=True)
    else:
        raise ValueError(family)
    if donor is not None:
        out["_donor_person_id"] = donor.values
    if pid is not None:
        out["person_id"] = pid.values
    return out
