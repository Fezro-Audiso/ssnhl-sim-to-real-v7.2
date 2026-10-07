"""
v6 cohort: index = first complete AC 4PTA; FU = last same-ear complete 4PTA within +365d.
Synthesis set S = PTA-trackable Case only (base+FU). F1/F2 share the same S.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
import clinical_constraints as clinical
from profile_tables import read_table


def _read_person(folder: Path, prefix: str) -> pd.DataFrame:
    p = pd.read_csv(
        folder / f"{prefix}_PERSON.xls",
        usecols=["person_id", "gender_concept_id", "year_of_birth"],
        dtype={"person_id": str},
        low_memory=False,
    )
    p["sex"] = (p["gender_concept_id"] == 8532).astype(int)
    p["yob"] = pd.to_numeric(p["year_of_birth"], errors="coerce")
    return p


def _ear_pta_on_date(day_df: pd.DataFrame, ear_map: dict) -> float:
    vals = []
    for hz, cid in ear_map.items():
        sub = day_df.loc[day_df["measurement_concept_id"].astype(str) == str(cid), "value_as_number"]
        if sub.empty or pd.isna(sub.iloc[0]):
            return float("nan")
        vals.append(float(sub.iloc[0]))
    return float(np.mean(vals))


def build_pta_index_cohort(meas: pd.DataFrame, person_ids: pd.Series) -> pd.DataFrame:
    """
    Index = first date with complete AC 4PTA on either ear (worse ear if both).
    pta_fu = last complete 4PTA on the same ear with date in (index, index+FOLLOWUP_DAYS].
    Returns one row per person who has at least a baseline complete PTA
    (pta_fu may still be NaN if not trackable).
    """
    ids = set(person_ids.astype(str))
    m = meas[meas["measurement_concept_id"].astype(str).isin(config.PTA_AC_IDS)].copy()
    m = m[m["person_id"].astype(str).isin(ids)]
    if m.empty:
        return pd.DataFrame(
            columns=["person_id", "index_date", "index_source", "pta_base", "pta_fu", "affected_ear", "pta_base_date"]
        )

    m["measurement_date"] = pd.to_datetime(m["measurement_date"], errors="coerce")
    m["value_as_number"] = pd.to_numeric(m["value_as_number"], errors="coerce")
    m = m.dropna(subset=["measurement_date"])
    fu_lim = config.FOLLOWUP_DAYS

    records = []
    for pid, g in m.groupby("person_id"):
        # chronologically find first complete ear-day
        best_base = None  # (date, ear, pta)
        for dt, day in g.groupby("measurement_date", sort=True):
            rt = _ear_pta_on_date(day, config.PTA_4FREQ_AC["right"])
            lt = _ear_pta_on_date(day, config.PTA_4FREQ_AC["left"])
            if np.isnan(rt) and np.isnan(lt):
                continue
            if np.isnan(rt):
                ear, pta = "left", lt
            elif np.isnan(lt):
                ear, pta = "right", rt
            else:
                ear, pta = ("right", rt) if rt >= lt else ("left", lt)
            best_base = (dt, ear, pta)
            break
        if best_base is None:
            continue
        base_dt, ear, pta_base = best_base
        ear_map = config.PTA_4FREQ_AC[ear]
        hi = base_dt + pd.Timedelta(days=fu_lim)
        fu_g = g[(g["measurement_date"] > base_dt) & (g["measurement_date"] <= hi)].sort_values(
            "measurement_date"
        )
        pta_fu = float("nan")
        for dt, day in fu_g.groupby("measurement_date", sort=True):
            val = _ear_pta_on_date(day, ear_map)
            if not np.isnan(val):
                pta_fu = val
        records.append(
            {
                "person_id": pid,
                "index_date": base_dt,
                "index_source": "first_complete_4pta",
                "pta_base": pta_base,
                "pta_fu": pta_fu,
                "affected_ear": ear,
                "pta_base_date": base_dt,
            }
        )
    return pd.DataFrame(records)

def flag_in_window(
    events: pd.DataFrame,
    person_index: pd.DataFrame,
    concept_col: str,
    concepts: set[str],
    date_col: str,
    side: str,
) -> pd.Series:
    """side: pre (index-LOOKBACK..index) or post (index..index+FOLLOWUP)."""
    out = pd.Series(0, index=person_index.index, dtype=int)
    if events.empty or not concepts:
        return out
    e = events[events[concept_col].astype(str).isin(concepts)].copy()
    e = e.merge(person_index.reset_index(drop=True), on="person_id", how="inner")
    e[date_col] = pd.to_datetime(e[date_col], errors="coerce")
    if side == "pre":
        lo = e["index_date"] - pd.Timedelta(days=config.LOOKBACK_DAYS)
        e = e[(e[date_col] < e["index_date"]) & (e[date_col] >= lo)]
    else:
        hi = e["index_date"] + pd.Timedelta(days=config.FOLLOWUP_DAYS)
        e = e[(e[date_col] >= e["index_date"]) & (e[date_col] <= hi)]
    flagged = set(e["person_id"].unique())
    return person_index["person_id"].isin(flagged).astype(int).reset_index(drop=True)


def slice_events_window(
    df: pd.DataFrame, person_index: pd.DataFrame, date_col: str
) -> pd.DataFrame:
    """Keep rows with date in [index-LOOKBACK, index+FOLLOWUP] for persons in index."""
    if df.empty or date_col not in df.columns:
        return df.iloc[0:0].copy() if len(df) == 0 else df.head(0)
    x = df.merge(person_index[["person_id", "index_date"]], on="person_id", how="inner")
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    lo = x["index_date"] - pd.Timedelta(days=config.LOOKBACK_DAYS)
    hi = x["index_date"] + pd.Timedelta(days=config.FOLLOWUP_DAYS)
    x = x[(x[date_col].isna()) | ((x[date_col] >= lo) & (x[date_col] <= hi))]
    return x.drop(columns=["index_date"], errors="ignore")


def age_band(age: float) -> int:
    if pd.isna(age):
        return -1
    return int(age // config.AGE_BAND) * config.AGE_BAND


def match_1to1(cases: pd.DataFrame, controls: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    cases = cases.copy()
    controls = controls.copy()
    cases["age_band"] = cases["age_at_index"].map(age_band)
    controls["age_band"] = controls["age_at_index"].map(age_band)
    picked, used = [], set()
    for _, row in cases.iterrows():
        pool = controls[
            (controls["age_band"] == row["age_band"])
            & (controls["sex"] == row["sex"])
            & (~controls["person_id"].isin(used))
        ]
        if pool.empty:
            continue
        choice = pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        used.add(choice["person_id"])
        picked.append(
            {
                "case_id": row["person_id"],
                "control_id": choice["person_id"],
                "age_band": row["age_band"],
                "sex": row["sex"],
            }
        )
    return pd.DataFrame(picked)


def build_v5_assets(out_dir: Path) -> dict:
    """Build cohort, S (PTA-trackable), spines, windowed event store for donors."""
    cohort_dir = out_dir / "cohort"
    profile_dir = out_dir / "profile"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    attrition = {}
    case_p = _read_person(config.CASE_DIR, "CASE")
    attrition["n_case_person"] = len(case_p)

    # Load large tables once
    print("Loading Case event tables...")
    cond = read_table("CONDITION_OCCURRENCE", dtype={"person_id": str, "condition_concept_id": str})
    drug = read_table("DRUG_EXPOSURE", dtype={"person_id": str, "drug_concept_id": str})
    proc = read_table("PROCEDURE_OCCURRENCE", dtype={"person_id": str, "procedure_concept_id": str})
    meas = read_table("MEASUREMENT", dtype={"person_id": str, "measurement_concept_id": str})

    print("Building PTA-index cohort (first complete 4PTA)...")
    pta_idx = build_pta_index_cohort(meas, case_p["person_id"])
    attrition["n_with_baseline_4pta"] = len(pta_idx)
    case = case_p.merge(pta_idx, on="person_id", how="inner")
    case["age_at_index"] = case["index_date"].dt.year - case["yob"]
    case = case[case["age_at_index"].between(0, 120)].copy()
    attrition["n_case_with_pta_index"] = len(case)

    # Optional: how many PTA-index cases also have SSNHL dx ever (descriptive only)
    dx = cond[cond["condition_concept_id"].astype(str).isin(config.SSNHL_DX_CONCEPT_IDS)]
    attrition["n_pta_index_with_ssnhl_dx_ever"] = int(case["person_id"].isin(set(dx["person_id"].astype(str))).sum())

    case["siegel"] = [clinical.siegel_class(b, f) for b, f in zip(case["pta_base"], case["pta_fu"])]
    case["recovered"] = case["siegel"].isin(["CR", "PR"]).astype(int)
    case["pta_complete"] = case["pta_base"].notna() & case["pta_fu"].notna() & case["siegel"].ne("NA")
    attrition["n_pta_trackable"] = int(case["pta_complete"].sum())
    attrition["n_pta_complete"] = attrition["n_pta_trackable"]  # alias

    # Synthesis set S = PTA-trackable (base + FU) — shared by F1 and F2
    S = case[case["pta_complete"]].copy()
    attrition["n_synthesis_set_S"] = len(S)
    attrition["index_rule"] = "first_complete_4pta"

    # Flags within ±365d
    S = S.reset_index(drop=True)
    pi = S[["person_id", "index_date"]]
    S["steroid_pre"] = flag_in_window(drug, pi, "drug_concept_id", config.STEROID_CONCEPT_IDS, "drug_exposure_start_date", "pre")
    S["steroid_post"] = flag_in_window(drug, pi, "drug_concept_id", config.STEROID_CONCEPT_IDS, "drug_exposure_start_date", "post")
    S["hbot_post"] = flag_in_window(proc, pi, "procedure_concept_id", config.HBOT_PROCEDURE_IDS, "procedure_date", "post")
    S["hear_dx_pre"] = flag_in_window(cond, pi, "condition_concept_id", config.F1_DIAGNOSIS_PROXY, "condition_start_date", "pre")
    S["htn_pre"] = flag_in_window(cond, pi, "condition_concept_id", config.F1_HTN, "condition_start_date", "pre")
    S["dm_pre"] = flag_in_window(cond, pi, "condition_concept_id", config.F1_DM, "condition_start_date", "pre")

    # Control for F1
    print("Building Control index + match...")
    ctrl_p = _read_person(config.CTRL_DIR, "CONTROL")
    ctrl_cond = pd.read_csv(
        config.CTRL_DIR / "CONTROL_CONDITION_OCCURRENCE.xls",
        dtype={"person_id": str, "condition_concept_id": str},
        low_memory=False,
    )
    # Control index: use first condition as placeholder then overwrite with matched case index
    ctrl_idx = (
        ctrl_cond.assign(condition_start_date=pd.to_datetime(ctrl_cond["condition_start_date"], errors="coerce"))
        .dropna(subset=["condition_start_date"])
        .sort_values("condition_start_date")
        .groupby("person_id", as_index=False)
        .first()[["person_id", "condition_start_date"]]
        .rename(columns={"condition_start_date": "index_date"})
    )
    ctrl = ctrl_p.merge(ctrl_idx, on="person_id", how="inner")
    ctrl["age_at_index"] = ctrl["index_date"].dt.year - ctrl["yob"]
    ctrl = ctrl[ctrl["age_at_index"].between(0, 120)].copy()

    rng = np.random.default_rng(config.SEED)
    # Match on S demographics (synthesis set)
    pairs = match_1to1(S, ctrl, rng)
    attrition["n_matched_pairs"] = len(pairs)

    # Assign control index = matched case index; rebuild control flags in window
    pair_map = pairs.rename(columns={"case_id": "person_id"})
    ctrl_m = pairs.merge(ctrl, left_on="control_id", right_on="person_id", how="inner", suffixes=("_pair", ""))
    if "sex" not in ctrl_m.columns and "sex_pair" in ctrl_m.columns:
        ctrl_m["sex"] = ctrl_m["sex_pair"]
    case_idx_map = S.set_index("person_id")["index_date"]
    ctrl_m["index_date"] = ctrl_m["case_id"].map(case_idx_map)
    ctrl_m["age_at_index"] = ctrl_m["index_date"].dt.year - ctrl_m["yob"]
    ctrl_m = ctrl_m.reset_index(drop=True)
    # ensure sex from control person table
    if "sex" not in ctrl_m.columns:
        ctrl_m = ctrl_m.merge(ctrl_p[["person_id", "sex"]], on="person_id", how="left")
    cpi = ctrl_m[["person_id", "index_date"]]
    ctrl_drug = pd.read_csv(
        config.CTRL_DIR / "CONTROL_DRUG_EXPOSURE.xls",
        dtype={"person_id": str, "drug_concept_id": str},
        low_memory=False,
        usecols=["person_id", "drug_concept_id", "drug_exposure_start_date"],
    )
    ctrl_m["steroid_pre"] = flag_in_window(
        ctrl_drug, cpi, "drug_concept_id", config.STEROID_CONCEPT_IDS, "drug_exposure_start_date", "pre"
    )
    ctrl_m["hear_dx_pre"] = flag_in_window(
        ctrl_cond, cpi, "condition_concept_id", config.F1_DIAGNOSIS_PROXY, "condition_start_date", "pre"
    )
    ctrl_m["htn_pre"] = flag_in_window(ctrl_cond, cpi, "condition_concept_id", config.F1_HTN, "condition_start_date", "pre")
    ctrl_m["dm_pre"] = flag_in_window(ctrl_cond, cpi, "condition_concept_id", config.F1_DM, "condition_start_date", "pre")
    ctrl_m["steroid_post"] = 0
    ctrl_m["hbot_post"] = 0
    ctrl_m["case"] = 0

    # F1 analytic: matched S cases + controls
    s_matched = S[S["person_id"].isin(pairs["case_id"])].copy()
    s_matched["case"] = 1
    f1_cols = ["person_id", "case", "age_at_index", "sex", "hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"]
    f1 = pd.concat(
        [
            s_matched[f1_cols],
            ctrl_m[f1_cols],
        ],
        ignore_index=True,
    )

    f2 = S[
        [
            "person_id",
            "age_at_index",
            "sex",
            "steroid_post",
            "hbot_post",
            "pta_base",
            "pta_fu",
            "affected_ear",
            "siegel",
            "recovered",
            "index_date",
        ]
    ].copy()

    # Train/hold on S
    strat = S["recovered"] if S["recovered"].nunique() > 1 else None
    S_train, S_hold = train_test_split(
        S, test_size=1 - config.TRAIN_FRAC, random_state=config.SEED, stratify=strat
    )
    S_train = S_train.reset_index(drop=True)
    S_hold = S_hold.reset_index(drop=True)
    attrition["n_S_train"] = len(S_train)
    attrition["n_S_hold"] = len(S_hold)

    # Windowed event store for donors (train persons only) — saved as parquet/csv per table
    donor_dir = cohort_dir / "donor_events"
    donor_dir.mkdir(parents=True, exist_ok=True)
    train_ids = set(S_train["person_id"].astype(str))
    pi_train = S_train[["person_id", "index_date"]]

    date_cols = {
        "CONDITION_OCCURRENCE": "condition_start_date",
        "DRUG_EXPOSURE": "drug_exposure_start_date",
        "PROCEDURE_OCCURRENCE": "procedure_date",
        "MEASUREMENT": "measurement_date",
        "OBSERVATION": "observation_date",
        "DEVICE_EXPOSURE": "device_exposure_start_date",
        "VISIT_OCCURRENCE": "visit_start_date",
        "VISIT_DETAIL": "visit_detail_start_date",
        "DEATH": "death_date",
        "OBSERVATION_PERIOD": "observation_period_start_date",
        "PERSON": None,
    }
    for table, dcol in date_cols.items():
        raw = read_table(table, dtype={"person_id": str})
        if raw.empty:
            raw.to_csv(donor_dir / f"{table}.csv", index=False)
            continue
        raw["person_id"] = raw["person_id"].astype(str)
        raw = raw[raw["person_id"].isin(train_ids)]
        if dcol and dcol in raw.columns:
            raw = slice_events_window(raw, pi_train, dcol)
        elif table == "PERSON":
            raw = raw[raw["person_id"].isin(train_ids)]
        raw.to_csv(donor_dir / f"{table}.csv", index=False)

    from profile_tables import profile_all, target_row_counts

    profile_all(train_ids, profile_dir)
    targets = target_row_counts(train_ids)
    (profile_dir / "target_row_counts_Strain.json").write_text(
        json.dumps(targets, indent=2), encoding="utf-8"
    )

    # Persist spines
    f1.to_csv(cohort_dir / "f1_analytic.csv", index=False)
    f2.to_csv(cohort_dir / "f2_analytic.csv", index=False)
    S.to_csv(cohort_dir / "synthesis_set_S.csv", index=False)
    S_train.to_csv(cohort_dir / "S_train.csv", index=False)
    S_hold.to_csv(cohort_dir / "S_hold.csv", index=False)
    pairs.to_csv(cohort_dir / "matched_pairs.csv", index=False)
    (cohort_dir / "attrition.json").write_text(json.dumps(attrition, indent=2), encoding="utf-8")

    print("Attrition:", attrition)
    return {
        "attrition": attrition,
        "S_train": S_train,
        "S_hold": S_hold,
        "f1": f1,
        "f2": f2,
        "pairs": pairs,
        "ctrl_m": ctrl_m,
        "target_row_counts": targets,
        "donor_dir": donor_dir,
    }
