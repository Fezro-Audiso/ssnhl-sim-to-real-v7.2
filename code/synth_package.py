"""Build 11-table synthetic Case package via donor bundles + exact row-count matching."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import clinical_constraints as clinical
import config


def _reassign_ids(df: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    if id_col not in df.columns or df.empty:
        return df
    out = df.copy()
    out[id_col] = [f"{prefix}_{i}" for i in range(len(out))]
    return out


def _match_row_count(df: pd.DataFrame, target_n: int, rng: np.random.Generator) -> pd.DataFrame:
    if target_n <= 0:
        return df.iloc[0:0].copy()
    if len(df) == target_n:
        return df.reset_index(drop=True)
    if len(df) == 0:
        return df
    if len(df) > target_n:
        return df.sample(target_n, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    extra = df.sample(target_n - len(df), replace=True, random_state=int(rng.integers(0, 1_000_000)))
    return pd.concat([df, extra], ignore_index=True)


def _preserve_missingness(df: pd.DataFrame, cols: list[str], target_rates: dict[str, float], rng: np.random.Generator):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        rate = float(target_rates.get(c, 0.0))
        if rate <= 0 or len(out) == 0:
            continue
        n_miss = int(round(rate * len(out)))
        # currently missing
        cur = out[c].isna()
        need = n_miss - int(cur.sum())
        if need > 0:
            candidates = out.index[~cur]
            if len(candidates) == 0:
                continue
            pick = rng.choice(candidates, size=min(need, len(candidates)), replace=False)
            out.loc[pick, c] = np.nan
        elif need < 0:
            # too many missing: leave as-is (donor already has missing)
            pass
    return out


def build_package(
    spine: pd.DataFrame,
    donor_dir: Path,
    target_counts: dict[str, int],
    profile: dict | None,
    out_dir: Path,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """
    Map each synth person -> donor; copy donor event rows; re-id; match table counts.
    """
    rng = rng or np.random.default_rng(config.SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "_donor_person_id" not in spine.columns:
        raise ValueError("spine requires _donor_person_id")

    # PERSON from donor rows, overwritten demographics from spine when available
    person_donor = pd.read_csv(donor_dir / "PERSON.csv", dtype={"person_id": str}, low_memory=False)
    person_map = []
    rows = []
    for i, r in spine.iterrows():
        donor = str(r["_donor_person_id"])
        syn_id = str(r["person_id"])
        person_map.append({"synth_person_id": syn_id, "donor_person_id": donor})
        src = person_donor[person_donor["person_id"].astype(str) == donor]
        if src.empty:
            row = {"person_id": syn_id, "gender_concept_id": 8507 if int(r.get("sex", 0)) == 0 else 8532}
            if "age_at_index" in r and pd.notna(r["age_at_index"]):
                # approximate YOB unknown — leave year_of_birth NaN
                row["year_of_birth"] = np.nan
            rows.append(row)
        else:
            row = src.iloc[0].to_dict()
            row["person_id"] = syn_id
            if "sex" in r and pd.notna(r["sex"]):
                row["gender_concept_id"] = 8532 if int(r["sex"]) == 1 else 8507
            rows.append(row)
    person_df = pd.DataFrame(rows)
    person_df = _match_row_count(person_df, target_counts.get("PERSON", len(person_df)), rng)
    # ensure person count equals spine after match: prefer exact spine length
    if len(person_df) != len(spine):
        # rebuild PERSON strictly from spine length
        person_df = pd.DataFrame(rows).reset_index(drop=True)
    person_df.to_csv(out_dir / "SYNTH_PERSON.csv", index=False)
    pd.DataFrame(person_map).to_csv(out_dir / "donor_map.csv", index=False)

    id_prefixes = {
        "OBSERVATION_PERIOD": ("observation_period_id", "op"),
        "CONDITION_OCCURRENCE": ("condition_occurrence_id", "cond"),
        "DRUG_EXPOSURE": ("drug_exposure_id", "drug"),
        "PROCEDURE_OCCURRENCE": ("procedure_occurrence_id", "proc"),
        "MEASUREMENT": ("measurement_id", "meas"),
        "OBSERVATION": ("observation_id", "obs"),
        "DEVICE_EXPOSURE": ("device_exposure_id", "dev"),
        "VISIT_OCCURRENCE": ("visit_occurrence_id", "vis"),
        "VISIT_DETAIL": ("visit_detail_id", "vd"),
        "DEATH": (None, None),
    }

    actual_counts = {"PERSON": len(person_df)}
    donor_to_synths: dict[str, list[str]] = {}
    for m in person_map:
        donor_to_synths.setdefault(m["donor_person_id"], []).append(m["synth_person_id"])

    steroid_rates = {}
    if profile and "tables" in profile:
        drug_prof = profile["tables"].get("DRUG_EXPOSURE", {}).get("columns", {})
        for c in config.STEROID_MISSINGNESS_COLS:
            if c in drug_prof:
                steroid_rates[c] = drug_prof[c].get("missing_rate", 0.0)

    for table, (id_col, prefix) in id_prefixes.items():
        path = donor_dir / f"{table}.csv"
        if not path.exists():
            pd.DataFrame().to_csv(out_dir / f"SYNTH_{table}.csv", index=False)
            actual_counts[table] = 0
            continue
        donor_df = pd.read_csv(path, dtype={"person_id": str}, low_memory=False)
        parts = []
        for donor, synth_ids in donor_to_synths.items():
            block = donor_df[donor_df["person_id"].astype(str) == donor]
            if block.empty:
                continue
            for sid in synth_ids:
                b = block.copy()
                b["person_id"] = sid
                parts.append(b)
        if not parts:
            out = donor_df.iloc[0:0].copy()
        else:
            out = pd.concat(parts, ignore_index=True)

        target = int(target_counts.get(table, len(out)))
        out = _match_row_count(out, target, rng)
        if id_col:
            out = _reassign_ids(out, id_col, prefix or table[:3])

        if table == "DRUG_EXPOSURE":
            # snap steroid concept rows' numeric fields missingness
            out = _preserve_missingness(out, config.STEROID_MISSINGNESS_COLS, steroid_rates, rng)
            # PTA-like: clinical range not applicable; quantity non-negative if present
            if "quantity" in out.columns:
                q = pd.to_numeric(out["quantity"], errors="coerce")
                out.loc[q.notna() & (q < 0), "quantity"] = 0

        if table == "MEASUREMENT":
            # snap PTA values on AC concepts
            mask = out["measurement_concept_id"].astype(str).isin(config.PTA_AC_IDS)
            if mask.any() and "value_as_number" in out.columns:
                vals = pd.to_numeric(out.loc[mask, "value_as_number"], errors="coerce")
                fixed = clinical.quantize(
                    vals.to_numpy(dtype=float),
                    clinical.PTA_STEP_SINGLE,
                    clinical.PTA_MIN,
                    clinical.PTA_MAX,
                )
                # restore nan
                nanmask = vals.isna().to_numpy()
                fixed = fixed.astype(float)
                fixed[nanmask] = np.nan
                out.loc[mask, "value_as_number"] = fixed

        out.to_csv(out_dir / f"SYNTH_{table}.csv", index=False)
        actual_counts[table] = len(out)

    # Assert counts
    mismatches = {t: (target_counts.get(t), actual_counts.get(t)) for t in config.CASE_TABLES if target_counts.get(t) != actual_counts.get(t)}
    # PERSON forced to spine length may differ from target if target was from S_train persons - should match
    manifest = {
        "target_counts": target_counts,
        "actual_counts": actual_counts,
        "mismatches": {k: {"target": v[0], "actual": v[1]} for k, v in mismatches.items()},
        "n_synth_persons": len(spine),
    }
    (out_dir / "count_check.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if mismatches:
        # For PERSON, allow equality to spine; force-fix other tables already matched
        # Re-check: only PERSON mismatch acceptable if target PERSON == n spine
        bad = {k: v for k, v in mismatches.items() if k != "PERSON" or v[1] != len(spine)}
        if bad and any(k != "PERSON" for k in bad):
            print("WARNING row-count mismatches:", bad)
    return actual_counts
