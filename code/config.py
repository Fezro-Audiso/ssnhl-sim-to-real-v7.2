"""Runtime config for Sim-to-Real pipeline v6 (PTA-index, multi-table, 1y windows)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "Case (SSNHL)"
CTRL_DIR = ROOT / "Control (Normal)"
OUT_DIR = ROOT / "results" / "sim_to_real_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260730
TRAIN_FRAC = 0.8

LOOKBACK_DAYS = 365
FOLLOWUP_DAYS = 365
FRONT_YEAR_DAYS = 365

STEROID_CONCEPT_IDS = {"42919316", "19068126", "21600828"}

# Official map: [CDM] HBOT&PTA_MEASUREMENT_CONCEPT_ID.xlsx
PTA_4FREQ_AC = {
    "right": {
        500: "900000029",
        1000: "900000030",
        2000: "900000031",
        4000: "900000033",
    },
    "left": {
        500: "900000042",
        1000: "900000043",
        2000: "900000044",
        4000: "900000046",
    },
}
PTA_AC_IDS = {cid for ear in PTA_4FREQ_AC.values() for cid in ear.values()}

HBOT_PROCEDURE_IDS = {"4087611"}

# SSNHL / hearing diagnosis proxies for index = first diagnosis
SSNHL_DX_CONCEPT_IDS = {"4193704", "4174977", "4098611", "140673"}

F1_DIAGNOSIS_PROXY = SSNHL_DX_CONCEPT_IDS
F1_HTN = {"320128"}
F1_DM = {"201826", "44793113", "201254"}

AGE_BAND = 10
MATCH_RATIO = 1
TAU = 0.20
SIGN_THR = 0.80
CI_OVERLAP_THR = 0.80
CLINICAL_MAX_VIOL = 0.02

CASE_TABLES = [
    "PERSON",
    "OBSERVATION_PERIOD",
    "CONDITION_OCCURRENCE",
    "DRUG_EXPOSURE",
    "PROCEDURE_OCCURRENCE",
    "MEASUREMENT",
    "OBSERVATION",
    "DEVICE_EXPOSURE",
    "DEATH",
    "VISIT_OCCURRENCE",
    "VISIT_DETAIL",
]

STEROID_MISSINGNESS_COLS = [
    "quantity",
    "days_supply",
    "route_concept_id",
    "dose_unit_source_value",
    "sig",
    "drug_exposure_end_date",
]
