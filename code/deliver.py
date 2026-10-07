"""Write Step2-adopted deliverable only; clean scratch generators."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config
from synth_package import build_package


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def adopt_and_deliver(
    family: str,
    eval_res: dict,
    generators: dict,
    donor_dir: Path,
    target_counts: dict,
    profile: dict,
    out_root: Path,
):
    adopted = eval_res.get("adopted_generator")
    deliv = out_root / "deliverable" / family
    if deliv.exists():
        try:
            shutil.rmtree(deliv)
        except PermissionError:
            # Windows: folder locked by Explorer/antivirus — write beside it.
            stamp = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
            deliv = out_root / "deliverable" / f"{family}_{stamp}"
    deliv.mkdir(parents=True, exist_ok=True)

    manifest = {
        "family": family,
        "step1_selected": eval_res.get("step1_selected"),
        "step2_co_primary": eval_res.get("step2_co_primary"),
        "adopted_generator": adopted,
        "adoption_rule": eval_res.get("adoption_rule"),
        "primary_overcome": eval_res.get("step2", {}).get("primary_overcome"),
        "or_overcome": eval_res.get("step2", {}).get("or_overcome"),
        "risk_causal_overcome": eval_res.get("step2", {}).get("risk_causal_overcome"),
    }

    if not adopted:
        manifest["status"] = "FAIL_NO_ADOPTED"
        (deliv / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    spine = generators[adopted]
    counts = build_package(
        spine=spine,
        donor_dir=donor_dir,
        target_counts=target_counts,
        profile=profile,
        out_dir=deliv,
    )
    manifest["status"] = "OK"
    manifest["actual_counts"] = counts
    manifest["target_counts"] = target_counts
    # spine snapshot
    spine.to_csv(deliv / "synth_spine.csv", index=False)
    (deliv / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return manifest


def write_report(summary: dict, attrition: dict, out_root: Path):
    lines = [
        "# Sim-to-Real v6 Results",
        "",
        "Index = first complete AC 4PTA; S = PTA-trackable (base+FU); F1/F2 share S;",
        "windows +/-365d; deliverable = F2 Step2-pass / F1 best Step2 among eligible.",
        "",
        "## Attrition",
    ]
    for k, v in attrition.items():
        lines.append(f"- {k}: **{v}**")
    lines += ["", "## Step x Family"]
    for fam in ["F1", "F2"]:
        if fam not in summary:
            continue
        r = summary[fam]
        st = r.get("step2", {})
        lines += [
            f"### {fam}",
            f"- Step1-selected: **{r.get('step1_selected')}**",
            f"- Step1-eligible: `{r.get('step1_eligible')}`",
            f"- OR overcome: **{st.get('or_overcome')}**",
            f"- Risk/causal overcome: **{st.get('risk_causal_overcome')}**",
            f"- Primary on selected: **{st.get('primary_overcome')}**",
            f"- Co-primary: **{st.get('co_primary_generator')}**",
            f"- Adoption rule: {r.get('adoption_rule')}",
            f"- **Adopted deliverable: {r.get('adopted_generator')}**",
            "",
        ]
    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
