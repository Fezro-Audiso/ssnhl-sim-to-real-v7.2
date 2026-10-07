# v7.2 code snapshot

Seed: `20260730` (`code/config.py`).  
Winner: sequential cascade, Y := Siegel after PTA snap.

This snapshot does **not** include hospital OMOP extracts.  
`cohort_v5.py` will not run without local Case/Control tables.

What a reader can do with the public deposit

1. Read the sequential generator and Overcome rules.
2. Recompute table-level metrics from `02_public_data/tables/` and `methods.csv`.
3. Inspect the sanitized spine (`sequential_spine_sanitized.csv`).

What they cannot do without a DUA

- Rebuild S from raw OMOP.
- Rebuild the 11-table NN-donor package.

Do not add `run_sim_to_real_v8.py` or other post-submission experiments to this snapshot.
