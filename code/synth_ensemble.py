"""v7 shared-spine generators: solo, structure ensembles, mixtures, positive controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import clinical_constraints as clinical
import config
from synth_spine import synth_ctgan, synth_dummy, synth_gaussian, synth_marginal

SHARED_COLS = [
    "age_at_index",
    "sex",
    "hear_dx_pre",
    "htn_pre",
    "dm_pre",
    "steroid_pre",
    "steroid_post",
    "hbot_post",
    "pta_base",
    "pta_fu",
    "recovered",
]
BINARY_COLS = [
    "sex",
    "hear_dx_pre",
    "htn_pre",
    "dm_pre",
    "steroid_pre",
    "steroid_post",
    "hbot_post",
    "recovered",
]
F2_TWO_WAY_PAIRS = [("recovered", "steroid_post"), ("recovered", "hbot_post")]
# Causal generators never sample recovered; Y := Siegel(PTA) after snap.
GEN_COLS = [c for c in SHARED_COLS if c != "recovered"]


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def assign_donors(train: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = syn.copy()
    real = _numeric(train, cols).fillna(0)
    fake = _numeric(out, cols).fillna(0)
    ss = StandardScaler().fit(real)
    nn = NearestNeighbors(n_neighbors=1).fit(ss.transform(real))
    _, ix = nn.kneighbors(ss.transform(fake))
    out["_donor_person_id"] = train.iloc[ix.ravel()]["person_id"].astype(str).values
    return out


def _snap_and_id(
    train: pd.DataFrame,
    syn: pd.DataFrame,
    prefix: str,
    cols: list[str],
    post=None,
    recompute_siegel_after_post: bool = False,
) -> pd.DataFrame:
    """Clinical snap (Siegel from PTA). I|Y post keeps Y; PTA|I post recomputes Siegel."""
    out = clinical.apply_shared_constraints(syn)
    if callable(post):
        out = post(out)
        for c in BINARY_COLS:
            if c in out.columns and c != "recovered":
                out[c] = clinical.snap_binary(out[c], fill_missing=False)
        if recompute_siegel_after_post:
            out = clinical.apply_f2_constraints(out, recompute_siegel=True)
    donor = None
    if "_donor_person_id" in out.columns:
        donor = out["_donor_person_id"].to_numpy()
    elif "_donor_person_id" in syn.columns:
        donor = syn["_donor_person_id"].to_numpy()
    keep = [c for c in cols if c in out.columns]
    out = out[keep].copy()
    if donor is not None:
        out["_donor_person_id"] = np.asarray(donor)[: len(out)].astype(str)
    else:
        out = assign_donors(train, out, keep)
    out["person_id"] = [f"synth_{prefix}_{i}" for i in range(len(out))]
    return out


def _is_binary_series(s: pd.Series) -> bool:
    u = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(u) > 0 and set(np.round(u.astype(float), 8)).issubset({0.0, 1.0})


def synth_copula(train: pd.DataFrame, n: int, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Gaussian copula with empirical margins; binaries thresholded to match prevalence (not 0.5)."""
    X = _numeric(train, cols)
    med = X.median()
    Xf = X.fillna(med)
    z_cols = []
    for c in cols:
        v = Xf[c].to_numpy(dtype=float)
        r = pd.Series(v).rank(method="average").to_numpy()
        u = np.clip((r - 0.5) / max(len(v), 1), 1e-6, 1 - 1e-6)
        z_cols.append(norm.ppf(u))
    Z = np.column_stack(z_cols)
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = 0.5 * (corr + corr.T) + np.eye(len(cols)) * 1e-6
    try:
        zs = rng.multivariate_normal(np.zeros(len(cols)), corr, size=n)
    except np.linalg.LinAlgError:
        zs = rng.multivariate_normal(np.zeros(len(cols)), np.eye(len(cols)), size=n)
    out = {}
    for j, c in enumerate(cols):
        u = np.clip(norm.cdf(zs[:, j]), 1e-6, 1 - 1e-6)
        ref = np.sort(Xf[c].to_numpy(dtype=float))
        if _is_binary_series(Xf[c]):
            p = float(np.clip(Xf[c].mean(), 1e-4, 1 - 1e-4))
            out[c] = (u > (1.0 - p)).astype(int)
        else:
            idx = np.clip((u * (len(ref) - 1)).astype(int), 0, len(ref) - 1)
            out[c] = ref[idx]
    return pd.DataFrame(out)


def _p_i_given_y(df: pd.DataFrame, ycol: str, icol: str) -> dict:
    y = pd.to_numeric(df[ycol], errors="coerce")
    i = pd.to_numeric(df[icol], errors="coerce")
    out = {}
    for yv in (0, 1):
        sub = i[y == yv].dropna()
        out[yv] = float(sub.mean()) if len(sub) else float(i.dropna().mean() if i.notna().any() else 0.0)
    return out


def apply_2way_ipf(syn: pd.DataFrame, train: pd.DataFrame, pairs: list[tuple[str, str]], rng: np.random.Generator) -> pd.DataFrame:
    """Keep Y; resample each I | Y to match real P(I=1|Y). Does not lock I×I or 3-way cells."""
    out = syn.copy()
    for ycol, icol in pairs:
        if ycol not in out.columns or icol not in out.columns:
            continue
        rates = _p_i_given_y(train, ycol, icol)
        y = pd.to_numeric(out[ycol], errors="coerce").fillna(0).astype(int)
        i_new = np.zeros(len(out), dtype=float)
        for yv, p in rates.items():
            mask = y.to_numpy() == int(yv)
            k = int(mask.sum())
            if k == 0:
                continue
            i_new[mask] = rng.binomial(1, np.clip(p, 0.0, 1.0), size=k).astype(float)
        out[icol] = i_new
    return out


def synth_ipf_2way(train: pd.DataFrame, n: int, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Marginal 1-way base; I|Y targeting is applied after Siegel snap in generate_v7_methods."""
    return synth_marginal(train, n, cols, rng)


def synth_ystrat_copula(train: pd.DataFrame, n: int, cols: list[str], ycol: str, rng: np.random.Generator) -> pd.DataFrame:
    """Stratify on Y; copula on non-PTA feats; joint-draw PTA pairs from stratum so Siegel keeps Y."""
    pta_cols = [c for c in ("pta_base", "pta_fu") if c in cols]
    feat = [c for c in cols if c != ycol and c not in pta_cols]
    parts = []
    total = len(train)
    y = pd.to_numeric(train[ycol], errors="coerce")
    for yv, grp in train.groupby(y.fillna(-1)):
        share = len(grp) / total if total else 0
        n_grp = max(1, int(round(n * share)))
        if feat:
            syn = synth_copula(grp, n_grp, feat, rng) if len(grp) >= 8 else grp[feat].sample(
                n_grp, replace=True, random_state=int(rng.integers(0, 1_000_000))
            ).reset_index(drop=True)
        else:
            syn = pd.DataFrame(index=range(n_grp))
        syn[ycol] = 0 if yv == 0 else (1 if yv == 1 else np.nan)
        if pta_cols:
            ix = rng.integers(0, len(grp), size=n_grp)
            for c in pta_cols:
                syn[c] = grp[c].to_numpy()[ix]
        parts.append(syn)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=config.SEED).reset_index(drop=True)
    elif len(out) < n:
        extra = out.sample(n - len(out), replace=True, random_state=config.SEED)
        out = pd.concat([out, extra], ignore_index=True)
    return out[cols]


def synth_partial_cell(train: pd.DataFrame, n: int, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Lock recovered × steroid_post; joint-resample other cols (incl. PTA pair) within that 2-way."""
    ycol, icol = "recovered", "steroid_post"
    feat_other = [c for c in cols if c not in {ycol, icol}]
    parts = []
    total = len(train)
    for keys, grp in train.groupby([ycol, icol], dropna=False):
        share = len(grp) / total if total else 0
        n_grp = max(1, int(round(n * share)))
        if n_grp <= 0 or grp.empty:
            continue
        ix = rng.integers(0, len(grp), size=n_grp)
        out = {c: grp[c].to_numpy()[ix] for c in feat_other}
        if not isinstance(keys, tuple):
            keys = (keys,)
        out[ycol] = keys[0]
        out[icol] = keys[1]
        parts.append(pd.DataFrame(out))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=config.SEED).reset_index(drop=True)
    elif len(out) < n:
        extra = out.sample(n - len(out), replace=True, random_state=config.SEED)
        out = pd.concat([out, extra], ignore_index=True)
    return out[cols]


def synth_cell_preserving_shared(train: pd.DataFrame, n: int, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Full Y × steroid × HBOT cell lock. Joint-resample other cols so PTA pair stays coherent with Y."""
    keys = [c for c in ["recovered", "steroid_post", "hbot_post"] if c in cols]
    other = [c for c in cols if c not in keys]
    parts = []
    total = len(train)
    for kvals, grp in train.groupby(keys, dropna=False):
        share = len(grp) / total if total else 0
        n_grp = max(1, int(round(n * share)))
        if n_grp <= 0 or grp.empty:
            continue
        ix = rng.integers(0, len(grp), size=n_grp)
        out = {c: grp[c].to_numpy()[ix] for c in other}
        if not isinstance(kvals, tuple):
            kvals = (kvals,)
        for c, v in zip(keys, kvals):
            out[c] = v
        parts.append(pd.DataFrame(out))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=config.SEED).reset_index(drop=True)
    elif len(out) < n:
        extra = out.sample(n - len(out), replace=True, random_state=config.SEED)
        out = pd.concat([out, extra], ignore_index=True)
    return out[cols]


def apply_pta_fu_given_i(syn: pd.DataFrame, train: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Joint-assign (pta_base, pta_fu) within treatment cells from real P(PTA pair | I).
    (Legacy name kept; maps the PTA pair, not fu alone — fu-only broke gain sign.)
    """
    out = syn.copy()
    if "pta_fu" not in out.columns or "pta_base" not in out.columns:
        return out
    tr = train.copy()
    use_keys = [k for k in keys if k in out.columns and k in tr.columns]
    if not use_keys:
        return out
    for k in use_keys:
        out[k] = pd.to_numeric(out[k], errors="coerce").round().clip(0, 1)
        tr[k] = pd.to_numeric(tr[k], errors="coerce").round().clip(0, 1)
    rng = np.random.default_rng(config.SEED + 91)
    for kvals, grp_real in tr.groupby(use_keys, dropna=False):
        pairs = grp_real[["pta_base", "pta_fu"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pairs) == 0:
            continue
        if not isinstance(kvals, tuple):
            kvals = (kvals,)
        mask = np.ones(len(out), dtype=bool)
        for c, v in zip(use_keys, kvals):
            mask &= pd.to_numeric(out[c], errors="coerce").to_numpy() == v
        ksyn = int(mask.sum())
        if ksyn == 0:
            continue
        pick = rng.integers(0, len(pairs), size=ksyn)
        out.loc[mask, "pta_base"] = pairs["pta_base"].to_numpy()[pick]
        out.loc[mask, "pta_fu"] = pairs["pta_fu"].to_numpy()[pick]
    return out


def synth_istrata_copula(
    train: pd.DataFrame,
    n: int,
    cols: list[str],
    rng: np.random.Generator,
    keys: tuple[str, ...] = ("steroid_post", "hbot_post"),
) -> pd.DataFrame:
    """Stratify on treatment I; Gaussian copula within stratum on PTA + covariates (not on Y)."""
    feat = [c for c in cols if c not in keys]
    parts = []
    total = len(train)
    for kvals, grp in train.groupby(list(keys), dropna=False):
        share = len(grp) / total if total else 0
        n_grp = max(1, int(round(n * share)))
        if n_grp <= 0 or grp.empty:
            continue
        if len(grp) < 8:
            syn = grp[feat].sample(n_grp, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        else:
            syn = synth_copula(grp, n_grp, feat, rng)
        if not isinstance(kvals, tuple):
            kvals = (kvals,)
        for c, v in zip(keys, kvals):
            syn[c] = v
        parts.append(syn)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=config.SEED).reset_index(drop=True)
    elif len(out) < n:
        extra = out.sample(n - len(out), replace=True, random_state=config.SEED)
        out = pd.concat([out, extra], ignore_index=True)
    return out[cols]


def synth_sequential(
    train: pd.DataFrame,
    n: int,
    cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Structure-ensemble cascade (Type C): X → I|X → PTA|I. Does not sample recovered.
    Stage1: Gaussian copula on demographics + F1 pre exposures.
    Stage2: sample (steroid_post, hbot_post) from empiric peers in age_band×sex.
    Stage3: within each I cell, Gaussian copula on (pta_base, pta_fu) from real I peers.
    Y := Siegel after clinical snap.
    """
    x_cols = [c for c in ["age_at_index", "sex", "hear_dx_pre", "htn_pre", "dm_pre", "steroid_pre"] if c in cols]
    i_cols = [c for c in ["steroid_post", "hbot_post"] if c in cols]
    pta_cols = [c for c in ["pta_base", "pta_fu"] if c in cols]
    if not x_cols or not i_cols or not pta_cols:
        raise ValueError("sequential needs X, I, and PTA columns")

    # --- Stage 1: X ---
    out = synth_copula(train, n, x_cols, rng)
    age = pd.to_numeric(out["age_at_index"], errors="coerce").fillna(train["age_at_index"].median())
    sex = pd.to_numeric(out["sex"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    out["age_at_index"] = age
    out["sex"] = sex
    band = (age // 10).astype(int) * 10

    tr = train.copy()
    tr_age = pd.to_numeric(tr["age_at_index"], errors="coerce")
    tr_sex = pd.to_numeric(tr["sex"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    tr_band = (tr_age // 10).astype(int) * 10

    # --- Stage 2: I | age_band, sex ---
    i_mat = np.zeros((n, len(i_cols)), dtype=float)
    for j in range(n):
        b, s = int(band.iloc[j]), int(sex.iloc[j])
        pool = tr[(tr_band == b) & (tr_sex == s)]
        if len(pool) < 5:
            pool = tr[tr_sex == s]
        if len(pool) == 0:
            pool = tr
        pick = pool.iloc[int(rng.integers(0, len(pool)))]
        for k, c in enumerate(i_cols):
            i_mat[j, k] = float(pd.to_numeric(pick[c], errors="coerce"))
    for k, c in enumerate(i_cols):
        out[c] = i_mat[:, k]

    # --- Stage 3: PTA | I (copula within treatment cell) ---
    for c in pta_cols:
        out[c] = np.nan
    for keys, syn_grp in out.groupby(i_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mask_real = np.ones(len(tr), dtype=bool)
        for c, v in zip(i_cols, keys):
            mask_real &= pd.to_numeric(tr[c], errors="coerce").to_numpy() == v
        real_grp = tr.loc[mask_real]
        idx = syn_grp.index.to_numpy()
        n_grp = len(idx)
        if n_grp == 0:
            continue
        if len(real_grp) < 8:
            samp = real_grp[pta_cols].sample(n_grp, replace=True, random_state=int(rng.integers(0, 1_000_000)))
            for c in pta_cols:
                out.loc[idx, c] = samp[c].to_numpy()
        else:
            samp = synth_copula(real_grp, n_grp, pta_cols, rng)
            for c in pta_cols:
                out.loc[idx, c] = samp[c].to_numpy()

    # fill any leftover PTA from global
    for c in pta_cols:
        miss = out[c].isna()
        if miss.any():
            out.loc[miss, c] = rng.choice(pd.to_numeric(tr[c], errors="coerce").dropna().to_numpy(), size=int(miss.sum()))

    # ensure column order / presence
    for c in cols:
        if c not in out.columns and c != "recovered":
            if c in tr.columns:
                out[c] = rng.choice(tr[c].to_numpy(), size=n, replace=True)
    keep = [c for c in cols if c != "recovered" and c in out.columns]
    return out[keep].reset_index(drop=True)


def synth_mixture(parts: list[pd.DataFrame], n: int, rng: np.random.Generator) -> pd.DataFrame:
    stacked = pd.concat(parts, ignore_index=True)
    if len(stacked) >= n:
        return stacked.sample(n, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    extra = stacked.sample(n - len(stacked), replace=True, random_state=config.SEED)
    return pd.concat([stacked, extra], ignore_index=True)


def generate_v7_methods(train: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    """Return snapped shared spines. Causal gens use GEN_COLS (no recovered). CTGAN failures omitted."""
    rng = np.random.default_rng(config.SEED)
    n = len(train)
    gcols = [c for c in GEN_COLS if c in train.columns]
    base_full = train[cols + (["person_id"] if "person_id" in train.columns else [])].copy()
    base_gen = train[gcols + (["person_id"] if "person_id" in train.columns else [])].copy()
    gens: dict[str, pd.DataFrame] = {}

    # --- PC ---
    gens["dummy"] = _snap_and_id(train, synth_dummy(base_full, n, rng), "dummy", cols)
    gens["cell_preserving"] = _snap_and_id(
        train, synth_cell_preserving_shared(train, n, cols, rng), "cell", cols
    )

    # --- AC: Y-peeking association targets (report only, cannot win) ---
    gens["ipf_2way"] = _snap_and_id(
        train,
        synth_ipf_2way(train, n, cols, rng),
        "ipf",
        cols,
        post=lambda d: apply_2way_ipf(d, train, F2_TWO_WAY_PAIRS, rng),
    )
    gens["ystrat_copula"] = _snap_and_id(
        train, synth_ystrat_copula(train, n, cols, "recovered", rng), "ystrat", cols
    )
    gens["partial_cell"] = _snap_and_id(
        train,
        synth_partial_cell(train, n, cols, rng),
        "pcell",
        cols,
        post=lambda d: apply_2way_ipf(d, train, [("recovered", "steroid_post")], rng),
    )

    # --- N + causal C: no free recovered ---
    gens["marginal"] = _snap_and_id(train, synth_marginal(base_gen, n, gcols, rng), "marg", cols)
    gens["gaussian"] = _snap_and_id(train, synth_gaussian(base_gen, n, gcols, rng), "gauss", cols)
    gens["copula"] = _snap_and_id(train, synth_copula(train, n, gcols, rng), "copula", cols)
    gens["istrata_copula"] = _snap_and_id(
        train, synth_istrata_copula(train, n, gcols, rng), "istrata", cols
    )
    print("  [v7] sequential cascade X->I->PTA...")
    gens["sequential"] = _snap_and_id(train, synth_sequential(train, n, gcols, rng), "seq", cols)
    gens["ipf_pta_given_i"] = _snap_and_id(
        train,
        synth_marginal(base_gen, n, gcols, rng),
        "ipfpta",
        cols,
        post=lambda d: apply_pta_fu_given_i(d, train, ["steroid_post", "hbot_post"]),
        recompute_siegel_after_post=True,
    )
    gens["partial_pta_steroid"] = _snap_and_id(
        train,
        synth_marginal(base_gen, n, gcols, rng),
        "ppta",
        cols,
        post=lambda d: apply_pta_fu_given_i(d, train, ["steroid_post"]),
        recompute_siegel_after_post=True,
    )

    try:
        print("  [v7.1] CTGAN (N3, no recovered)...")
        gens["ctgan"] = _snap_and_id(train, synth_ctgan(base_gen, n, gcols, config.SEED), "ctgan", cols)
    except Exception as e:
        print(f"  [v7.1] CTGAN failed: {e}")

    seed_parts = []
    n_seeds = 5
    take = max(1, n // n_seeds)
    for k in range(n_seeds):
        try:
            print(f"  [v7.1] CTGAN seed {k + 1}/{n_seeds}...")
            part = synth_ctgan(base_gen, take + (1 if k < n % n_seeds else 0), gcols, config.SEED + 17 * (k + 1))
            seed_parts.append(part)
        except Exception as e:
            print(f"  [v7.1] CTGAN seed {k + 1} failed: {e}")
    if seed_parts:
        mix = synth_mixture(seed_parts, n, rng)
        gens["ctgan_seed_ensemble"] = _snap_and_id(train, mix, "ctganens", cols)

    n1 = n // 2
    mix_mg = pd.concat(
        [synth_marginal(base_gen, n1, gcols, rng), synth_gaussian(base_gen, n - n1, gcols, rng)],
        ignore_index=True,
    )
    gens["mix_marginal_gaussian"] = _snap_and_id(train, mix_mg, "mixmg", cols)
    mix_cm = pd.concat(
        [synth_copula(train, n1, gcols, rng), synth_marginal(base_gen, n - n1, gcols, rng)],
        ignore_index=True,
    )
    gens["mix_copula_marginal"] = _snap_and_id(train, mix_cm, "mixcm", cols)

    return gens


POSITIVE_CONTROLS = frozenset({"dummy", "cell_preserving"})
ASSOC_CONTROLS = frozenset({"ystrat_copula", "ipf_2way", "partial_cell"})
STRUCTURE_IDS = ("copula", "sequential", "istrata_copula", "ipf_pta_given_i", "partial_pta_steroid")
SOLO_IDS = ("marginal", "gaussian", "ctgan")
SENS_IDS = ("ctgan_seed_ensemble", "mix_marginal_gaussian", "mix_copula_marginal")
CONSTRAINT_RANK = {
    "copula": 0,
    "sequential": 1,
    "istrata_copula": 2,
    "ipf_pta_given_i": 3,
    "partial_pta_steroid": 4,
    "marginal": 5,
    "gaussian": 5,
    "ctgan": 5,
    "ctgan_seed_ensemble": 6,
    "mix_marginal_gaussian": 6,
    "mix_copula_marginal": 6,
}
WINNER_POOL = frozenset(STRUCTURE_IDS + SOLO_IDS + SENS_IDS)
