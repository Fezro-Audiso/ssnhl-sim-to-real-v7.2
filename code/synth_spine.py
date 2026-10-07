"""Person-spine generators + clinical snap for v5."""
from __future__ import annotations

import numpy as np
import pandas as pd

import clinical_constraints as clinical
import config


def synth_dummy(train: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.integers(0, len(train), size=n)
    out = train.iloc[idx].copy().reset_index(drop=True)
    out["person_id"] = [f"synth_dummy_{i}" for i in range(n)]
    out["_donor_person_id"] = train.iloc[idx]["person_id"].astype(str).values
    return out


def synth_marginal(train: pd.DataFrame, n: int, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = {c: rng.choice(train[c].to_numpy(), size=n, replace=True) for c in cols}
    df = pd.DataFrame(out)
    # donor: nearest by age/sex if available else random
    donors = rng.choice(train["person_id"].astype(str).to_numpy(), size=n, replace=True)
    df["person_id"] = [f"synth_marg_{i}" for i in range(n)]
    df["_donor_person_id"] = donors
    return df


def synth_gaussian(train: pd.DataFrame, n: int, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    x = train[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mu = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(mu, inds[1])
    cov = np.cov(x, rowvar=False) + np.eye(len(cols)) * 1e-6
    samples = rng.multivariate_normal(mu, cov, size=n)
    df = pd.DataFrame(samples, columns=cols)
    for c in cols:
        if set(pd.Series(train[c].dropna().unique())).issubset({0, 1}):
            df[c] = (df[c] >= 0.5).astype(int)
    # NN donor in standardized feature space
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    real = StandardScaler().fit_transform(train[cols].apply(pd.to_numeric, errors="coerce").fillna(0))
    syn = StandardScaler().fit(train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)).transform(
        df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    nn = NearestNeighbors(n_neighbors=1).fit(real)
    _, ix = nn.kneighbors(syn)
    df["person_id"] = [f"synth_gauss_{i}" for i in range(n)]
    df["_donor_person_id"] = train.iloc[ix.ravel()]["person_id"].astype(str).values
    return df


def synth_ctgan(train: pd.DataFrame, n: int, cols: list[str], seed: int) -> pd.DataFrame:
    from ctgan import CTGAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    discrete = [c for c in cols if set(pd.Series(train[c].dropna().unique())).issubset({0, 1})]
    model = CTGAN(epochs=50, verbose=False)
    model.set_random_state(seed)
    model.fit(train[cols], discrete)
    df = model.sample(n)
    real = StandardScaler().fit_transform(train[cols].apply(pd.to_numeric, errors="coerce").fillna(0))
    syn = StandardScaler().fit(train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)).transform(
        df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    nn = NearestNeighbors(n_neighbors=1).fit(real)
    _, ix = nn.kneighbors(syn)
    df["person_id"] = [f"synth_ctgan_{i}" for i in range(n)]
    df["_donor_person_id"] = train.iloc[ix.ravel()]["person_id"].astype(str).values
    return df


def synth_stratified(train, n, cols, ycol, rng, base):
    parts = []
    total = len(train)
    for yval, grp in train.groupby(ycol):
        share = len(grp) / total if total else 0
        n_grp = max(1, int(round(n * share)))
        feat = [c for c in cols if c != ycol]
        if base == "dummy":
            syn = synth_dummy(grp, n_grp, rng)
        elif base == "marginal":
            syn = synth_marginal(grp, n_grp, feat, rng)
        elif base == "gaussian":
            syn = synth_gaussian(grp, n_grp, feat, rng)
        elif base == "ctgan":
            syn = synth_ctgan(grp, n_grp, feat, config.SEED)
        else:
            raise ValueError(base)
        syn[ycol] = yval
        parts.append(syn)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=config.SEED).reset_index(drop=True)
    elif len(out) < n:
        extra = out.sample(n - len(out), replace=True, random_state=config.SEED)
        out = pd.concat([out, extra], ignore_index=True)
    out["person_id"] = [f"synth_{base}_strat_{i}" for i in range(len(out))]
    return out


def synth_cell_preserving(train, n, ycol, exposure_cols, rng):
    feat_cols = [c for c in train.columns if c not in {"person_id", "_donor_person_id"}]
    resample_cols = [c for c in feat_cols if c not in [ycol] + exposure_cols]
    total = len(train)
    parts = []
    for keys, grp in train.groupby([ycol] + exposure_cols, dropna=False):
        share = len(grp) / total if total else 0
        n_grp = max(1, int(round(n * share)))
        if n_grp <= 0 or grp.empty:
            continue
        # Joint row draw so PTA pairs (and other joints) stay coherent with locked Y×I.
        ix = rng.integers(0, len(grp), size=n_grp)
        out = {c: grp[c].to_numpy()[ix] for c in resample_cols}
        donors = grp["person_id"].astype(str).to_numpy()[ix]
        if not isinstance(keys, tuple):
            keys = (keys,)
        for c, v in zip([ycol] + exposure_cols, keys):
            out[c] = v
        df = pd.DataFrame(out)
        df["_donor_person_id"] = donors
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=config.SEED).reset_index(drop=True)
    elif len(out) < n:
        extra = out.sample(n - len(out), replace=True, random_state=config.SEED)
        out = pd.concat([out, extra], ignore_index=True)
    out["person_id"] = [f"synth_cell_{i}" for i in range(len(out))]
    return out


def generate_all(train: pd.DataFrame, feature_cols: list[str], family: str, ycol: str, exposures: list[str]):
    rng = np.random.default_rng(config.SEED)
    n = len(train)
    need = feature_cols + (["person_id"] if "person_id" in train.columns else [])
    base = train[need].copy() if "person_id" in train.columns else train[feature_cols].assign(person_id=np.arange(len(train)).astype(str))
    gens = {
        "dummy": synth_dummy(base, n, rng),
        "marginal": synth_marginal(base, n, feature_cols, rng),
        "gaussian": synth_gaussian(base, n, feature_cols, rng),
    }
    try:
        print(f"  [{family}] CTGAN...")
        gens["ctgan"] = synth_ctgan(base, n, feature_cols, config.SEED)
    except Exception as e:
        print(f"  [{family}] CTGAN failed: {e}")

    if family == "F2":
        for b in ["dummy", "marginal", "gaussian"]:
            gens[f"{b}_strat"] = synth_stratified(base, n, feature_cols, ycol, rng, b)
        gens["cell_preserving"] = synth_cell_preserving(base, n, ycol, exposures, rng)
        try:
            gens["ctgan_strat"] = synth_stratified(base, n, feature_cols, ycol, rng, "ctgan")
        except Exception as e:
            print(f"  [{family}] ctgan_strat failed: {e}")

    snapped = {}
    for name, syn in gens.items():
        snapped[name] = clinical.apply_constraints(syn, family)
        if "_donor_person_id" in syn.columns and "_donor_person_id" not in snapped[name].columns:
            snapped[name]["_donor_person_id"] = syn["_donor_person_id"].values
        if "person_id" in syn.columns:
            snapped[name]["person_id"] = syn["person_id"].values
    return snapped
