#!/usr/bin/env python3
"""
ariel_ranking_no_waic_with_mags.py

Fixes requested
---------------
1) Removes WAIC computation (and ArviZ dependency) for now.
2) Adds star magnitudes as features (J/H/Ks/W1/W2/TESS).
3) Uses separate models for Transit and Eclipse, with method-appropriate targets:
   - Transit model  : predicts log(Tier 1 Transits)
   - Eclipse model  : predicts log(Tier 1 Eclipses)
4) Uses ONLY columns available in BOTH Known and TPC, including the transit timing
   + depth uncertainties you listed (and transit duration [hr]/[hrs] + its errors).

Ranking (TPC only)
------------------
For each method subset, we predict a posterior over required events (Tier-1 count),
then score each candidate by

    score = P(pred_required <= Available_{Transits/Eclipses}) / median(pred_required)

So higher is better: fewer required events and feasible within the available window.

Dependencies
------------
pip install "jax[cpu]" numpyro pandas numpy matplotlib scikit-learn

Example
-------
python ariel_ranking_no_waic_with_mags.py \
  --known Ariel_MCS_Known_2024-07-09.csv \
  --tpc   Ariel_MCS_TPCs_2024-07-09.csv \
  --model bnn --inference svi \
  --outputs outputs_ariel_spec_bnn

Notes
-----
- "Transit Mid Time" is included because you asked; absolute epoch is not physical,
  so we center it (median subtraction).
- We median-impute missing magnitudes (e.g., W1/W2 sometimes missing) before scaling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.optim import Adam


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def sym_sigma(lower: pd.Series, upper: pd.Series) -> np.ndarray:
    lo = np.abs(_num(lower).to_numpy(float))
    hi = np.abs(_num(upper).to_numpy(float))
    return 0.5 * (lo + hi)

def safe_log(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log(np.maximum(x, floor))

def mag_to_logflux(mag: np.ndarray) -> np.ndarray:
    """
    Relative flux F ~ 10^(-0.4 m). Return ln(F).
    This is linear in magnitude: ln F = -0.4 ln(10) m.
    """
    m = np.asarray(mag, dtype=float)
    out = np.full_like(m, np.nan, dtype=float)
    ok = np.isfinite(m)
    out[ok] = -0.4 * np.log(10.0) * m[ok]
    return out

def percentile_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    ok = np.isfinite(x)
    if np.sum(ok) < 2:
        return out
    r = x[ok].argsort().argsort().astype(float)
    out[ok] = r / (len(r) - 1)
    return out

def require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} is missing required columns: {', '.join(miss)}")

def get_any(df: pd.DataFrame, names: List[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    # return all-NaN series of correct length
    return pd.Series([np.nan] * len(df))


# -------------------------
# Feature builders
# -------------------------
MAG_COLS = ["Star J Mag", "Star H Mag", "Star Ks Mag", "Star W1 Mag", "Star W2 Mag", "Star TESS Mag"]

BASE_COLS = ["Preferred Method"] + MAG_COLS

# Transit columns (note: Known uses [hrs] plural; TPC uses [hr] singular)
TRANSIT_COLS = [
    "Transit Mid Time",
    "Transit Mid Time Error Lower [days]",
    "Transit Mid Time Error Upper [days]",
    "Transit Depth [%]",
    "Transit Depth Error Lower [%]",
    "Transit Depth Error Upper [%]",
    "Transit Duration T14 [s]",
    "Available Transits",
]
TRANSIT_DUR_HR = ["Transit Duration [hr]", "Transit Duration [hrs]"]
TRANSIT_DUR_HR_LO = ["Transit Duration Error Lower [hr]", "Transit Duration Error Lower [hrs]"]
TRANSIT_DUR_HR_HI = ["Transit Duration Error Upper [hr]", "Transit Duration Error Upper [hrs]"]

# Eclipse columns
ECLIPSE_COLS = [
    "Eclipse Mid Time",
    "Eclipse Duration E14 [s]",
    "Available Eclipses",
]

# Targets (Known must have these)
TARGET_TRANSIT = "Tier 1 Transits"
TARGET_ECLIPSE = "Tier 1 Eclipses"


def build_transit_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    require_cols(df, BASE_COLS + TRANSIT_COLS, "DataFrame (transit)")

    d = df.copy()

    # Depth + uncertainty
    depth = _num(d["Transit Depth [%]"]).to_numpy(float) / 100.0
    s_depth = sym_sigma(d["Transit Depth Error Lower [%]"], d["Transit Depth Error Upper [%]"]) / 100.0
    s_depth_rel = s_depth / np.maximum(depth, 1e-12)

    # Mid-time + uncertainty
    t0 = _num(d["Transit Mid Time"]).to_numpy(float)
    s_t0 = sym_sigma(d["Transit Mid Time Error Lower [days]"], d["Transit Mid Time Error Upper [days]"])

    # Duration (hr) + uncertainty (if present in both; it is)
    dur_hr = _num(get_any(d, TRANSIT_DUR_HR)).to_numpy(float)
    s_dur = sym_sigma(get_any(d, TRANSIT_DUR_HR_LO), get_any(d, TRANSIT_DUR_HR_HI))

    # T14 (s) -> hr
    t14 = _num(d["Transit Duration T14 [s]"]).to_numpy(float)
    t14_hr = t14 / 3600.0

    # Available transits
    n_av = _num(d["Available Transits"]).to_numpy(float)

    # Magnitudes -> log fluxes
    for c in MAG_COLS:
        d[f"feat_logF_{c.split()[1]}"] = mag_to_logflux(_num(d[c]).to_numpy(float))

    # Engineered features (requested errors emphasized)
    d["feat_log_depth"] = safe_log(depth)
    d["feat_log_depth_snr"] = safe_log(depth / np.maximum(s_depth, 1e-12))
    d["feat_log_depth_relunc"] = safe_log(s_depth_rel)

    d["feat_log_t0_sigma"] = safe_log(np.maximum(s_t0, 1e-12))
    d["feat_t0_centered"] = t0 - np.nanmedian(t0)

    d["feat_log_dur_hr"] = safe_log(np.maximum(dur_hr, 1e-12))
    d["feat_log_dur_snr"] = safe_log(np.maximum(dur_hr, 1e-12) / np.maximum(s_dur, 1e-12))
    d["feat_log_t14_hr"] = safe_log(np.maximum(t14_hr, 1e-12))

    d["feat_log_avail_transits"] = safe_log(n_av + 1.0)

    feat_cols = [
        "feat_log_depth",
        "feat_log_depth_snr",
        "feat_log_depth_relunc",
        "feat_log_t0_sigma",
        "feat_t0_centered",
        "feat_log_dur_hr",
        "feat_log_dur_snr",
        "feat_log_t14_hr",
        "feat_log_avail_transits",
    ] + [f"feat_logF_{c.split()[1]}" for c in MAG_COLS]

    X = d[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    return X, feat_cols, d


def build_eclipse_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    require_cols(df, BASE_COLS + ECLIPSE_COLS, "DataFrame (eclipse)")

    d = df.copy()

    e0 = _num(d["Eclipse Mid Time"]).to_numpy(float)
    e14 = _num(d["Eclipse Duration E14 [s]"]).to_numpy(float) / 3600.0
    n_av = _num(d["Available Eclipses"]).to_numpy(float)

    # Magnitudes -> log fluxes
    for c in MAG_COLS:
        d[f"feat_logF_{c.split()[1]}"] = mag_to_logflux(_num(d[c]).to_numpy(float))

    d["feat_e0_centered"] = e0 - np.nanmedian(e0)
    d["feat_log_e14_hr"] = safe_log(np.maximum(e14, 1e-12))
    d["feat_log_avail_eclipses"] = safe_log(n_av + 1.0)

    feat_cols = [
        "feat_e0_centered",
        "feat_log_e14_hr",
        "feat_log_avail_eclipses",
    ] + [f"feat_logF_{c.split()[1]}" for c in MAG_COLS]

    X = d[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    return X, feat_cols, d


# -------------------------
# NumPyro models (Student-T regression on log target)
# -------------------------
def linear_student_t(x: jnp.ndarray, y: Optional[jnp.ndarray] = None):
    n, p = x.shape
    b0 = numpyro.sample("b0", dist.Normal(0.0, 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand((p,)))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    nu = numpyro.sample("nu", dist.Exponential(1/10.0)) + 2.0
    mu = b0 + jnp.dot(x, beta)
    numpyro.sample("y", dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)

def bnn_student_t(x: jnp.ndarray, y: Optional[jnp.ndarray] = None, hidden: int = 16):
    n, p = x.shape
    w1 = numpyro.sample("w1", dist.Normal(0.0, 1.0).expand((p, hidden)))
    b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0).expand((hidden,)))
    w2 = numpyro.sample("w2", dist.Normal(0.0, 1.0).expand((hidden, 1)))
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    nu = numpyro.sample("nu", dist.Exponential(1/10.0)) + 2.0
    h = jnp.tanh(jnp.dot(x, w1) + b1)
    mu = jnp.squeeze(jnp.dot(h, w2), axis=-1) + b2
    numpyro.sample("y", dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)


# -------------------------
# Fit + predict
# -------------------------
@dataclass
class Fitted:
    model: str
    inference: str
    imputer: SimpleImputer
    scaler: StandardScaler
    feature_cols: List[str]
    guide: Optional[AutoDiagonalNormal]
    params: Dict[str, jnp.ndarray]
    posterior: Dict[str, jnp.ndarray]   # NUTS


def fit_numpyro(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: str,
    inference: str,
    seed: int,
    hidden: int,
    svi_steps: int,
    lr: float,
) -> Tuple[Fitted, Dict[str, float]]:
    rng = jax.random.PRNGKey(seed)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X_train)
    Xs = scaler.fit_transform(X_imp)

    x = jnp.array(Xs)
    y = jnp.array(y_train)

    if model == "linear":
        model_fn = linear_student_t
        model_kwargs = {}
    elif model == "bnn":
        model_fn = bnn_student_t
        model_kwargs = {"hidden": hidden}
    else:
        raise ValueError("model must be 'linear' or 'bnn'")

    metrics: Dict[str, float] = {}

    if inference == "nuts":
        kernel = NUTS(model_fn)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=1500, num_chains=2, progress_bar=True)
        mcmc.run(rng, x=x, y=y, **model_kwargs)
        posterior = mcmc.get_samples(group_by_chain=False)
        metrics["n_draws"] = float(posterior[next(iter(posterior.keys()))].shape[0])
        return Fitted(model, inference, imputer, scaler, [], None, {}, posterior), metrics

    if inference == "svi":
        guide = AutoDiagonalNormal(model_fn)
        svi = SVI(model_fn, guide, Adam(lr), loss=Trace_ELBO())
        state = svi.init(rng, x=x, y=y, **model_kwargs)
        last = None
        for _ in range(svi_steps):
            state, loss = svi.update(state, x=x, y=y, **model_kwargs)
            last = float(loss)
        params = svi.get_params(state)
        metrics["final_elbo"] = -last if last is not None else np.nan
        return Fitted(model, inference, imputer, scaler, [], guide, params, {}), metrics

    raise ValueError("inference must be 'nuts' or 'svi'")


def predict_logy_draws(
    fit: Fitted,
    X: np.ndarray,
    model: str,
    hidden: int,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    rng = jax.random.PRNGKey(seed)

    X_imp = fit.imputer.transform(X)
    Xs = fit.scaler.transform(X_imp)
    x = jnp.array(Xs)

    model_fn = linear_student_t if model == "linear" else bnn_student_t
    model_kwargs = {} if model == "linear" else {"hidden": hidden}

    if fit.inference == "nuts":
        pred = Predictive(model_fn, posterior_samples=fit.posterior, num_samples=n_draws)
        out = pred(rng, x=x, y=None, **model_kwargs)
        ydraw = np.array(out["y"])  # (draws, N)
    else:
        assert fit.guide is not None
        pred = Predictive(model_fn, guide=fit.guide, params=fit.params, num_samples=n_draws)
        out = pred(rng, x=x, y=None, **model_kwargs)
        ydraw = np.array(out["y"])  # (draws, N)

    if ydraw.shape[0] == n_draws:
        ydraw = ydraw.T
    return ydraw  # (N, draws)


def summarize_required_counts(logy_draws: np.ndarray, clip_min: int = 1, clip_max: Optional[int] = None):
    req = np.rint(np.exp(logy_draws)).astype(int)
    req = np.maximum(req, clip_min)
    if clip_max is not None:
        req = np.minimum(req, clip_max)
    med = np.median(req, axis=1)
    p16 = np.quantile(req, 0.16, axis=1)
    p84 = np.quantile(req, 0.84, axis=1)
    return req, med, p16, p84


# -------------------------
# Plotting
# -------------------------
def plot_calibration(y_true: np.ndarray, y_med: np.ndarray, outpath: Path, title: str):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_med, alpha=0.6)
    lo = min(np.min(y_true), np.min(y_med))
    hi = max(np.max(y_true), np.max(y_med))
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("True target (count)")
    plt.ylabel("Predicted median (count)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_feature_weights_linear(posterior: Dict[str, jnp.ndarray], feature_cols: List[str], outpath: Path, title: str):
    if "beta" not in posterior:
        return
    beta = np.array(posterior["beta"])  # (draws, p)
    mu = beta.mean(axis=0)
    lo = np.quantile(beta, 0.025, axis=0)
    hi = np.quantile(beta, 0.975, axis=0)

    order = np.argsort(mu)
    plt.figure(figsize=(9,5))
    y = np.arange(len(feature_cols))
    plt.errorbar(mu[order], y, xerr=[mu[order]-lo[order], hi[order]-mu[order]], fmt="o", alpha=0.9)
    plt.yticks(y, [feature_cols[i] for i in order])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(title + " (posterior β with 95% CI)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_top_scatter(df: pd.DataFrame, score_col: str, outpath: Path, title: str):
    plt.figure(figsize=(7,5))
    plt.scatter(df["depth_frac"], df[score_col], alpha=0.25)
    top = df.sort_values(score_col, ascending=False).head(80)
    plt.scatter(top["depth_frac"], top[score_col], alpha=0.9)
    plt.xscale("log")
    plt.xlabel("Transit Depth (fraction)")
    plt.ylabel(score_col)
    plt.title(title + " (top 80 highlighted)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_score_hist(df: pd.DataFrame, score_col: str, outpath: Path, title: str):
    plt.figure(figsize=(7,4.5))
    x = df[score_col].to_numpy(float)
    x = x[np.isfinite(x)]
    plt.hist(x, bins=40, alpha=0.9)
    plt.xlabel(score_col)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------
# Run pipeline per method
# -------------------------
def run_transit(
    known: pd.DataFrame,
    tpc: pd.DataFrame,
    outputs: Path,
    model: str,
    inference: str,
    seed: int,
    hidden: int,
    svi_steps: int,
    lr: float,
    n_draws_pred: int,
) -> None:
    plots = outputs / "plots"
    ensure_dir(plots)

    # Filter subsets
    known_m = known[known["Preferred Method"].isin(["Transit", "Either"])].copy()
    tpc_m = tpc[tpc["Preferred Method"].isin(["Transit", "Either"])].copy()

    require_cols(known_m, [TARGET_TRANSIT], "Known CSV (Transit target)")
    Xk, feat_cols, known_m = build_transit_features(known_m)
    Xt, _, tpc_m = build_transit_features(tpc_m)

    yk_count = _num(known_m[TARGET_TRANSIT]).to_numpy(float)
    ok_k = np.isfinite(yk_count) & np.isfinite(Xk).any(axis=1)
    Xk = Xk[ok_k]
    yk_count = yk_count[ok_k]
    known_m = known_m.loc[ok_k].copy()

    # log target
    yk = np.log(np.maximum(yk_count, 1.0))

    # split for calibration
    X_tr, X_te, y_tr, y_te, yc_tr, yc_te = train_test_split(
        Xk, yk, yk_count, test_size=0.2, random_state=seed
    )

    # fit
    fit, fit_metrics = fit_numpyro(
        X_train=X_tr, y_train=y_tr,
        model=model, inference=inference,
        seed=seed, hidden=hidden,
        svi_steps=svi_steps, lr=lr,
    )
    fit.feature_cols = feat_cols

    # calibration on known test
    ydraw_te = predict_logy_draws(fit, X_te, model=model, hidden=hidden, n_draws=n_draws_pred, seed=seed + 21)
    _, med_te, _, _ = summarize_required_counts(ydraw_te, clip_min=1, clip_max=5000)
    mae = float(np.mean(np.abs(med_te - yc_te)))
    rmse_log = float(np.sqrt(np.mean((np.median(ydraw_te, axis=1) - y_te) ** 2)))

    plot_calibration(
        y_true=yc_te,
        y_med=med_te,
        outpath=plots / "transit_calibration_known_test.png",
        title=f"Transit: Known test calibration | MAE={mae:.2f} | RMSE(log)={rmse_log:.3f}",
    )
    if model == "linear" and inference == "nuts":
        plot_feature_weights_linear(fit.posterior, feat_cols, plots / "transit_linear_beta.png", "Transit linear model")

    # predict on TPC
    ydraw_tpc = predict_logy_draws(fit, Xt, model=model, hidden=hidden, n_draws=n_draws_pred, seed=seed + 31)
    req_tpc, med_tpc, p16_tpc, p84_tpc = summarize_required_counts(ydraw_tpc, clip_min=1, clip_max=5000)

    n_av = _num(tpc_m["Available Transits"]).to_numpy(float)
    p_feasible = np.mean(req_tpc <= n_av[:, None], axis=1)
    score = p_feasible / np.maximum(med_tpc, 1.0)

    out = tpc_m.copy()
    out["pred_Tier1Transits_med"] = med_tpc
    out["pred_Tier1Transits_p16"] = p16_tpc
    out["pred_Tier1Transits_p84"] = p84_tpc
    out["p_pred_le_available_transits"] = p_feasible
    out["transit_spectroscopy_score"] = score
    out["depth_frac"] = _num(out["Transit Depth [%]"]).to_numpy(float) / 100.0

    out = out.sort_values("transit_spectroscopy_score", ascending=False)
    out.to_csv(outputs / "tpc_transit_ranked.csv", index=False)

    plot_top_scatter(out, "transit_spectroscopy_score", plots / "transit_score_vs_depth.png", "Transit score vs depth")
    plot_score_hist(out, "transit_spectroscopy_score", plots / "transit_score_hist.png", "Transit score distribution")

    metrics_txt = "\n".join([
        "method: Transit",
        f"model: {model}",
        f"inference: {inference}",
        *[f"{k}: {v}" for k, v in fit_metrics.items()],
        f"known_test_mae_counts: {mae}",
        f"known_test_rmse_log: {rmse_log}",
        f"n_known_used: {len(known_m)}",
        f"n_tpc_used: {len(tpc_m)}",
        "",
        "Features (median-imputed then standardized):",
        *[f" - {c}" for c in feat_cols],
        "",
        "Ranking score: P(pred_required <= Available Transits) / median(pred_required)",
    ])
    (outputs / "model_metrics_transit.txt").write_text(metrics_txt)


def run_eclipse(
    known: pd.DataFrame,
    tpc: pd.DataFrame,
    outputs: Path,
    model: str,
    inference: str,
    seed: int,
    hidden: int,
    svi_steps: int,
    lr: float,
    n_draws_pred: int,
) -> None:
    plots = outputs / "plots"
    ensure_dir(plots)

    known_m = known[known["Preferred Method"].isin(["Eclipse", "Either"])].copy()
    tpc_m = tpc[tpc["Preferred Method"].isin(["Eclipse", "Either"])].copy()

    require_cols(known_m, [TARGET_ECLIPSE], "Known CSV (Eclipse target)")
    Xk, feat_cols, known_m = build_eclipse_features(known_m)
    Xt, _, tpc_m = build_eclipse_features(tpc_m)

    yk_count = _num(known_m[TARGET_ECLIPSE]).to_numpy(float)
    ok_k = np.isfinite(yk_count) & np.isfinite(Xk).any(axis=1)
    Xk = Xk[ok_k]
    yk_count = yk_count[ok_k]
    known_m = known_m.loc[ok_k].copy()

    yk = np.log(np.maximum(yk_count, 1.0))

    X_tr, X_te, y_tr, y_te, yc_tr, yc_te = train_test_split(
        Xk, yk, yk_count, test_size=0.2, random_state=seed
    )

    fit, fit_metrics = fit_numpyro(
        X_train=X_tr, y_train=y_tr,
        model=model, inference=inference,
        seed=seed + 100, hidden=hidden,
        svi_steps=svi_steps, lr=lr,
    )
    fit.feature_cols = feat_cols

    ydraw_te = predict_logy_draws(fit, X_te, model=model, hidden=hidden, n_draws=n_draws_pred, seed=seed + 121)
    _, med_te, _, _ = summarize_required_counts(ydraw_te, clip_min=1, clip_max=2000)
    mae = float(np.mean(np.abs(med_te - yc_te)))
    rmse_log = float(np.sqrt(np.mean((np.median(ydraw_te, axis=1) - y_te) ** 2)))

    plot_calibration(
        y_true=yc_te,
        y_med=med_te,
        outpath=plots / "eclipse_calibration_known_test.png",
        title=f"Eclipse: Known test calibration | MAE={mae:.2f} | RMSE(log)={rmse_log:.3f}",
    )
    if model == "linear" and inference == "nuts":
        plot_feature_weights_linear(fit.posterior, feat_cols, plots / "eclipse_linear_beta.png", "Eclipse linear model")

    ydraw_tpc = predict_logy_draws(fit, Xt, model=model, hidden=hidden, n_draws=n_draws_pred, seed=seed + 131)
    req_tpc, med_tpc, p16_tpc, p84_tpc = summarize_required_counts(ydraw_tpc, clip_min=1, clip_max=2000)

    n_av = _num(tpc_m["Available Eclipses"]).to_numpy(float)
    p_feasible = np.mean(req_tpc <= n_av[:, None], axis=1)
    score = p_feasible / np.maximum(med_tpc, 1.0)

    out = tpc_m.copy()
    out["pred_Tier1Eclipses_med"] = med_tpc
    out["pred_Tier1Eclipses_p16"] = p16_tpc
    out["pred_Tier1Eclipses_p84"] = p84_tpc
    out["p_pred_le_available_eclipses"] = p_feasible
    out["eclipse_spectroscopy_score"] = score

    # still plot vs transit depth since you care about it and it's in both; eclipse subset still has Transit Depth
    out["depth_frac"] = _num(out["Transit Depth [%]"]).to_numpy(float) / 100.0

    out = out.sort_values("eclipse_spectroscopy_score", ascending=False)
    out.to_csv(outputs / "tpc_eclipse_ranked.csv", index=False)

    plot_top_scatter(out, "eclipse_spectroscopy_score", plots / "eclipse_score_vs_depth.png", "Eclipse score vs depth")
    plot_score_hist(out, "eclipse_spectroscopy_score", plots / "eclipse_score_hist.png", "Eclipse score distribution")

    metrics_txt = "\n".join([
        "method: Eclipse",
        f"model: {model}",
        f"inference: {inference}",
        *[f"{k}: {v}" for k, v in fit_metrics.items()],
        f"known_test_mae_counts: {mae}",
        f"known_test_rmse_log: {rmse_log}",
        f"n_known_used: {len(known_m)}",
        f"n_tpc_used: {len(tpc_m)}",
        "",
        "Features (median-imputed then standardized):",
        *[f" - {c}" for c in feat_cols],
        "",
        "Ranking score: P(pred_required <= Available Eclipses) / median(pred_required)",
    ])
    (outputs / "model_metrics_eclipse.txt").write_text(metrics_txt)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--known", required=True)
    p.add_argument("--tpc", required=True)
    p.add_argument("--outputs", default="outputs_ariel_spec_no_waic")

    p.add_argument("--model", choices=["linear", "bnn"], default="bnn")
    p.add_argument("--inference", choices=["nuts", "svi"], default="svi")
    p.add_argument("--hidden", type=int, default=16)

    p.add_argument("--svi-steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-2)

    p.add_argument("--n-draws-pred", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    ensure_dir(outputs)
    ensure_dir(outputs / "plots")

    known = pd.read_csv(args.known)
    tpc = pd.read_csv(args.tpc)

    # Verify minimum required columns
    require_cols(known, ["Preferred Method", TARGET_TRANSIT, TARGET_ECLIPSE], "Known CSV")
    require_cols(known, MAG_COLS, "Known CSV (magnitudes)")
    require_cols(tpc, ["Preferred Method"], "TPC CSV")
    require_cols(tpc, MAG_COLS, "TPC CSV (magnitudes)")

    # Run
    run_transit(
        known=known, tpc=tpc, outputs=outputs,
        model=args.model, inference=args.inference,
        seed=args.seed, hidden=args.hidden,
        svi_steps=args.svi_steps, lr=args.lr,
        n_draws_pred=args.n_draws_pred,
    )
    run_eclipse(
        known=known, tpc=tpc, outputs=outputs,
        model=args.model, inference=args.inference,
        seed=args.seed, hidden=args.hidden,
        svi_steps=args.svi_steps, lr=args.lr,
        n_draws_pred=args.n_draws_pred,
    )

    print(f"\nWrote outputs to: {outputs.resolve()}")
    print("Key outputs:")
    print(" - tpc_transit_ranked.csv")
    print(" - tpc_eclipse_ranked.csv")
    print(" - model_metrics_transit.txt / model_metrics_eclipse.txt")
    print(" - plots/*.png")


if __name__ == "__main__":
    numpyro.set_host_device_count(1)
    main()
