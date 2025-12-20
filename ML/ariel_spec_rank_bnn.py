#!/usr/bin/env python3
"""
ariel_spec_rank_bnn.py

Ariel spectroscopy target ranking (TRANSIT vs ECLIPSE) using a Bayesian Neural Network (NumPyro).

Key ideas (matches your constraints):
- NO masses, NO RV, NO ephemeris drift horizon propagation.
- Uses only spectroscopy-relevant columns:
  brightness (mags -> flux proxy), transit depth + depth uncertainty, duration + uncertainty,
  timing uncertainty (directly), and number of available events.
- Trains separate BNNs for Transit and Eclipse on the KNOWN sample.
- Scores ONLY the TPCs, producing ranked lists + plots.

Model:
  y = log1p(required Tier-1 events)  (Tier 1 Transits or Tier 1 Eclipses)
  y ~ StudentT(nu, mu_BNN(x), sigma_int)

Score (per candidate):
  score_raw = P(required <= available) * (yield_proxy) * sqrt(available) * (uncertainty_penalty) / (required_med + eps)

  yield_proxy (Transit)  ~ sqrt(flux_proxy) * depth_SNR
  yield_proxy (Eclipse)  ~ sqrt(flux_proxy) * depth_SNR * sqrt(Teq/median_Teq)   (Teq term only if available)

  uncertainty_penalty downweights large timing/duration uncertainties.

Outputs:
  outputs_dir/
    tpc_transit_ranked.csv
    tpc_eclipse_ranked.csv
    plots/*.png
    metrics_transit.txt
    metrics_eclipse.txt
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

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.optim import Adam


# -------------------------
# utils
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
    return np.log(np.clip(x, floor, np.inf))


def mag_flux(m: np.ndarray) -> np.ndarray:
    """Relative flux proxy from magnitudes: F ~ 10^{-0.4 m}."""
    m = np.asarray(m, dtype=float)
    out = np.full_like(m, np.nan, dtype=float)
    ok = np.isfinite(m)
    out[ok] = 10.0 ** (-0.4 * m[ok])
    return out


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Map x -> [0,1] by rank (higher is better). NaNs -> 0."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    ok = np.isfinite(x)
    if np.sum(ok) < 2:
        return out
    r = x[ok].argsort().argsort().astype(float)
    out[ok] = r / (len(r) - 1)
    return out


def get_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return _num(df[col]).to_numpy(float)
    return np.full(len(df), np.nan, dtype=float)


def flux_proxy_from_mags(df: pd.DataFrame) -> np.ndarray:
    """
    Prefer IR mags (J/H/Ks/W1/W2), fallback to TESS mag.
    Returns a relative flux proxy (higher is brighter).
    """
    mags = []
    for c in ["Star J Mag", "Star H Mag", "Star Ks Mag", "Star W1 Mag", "Star W2 Mag"]:
        if c in df.columns:
            mags.append(mag_flux(get_col(df, c)))
    if len(mags) > 0:
        f = np.nanmean(np.vstack(mags), axis=0)
    else:
        f = np.full(len(df), np.nan, dtype=float)

    # fallback
    if "Star TESS Mag" in df.columns:
        f_fb = mag_flux(get_col(df, "Star TESS Mag"))
        f = np.where(np.isfinite(f), f, f_fb)
    return f


# -------------------------
# Feature engineering
# -------------------------
def build_features(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds engineered features (columns starting with feat_) and returns feature list.

    Required raw columns (both Known and TPC have them in your files):
      - Transit Depth [%] and its errors
      - Transit Duration [hr] and its errors
      - Transit Mid Time error bounds
      - Available Transits / Available Eclipses
      - Star mags (at least TESS Mag is present in TPC; Known has many)
      - Planet Temperature [K] (used as optional extra for eclipse yield)
    """
    out = df.copy()

    # depth (fraction) and uncertainty
    depth_frac = get_col(out, "Transit Depth [%]") / 100.0
    sig_depth_frac = np.full(len(out), np.nan, dtype=float)
    if ("Transit Depth Error Lower [%]" in out.columns) and ("Transit Depth Error Upper [%]" in out.columns):
        sig_depth_frac = sym_sigma(out["Transit Depth Error Lower [%]"], out["Transit Depth Error Upper [%]"]) / 100.0

    # duration and uncertainty (hours)
    dur_hr = get_col(out, "Transit Duration [hr]")
    sig_dur_hr = np.full(len(out), np.nan, dtype=float)
    if ("Transit Duration Error Lower [hr]" in out.columns) and ("Transit Duration Error Upper [hr]" in out.columns):
        sig_dur_hr = sym_sigma(out["Transit Duration Error Lower [hr]"], out["Transit Duration Error Upper [hr]"])

    # timing uncertainty (days) — direct (no horizon propagation)
    sig_t0_days = np.full(len(out), np.nan, dtype=float)
    if ("Transit Mid Time Error Lower [days]" in out.columns) and ("Transit Mid Time Error Upper [days]" in out.columns):
        sig_t0_days = sym_sigma(out["Transit Mid Time Error Lower [days]"], out["Transit Mid Time Error Upper [days]"])

    # availability (method-specific)
    if method.lower() == "transit":
        avail = get_col(out, "Available Transits")
    else:
        avail = get_col(out, "Available Eclipses")

    # brightness proxy
    f = flux_proxy_from_mags(out)

    # depth SNR (uses uncertainty)
    depth_snr = np.full(len(out), np.nan, dtype=float)
    ok = np.isfinite(depth_frac) & np.isfinite(sig_depth_frac) & (sig_depth_frac > 0)
    depth_snr[ok] = depth_frac[ok] / sig_depth_frac[ok]

    # optional Teq
    teq = get_col(out, "Planet Temperature [K]")

    # engineered log-features (stable ranges for BNN)
    out["feat_log_depth"] = safe_log(depth_frac, floor=1e-12)
    out["feat_log_depth_snr"] = safe_log(depth_snr, floor=1e-6)
    out["feat_log_flux"] = safe_log(f, floor=1e-20)
    out["feat_log_avail"] = safe_log(np.clip(avail, 0, np.inf) + 1.0, floor=1e-12)  # log1p
    out["feat_log_dur"] = safe_log(dur_hr, floor=1e-6)
    out["feat_log_sig_dur"] = safe_log(sig_dur_hr, floor=1e-6)
    out["feat_log_sig_t0"] = safe_log(sig_t0_days, floor=1e-8)
    out["feat_log_teq"] = safe_log(teq, floor=1.0)

    # median-impute any missing engineered features so we don't drop large chunks of TPCs
    for c in [k for k in out.columns if k.startswith("feat_")]:
        v = out[c].to_numpy(float)
        v[~np.isfinite(v)] = np.nan
        med = np.nanmedian(v)
        if not np.isfinite(med):
            med = 0.0
        out[c] = np.where(np.isfinite(out[c].to_numpy(float)), out[c].to_numpy(float), med)

    # Keep Teq in the feature set (it helps eclipse especially), but it is optional:
    feature_cols = [
        "feat_log_depth",
        "feat_log_depth_snr",
        "feat_log_flux",
        "feat_log_avail",
        "feat_log_dur",
        "feat_log_sig_dur",
        "feat_log_sig_t0",
        "feat_log_teq",
    ]
    return out, feature_cols


# -------------------------
# NumPyro BNN
# -------------------------
def bnn_student_t_regression(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    hidden1: int = 32,
    hidden2: int = 16,
    w_scale: float = 1.0,
):
    """
    2-hidden-layer Bayesian NN with Student-T likelihood.
    y is continuous (we use y = log1p(count)).

    Prior: Normal(0, w_scale) on weights/biases.
    Likelihood: StudentT(df=nu, loc=mu, scale=sigma_int)
    """
    n, p = x.shape

    w1 = numpyro.sample("w1", dist.Normal(0.0, w_scale).expand((p, hidden1)))
    b1 = numpyro.sample("b1", dist.Normal(0.0, w_scale).expand((hidden1,)))
    w2 = numpyro.sample("w2", dist.Normal(0.0, w_scale).expand((hidden1, hidden2)))
    b2 = numpyro.sample("b2", dist.Normal(0.0, w_scale).expand((hidden2,)))
    w3 = numpyro.sample("w3", dist.Normal(0.0, w_scale).expand((hidden2, 1)))
    b3 = numpyro.sample("b3", dist.Normal(0.0, w_scale))

    sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(0.5))
    nu = numpyro.sample("nu", dist.Exponential(1 / 10.0)) + 1.0

    h = jnp.tanh(jnp.dot(x, w1) + b1)
    h = jnp.tanh(jnp.dot(h, w2) + b2)
    mu = jnp.squeeze(jnp.dot(h, w3), axis=-1) + b3

    with numpyro.plate("data", n):
        numpyro.sample("y", dist.StudentT(df=nu, loc=mu, scale=sigma_int), obs=y)


@dataclass
class Fit:
    params: Dict[str, jnp.ndarray]
    guide: AutoDiagonalNormal
    scaler: StandardScaler
    feature_cols: List[str]


def fit_bnn_svi(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    svi_steps: int,
    lr: float,
    hidden1: int,
    hidden2: int,
    w_scale: float,
) -> Tuple[Fit, Dict[str, float]]:
    rng_key = jax.random.PRNGKey(seed)
    x = jnp.array(x_tr)
    y = jnp.array(y_tr)

    guide = AutoDiagonalNormal(bnn_student_t_regression)
    optim = Adam(lr)
    svi = SVI(bnn_student_t_regression, guide, optim, loss=Trace_ELBO())
    svi_state = svi.init(rng_key, x=x, y=y, hidden1=hidden1, hidden2=hidden2, w_scale=w_scale)

    last = None
    for _ in range(svi_steps):
        svi_state, loss = svi.update(svi_state, x=x, y=y, hidden1=hidden1, hidden2=hidden2, w_scale=w_scale)
        last = float(loss)

    params = svi.get_params(svi_state)
    metrics = {"final_elbo": float(-last) if last is not None else np.nan}
    return Fit(params=params, guide=guide, scaler=StandardScaler(), feature_cols=[]), metrics


def predictive_draws_log_required(
    fit: Fit,
    x_std: np.ndarray,
    n_draws: int,
    seed: int,
    hidden1: int,
    hidden2: int,
    w_scale: float,
) -> np.ndarray:
    """
    Returns posterior predictive draws for y = log1p(required), shape (N, n_draws).
    """
    rng_key = jax.random.PRNGKey(seed)
    x = jnp.array(x_std)
    pred = Predictive(
        bnn_student_t_regression,
        guide=fit.guide,
        params=fit.params,
        num_samples=n_draws,
    )
    out = pred(rng_key, x=x, y=None, hidden1=hidden1, hidden2=hidden2, w_scale=w_scale)
    y = np.array(out["y"])  # (n_draws, N)
    if y.shape[0] == n_draws:
        y = y.T
    return y


# -------------------------
# Train / evaluate per method
# -------------------------
def prepare_known_method(
    known: pd.DataFrame,
    method: str,
    y_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    df = known.copy()

    if "Preferred Method" in df.columns:
        df = df[df["Preferred Method"].astype(str).str.upper() == method.upper()].copy()

    df, feature_cols = build_features(df, method=method)

    # target y = log1p(count)
    y_count = _num(df[y_col]).to_numpy(float)
    y = np.log1p(np.clip(y_count, 0.0, np.inf))

    # assemble X and clean
    X = df[feature_cols].to_numpy(float)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    df = df.loc[keep].copy()
    X = X[keep]
    y = y[keep]

    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    return df, feature_cols, x_tr_s, x_te_s, y_tr, y_te, scaler


def calibration_plot(
    method: str,
    y_te: np.ndarray,
    ydraws_te: np.ndarray,
    outpath: Path,
) -> Dict[str, float]:
    """
    y are log1p(count). We report:
      - MAE in count space
      - RMSE in log space
      - 68% interval coverage in log space
    """
    pred_med = np.median(ydraws_te, axis=1)
    pred_lo = np.quantile(ydraws_te, 0.16, axis=1)
    pred_hi = np.quantile(ydraws_te, 0.84, axis=1)

    true_count = np.expm1(y_te)
    pred_count = np.expm1(pred_med)

    mae = float(np.mean(np.abs(pred_count - true_count)))
    rmse_log = float(np.sqrt(np.mean((pred_med - y_te) ** 2)))
    cov68 = float(np.mean((y_te >= pred_lo) & (y_te <= pred_hi)))

    # plot in count space for readability
    plt.figure(figsize=(7, 6))
    plt.scatter(true_count, pred_count, alpha=0.7)
    mx = max(np.nanmax(true_count), np.nanmax(pred_count))
    plt.plot([0, mx], [0, mx], "--")
    plt.xlabel("True required Tier-1 events (count)")
    plt.ylabel("Predicted median required (count)")
    plt.title(f"{method.title()}: Known test calibration | MAE={mae:.2f} | RMSE(log)={rmse_log:.3f} | cov68={cov68:.3f}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return {"mae_count": mae, "rmse_log": rmse_log, "cov68_log": cov68}


# -------------------------
# Scoring TPCs per method
# -------------------------
def score_tpc_method(
    tpc: pd.DataFrame,
    method: str,
    y_col: str,
    feature_cols: List[str],
    scaler: StandardScaler,
    fit: Fit,
    outputs: Path,
    n_draws: int,
    seed: int,
    hidden1: int,
    hidden2: int,
    w_scale: float,
    top_highlight: int,
) -> pd.DataFrame:
    df = tpc.copy()

    if "Preferred Method" in df.columns:
        df = df[df["Preferred Method"].astype(str).str.upper() == method.upper()].copy()

    df, _ = build_features(df, method=method)

    # Need these for scoring + plotting
    needed_raw = [
        "Planet Name", "Star Name",
        "Transit Depth [%]",
        "Transit Depth Error Lower [%]", "Transit Depth Error Upper [%]",
        "Transit Duration [hr]",
        "Transit Duration Error Lower [hr]", "Transit Duration Error Upper [hr]",
        "Transit Mid Time Error Lower [days]", "Transit Mid Time Error Upper [days]",
        "Star TESS Mag",
        "Star J Mag", "Star H Mag", "Star Ks Mag", "Star W1 Mag", "Star W2 Mag",
        "Planet Temperature [K]",
        "Available Transits", "Available Eclipses",
    ]
    for c in needed_raw:
        if c not in df.columns:
            df[c] = np.nan

    # Clean for features
    X = df[feature_cols].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    df = df.loc[ok].copy()
    X = X[ok]

    if len(df) == 0:
        raise RuntimeError(f"No TPC rows available for method={method} after feature cleaning.")

    Xs = scaler.transform(X)

    # posterior predictive draws for y = log1p(required)
    ydraws = predictive_draws_log_required(
        fit=fit,
        x_std=Xs,
        n_draws=n_draws,
        seed=seed,
        hidden1=hidden1,
        hidden2=hidden2,
        w_scale=w_scale,
    )

    # required draws in count space
    req_draws = np.expm1(ydraws)
    req_draws = np.clip(req_draws, 0.0, np.inf)

    req_med = np.median(req_draws, axis=1)
    req_p16 = np.quantile(req_draws, 0.16, axis=1)
    req_p84 = np.quantile(req_draws, 0.84, axis=1)

    # availability
    if method.lower() == "transit":
        avail = get_col(df, "Available Transits")
    else:
        avail = get_col(df, "Available Eclipses")
    avail = np.clip(avail, 0.0, np.inf)

    p_feas = np.mean(req_draws <= avail[:, None], axis=1)

    # build score factors
    depth_frac = get_col(df, "Transit Depth [%]") / 100.0
    sig_depth_frac = sym_sigma(df["Transit Depth Error Lower [%]"], df["Transit Depth Error Upper [%]"]) / 100.0
    depth_snr = np.divide(
        depth_frac,
        np.clip(sig_depth_frac, 1e-12, np.inf),
        out=np.full_like(depth_frac, np.nan),
        where=np.isfinite(depth_frac) & np.isfinite(sig_depth_frac),
    )

    f = flux_proxy_from_mags(df)
    dur_sig = sym_sigma(df["Transit Duration Error Lower [hr]"], df["Transit Duration Error Upper [hr]"])
    t0_sig = sym_sigma(df["Transit Mid Time Error Lower [days]"], df["Transit Mid Time Error Upper [days]"])
    teq = get_col(df, "Planet Temperature [K]")

    # uncertainty penalty (normalize by typical scales in this method subset)
    med_t0 = np.nanmedian(t0_sig) if np.isfinite(np.nanmedian(t0_sig)) else 1.0
    med_dur = np.nanmedian(dur_sig) if np.isfinite(np.nanmedian(dur_sig)) else 1.0
    t0_n = np.divide(t0_sig, med_t0, out=np.ones_like(t0_sig), where=np.isfinite(t0_sig) & (med_t0 > 0))
    dur_n = np.divide(dur_sig, med_dur, out=np.ones_like(dur_sig), where=np.isfinite(dur_sig) & (med_dur > 0))
    uncert_pen = 1.0 / (1.0 + t0_n**2 + dur_n**2)

    # yield proxy (method-specific)
    yield_proxy = np.sqrt(np.clip(f, 1e-30, np.inf)) * np.clip(depth_snr, 1e-6, np.inf)
    if method.lower() == "eclipse":
        med_teq = np.nanmedian(teq) if np.isfinite(np.nanmedian(teq)) else np.nan
        if np.isfinite(med_teq) and med_teq > 0:
            yield_proxy = yield_proxy * np.sqrt(np.clip(teq / med_teq, 0.1, 10.0))

    # characterization proxy from availability
    char_proxy = np.sqrt(np.clip(avail, 0.0, np.inf))

    eps = 1e-6
    score_raw = p_feas * yield_proxy * char_proxy * uncert_pen / (req_med + eps)

    # scale to [0,1] for readability
    score = percentile_rank(score_raw)

    out = df.copy()
    out[f"pred_required_{method.lower()}_med"] = req_med
    out[f"pred_required_{method.lower()}_p16"] = req_p16
    out[f"pred_required_{method.lower()}_p84"] = req_p84
    out["p_pred_le_available"] = p_feas
    out["flux_proxy"] = f
    out["depth_snr"] = depth_snr
    out["uncert_pen"] = uncert_pen
    out["score_raw"] = score_raw
    out[f"{method.lower()}_spectroscopy_score"] = score

    out = out.sort_values(f"{method.lower()}_spectroscopy_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    # write CSV
    out_csv = outputs / f"tpc_{method.lower()}_ranked.csv"
    out.to_csv(out_csv, index=False)

    # plots
    plots = outputs / "plots"
    ensure_dir(plots)

    # score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(out[f"{method.lower()}_spectroscopy_score"].to_numpy(float), bins=35, alpha=0.9)
    plt.xlabel(f"{method.lower()}_spectroscopy_score")
    plt.ylabel("Count")
    plt.title(f"{method.title()} score distribution")
    plt.tight_layout()
    plt.savefig(plots / f"{method.lower()}_score_hist.png", dpi=200)
    plt.close()

    # score vs depth (top highlighted)
    depth_frac_all = out["Transit Depth [%]"].to_numpy(float) / 100.0
    plt.figure(figsize=(8, 5))
    plt.scatter(depth_frac_all, out[f"{method.lower()}_spectroscopy_score"], alpha=0.25, label="All")
    top = out.head(top_highlight)
    plt.scatter(top["Transit Depth [%]"].to_numpy(float) / 100.0, top[f"{method.lower()}_spectroscopy_score"], alpha=0.9, label=f"Top {top_highlight}")
    plt.xscale("log")
    plt.xlabel("Transit Depth (fraction)")
    plt.ylabel(f"{method.lower()}_spectroscopy_score")
    plt.title(f"{method.title()} score vs depth (top {top_highlight} highlighted)")
    plt.tight_layout()
    plt.savefig(plots / f"{method.lower()}_score_vs_depth.png", dpi=200)
    plt.close()

    # feasibility vs required
    plt.figure(figsize=(7, 5))
    plt.scatter(out[f"pred_required_{method.lower()}_med"], out["p_pred_le_available"], alpha=0.3)
    top = out.head(top_highlight)
    plt.scatter(top[f"pred_required_{method.lower()}_med"], top["p_pred_le_available"], alpha=0.9)
    plt.xlabel("Predicted required Tier-1 events (median)")
    plt.ylabel("P(required ≤ available)")
    plt.title(f"{method.title()}: Feasibility vs required (top {top_highlight} highlighted)")
    plt.tight_layout()
    plt.savefig(plots / f"{method.lower()}_feasible_vs_required.png", dpi=200)
    plt.close()

    # depth vs brightness, marker size ~ score
    tess = out["Star TESS Mag"].to_numpy(float)
    plt.figure(figsize=(8, 6))
    sizes = 20.0 + 280.0 * out[f"{method.lower()}_spectroscopy_score"].to_numpy(float)
    plt.scatter(depth_frac_all, tess, s=sizes, alpha=0.25, label=method.title())
    plt.gca().invert_yaxis()
    plt.xscale("log")
    plt.xlabel("Transit Depth (fraction)")
    plt.ylabel("Star TESS Mag (brighter → higher)")
    plt.title(f"{method.title()}: Depth vs brightness (marker size ∝ score)")
    plt.tight_layout()
    plt.savefig(plots / f"{method.lower()}_depth_vs_brightness_size_score.png", dpi=200)
    plt.close()

    return out


def combined_plots(transit_ranked: pd.DataFrame, eclipse_ranked: pd.DataFrame, outputs: Path) -> None:
    plots = outputs / "plots"
    ensure_dir(plots)

    # 1) score drop-off
    plt.figure(figsize=(9, 5))
    plt.plot(transit_ranked["rank"], transit_ranked["transit_spectroscopy_score"], label="Transit")
    plt.plot(eclipse_ranked["rank"], eclipse_ranked["eclipse_spectroscopy_score"], label="Eclipse")
    plt.xlabel("Rank (1 = best)")
    plt.ylabel("spectroscopy score")
    plt.title("Score drop-off across ranked lists (Transit vs Eclipse)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "01_combined_score_dropoff.png", dpi=200)
    plt.close()

    # 2) score distributions overlay
    plt.figure(figsize=(9, 5))
    plt.hist(transit_ranked["transit_spectroscopy_score"].to_numpy(float), bins=35, alpha=0.6, label="Transit")
    plt.hist(eclipse_ranked["eclipse_spectroscopy_score"].to_numpy(float), bins=35, alpha=0.6, label="Eclipse")
    plt.xlabel("spectroscopy score")
    plt.ylabel("count")
    plt.title("Score distributions (Transit vs Eclipse)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "02_combined_score_hist.png", dpi=200)
    plt.close()

    # 3) depth vs brightness combined (size ~ score)
    plt.figure(figsize=(9, 6))
    d_t = transit_ranked["Transit Depth [%]"].to_numpy(float) / 100.0
    m_t = transit_ranked["Star TESS Mag"].to_numpy(float)
    s_t = 20.0 + 260.0 * transit_ranked["transit_spectroscopy_score"].to_numpy(float)
    plt.scatter(d_t, m_t, s=s_t, alpha=0.25, label="Transit")

    d_e = eclipse_ranked["Transit Depth [%]"].to_numpy(float) / 100.0
    m_e = eclipse_ranked["Star TESS Mag"].to_numpy(float)
    s_e = 20.0 + 260.0 * eclipse_ranked["eclipse_spectroscopy_score"].to_numpy(float)
    plt.scatter(d_e, m_e, s=s_e, alpha=0.25, label="Eclipse")

    plt.gca().invert_yaxis()
    plt.xscale("log")
    plt.xlabel("Transit Depth (fraction)")
    plt.ylabel("Star TESS Mag (brighter → higher)")
    plt.title("Depth vs brightness (marker size ∝ score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "03_combined_depth_vs_brightness_size_score.png", dpi=200)
    plt.close()

    # 4) feasibility vs required combined
    plt.figure(figsize=(8, 6))
    plt.scatter(
        transit_ranked["pred_required_transit_med"].to_numpy(float),
        transit_ranked["p_pred_le_available"].to_numpy(float),
        alpha=0.25,
        label="Transit",
    )
    plt.scatter(
        eclipse_ranked["pred_required_eclipse_med"].to_numpy(float),
        eclipse_ranked["p_pred_le_available"].to_numpy(float),
        alpha=0.25,
        label="Eclipse",
    )
    plt.xlabel("Predicted required Tier-1 events (median)")
    plt.ylabel("P(required ≤ available)")
    plt.title("Feasibility vs required (Transit vs Eclipse)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "04_combined_feasible_vs_required.png", dpi=200)
    plt.close()


# -------------------------
# CLI main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--known", required=True, help="Known MCS CSV")
    p.add_argument("--tpc", required=True, help="TPC CSV")
    p.add_argument("--outputs", default="outputs_ariel_spec_bnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden1", type=int, default=32)
    p.add_argument("--hidden2", type=int, default=16)
    p.add_argument("--w-scale", type=float, default=1.0)
    p.add_argument("--svi-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-draws", type=int, default=2000)
    p.add_argument("--n-draws-test", type=int, default=1000)
    p.add_argument("--top-highlight", type=int, default=80)
    return p.parse_args()


def run_one_method(
    known: pd.DataFrame,
    tpc: pd.DataFrame,
    method: str,
    y_col: str,
    outputs: Path,
    seed: int,
    hidden1: int,
    hidden2: int,
    w_scale: float,
    svi_steps: int,
    lr: float,
    n_draws: int,
    n_draws_test: int,
    top_highlight: int,
) -> pd.DataFrame:
    plots = outputs / "plots"
    ensure_dir(plots)

    # prepare known training data
    df_known, feature_cols, x_tr_s, x_te_s, y_tr, y_te, scaler = prepare_known_method(
        known=known,
        method=method,
        y_col=y_col,
        seed=seed,
    )

    # fit BNN
    fit, fit_metrics = fit_bnn_svi(
        x_tr=x_tr_s,
        y_tr=y_tr,
        seed=seed,
        svi_steps=svi_steps,
        lr=lr,
        hidden1=hidden1,
        hidden2=hidden2,
        w_scale=w_scale,
    )
    fit.scaler = scaler
    fit.feature_cols = feature_cols

    # calibration on held-out known
    ydraws_te = predictive_draws_log_required(
        fit=fit,
        x_std=x_te_s,
        n_draws=n_draws_test,
        seed=seed + 10,
        hidden1=hidden1,
        hidden2=hidden2,
        w_scale=w_scale,
    )
    cal = calibration_plot(
        method=method,
        y_te=y_te,
        ydraws_te=ydraws_te,
        outpath=plots / f"{method.lower()}_calibration_known_test.png",
    )

    # write metrics
    metrics_path = outputs / f"metrics_{method.lower()}.txt"
    metrics_path.write_text(
        "\n".join([
            f"method: {method}",
            f"target_col: {y_col}",
            f"n_known_used: {len(df_known)}",
            f"n_train: {len(y_tr)}",
            f"n_test: {len(y_te)}",
            f"final_elbo: {fit_metrics.get('final_elbo', np.nan):.3f}",
            f"MAE_count: {cal['mae_count']:.3f}",
            f"RMSE_log: {cal['rmse_log']:.3f}",
            f"cov68_log: {cal['cov68_log']:.3f}",
            "feature_cols: " + ", ".join(feature_cols),
        ]) + "\n"
    )

    # score TPCs
    ranked = score_tpc_method(
        tpc=tpc,
        method=method,
        y_col=y_col,
        feature_cols=feature_cols,
        scaler=scaler,
        fit=fit,
        outputs=outputs,
        n_draws=n_draws,
        seed=seed + 100,
        hidden1=hidden1,
        hidden2=hidden2,
        w_scale=w_scale,
        top_highlight=top_highlight,
    )

    return ranked


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    ensure_dir(outputs)
    ensure_dir(outputs / "plots")

    known = pd.read_csv(args.known)
    tpc = pd.read_csv(args.tpc)

    # Run both methods
    transit_ranked = run_one_method(
        known=known,
        tpc=tpc,
        method="Transit",
        y_col="Tier 1 Transits",
        outputs=outputs,
        seed=args.seed,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        w_scale=args.w_scale,
        svi_steps=args.svi_steps,
        lr=args.lr,
        n_draws=args.n_draws,
        n_draws_test=args.n_draws_test,
        top_highlight=args.top_highlight,
    )

    eclipse_ranked = run_one_method(
        known=known,
        tpc=tpc,
        method="Eclipse",
        y_col="Tier 1 Eclipses",
        outputs=outputs,
        seed=args.seed + 1,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        w_scale=args.w_scale,
        svi_steps=args.svi_steps,
        lr=args.lr,
        n_draws=args.n_draws,
        n_draws_test=args.n_draws_test,
        top_highlight=args.top_highlight,
    )

    # Combined plots
    combined_plots(transit_ranked, eclipse_ranked, outputs)

    print(f"\nWrote outputs to: {outputs.resolve()}")
    print("Key outputs:")
    print(" - tpc_transit_ranked.csv")
    print(" - tpc_eclipse_ranked.csv")
    print(" - plots/*.png")
    print(" - metrics_transit.txt, metrics_eclipse.txt")


if __name__ == "__main__":
    numpyro.set_host_device_count(1)
    main()
