#!/usr/bin/env python3
"""
ariel_spectroscopy_ranking_numpyro.py

Goal
----
Rank TESS Planet Candidates (TPCs) for ARIEL **Transit** and **Eclipse** spectroscopy
WITHOUT predicting masses and WITHOUT RV/ephemeris propagation.

Key idea
--------
Use the ARIEL-provided "Tier 1" metrics as the supervised target:
  - Transit model target  : log(Tier 1 Transits)
  - Eclipse model target  : log(Tier 1 Observations)

Train separate Bayesian models on the Known MCS, using ONLY columns present in BOTH
Known and TPC and explicitly including the transit measurement uncertainties:

Inputs (common columns used):
  - Transit Depth [%] + (lower/upper errors)
  - Transit Mid Time + (lower/upper errors)
  - Transit Duration T14 [s]
  - Available Transits
  - (plus: these are present in both and are ARIEL-relevant targets, NOT inputs)
    Tier 1 Observations, Tier 1 Transits

Outputs
-------
Writes in --outputs folder:
  - tpc_transit_ranked.csv
  - tpc_eclipse_ranked.csv
  - plots/*.png
  - model_metrics_transit.txt, model_metrics_eclipse.txt

Dependencies
------------
pip install "jax[cpu]" numpyro arviz pandas numpy matplotlib scikit-learn

Example
-------
python ariel_spectroscopy_ranking_numpyro.py \
  --known /path/Ariel_MCS_Known_2024-07-09.csv \
  --tpc   /path/Ariel_MCS_TPCs_2024-07-09.csv \
  --model linear --inference nuts --outputs outputs_ariel_spec

Notes
-----
- "Transit Mid Time" is included because you requested it; in practice, its absolute
  value should not matter physically, so the model may learn ~0 weight for it.
- Duration *errors* are not available in the Known CSV; we therefore use T14 only.
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

from numpyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.optim import Adam

import arviz as az
from numpyro.infer.util import log_likelihood


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

def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Map x to [0,1] by rank (higher is better). NaNs -> 0."""
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

# -------------------------
# Feature engineering (ONLY common columns)
# -------------------------
REQ_COMMON_INPUTS = [
    "Preferred Method",
    "Transit Mid Time",
    "Transit Mid Time Error Lower [days]",
    "Transit Mid Time Error Upper [days]",
    "Transit Depth [%]",
    "Transit Depth Error Lower [%]",
    "Transit Depth Error Upper [%]",
    "Transit Duration T14 [s]",
    "Available Transits",
]

REQ_TARGETS = [
    "Tier 1 Observations",
    "Tier 1 Transits",
]

def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Returns:
      X (N,p), feature_names, df_with_feature_cols (copy)
    """
    require_cols(df, REQ_COMMON_INPUTS + REQ_TARGETS, "DataFrame")

    d = df.copy()

    depth = _num(d["Transit Depth [%]"]).to_numpy(float) / 100.0  # fraction
    s_depth = sym_sigma(d["Transit Depth Error Lower [%]"], d["Transit Depth Error Upper [%]"]) / 100.0
    s_depth_rel = s_depth / np.maximum(depth, 1e-12)

    t14 = _num(d["Transit Duration T14 [s]"]).to_numpy(float)  # seconds
    t14_hr = t14 / 3600.0

    n_avail = _num(d["Available Transits"]).to_numpy(float)

    t0 = _num(d["Transit Mid Time"]).to_numpy(float)
    s_t0 = sym_sigma(d["Transit Mid Time Error Lower [days]"], d["Transit Mid Time Error Upper [days]"])

    # robust / monotonic features
    d["feat_log_depth"] = safe_log(depth)
    d["feat_log_t14_hr"] = safe_log(t14_hr)
    d["feat_log_avail"] = safe_log(n_avail + 1.0)     # +1 to handle 0
    d["feat_log_depth_snr"] = safe_log(depth / np.maximum(s_depth, 1e-12))
    d["feat_log_depth_relunc"] = safe_log(s_depth_rel)
    d["feat_log_t0_sigma"] = safe_log(np.maximum(s_t0, 1e-12))
    d["feat_t0_centered"] = t0 - np.nanmedian(t0)     # absolute epoch shouldn't matter; center anyway

    feat_cols = [
        "feat_log_depth",
        "feat_log_t14_hr",
        "feat_log_avail",
        "feat_log_depth_snr",
        "feat_log_depth_relunc",
        "feat_log_t0_sigma",
        "feat_t0_centered",
    ]

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
# Fitting + WAIC + prediction
# -------------------------
@dataclass
class Fitted:
    model: str
    inference: str
    scaler: StandardScaler
    feature_cols: List[str]
    guide: Optional[AutoDiagonalNormal]
    params: Dict[str, jnp.ndarray]
    posterior: Dict[str, jnp.ndarray]   # NUTS

def _model_fn(model: str):
    if model == "linear":
        return linear_student_t, {}
    if model == "bnn":
        return bnn_student_t, {}
    raise ValueError("model must be 'linear' or 'bnn'")

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

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    x = jnp.array(Xs)
    y = jnp.array(y_train)

    if model == "linear":
        model_fn = linear_student_t
        model_kwargs = {}
    else:
        model_fn = bnn_student_t
        model_kwargs = {"hidden": hidden}

    metrics: Dict[str, float] = {}

    if inference == "nuts":
        kernel = NUTS(model_fn)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=1500, num_chains=2, progress_bar=True)
        mcmc.run(rng, x=x, y=y, **model_kwargs)
        posterior = mcmc.get_samples(group_by_chain=False)
        metrics["n_draws"] = float(posterior[next(iter(posterior.keys()))].shape[0])
        return Fitted(model, inference, scaler, [], None, {}, posterior), metrics

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
        return Fitted(model, inference, scaler, [], guide, params, {}), metrics

    raise ValueError("inference must be 'nuts' or 'svi'")

def sample_posterior_params(fit: Fitted, n_draws: int, seed: int, model: str, hidden: int):
    rng = jax.random.PRNGKey(seed)

    if fit.inference == "nuts":
        return fit.posterior

    assert fit.guide is not None
    model_fn = linear_student_t if model == "linear" else bnn_student_t
    model_kwargs = {} if model == "linear" else {"hidden": hidden}
    # guide returns a dict of latent sites
    samples = fit.guide.sample_posterior(rng, fit.params, sample_shape=(n_draws,), **model_kwargs)
    return samples

def compute_waic(
    fit: Fitted,
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    hidden: int,
    n_draws: int,
    seed: int,
) -> Dict[str, float]:
    """
    Approximate WAIC with posterior draws (NUTS exact; SVI approximate).
    Returns a dict with waic and p_waic.
    """
    Xs = fit.scaler.transform(X)
    x = jnp.array(Xs)
    yj = jnp.array(y)

    model_fn = linear_student_t if model == "linear" else bnn_student_t
    model_kwargs = {} if model == "linear" else {"hidden": hidden}

    posterior = sample_posterior_params(fit, n_draws=n_draws, seed=seed, model=model, hidden=hidden)
    ll = log_likelihood(model_fn, posterior, x=x, y=yj, **model_kwargs)["y"]  # (draws, N)

    # ArviZ expects chain, draw, obs dims; we'll add chain=1
    ll_np = np.array(ll)[None, ...]  # (1, draws, N)
    idata = az.from_dict(log_likelihood={"y": ll_np})

    waic = az.waic(idata, var_name="y", scale="deviance")
    out = {
        "waic": float(waic.waic),
        "p_waic": float(waic.p_waic),
        "waic_se": float(waic.waic_se),
    }
    return out

def predict_logy_draws(
    fit: Fitted,
    X: np.ndarray,
    model: str,
    hidden: int,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """
    Returns draws of log(y_target) with shape (N, n_draws).
    """
    rng = jax.random.PRNGKey(seed)
    Xs = fit.scaler.transform(X)
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
    plt.figure(figsize=(8,4.5))
    y = np.arange(len(feature_cols))
    plt.errorbar(mu[order], y, xerr=[mu[order]-lo[order], hi[order]-mu[order]], fmt="o", alpha=0.9)
    plt.yticks(y, [feature_cols[i] for i in order])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(title + " (posterior β with 95% CI)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_top_scatter(df: pd.DataFrame, score_col: str, depth_col: str, outpath: Path, title: str):
    plt.figure(figsize=(7,5))
    plt.scatter(df[depth_col], df[score_col], alpha=0.25)
    top = df.sort_values(score_col, ascending=False).head(80)
    plt.scatter(top[depth_col], top[score_col], alpha=0.9)
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
# Main pipeline per method
# -------------------------
def run_one_method(
    method: str,
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
    n_draws_waic: int,
) -> None:
    """
    method: "Transit" or "Eclipse"
    """
    plots = outputs / "plots"
    ensure_dir(plots)

    # Filter
    known_m = known[known["Preferred Method"].isin([method, "Either"])].copy()
    tpc_m = tpc[tpc["Preferred Method"].isin([method, "Either"])].copy()

    # Build common features
    Xk, feat_cols, known_m = build_features(known_m)
    Xt, _, tpc_m = build_features(tpc_m)

    # Targets (counts)
    if method == "Transit":
        yk_count = _num(known_m["Tier 1 Transits"]).to_numpy(float)
        yt_count = _num(tpc_m["Tier 1 Transits"]).to_numpy(float)
        target_name = "Tier 1 Transits"
        clip_max = 5000
    else:
        yk_count = _num(known_m["Tier 1 Observations"]).to_numpy(float)
        yt_count = _num(tpc_m["Tier 1 Observations"]).to_numpy(float)
        target_name = "Tier 1 Observations"
        clip_max = 50

    # Clean
    ok_k = np.isfinite(Xk).all(axis=1) & np.isfinite(yk_count)
    ok_t = np.isfinite(Xt).all(axis=1) & np.isfinite(yt_count)

    known_m = known_m.loc[ok_k].copy()
    tpc_m = tpc_m.loc[ok_t].copy()
    Xk = Xk[ok_k]
    Xt = Xt[ok_t]
    yk_count = yk_count[ok_k]
    yt_count = yt_count[ok_t]

    # Log target for regression
    yk = np.log(np.maximum(yk_count, 1.0))
    yt = np.log(np.maximum(yt_count, 1.0))

    # Train/test split on known for sanity
    X_tr, X_te, y_tr, y_te, yc_tr, yc_te = train_test_split(
        Xk, yk, yk_count, test_size=0.2, random_state=seed
    )

    # Fit
    fit, fit_metrics = fit_numpyro(
        X_train=X_tr, y_train=y_tr,
        model=model, inference=inference,
        seed=seed, hidden=hidden,
        svi_steps=svi_steps, lr=lr,
    )
    fit.feature_cols = feat_cols

    # WAIC on known (full known subset)
    waic = compute_waic(
        fit=fit,
        X=Xk, y=yk,
        model=model, hidden=hidden,
        n_draws=n_draws_waic,
        seed=seed + 11,
    )

    # Predict on known test split (counts)
    ydraw_te = predict_logy_draws(fit, X_te, model=model, hidden=hidden, n_draws=n_draws_pred, seed=seed + 21)
    req_te, med_te, _, _ = summarize_required_counts(ydraw_te, clip_min=1, clip_max=clip_max)

    mae = float(np.mean(np.abs(med_te - yc_te)))
    rmse_log = float(np.sqrt(np.mean((np.median(ydraw_te, axis=1) - y_te) ** 2)))

    # Predict on ALL TPC subset
    ydraw_tpc = predict_logy_draws(fit, Xt, model=model, hidden=hidden, n_draws=n_draws_pred, seed=seed + 31)
    req_tpc, med_tpc, p16_tpc, p84_tpc = summarize_required_counts(ydraw_tpc, clip_min=1, clip_max=clip_max)

    n_av = _num(tpc_m["Available Transits"]).to_numpy(float)
    p_feasible = np.mean(req_tpc <= n_av[:, None], axis=1)

    # Final score: higher is better.
    # - encourage lower required count
    # - encourage feasibility given available transits
    score = p_feasible / np.maximum(med_tpc, 1.0)

    # Attach outputs
    out = tpc_m.copy()
    out[f"pred_{target_name}_med"] = med_tpc
    out[f"pred_{target_name}_p16"] = p16_tpc
    out[f"pred_{target_name}_p84"] = p84_tpc
    out[f"p_{target_name}_le_available"] = p_feasible
    out[f"{method.lower()}_spectroscopy_score"] = score

    # Include a couple raw columns for plotting convenience
    out["depth_frac"] = _num(out["Transit Depth [%]"]).to_numpy(float) / 100.0

    # Rank + save
    out = out.sort_values(f"{method.lower()}_spectroscopy_score", ascending=False)
    out_csv = outputs / f"tpc_{method.lower()}_ranked.csv"
    out.to_csv(out_csv, index=False)

    # Plots
    plot_calibration(
        y_true=yc_te,
        y_med=med_te,
        outpath=plots / f"{method.lower()}_calibration_known_test.png",
        title=f"{method}: calibration on Known (test split) | MAE={mae:.2f} counts | RMSE(log)={rmse_log:.3f}",
    )

    if model == "linear" and inference == "nuts":
        plot_feature_weights_linear(
            posterior=fit.posterior,
            feature_cols=feat_cols,
            outpath=plots / f"{method.lower()}_linear_beta.png",
            title=f"{method} linear model",
        )

    plot_top_scatter(
        df=out,
        score_col=f"{method.lower()}_spectroscopy_score",
        depth_col="depth_frac",
        outpath=plots / f"{method.lower()}_score_vs_depth.png",
        title=f"{method}: score vs transit depth",
    )

    plot_score_hist(
        df=out,
        score_col=f"{method.lower()}_spectroscopy_score",
        outpath=plots / f"{method.lower()}_score_hist.png",
        title=f"{method}: spectroscopy score distribution (TPC subset)",
    )

    # Write metrics
    metrics_txt = "\n".join([
        f"method: {method}",
        f"model: {model}",
        f"inference: {inference}",
        *[f"{k}: {v}" for k, v in fit_metrics.items()],
        f"known_test_mae_counts: {mae}",
        f"known_test_rmse_log: {rmse_log}",
        f"waic_deviance: {waic['waic']}",
        f"p_waic: {waic['p_waic']}",
        f"waic_se: {waic['waic_se']}",
        f"n_known_used: {len(known_m)}",
        f"n_tpc_used: {len(tpc_m)}",
        "",
        "Features (standardized):",
        *[f" - {c}" for c in feat_cols],
        "",
        f"Ranking score used: score = P(pred_required <= Available Transits) / median(pred_required)",
    ])
    (outputs / f"model_metrics_{method.lower()}.txt").write_text(metrics_txt)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--known", required=True)
    p.add_argument("--tpc", required=True)
    p.add_argument("--outputs", default="outputs_ariel_spectroscopy_numpyro")

    p.add_argument("--model", choices=["linear", "bnn"], default="linear")
    p.add_argument("--inference", choices=["nuts", "svi"], default="nuts")
    p.add_argument("--hidden", type=int, default=16)

    p.add_argument("--svi-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-2)

    p.add_argument("--n-draws-pred", type=int, default=1500)
    p.add_argument("--n-draws-waic", type=int, default=1000)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    ensure_dir(outputs)

    known = pd.read_csv(args.known)
    tpc = pd.read_csv(args.tpc)

    # Basic validation
    require_cols(known, REQ_COMMON_INPUTS + REQ_TARGETS, "Known CSV")
    require_cols(tpc, REQ_COMMON_INPUTS + REQ_TARGETS, "TPC CSV")

    # Run separate models
    for method in ["Transit", "Eclipse"]:
        run_one_method(
            method=method,
            known=known,
            tpc=tpc,
            outputs=outputs,
            model=args.model,
            inference=args.inference,
            seed=args.seed,
            hidden=args.hidden,
            svi_steps=args.svi_steps,
            lr=args.lr,
            n_draws_pred=args.n_draws_pred,
            n_draws_waic=args.n_draws_waic,
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
