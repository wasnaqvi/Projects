#!/usr/bin/env python3
"""
exoplanet_mass_pgm.py

Probabilistic Statistical Learning (PGM):
Train an uncertainty-aware Bayesian regression model for log10(planet mass)
using "known" planets with measured masses, then apply to TESS planet candidates
to produce posterior predictive mass distributions (not just point estimates).

Outputs:
- Cleaned training table
- Candidate predictions table (median + 68% credible interval)
- EDA plots + calibration plots saved to ./outputs/

Run examples:
  python exoplanet_mass_pgm.py --known Ariel_MCS_Known_2024-07-09.csv --tpc Ariel_MCS_TPCs_2024-07-09.csv
  python exoplanet_mass_pgm.py --feature-set broad
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------
# Feature sets
# -------------------------
STRICT_FEATURES = [
    "log_rp", "log_p", "log_teq",
    "Star Metallicity", "Star Age [Gyr]",
    "Star Mass [Ms]", "Star Temperature [K]", "Star log(g)",
]

BROAD_FEATURES = [
    "log_rp", "log_p", "log_teq",
    "Star Mass [Ms]", "Star Temperature [K]", "Star log(g)",
]


# -------------------------
# Utility functions
# -------------------------
def _safe_log10(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0)
    return np.log10(x)


def sym_sigma(lower: pd.Series, upper: pd.Series) -> np.ndarray:
    """
    Make a symmetric 1-sigma from lower/upper errors.
    Many catalogs store the lower error as negative; use absolute values.
    """
    lo = np.abs(pd.to_numeric(lower, errors="coerce").to_numpy(float))
    hi = np.abs(pd.to_numeric(upper, errors="coerce").to_numpy(float))
    return 0.5 * (lo + hi)


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Target
    df["log_mp"] = _safe_log10(df["Planet Mass [Me]"])
    # Predictors
    df["log_rp"] = _safe_log10(df["Planet Radius [Re]"])
    df["log_p"] = _safe_log10(df["Planet Period [days]"])
    df["log_teq"] = _safe_log10(df["Planet Temperature [K]"])
    return df


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Data prep
# -------------------------
@dataclass
class TrainingData:
    x_tr_s: np.ndarray
    x_te_s: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    s_tr: np.ndarray
    s_te: np.ndarray
    scaler: StandardScaler
    feature_cols: List[str]
    train_table: pd.DataFrame


def prepare_training(
    known: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> TrainingData:
    """
    Train only on known planets with measured masses: Mass Flag == False.
    Uses log10 mass as target and combines measurement error with intrinsic scatter in the model.
    """
    df = known.loc[known["Mass Flag"] == False].copy()
    df = add_log_features(df)

    sig_m = sym_sigma(df["Planet Mass Error Lower [Me]"], df["Planet Mass Error Upper [Me]"])
    df["sigma_log_mp"] = sig_m / (pd.to_numeric(df["Planet Mass [Me]"], errors="coerce") * np.log(10))

    keep = feature_cols + ["log_mp", "sigma_log_mp", "Planet Name", "Star Name",
                           "Planet Mass [Me]", "Planet Radius [Re]", "Planet Period [days]", "Planet Temperature [K]"]
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna()

    # Optional trimming of extreme outliers (edit/remove as desired)
    df = df[(df["Planet Mass [Me]"] < 5000) & (df["Planet Radius [Re]"] < 30)]

    x = df[feature_cols].to_numpy(float)
    y = df["log_mp"].to_numpy(float)
    s = df["sigma_log_mp"].to_numpy(float)

    x_tr, x_te, y_tr, y_te, s_tr, s_te = train_test_split(
        x, y, s, test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    return TrainingData(
        x_tr_s=x_tr_s,
        x_te_s=x_te_s,
        y_tr=y_tr,
        y_te=y_te,
        s_tr=s_tr,
        s_te=s_te,
        scaler=scaler,
        feature_cols=feature_cols,
        train_table=df,
    )


# -------------------------
# EDA plots
# -------------------------
def plot_eda(train_table: pd.DataFrame, outputs: Path) -> None:
    ensure_dir(outputs)

    # Mass–Radius with metallicity color + age size (if available)
    if ("Star Metallicity" in train_table.columns) and ("Star Age [Gyr]" in train_table.columns):
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(
            train_table["Planet Radius [Re]"].to_numpy(float),
            train_table["Planet Mass [Me]"].to_numpy(float),
            c=train_table["Star Metallicity"].to_numpy(float),
            s=10 + 6 * train_table["Star Age [Gyr]"].to_numpy(float),
            alpha=0.7,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Planet Radius [Re]")
        plt.ylabel("Planet Mass [Me]")
        cb = plt.colorbar(sc)
        cb.set_label("Star Metallicity [dex]")
        plt.title("Known planets: Mass–Radius (color=[Fe/H], size=Age)")
        plt.tight_layout()
        plt.savefig(outputs / "eda_mass_radius_metallicity_age.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.scatter(train_table["Star Metallicity"], train_table["Star Age [Gyr]"], alpha=0.6)
        plt.xlabel("Star Metallicity [dex]")
        plt.ylabel("Star Age [Gyr]")
        plt.title("Known planets: stellar age–metallicity plane")
        plt.tight_layout()
        plt.savefig(outputs / "eda_age_vs_metallicity.png", dpi=200)
        plt.close()

    # Correlation heatmap (numeric columns only)
    cols = [c for c in train_table.columns if c not in ("Planet Name", "Star Name")]
    corr = train_table[cols].corr(numeric_only=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(corr.values, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation")
    plt.title("Correlation heatmap (training table)")
    plt.tight_layout()
    plt.savefig(outputs / "eda_correlation_heatmap.png", dpi=200)
    plt.close()


# -------------------------
# Bayesian model (PyMC)
# -------------------------
@dataclass
class PosteriorSamples:
    b0: np.ndarray      # (S,)
    beta: np.ndarray    # (p, S)
    sigma_int: np.ndarray  # (S,)
    nu: np.ndarray      # (S,)


def fit_pymc_student_t(x_tr_s: np.ndarray, y_tr: np.ndarray, s_tr: np.ndarray, random_seed: int = 42):
    """
    Student-T regression with intrinsic scatter:
      y ~ StudentT(nu, mu=b0 + X beta, sigma=sqrt(sigma_int^2 + s_obs^2))
    """
    try:
        import pymc as pm
        import arviz as az
    except Exception as e:
        raise RuntimeError(
            "PyMC/ArviZ not found. Install with: pip install pymc arviz pytensor\n"
            f"Import error: {e}"
        )

    with pm.Model() as model:
        b0 = pm.Normal("intercept", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0, shape=x_tr_s.shape[1])
        sigma_int = pm.HalfNormal("sigma_int", 0.5)
        nu = pm.Exponential("nu_minus1", 1 / 10) + 1.0

        mu = b0 + pm.math.dot(x_tr_s, beta)
        sigma_tot = pm.math.sqrt(sigma_int**2 + s_tr**2)

        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma_tot, observed=y_tr)

        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=random_seed,
        )

    return idata


def extract_posterior(idata) -> PosteriorSamples:
    post = idata.posterior
    beta = post["beta"].stack(sample=("chain", "draw")).values
    b0 = post["intercept"].stack(sample=("chain", "draw")).values
    sigma_int = post["sigma_int"].stack(sample=("chain", "draw")).values
    nu = (post["nu_minus1"].stack(sample=("chain", "draw")).values + 1.0)
    return PosteriorSamples(b0=b0, beta=beta, sigma_int=sigma_int, nu=nu)


def posterior_predict_logmass(
    samples: PosteriorSamples,
    x_std: np.ndarray,
    sigma_obs: np.ndarray,
    n_draws: int = 2000,
    rng_seed: int = 0,
) -> np.ndarray:
    """
    Draw posterior predictive samples for log10 mass.
    Returns yrep with shape (N, n_draws).
    """
    rng = np.random.default_rng(rng_seed)
    s = samples.b0.shape[0]
    idx = rng.integers(0, s, size=n_draws)

    mu = samples.b0[idx] + x_std @ samples.beta[:, idx]          # (N, n_draws)
    sig = np.sqrt(samples.sigma_int[idx] ** 2 + sigma_obs[:, None] ** 2)

    eps = rng.standard_t(df=samples.nu[idx], size=mu.shape)
    yrep = mu + sig * eps
    return yrep


# -------------------------
# Evaluation + candidate application
# -------------------------
def evaluate_on_test(
    y_te: np.ndarray,
    yrep_te: np.ndarray,
    outputs: Path,
) -> Dict[str, float]:
    ensure_dir(outputs)

    pred_med = np.median(yrep_te, axis=1)
    pred_lo = np.quantile(yrep_te, 0.16, axis=1)
    pred_hi = np.quantile(yrep_te, 0.84, axis=1)

    rmse = float(np.sqrt(np.mean((pred_med - y_te) ** 2)))
    coverage = float(np.mean((y_te >= pred_lo) & (y_te <= pred_hi)))

    plt.figure(figsize=(6, 6))
    plt.errorbar(y_te, pred_med, yerr=[pred_med - pred_lo, pred_hi - pred_med], fmt="o", alpha=0.6)
    lims = [min(y_te.min(), pred_lo.min()) - 0.1, max(y_te.max(), pred_hi.max()) + 0.1]
    plt.plot(lims, lims, "--")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True log10(Mp/Me) (test)")
    plt.ylabel("Predicted log10(Mp/Me)")
    plt.title(f"Posterior predictive (held-out): RMSE={rmse:.3f} dex, 68% cov={coverage:.3f}")
    plt.tight_layout()
    plt.savefig(outputs / "calibration_pred_vs_true.png", dpi=200)
    plt.close()

    return {"rmse_dex": rmse, "coverage_68": coverage}


def apply_to_candidates(
    tpc: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    samples: PosteriorSamples,
    outputs: Path,
    n_draws: int = 2000,
) -> pd.DataFrame:
    """
    Apply trained model to candidates with complete features.
    Produces median and 16/84% in both log10 and linear mass.
    """
    ensure_dir(outputs)

    df = add_log_features(tpc)

    need = ["Planet Name", "Star Name"] + feature_cols + [
        "Planet Mass [Me]", "Planet Radius [Re]", "Planet Period [days]", "Planet Temperature [K]"
    ]
    df = df[need].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return df

    x = df[feature_cols].to_numpy(float)
    x_s = scaler.transform(x)

    # Candidate mass errors are often absent; default to 0.10 dex if missing.
    if ("Planet Mass Error Lower [Me]" in tpc.columns) and ("Planet Mass Error Upper [Me]" in tpc.columns):
        sig_m = sym_sigma(df["Planet Mass Error Lower [Me]"], df["Planet Mass Error Upper [Me]"])
        sigma_log = sig_m / (pd.to_numeric(df["Planet Mass [Me]"], errors="coerce") * np.log(10))
        sigma_log = np.asarray(sigma_log, dtype=float)
        sigma_log = np.where(np.isfinite(sigma_log), sigma_log, 0.10)
    else:
        sigma_log = np.full(len(df), 0.10)

    yrep = posterior_predict_logmass(samples, x_s, sigma_log, n_draws=n_draws, rng_seed=2)

    df = df.copy()
    df["logMp_pred_med"] = np.median(yrep, axis=1)
    df["logMp_pred_p16"] = np.quantile(yrep, 0.16, axis=1)
    df["logMp_pred_p84"] = np.quantile(yrep, 0.84, axis=1)

    df["Mp_pred_med"] = 10 ** df["logMp_pred_med"]
    df["Mp_pred_p16"] = 10 ** df["logMp_pred_p16"]
    df["Mp_pred_p84"] = 10 ** df["logMp_pred_p84"]

    # Plot predicted mass vs radius
    plt.figure(figsize=(7, 5))
    rp = df["Planet Radius [Re]"].to_numpy(float)
    mp_med = df["Mp_pred_med"].to_numpy(float)
    mp_lo = df["Mp_pred_p16"].to_numpy(float)
    mp_hi = df["Mp_pred_p84"].to_numpy(float)
    plt.errorbar(rp, mp_med, yerr=[mp_med - mp_lo, mp_hi - mp_med], fmt="o", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Planet Radius [Re]")
    plt.ylabel("Predicted Planet Mass [Me]")
    plt.title("TESS candidates: posterior predictive masses (median ± 68%)")
    plt.tight_layout()
    plt.savefig(outputs / "tpc_pred_mass_vs_radius.png", dpi=200)
    plt.close()

    # Plot predicted vs catalog mass (if catalog masses exist & positive)
    mp_cat = pd.to_numeric(df["Planet Mass [Me]"], errors="coerce").to_numpy(float)
    good = np.isfinite(mp_cat) & (mp_cat > 0)
    if np.sum(good) > 10:
        plt.figure(figsize=(6, 6))
        plt.scatter(mp_cat[good], mp_med[good], alpha=0.6)
        lims = [min(mp_cat[good].min(), mp_lo[good].min()) * 0.8, max(mp_cat[good].max(), mp_hi[good].max()) * 1.2]
        plt.plot(lims, lims, "--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(lims)
        plt.ylim(lims)
        plt.xlabel("Catalog Mass [Me] (TPC)")
        plt.ylabel("Predicted Mass [Me]")
        plt.title("TPC: predicted vs catalog mass")
        plt.tight_layout()
        plt.savefig(outputs / "tpc_pred_vs_catalog_mass.png", dpi=200)
        plt.close()

    return df


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--known", type=str, default="Ariel_MCS_Known_2024-07-09.csv", help="Known planets CSV")
    p.add_argument("--tpc", type=str, default="Ariel_MCS_TPCs_2024-07-09.csv", help="TESS planet candidates CSV")
    p.add_argument("--feature-set", type=str, default="strict", choices=["strict", "broad"],
                   help="strict includes Age+Metallicity; broad drops them to maximize candidate coverage")
    p.add_argument("--outputs", type=str, default="outputs", help="Output directory for plots/tables")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-draws", type=int, default=2000, help="Posterior predictive draws")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    ensure_dir(outputs)

    known = pd.read_csv(args.known)
    tpc = pd.read_csv(args.tpc)

    feature_cols = STRICT_FEATURES if args.feature_set == "strict" else BROAD_FEATURES

    # 1) Prepare training data + EDA
    td = prepare_training(known, feature_cols=feature_cols, random_seed=args.random_seed)
    td.train_table.to_csv(outputs / "train_clean.csv", index=False)
    plot_eda(td.train_table, outputs)

    # 2) Fit Bayesian model
    idata = fit_pymc_student_t(td.x_tr_s, td.y_tr, td.s_tr, random_seed=args.random_seed)

    # Save quick summary
    try:
        import arviz as az
        summ = az.summary(idata, var_names=["intercept", "beta", "sigma_int", "nu_minus1"])
        summ.to_csv(outputs / "posterior_summary.csv")
    except Exception:
        pass

    # 3) Evaluate
    samples = extract_posterior(idata)
    yrep_te = posterior_predict_logmass(samples, td.x_te_s, td.s_te, n_draws=args.n_draws, rng_seed=args.random_seed + 1)
    metrics = evaluate_on_test(td.y_te, yrep_te, outputs)
    (outputs / "metrics.txt").write_text(
        f"RMSE (dex): {metrics['rmse_dex']:.4f}\n68% interval coverage: {metrics['coverage_68']:.4f}\n"
    )

    # 4) Apply to candidates
    pred = apply_to_candidates(
        tpc,
        feature_cols=feature_cols,
        scaler=td.scaler,
        samples=samples,
        outputs=outputs,
        n_draws=args.n_draws,
    )
    pred.to_csv(outputs / "tpc_predictions.csv", index=False)

    print("Done.")
    print(f"Outputs written to: {outputs.resolve()}")
    print("Key files:")
    print(" - train_clean.csv")
    print(" - posterior_summary.csv (if arviz available)")
    print(" - metrics.txt")
    print(" - tpc_predictions.csv")
    print(" - *.png plots")


if __name__ == "__main__":
    main()
