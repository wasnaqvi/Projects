"""
ariel_spectroscopy_bayes.py

Bayesian ranking of Ariel targets using ONLY transit / eclipse observables.
Separate models for Transit and Eclipse spectroscopy.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

# -----------------------
# Utilities
# -----------------------
def sym_sigma(lo, hi):
    lo = np.abs(pd.to_numeric(lo, errors="coerce"))
    hi = np.abs(pd.to_numeric(hi, errors="coerce"))
    return 0.5 * (lo + hi)

def log_safe(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.log10(x.where(x > 0))

# -----------------------
# Feature construction
# -----------------------
def build_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace("\u202f", " ", regex=False)

    rename = {
        "Transit Depth [%]": "depth",
        "Transit Depth Error Lower [%]": "depth_lo",
        "Transit Depth Error Upper [%]": "depth_hi",
        "Transit Duration [hr]": "duration",
        "Transit Duration Error Lower [hr]": "duration_lo",
        "Transit Duration Error Upper [hr]": "duration_hi",
        "Transit Duration T14 [s]": "t14",
        "Transit Mid Time Error Lower [days]": "t0_lo",
        "Transit Mid Time Error Upper [days]": "t0_hi",
        "Available Transits": "ntr",
    }
    df = df.rename(columns=rename)

    df["log_depth"] = log_safe(df["depth"])
    df["log_duration"] = log_safe(df["duration"])
    df["log_t14"] = log_safe(df["t14"])
    df["log_ntr"] = log_safe(df["ntr"])

    df["sigma_depth"] = sym_sigma(df["depth_lo"], df["depth_hi"])
    df["sigma_duration"] = sym_sigma(df["duration_lo"], df["duration_hi"])
    df["sigma_t0"] = sym_sigma(df["t0_lo"], df["t0_hi"]) * 24.0  # days → hr

    return df


FEATURES = [
    "log_depth",
    "log_duration",
    "log_t14",
    "log_ntr",
    "sigma_depth",
    "sigma_duration",
    "sigma_t0",
]

# -----------------------
# Bayesian spectroscopy model
# -----------------------
def spectroscopy_model(x, sigma_obs, y=None):
    n, p = x.shape
    b0 = numpyro.sample("b0", dist.Normal(0, 1))
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand((p,)))
    sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(0.5))
    nu = numpyro.sample("nu", dist.Exponential(1 / 10)) + 1

    mu = b0 + jnp.dot(x, beta)
    sigma = jnp.sqrt(sigma_int**2 + sigma_obs**2)

    numpyro.sample("y", dist.StudentT(nu, mu, sigma), obs=y)

# -----------------------
# Fit + WAIC
# -----------------------
def fit_model(df, label, outdir):
    X = df[FEATURES].to_numpy(float)
    mu_X, sd_X = X.mean(0), X.std(0)
    X = (X - mu_X) / sd_X

    y = (
        df["Transit Depth [%]"]
        * df["Transit Duration [hr]"]
        * np.sqrt(df["Available Transits(Number of Transits available)"])
    )
    y = log_safe(y).to_numpy(float)

    sigma_obs = (
        df["sigma_depth"]
        + df["sigma_duration"]
        + df["sigma_t0"]
    ).to_numpy(float)

    mask = np.isfinite(X).all(1) & np.isfinite(y) & np.isfinite(sigma_obs)
    X, y, sigma_obs = X[mask], y[mask], sigma_obs[mask]

    rng = jax.random.PRNGKey(0)
    mcmc = MCMC(NUTS(spectroscopy_model), num_warmup=1000, num_samples=2000)
    mcmc.run(rng, x=jnp.array(X), sigma_obs=jnp.array(sigma_obs), y=jnp.array(y))

    samples = mcmc.get_samples()

    ### FIX: compute pointwise log-likelihood
    log_likelihood = (
        Predictive(
            spectroscopy_model,
            samples,
            return_sites=["y"],
        )(
            rng,
            x=jnp.array(X),
            sigma_obs=jnp.array(sigma_obs),
            y=None,
        )["y"]
    )

    ### FIX: wrap in ArviZ InferenceData
    idata = az.from_dict(
        posterior=samples,
        log_likelihood={"y": np.array(log_likelihood)},
    )

    w = az.waic(idata)
    (outdir / f"waic_{label}.txt").write_text(str(w))

    return samples, mu_X, sd_X, w

# -----------------------
# Rank TPCs
# -----------------------
def rank_tpcs(tpc, samples, mu, sd, label, outdir):
    X = tpc[FEATURES].to_numpy(float)
    X = (X - mu) / sd

    sigma_obs = (
        tpc["sigma_depth"]
        + tpc["sigma_duration"]
        + tpc["sigma_t0"]
    ).to_numpy(float)

    pred = Predictive(
        spectroscopy_model, samples, num_samples=2000
    )(
        jax.random.PRNGKey(1),
        x=jnp.array(X),
        sigma_obs=jnp.array(sigma_obs),
    )

    score = np.median(pred["y"], axis=0)
    tpc[f"spectroscopy_score_{label}"] = score

    tpc.sort_values(
        f"spectroscopy_score_{label}", ascending=False
    ).to_csv(outdir / f"ranked_{label}.csv", index=False)

# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--known", required=True)
    p.add_argument("--tpc", required=True)
    p.add_argument("--out", default="ariel_spectroscopy_outputs")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    known = build_features(pd.read_csv(args.known))
    tpc = build_features(pd.read_csv(args.tpc))

    for mode in ["Transit", "Eclipse"]:
        known_m = known[known["Preferred Method"] == mode]
        tpc_m = tpc[tpc["Preferred Method"] == mode]

        samples, mu, sd, w = fit_model(known_m, mode, outdir)
        rank_tpcs(tpc_m, samples, mu, sd, mode, outdir)

        print(f"{mode} WAIC:")
        print(w)

if __name__ == "__main__":
    numpyro.set_host_device_count(1)
    main()
