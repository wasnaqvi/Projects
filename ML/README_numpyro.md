# Ariel follow-up targeting (JAX/NumPyro)

This project trains a probabilistic model on **Known MCS** planets with *measured* masses,
applies it to **TPCs**, and ranks candidates for **Ariel (0.5–7.8 μm)** science + pre-launch follow-up.

## Why probabilistic?
You want **uncertainty-aware** target lists:
- predict a *distribution* for planet mass, not a point
- propagate that into RV semi-amplitude K and atmospheric signal proxies
- rank by probabilities (e.g., P(K > 2 m/s)) + science yield + feasibility

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_numpyro.txt
```

## Run (recommended)
Broad features maximize TPC coverage:
```bash
python ariel_targeting_numpyro.py   --known Ariel_MCS_Known_2024-07-09.csv   --tpc Ariel_MCS_TPCs_2024-07-09.csv   --feature-set broad   --model linear   --inference svi   --outputs outputs_ariel_numpyro
```

## Try the Bayesian Neural Net (BNN)
```bash
python ariel_targeting_numpyro.py   --known Ariel_MCS_Known_2024-07-09.csv   --tpc Ariel_MCS_TPCs_2024-07-09.csv   --feature-set broad   --model bnn   --inference svi   --hidden 32   --outputs outputs_ariel_bnn
```

## Full MCMC (linear only)
```bash
python ariel_targeting_numpyro.py   --known Ariel_MCS_Known_2024-07-09.csv   --tpc Ariel_MCS_TPCs_2024-07-09.csv   --feature-set broad   --model linear   --inference nuts   --outputs outputs_ariel_nuts
```

## Outputs
In the output folder:
- `tpc_with_scores.csv` (all targets + predicted masses + follow-up metrics + scores)
- `target_list_followup_priority.csv` (best to follow up now)
- `target_list_ariel_science.csv` (best Ariel atmospheric yield proxy)
- `target_list_rv_for_ariel.csv` (best RV among top science)
- `target_list_ephemeris_for_ariel.csv` (best ephemeris maintenance among top science)
- `plots/` includes calibration + targeting plots
