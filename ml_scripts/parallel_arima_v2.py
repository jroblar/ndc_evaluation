# -*- coding: utf-8 -*-
"""
Refactored future-ensemble generator (country-wise).
Fixes:
- Never "forecast" an observed year (start_year defaults to last_year + 1)
- ARIMA/SARIMAX simulated outputs for d>0 are integrated back to LEVELS
- BOUNDED (%/shares) are simulated in logit-space with mean reversion (no clipping-to-zero artifacts)
- No Latin Hypercube recombination (preserves time consistency)
- Produces a tidy long DF: iso_alpha_3, future_id, year, <vars...>
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

import pyarrow as pa
import pyarrow.parquet as pq

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ----------------------------
# Warnings
# ----------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
    module=r"sklearn",
)

# ----------------------------
# Variable lists (your current setup)
# ----------------------------
ARIMA_VARS = [
    "log_gdp_2021_ppp_intl_usd",
    "log_pop_total",
    "log_energy_use_kg_of_oil_equivalent_per_capita",
    "log_electricity_consumption_kwh_per_capita",
    "log_gdp_per_capita_2021_ppp_intl_usd",
    "log_cereal_yield_kg_per_hectare",
]

BOUNDED_VARS = [
    "fossil_fuel_energy_consumption_pct",
    "renewable_energy_consumption_pct",
    "industry_pct_of_gdp",
    "exports_pct_of_gdp",
    "manufacturing_pct_of_gdp",
    "forest_area_pct",
    "arable_land_pct",
    "agricultural_land_pct",
]

POLICY_VAR = "log_policy_flow_jur_stock"

POLICY_SCENARIOS = {
    "baseline": 0.00,
    "moderate": 0.05,
    "ambitious": 0.10,
}


# ----------------------------
# Helpers
# ----------------------------
def _to_period_year_index(series: pd.Series, full_years: np.ndarray) -> pd.Series:
    """
    Reindex a Series to a complete set of years and return with PeriodIndex.
    """
    s = pd.to_numeric(series, errors="coerce")

    # Reindex FIRST (this inserts NaNs for missing years)
    s = s.reindex(full_years)

    # Now lengths match → safe
    s.index = pd.PeriodIndex(full_years, freq="Y")

    return s



def _safe_last_valid(series: pd.Series, default: float = np.nan) -> float:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    return float(s.iloc[-1]) if len(s) else float(default)


def _fit_and_simulate_arima_levels(
    hist_levels: pd.Series,
    horizon: int,
    n_scenarios: int,
    rand_seed: int,
    arima_order: Tuple[int, int, int],
    auto_tune: bool,
    max_p: int,
    max_d: int,
    max_q: int,
    min_points_for_tune: int = 20,
) -> np.ndarray:
    """
    Fits a SARIMAX(order=p,d,q) on a *level* series and returns simulated *levels*.
    IMPORTANT:
      - If d>0 and simple_differencing=True, fit.simulate() returns differenced values.
      - We integrate back to levels using cumulative sum + last observed level.
    """
    rng = np.random.default_rng(rand_seed)

    # Clean history
    hist = pd.to_numeric(hist_levels, errors="coerce").dropna().astype(float)

    # If too short / nearly constant: random walk around last
    if len(hist) < 10 or hist.nunique() < 3:
        last = hist.iloc[-1] if len(hist) else np.nan
        noise = rng.normal(0, 0.002, size=(n_scenarios, horizon))
        return np.full((n_scenarios, horizon), last) + noise

    # Choose order (optional)
    order = arima_order
    if auto_tune and len(hist) >= min_points_for_tune and hist.nunique() >= 3:
        try:
            order = auto_arima(
                hist,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
            ).order
        except Exception:
            order = arima_order

    p, d, q = order

    # Fit SARIMAX with stable settings
    try:
        model = SARIMAX(
            hist,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=True if d > 0 else False,
            initialization="approximate_diffuse",
        )
        fit = model.fit(disp=False)

        sims = np.vstack(
            [fit.simulate(nsimulations=horizon, anchor="end").values for _ in range(n_scenarios)]
        ).astype(float)

    except Exception:
        # Fallback to (0,1,0) if possible, else flat+noise
        try:
            model = SARIMAX(
                hist,
                order=(0, 1, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True,
                initialization="approximate_diffuse",
            )
            fit = model.fit(disp=False)
            sims = np.vstack(
                [fit.simulate(nsimulations=horizon, anchor="end").values for _ in range(n_scenarios)]
            ).astype(float)
            p, d, q = (0, 1, 0)
        except Exception:
            last = hist.iloc[-1]
            noise = rng.normal(0, 0.02, size=(n_scenarios, horizon))
            return np.full((n_scenarios, horizon), last) + noise

    # Replace invalid values with last observed
    last_level = float(hist.iloc[-1])
    sims[~np.isfinite(sims)] = np.nan
    sims = np.where(np.isnan(sims), last_level, sims)

    # If differenced, integrate back to level space
    if d > 0:
        sims = np.cumsum(sims, axis=1) + last_level

    # Guard: for log variables, avoid insane negatives (optional)
    # (This is conservative; adjust if you want)
    sims = np.maximum(sims, -50.0)

    return sims


def _simulate_bounded_logit_mean_reverting(
    last_pct: float,
    horizon: int,
    n_scenarios: int,
    rand_seed: int,
    sigma: float = 0.35,     # shock scale in logit space
    phi: float = 0.85,       # mean reversion strength
    eps: float = 1e-3,       # prevents 0/100 singularities
) -> np.ndarray:
    """
    Simulate bounded percent variable in logit space with mean reversion to last observed value.
    Returns values in [0, 100] without clipping artifacts.
    """
    rng = np.random.default_rng(rand_seed)

    # If missing, just return NaNs (caller can fill)
    if not np.isfinite(last_pct):
        return np.full((n_scenarios, horizon), np.nan, dtype=float)

    # Convert pct to prob and logit
    p0 = np.clip(last_pct / 100.0, eps, 1.0 - eps)
    z0 = np.log(p0 / (1.0 - p0))

    # AR(1) in logit space around z0
    z = np.empty((n_scenarios, horizon), dtype=float)
    z[:, 0] = z0 + rng.normal(0, sigma, size=n_scenarios)

    for t in range(1, horizon):
        shock = rng.normal(0, sigma, size=n_scenarios)
        z[:, t] = z0 + phi * (z[:, t - 1] - z0) + shock

    # Inverse logit back to pct
    p = 1.0 / (1.0 + np.exp(-z))
    return 100.0 * p


@dataclass
class EnsembleConfig:
    end_year: int
    n_scenarios: int
    policy_scenario: str
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    auto_tune_arima: bool = True
    max_p: int = 3
    max_d: int = 1
    max_q: int = 3
    random_state: int = 42

    # bounded simulation parameters
    bounded_sigma: float = 0.35
    bounded_phi: float = 0.85

    # if you want a baseline row for the last observed year included in output
    include_last_observed_row: bool = True


def simulate_country_ensemble(
    df_country: pd.DataFrame,
    iso: str,
    years_future: np.ndarray,
    config: EnsembleConfig,
    seed: int,
    arima_vars: List[str],
    bounded_vars: List[str],
    policy_var: str,
) -> pd.DataFrame:
    """
    Returns a long dataframe for one country with n_scenarios simulated futures.
    """
    rng = np.random.default_rng(seed)

    dfc = df_country.sort_values("year").reset_index(drop=True)

    last_year = int(dfc["year"].max())
    last_policy = _safe_last_valid(dfc[policy_var], default=np.nan)
    policy_growth = POLICY_SCENARIOS[config.policy_scenario]

    # --- 1) ARIMA vars: simulate LEVELS ---
    sims_arima: Dict[str, np.ndarray] = {}
    for k, var in enumerate(arima_vars):
        hist = pd.to_numeric(dfc[var], errors="coerce")
        hist.index = dfc["year"].astype(int).values

        # ensure continuous year span for the fit
        full_years = np.arange(int(dfc["year"].min()), last_year + 1)
        hist = _to_period_year_index(hist, full_years)
        hist_non_na = hist.dropna()

        sims_arima[var] = _fit_and_simulate_arima_levels(
            hist_levels=hist_non_na,
            horizon=len(years_future),
            n_scenarios=config.n_scenarios,
            rand_seed=int((seed + 1009 * (k + 1)) % 2**32),
            arima_order=config.arima_order,
            auto_tune=config.auto_tune_arima,
            max_p=config.max_p,
            max_d=config.max_d,
            max_q=config.max_q,
        )

    # --- 2) Bounded vars: logit mean-reverting around last observed ---
    sims_bounded: Dict[str, np.ndarray] = {}
    for k, var in enumerate(bounded_vars):
        last_pct = _safe_last_valid(dfc[var], default=np.nan)
        sims_bounded[var] = _simulate_bounded_logit_mean_reverting(
            last_pct=last_pct,
            horizon=len(years_future),
            n_scenarios=config.n_scenarios,
            rand_seed=int((seed + 2003 * (k + 1)) % 2**32),
            sigma=config.bounded_sigma,
            phi=config.bounded_phi,
        )

    # --- 3) Policy path (deterministic scenario) ---
    # We start policy updates only after last observed year.
    policy_path = np.empty(len(years_future), dtype=float)
    current = last_policy
    for i, y in enumerate(years_future):
        if y > last_year:
            current = current + policy_growth
        policy_path[i] = current

    # --- 4) Build output rows ---
    rows = []
    for s in range(config.n_scenarios):
        fid = f"id_{iso}_{s+1}_{config.policy_scenario}"

        if config.include_last_observed_row:
            # add the last observed year row (as "baseline") so year alignment is explicit
            base_row = {
                "iso_alpha_3": iso,
                "future_id": fid,
                "year": last_year,
                policy_var: last_policy,
            }
            # populate observed values where available
            for var in arima_vars + bounded_vars:
                base_row[var] = _safe_last_valid(dfc[var], default=np.nan)
            rows.append(base_row)

        for t, y in enumerate(years_future):
            row = {
                "iso_alpha_3": iso,
                "future_id": fid,
                "year": int(y),
                policy_var: float(policy_path[t]),
            }
            for var in arima_vars:
                row[var] = float(sims_arima[var][s, t])
            for var in bounded_vars:
                row[var] = float(sims_bounded[var][s, t])
            rows.append(row)

    out = pd.DataFrame(rows)

    # Final guardrails: bounded vars in [0,100]
    out[bounded_vars] = out[bounded_vars].clip(lower=0.0, upper=100.0)

    # Optional: enforce non-crazy log values (you can tune thresholds)
    for v in arima_vars:
        if v.startswith("log_"):
            out[v] = out[v].clip(lower=-50.0, upper=60.0)

    return out


def generate_ensemble(
    df: pd.DataFrame,
    out_path: str,
    config: EnsembleConfig,
    arima_vars: Optional[List[str]] = None,
    bounded_vars: Optional[List[str]] = None,
    policy_var: str = POLICY_VAR,
    n_jobs: int = -1,
) -> None:
    """
    Country-parallel ensemble generator.
    By default, future years begin at last_observed_year + 1 (per country).
    Output includes last observed year row per scenario if config.include_last_observed_row=True.
    """
    arima_vars = arima_vars or ARIMA_VARS
    bounded_vars = bounded_vars or BOUNDED_VARS

    required = {"iso_alpha_3", "year", policy_var, *arima_vars, *bounded_vars}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # group countries
    iso_groups = {iso: g.copy() for iso, g in df.groupby("iso_alpha_3")}
    iso_list = sorted(iso_groups.keys())

    # deterministic per-iso seeds
    seeds = {iso: (hash((iso, config.random_state)) % 2**32) for iso in iso_list}

    def _worker(iso: str) -> pd.DataFrame:
        g = iso_groups[iso].sort_values("year")
        last_year = int(g["year"].max())

        # IMPORTANT: start forecasts AFTER last observed year
        years_future = np.arange(last_year + 1, config.end_year + 1, dtype=int)
        if len(years_future) == 0:
            # If end_year <= last_year, still emit baseline row (optional)
            years_future = np.array([], dtype=int)

        return simulate_country_ensemble(
            df_country=g,
            iso=iso,
            years_future=years_future,
            config=config,
            seed=int(seeds[iso]),
            arima_vars=arima_vars,
            bounded_vars=bounded_vars,
            policy_var=policy_var,
        )

    dfs = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_worker)(iso) for iso in iso_list
    )

    ensemble_df = pd.concat(dfs, ignore_index=True)

    # Write parquet
    table = pa.Table.from_pandas(ensemble_df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    print(
        f"Saved ensemble: {len(iso_list)} countries × {config.n_scenarios} scenarios "
        f"({config.policy_scenario}) → {out_path}"
    )


# ----------------------------
# Example usage (match your folder layout)
# ----------------------------
if __name__ == "__main__":
    start_time = time.time()

    SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)

    DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, "data")
    PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "processed_data")
    OUTPUT_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "output")
    ENSEMBLE_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "ensemble")
    TRAINING_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "training")

    os.makedirs(ENSEMBLE_DIR_PATH, exist_ok=True)

    training_df_log_transformed = pd.read_csv(
        os.path.join(TRAINING_DIR_PATH, "training_df_top15_preds.csv")
    )

    n_scenarios = 1000

    cfg = EnsembleConfig(
        end_year=2030,
        n_scenarios=n_scenarios,
        policy_scenario="moderate",
        arima_order=(1, 1, 1),
        auto_tune_arima=True,
        max_p=3,
        max_d=1,
        max_q=3,
        random_state=42,
        bounded_sigma=0.35,
        bounded_phi=0.85,
        include_last_observed_row=True,  # includes last observed year per scenario
    )

    out_path = os.path.join(ENSEMBLE_DIR_PATH, f"ensemble_arima_{n_scenarios}.parquet")

    generate_ensemble(
        df=training_df_log_transformed,
        out_path=out_path,
        config=cfg,
        n_jobs=-1,
    )

    ensemble_df = pd.read_parquet(out_path)
    print("Shape:", ensemble_df.shape)
    print(ensemble_df.head(10))

    print(f"Execution time: {time.time() - start_time:.2f} seconds")
