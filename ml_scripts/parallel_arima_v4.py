# -*- coding: utf-8 -*-
"""
Refactored future-ensemble generator (country-wise).
Fixes:
- Never "forecast" an observed year (start_year defaults to last_year + 1)
- ARIMA/SARIMAX simulated outputs for d>0 are integrated back to LEVELS
- No Latin Hypercube recombination (preserves time consistency)
- Produces a tidy long DF: iso_alpha_3, future_id, year, <vars...>

Migration note:
- Supports JSON-driven variable projection constraints (binary/count/bounds/monotonicity).
"""

import os
import json
import time
import warnings
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import yaml

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



SUPPORTED_RULE_CATEGORIES = {
    "unconstrained",
    "log_unbounded",
    "binary",
    "count",
    "cumulative_count",
    "cumulative_binary",
    "bounded_0_1",
    "bounded_0_100",
    "non_negative",
    "non_decreasing",
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


def _signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(z: pd.Series) -> pd.Series:
    # keep expm1 stable for extreme values
    z = z.clip(lower=-700.0, upper=700.0)
    return np.sign(z) * np.expm1(np.abs(z))


def _stable_unit_interval(*parts: object) -> float:
    """
    Deterministic pseudo-random number in [0, 1) derived from input parts.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    as_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return as_int / float(2**64)


def load_projection_rules(rule_path: str) -> Dict[str, object]:
    """
    Load variable projection rules from a JSON file.
    Expected top-level keys:
      - default_category: str
      - categories: {category_name: [var1, var2, ...]}
      - prefix_rules: [{"prefix": "inc_has_", "category": "binary"}, ...]
      - overrides: {var_name: category_name}
    """
    with open(rule_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    default_category = spec.get("default_category", "unconstrained")
    if default_category not in SUPPORTED_RULE_CATEGORIES:
        raise ValueError(
            f"Unsupported default category '{default_category}'. "
            f"Allowed categories: {sorted(SUPPORTED_RULE_CATEGORIES)}"
        )

    categories = spec.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("'categories' must be a dict mapping category -> list of variables.")

    category_lookup: Dict[str, str] = {}
    for category, vars_list in categories.items():
        if category not in SUPPORTED_RULE_CATEGORIES:
            raise ValueError(
                f"Unsupported category '{category}' in JSON. "
                f"Allowed categories: {sorted(SUPPORTED_RULE_CATEGORIES)}"
            )
        if not isinstance(vars_list, list):
            raise ValueError(f"'categories.{category}' must be a list.")
        for var in vars_list:
            if not isinstance(var, str):
                raise ValueError(f"Variable in 'categories.{category}' must be a string.")
            category_lookup[var] = category

    prefix_rules = spec.get("prefix_rules", [])
    if not isinstance(prefix_rules, list):
        raise ValueError("'prefix_rules' must be a list.")
    parsed_prefix_rules: List[Tuple[str, str]] = []
    for i, item in enumerate(prefix_rules):
        if not isinstance(item, dict):
            raise ValueError(f"'prefix_rules[{i}]' must be an object.")
        prefix = item.get("prefix")
        category = item.get("category")
        if not isinstance(prefix, str) or not isinstance(category, str):
            raise ValueError(f"'prefix_rules[{i}]' must contain string keys 'prefix' and 'category'.")
        if category not in SUPPORTED_RULE_CATEGORIES:
            raise ValueError(
                f"Unsupported category '{category}' in prefix rule {i}. "
                f"Allowed categories: {sorted(SUPPORTED_RULE_CATEGORIES)}"
            )
        parsed_prefix_rules.append((prefix, category))

    overrides = spec.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("'overrides' must be a dict mapping variable -> category.")
    for var, category in overrides.items():
        if not isinstance(var, str) or not isinstance(category, str):
            raise ValueError("Every override key/value must be a string.")
        if category not in SUPPORTED_RULE_CATEGORIES:
            raise ValueError(
                f"Unsupported override category '{category}' for '{var}'. "
                f"Allowed categories: {sorted(SUPPORTED_RULE_CATEGORIES)}"
            )
        category_lookup[var] = category

    return {
        "default_category": default_category,
        "category_lookup": category_lookup,
        "prefix_rules": parsed_prefix_rules,
    }


def _lookup_category_from_rulebook(var_name: str, rulebook: Optional[Dict[str, object]]) -> str:
    if not rulebook:
        return "log_unbounded" if var_name.startswith("log_") else "unconstrained"

    category_lookup = rulebook["category_lookup"]
    if var_name in category_lookup:
        return str(category_lookup[var_name])

    for prefix, category in rulebook["prefix_rules"]:
        if var_name.startswith(prefix):
            return str(category)

    default_category = str(rulebook["default_category"])
    return default_category


def _get_variable_category(var_name: str, rulebook: Optional[Dict[str, object]]) -> str:
    category = _lookup_category_from_rulebook(var_name, rulebook)
    if (
        rulebook
        and var_name.startswith("x_log_signed_")
        and category == str(rulebook["default_category"])
    ):
        base_var = var_name[len("x_log_signed_"):]
        return _lookup_category_from_rulebook(base_var, rulebook)
    return category


def _apply_projection_rules(
    out: pd.DataFrame,
    arima_vars: List[str],
    rulebook: Optional[Dict[str, object]],
) -> pd.DataFrame:
    """
    Apply category-specific constraints to simulated paths.
    Rules are applied per country/future_id trajectory so monotonic constraints
    respect temporal order.
    """
    if out.empty or not arima_vars:
        return out

    out = out.copy()

    monotonic_vars: List[str] = []
    binary_vars: List[str] = []
    binary_latent: Dict[str, pd.Series] = {}
    for var in arima_vars:
        if var not in out.columns:
            continue

        category = _get_variable_category(var, rulebook)
        s = pd.to_numeric(out[var], errors="coerce").astype(float)
        is_signed_log = var.startswith("x_log_signed_")
        values = _signed_expm1(s) if is_signed_log else s

        if category == "log_unbounded":
            values = values.clip(lower=-50.0, upper=60.0)
        elif category == "binary":
            # Keep latent values for controlled one-time switching logic below.
            values = values.clip(lower=0.0, upper=1.0)
            binary_vars.append(var)
            binary_latent[var] = values.astype(float)
            values = np.rint(values).clip(lower=0.0, upper=1.0)
        elif category == "count":
            values = np.rint(values).clip(lower=0.0)
        elif category == "cumulative_count":
            values = np.rint(values).clip(lower=0.0)
            monotonic_vars.append(var)
        elif category == "cumulative_binary":
            values = np.rint(values).clip(lower=0.0, upper=1.0)
            monotonic_vars.append(var)
        elif category == "bounded_0_1":
            values = values.clip(lower=0.0, upper=1.0)
        elif category == "bounded_0_100":
            values = values.clip(lower=0.0, upper=100.0)
        elif category == "non_negative":
            values = values.clip(lower=0.0)
        elif category == "non_decreasing":
            monotonic_vars.append(var)
        elif category == "unconstrained":
            pass
        else:
            raise ValueError(
                f"Unsupported category '{category}' for variable '{var}'. "
                f"Allowed categories: {sorted(SUPPORTED_RULE_CATEGORIES)}"
            )

        out[var] = _signed_log1p(values).astype(float) if is_signed_log else values.astype(float)

    if binary_vars or monotonic_vars:
        out["_row_order"] = np.arange(len(out))
        out = out.sort_values(["iso_alpha_3", "future_id", "year"]).reset_index(drop=True)

    if binary_vars:
        group_keys = out[["iso_alpha_3", "future_id"]].drop_duplicates()
        for var in binary_vars:
            latent_sorted = pd.to_numeric(binary_latent[var], errors="coerce").reset_index(drop=True)
            out[var] = pd.to_numeric(out[var], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

            for _, key_row in group_keys.iterrows():
                iso = key_row["iso_alpha_3"]
                fid = key_row["future_id"]
                idx = out.index[(out["iso_alpha_3"] == iso) & (out["future_id"] == fid)].to_numpy()
                if len(idx) <= 1:
                    continue

                path = out.loc[idx, var].to_numpy(dtype=float)
                latent = latent_sorted.iloc[idx].to_numpy(dtype=float)

                # Treat first row as anchor state; future can switch at most once and then persist.
                start_state = int(np.rint(path[0]))
                future_latent = latent[1:]
                n_future = len(future_latent)
                if n_future == 0:
                    continue

                if start_state == 0:
                    crossing = np.where(future_latent >= 0.65)[0]
                    drift = max(float(np.nanmean(future_latent) - 0.5), 0.0)
                else:
                    crossing = np.where(future_latent <= 0.35)[0]
                    drift = max(float(0.5 - np.nanmean(future_latent)), 0.0)

                switch_prob = min(0.45, 0.02 + 0.90 * drift)
                u_switch = _stable_unit_interval(iso, fid, var, "switch")
                should_switch = (len(crossing) > 0) or (u_switch < switch_prob)

                if not should_switch:
                    path[1:] = start_state
                else:
                    if len(crossing) > 0:
                        switch_at = int(crossing[0]) + 1
                    else:
                        u_year = _stable_unit_interval(iso, fid, var, "switch_year")
                        switch_at = int(np.floor(u_year * n_future)) + 1
                    switch_state = 1 - start_state
                    path[1:switch_at] = start_state
                    path[switch_at:] = switch_state

                out.loc[idx, var] = path

    if monotonic_vars:
        g = out.groupby(["iso_alpha_3", "future_id"], sort=False)
        for var in monotonic_vars:
            out[var] = g[var].cummax()

    if binary_vars or monotonic_vars:
        out = out.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)

    return out


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


@dataclass
class EnsembleConfig:
    end_year: int
    n_scenarios: int
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    auto_tune_arima: bool = True
    max_p: int = 3
    max_d: int = 1
    max_q: int = 3
    random_state: int = 42

    # if you want a baseline row for the last observed year included in output
    include_last_observed_row: bool = True
    projection_rules_path: Optional[str] = None


def simulate_country_ensemble(
    df_country: pd.DataFrame,
    iso: str,
    years_future: np.ndarray,
    config: EnsembleConfig,
    seed: int,
    arima_vars: List[str],
    rulebook: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Returns a long dataframe for one country with n_scenarios simulated futures.
    """
    dfc = df_country.sort_values("year").reset_index(drop=True)

    last_year = int(dfc["year"].max())

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

    # --- 2) Build output rows ---
    rows = []
    for s in range(config.n_scenarios):
        fid = f"id_{iso}_{s+1}"

        if config.include_last_observed_row:
            # add the last observed year row (as "baseline") so year alignment is explicit
            base_row = {
                "iso_alpha_3": iso,
                "future_id": fid,
                "year": last_year,
            }
            # populate observed values where available
            for var in arima_vars:
                base_row[var] = _safe_last_valid(dfc[var], default=np.nan)
            rows.append(base_row)

        for t, y in enumerate(years_future):
            row = {
                "iso_alpha_3": iso,
                "future_id": fid,
                "year": int(y),
            }
            for var in arima_vars:
                row[var] = float(sims_arima[var][s, t])
            rows.append(row)

    out = pd.DataFrame(rows)

    return _apply_projection_rules(out, arima_vars=arima_vars, rulebook=rulebook)


def generate_ensemble(
    df: pd.DataFrame,
    out_path: str,
    config: EnsembleConfig,
    arima_vars: Optional[List[str]] = None,
    n_jobs: int = -1,
) -> None:
    """
    Country-parallel ensemble generator.
    By default, future years begin at last_observed_year + 1 (per country).
    Output includes last observed year row per scenario if config.include_last_observed_row=True.
    """
    if arima_vars is None:
        arima_vars = [
            c for c in df.columns
            if c not in {"iso_alpha_3", "year", "future_id"}
        ]
    else:
        arima_vars = list(arima_vars)

    if not arima_vars:
        raise ValueError(
            "No variables selected for projection. "
            "Pass arima_vars explicitly or include feature columns in df."
        )

    rulebook = load_projection_rules(config.projection_rules_path) if config.projection_rules_path else None

    required = {"iso_alpha_3", "year", *arima_vars}
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
            rulebook=rulebook,
        )

    dfs = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_worker)(iso) for iso in iso_list
    )

    ensemble_df = pd.concat(dfs, ignore_index=True)

    # Write parquet
    table = pa.Table.from_pandas(ensemble_df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    print(
        f"Saved ensemble: {len(iso_list)} countries × {config.n_scenarios} scenarios → {out_path}"
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
    CONFIG_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "config")

    os.makedirs(ENSEMBLE_DIR_PATH, exist_ok=True)

    # Obtain file names
    with open(os.path.join(CONFIG_DIR_PATH, "arima_projections_config.yaml")) as stream:
        try:
            arima_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    # File names
    data_to_project = arima_config["data_to_project"]
    variable_projections_rules_file = arima_config["variable_projections_rules_file"] 
    run_id = arima_config["run_id"]
    n_scenarios = arima_config["n_scenarios"]
    end_year = arima_config["end_year"]
    
    
    training_df_log_transformed = pd.read_csv(
        os.path.join(TRAINING_DIR_PATH, data_to_project)
    )

    cfg = EnsembleConfig(
        end_year=end_year,
        n_scenarios=n_scenarios,
        arima_order=(1, 1, 1),
        auto_tune_arima=True,
        max_p=3,
        max_d=1,
        max_q=3,
        random_state=42,
        include_last_observed_row=True,  # includes last observed year per scenario
        projection_rules_path=os.path.join(CONFIG_DIR_PATH, variable_projections_rules_file),
    )

    out_path = os.path.join(ENSEMBLE_DIR_PATH, f"ensemble_arima_{run_id}.parquet")

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
    print("------- Config ------")
    print(arima_config)
