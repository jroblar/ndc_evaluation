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
from dataclasses import dataclass, field
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


def _safe_value_at_year(
    df_country: pd.DataFrame,
    var_name: str,
    year: int,
    default: float = np.nan,
) -> float:
    if var_name not in df_country.columns:
        return float(default)
    mask = pd.to_numeric(df_country["year"], errors="coerce").astype("Int64") == int(year)
    vals = pd.to_numeric(df_country.loc[mask, var_name], errors="coerce").dropna()
    if len(vals):
        return float(vals.iloc[-1])
    return _safe_last_valid(df_country[var_name], default=default)


def _as_year_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


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
    # Keep specific features fixed at a given historical year value for each country
    # e.g. {"emissions_anchor_2022": 2022}
    constant_feature_years: Dict[str, int] = field(default_factory=dict)
    # Deterministic year features e.g. {"years_since_2022": 2022}
    deterministic_year_features: Dict[str, int] = field(default_factory=dict)
    # Deterministic derived features as products e.g. {"trend_year_interaction": ["em_trend_5y","years_since_2022"]}
    derived_multiplicative_features: Dict[str, List[str]] = field(default_factory=dict)
    # Deterministic lag features.
    # Simple form: {"em_lag_1y": "x_log_signed_con_edgar_ghg_mt"}
    # Advanced form:
    # {"em_lag_1y": {"source_col":"x_log_signed_con_edgar_ghg_mt","mode":"trend_guided",
    #                "anchor_col":"emissions_anchor_2022","trend_col":"em_trend_5y",
    #                "years_since_col":"years_since_2022","blend":0.8}}
    lag_features: Dict[str, object] = field(default_factory=dict)
    # Rolling slope features.
    # Example:
    # {"em_trend_3y": {"source_col":"x_log_signed_con_edgar_ghg_mt","window":3,"min_periods":2,"shift":1}}
    rolling_slope_features: Dict[str, object] = field(default_factory=dict)
    # Rolling std features.
    # Example:
    # {"em_volatility_5y": {"source_col":"x_log_signed_con_edgar_ghg_mt","window":5,"min_periods":3,"shift":1}}
    rolling_std_features: Dict[str, object] = field(default_factory=dict)
    # Derived difference features.
    # Example: {"em_acceleration": ["em_trend_3y", "em_trend_5y"]}
    difference_features: Dict[str, List[str]] = field(default_factory=dict)
    # Per-feature spread scaling around last observed level (1.0=no change, <1 tighter, >1 wider)
    feature_innovation_scale: Dict[str, float] = field(default_factory=dict)


def _collect_non_simulated_feature_names(config: EnsembleConfig) -> set:
    return (
        set((config.constant_feature_years or {}).keys())
        | set((config.deterministic_year_features or {}).keys())
        | set((config.derived_multiplicative_features or {}).keys())
        | set((config.lag_features or {}).keys())
        | set((config.rolling_slope_features or {}).keys())
        | set((config.rolling_std_features or {}).keys())
        | set((config.difference_features or {}).keys())
    )


def _compute_slope_raw(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if int(mask.sum()) < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]
    return float(np.polyfit(x, y, 1)[0])


def _apply_feature_scale(series: pd.Series, anchor: float, scale: float) -> pd.Series:
    if not np.isfinite(anchor):
        return series
    if not np.isfinite(scale) or scale == 1.0:
        return series
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return anchor + (s - anchor) * scale


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
    constant_feature_years = config.constant_feature_years or {}
    deterministic_year_features = config.deterministic_year_features or {}
    derived_multiplicative_features = config.derived_multiplicative_features or {}
    lag_features = config.lag_features or {}
    rolling_slope_features = config.rolling_slope_features or {}
    rolling_std_features = config.rolling_std_features or {}
    difference_features = config.difference_features or {}
    derived_feature_names = set(derived_multiplicative_features.keys())
    deterministic_feature_names = set(constant_feature_years.keys()) | set(deterministic_year_features.keys())
    lag_feature_names = set(lag_features.keys())
    rolling_slope_feature_names = set(rolling_slope_features.keys())
    rolling_std_feature_names = set(rolling_std_features.keys())
    difference_feature_names = set(difference_features.keys())
    non_simulated_vars = (
        deterministic_feature_names
        | derived_feature_names
        | lag_feature_names
        | rolling_slope_feature_names
        | rolling_std_feature_names
        | difference_feature_names
    )
    simulated_vars = [v for v in arima_vars if v not in non_simulated_vars]

    # --- 1) ARIMA vars: simulate LEVELS ---
    sims_arima: Dict[str, np.ndarray] = {}
    for k, var in enumerate(simulated_vars):
        hist = pd.to_numeric(dfc[var], errors="coerce")
        hist.index = dfc["year"].astype(int).values

        # ensure continuous year span for the fit
        full_years = np.arange(int(dfc["year"].min()), last_year + 1)
        hist = _to_period_year_index(hist, full_years)
        hist_non_na = hist.dropna()

        sims = _fit_and_simulate_arima_levels(
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
        # Optional per-feature damping to keep trajectories near historical behavior.
        scale = float(config.feature_innovation_scale.get(var, 1.0))
        if np.isfinite(scale) and scale != 1.0:
            last_val = _safe_last_valid(hist_non_na, default=np.nan)
            if np.isfinite(last_val):
                sims = last_val + (sims - last_val) * scale
        sims_arima[var] = sims

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
            for var in simulated_vars:
                base_row[var] = _safe_last_valid(dfc[var], default=np.nan)
            rows.append(base_row)

        for t, y in enumerate(years_future):
            row = {
                "iso_alpha_3": iso,
                "future_id": fid,
                "year": int(y),
            }
            for var in simulated_vars:
                row[var] = float(sims_arima[var][s, t])
            rows.append(row)

    out = pd.DataFrame(rows)

    # --- 3) Deterministic feature injection ---
    for var, anchor_year in constant_feature_years.items():
        fixed_val = _safe_value_at_year(dfc, var_name=var, year=int(anchor_year), default=np.nan)
        out[var] = float(fixed_val) if np.isfinite(fixed_val) else np.nan

    for var, base_year in deterministic_year_features.items():
        out[var] = pd.to_numeric(out["year"], errors="coerce").astype(float) - float(base_year)

    for out_var, inputs in derived_multiplicative_features.items():
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Invalid derived_multiplicative_features spec for '{out_var}'. Expected [left_col, right_col]."
            )
        left_col, right_col = inputs
        if left_col not in out.columns or right_col not in out.columns:
            out[out_var] = np.nan
            continue
        out[out_var] = pd.to_numeric(out[left_col], errors="coerce") * pd.to_numeric(out[right_col], errors="coerce")

    # Lag/rolling features are computed per trajectory in temporal order.
    if lag_features or rolling_slope_features or rolling_std_features:
        out = out.sort_values(["future_id", "year"]).reset_index(drop=True)
        out_year_int = _as_year_int(out["year"])

        for lag_col, lag_spec in lag_features.items():
            if isinstance(lag_spec, str):
                source_col = lag_spec
                mode = "source_lag"
                anchor_col = None
                trend_col = None
                years_since_col = None
                blend = 1.0
            elif isinstance(lag_spec, dict):
                source_col = str(lag_spec.get("source_col", ""))
                mode = str(lag_spec.get("mode", "source_lag")).lower()
                anchor_col = lag_spec.get("anchor_col")
                trend_col = lag_spec.get("trend_col")
                years_since_col = lag_spec.get("years_since_col")
                blend = float(lag_spec.get("blend", 0.8))
            else:
                raise ValueError(f"Invalid lag_features spec for '{lag_col}'.")

            if source_col not in out.columns:
                out[lag_col] = np.nan
                continue

            source_series = pd.to_numeric(out[source_col], errors="coerce")
            lag_input = source_series.copy()

            # Optional trend-guided lag input: blend ARIMA source with anchor + trend * years_since.
            if mode == "trend_guided":
                if (
                    anchor_col in out.columns
                    and trend_col in out.columns
                    and years_since_col in out.columns
                ):
                    guided = (
                        pd.to_numeric(out[anchor_col], errors="coerce")
                        + pd.to_numeric(out[trend_col], errors="coerce")
                        * pd.to_numeric(out[years_since_col], errors="coerce")
                    )
                    b = float(np.clip(blend, 0.0, 1.0))
                    lag_input = (b * guided) + ((1.0 - b) * source_series)

            # shift on explicit series (avoids column-name coupling)
            lagged = lag_input.groupby(out["future_id"], sort=False).shift(1)

            # If last observed row is excluded, seed first projected year with last historical value.
            if not config.include_last_observed_row:
                seed_val = _safe_last_valid(dfc[source_col], default=np.nan) if source_col in dfc.columns else np.nan
                first_idx = out.groupby("future_id", sort=False).head(1).index
                if len(first_idx) > 0:
                    lagged.loc[first_idx] = seed_val
            else:
                # If baseline row exists (e.g., 2022), set its lag from previous historical year (e.g., 2021)
                first_rows = out.groupby("future_id", sort=False).head(1).copy()
                for idx, y0 in zip(first_rows.index, _as_year_int(first_rows["year"])):
                    if pd.isna(y0):
                        continue
                    prev_val = _safe_value_at_year(dfc, var_name=source_col, year=int(y0) - 1, default=np.nan)
                    if np.isfinite(prev_val):
                        lagged.loc[idx] = float(prev_val)

            out[lag_col] = pd.to_numeric(lagged, errors="coerce")
            hist_lag = pd.to_numeric(dfc[source_col], errors="coerce").shift(1)
            hist_anchor = _safe_last_valid(hist_lag, default=np.nan)
            scale = float(config.feature_innovation_scale.get(lag_col, 1.0))
            out[lag_col] = _apply_feature_scale(out[lag_col], anchor=hist_anchor, scale=scale)

        for out_col, slope_spec in rolling_slope_features.items():
            if not isinstance(slope_spec, dict):
                continue
            source_col = str(slope_spec.get("source_col", "x_log_signed_con_edgar_ghg_mt"))
            window = int(slope_spec.get("window", 3))
            min_periods = int(slope_spec.get("min_periods", max(2, window - 1)))
            shift = int(slope_spec.get("shift", 1))
            if source_col not in out.columns:
                continue
            out[out_col] = np.nan

            hist_source = pd.to_numeric(dfc[source_col], errors="coerce").astype(float)
            hist_rolled = (
                hist_source.rolling(window=window, min_periods=min_periods)
                .apply(_compute_slope_raw, raw=True)
            )
            if shift > 0:
                hist_rolled = hist_rolled.shift(shift)
            hist_anchor = _safe_last_valid(hist_rolled, default=np.nan)

            for fid, g in out.groupby("future_id", sort=False):
                idx = g.index
                source_future = pd.to_numeric(g[source_col], errors="coerce").astype(float).values
                first_future_year = int(pd.to_numeric(g["year"], errors="coerce").min())
                hist_prefix = (
                    pd.to_numeric(
                        dfc.loc[pd.to_numeric(dfc["year"], errors="coerce") < first_future_year, source_col],
                        errors="coerce",
                    )
                    .astype(float)
                    .values
                )
                combined = np.concatenate([hist_prefix, source_future]) if len(hist_prefix) else source_future
                rolled_combined = (
                    pd.Series(combined, dtype=float)
                    .rolling(window=window, min_periods=min_periods)
                    .apply(_compute_slope_raw, raw=True)
                )
                if shift > 0:
                    rolled_combined = rolled_combined.shift(shift)
                out.loc[idx, out_col] = rolled_combined.iloc[-len(idx):].values

            scale = float(config.feature_innovation_scale.get(out_col, 1.0))
            out[out_col] = _apply_feature_scale(out[out_col], anchor=hist_anchor, scale=scale)

        for out_col, std_spec in rolling_std_features.items():
            if not isinstance(std_spec, dict):
                continue
            source_col = str(std_spec.get("source_col", "x_log_signed_con_edgar_ghg_mt"))
            window = int(std_spec.get("window", 5))
            min_periods = int(std_spec.get("min_periods", max(2, window - 2)))
            shift = int(std_spec.get("shift", 1))
            if source_col not in out.columns:
                continue
            out[out_col] = np.nan

            hist_source = pd.to_numeric(dfc[source_col], errors="coerce").astype(float)
            hist_rolled = hist_source.rolling(window=window, min_periods=min_periods).std()
            if shift > 0:
                hist_rolled = hist_rolled.shift(shift)
            hist_anchor = _safe_last_valid(hist_rolled, default=np.nan)

            for fid, g in out.groupby("future_id", sort=False):
                idx = g.index
                source_future = pd.to_numeric(g[source_col], errors="coerce").astype(float).values
                first_future_year = int(pd.to_numeric(g["year"], errors="coerce").min())
                hist_prefix = (
                    pd.to_numeric(
                        dfc.loc[pd.to_numeric(dfc["year"], errors="coerce") < first_future_year, source_col],
                        errors="coerce",
                    )
                    .astype(float)
                    .values
                )
                combined = np.concatenate([hist_prefix, source_future]) if len(hist_prefix) else source_future
                rolled_combined = pd.Series(combined, dtype=float).rolling(window=window, min_periods=min_periods).std()
                if shift > 0:
                    rolled_combined = rolled_combined.shift(shift)
                out.loc[idx, out_col] = rolled_combined.iloc[-len(idx):].values

            scale = float(config.feature_innovation_scale.get(out_col, 1.0))
            out[out_col] = _apply_feature_scale(out[out_col], anchor=hist_anchor, scale=scale)

    for out_var, inputs in difference_features.items():
        if not isinstance(inputs, list) or len(inputs) != 2:
            continue
        left_col, right_col = inputs
        if left_col not in out.columns or right_col not in out.columns:
            continue
        out[out_var] = pd.to_numeric(out[left_col], errors="coerce") - pd.to_numeric(out[right_col], errors="coerce")
        scale = float(config.feature_innovation_scale.get(out_var, 1.0))
        baseline_vals = pd.to_numeric(
            out.loc[_as_year_int(out["year"]) == int(last_year), out_var],
            errors="coerce",
        )
        baseline_anchor = _safe_last_valid(baseline_vals, default=np.nan)
        out[out_var] = _apply_feature_scale(out[out_var], anchor=baseline_anchor, scale=scale)

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

    # Ensure deterministic/derived guidance features are retained in the output variable list.
    extra_vars = (
        list((config.constant_feature_years or {}).keys())
        + list((config.deterministic_year_features or {}).keys())
        + list((config.derived_multiplicative_features or {}).keys())
        + list((config.lag_features or {}).keys())
        + list((config.rolling_slope_features or {}).keys())
        + list((config.rolling_std_features or {}).keys())
        + list((config.difference_features or {}).keys())
    )
    for c in extra_vars:
        if c not in arima_vars:
            arima_vars.append(c)

    rulebook = load_projection_rules(config.projection_rules_path) if config.projection_rules_path else None

    non_simulated_vars = _collect_non_simulated_feature_names(config)
    simulated_vars = [v for v in arima_vars if v not in non_simulated_vars]

    required = {"iso_alpha_3", "year", *simulated_vars}
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
    with open(os.path.join(CONFIG_DIR_PATH, "arima_projections_with_lags_config.yaml")) as stream:
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
        constant_feature_years=arima_config.get("constant_feature_years", {}),
        deterministic_year_features=arima_config.get("deterministic_year_features", {}),
        derived_multiplicative_features=arima_config.get("derived_multiplicative_features", {}),
        lag_features=arima_config.get("lag_features", {}),
        rolling_slope_features=arima_config.get("rolling_slope_features", {}),
        rolling_std_features=arima_config.get("rolling_std_features", {}),
        difference_features=arima_config.get("difference_features", {}),
        feature_innovation_scale=arima_config.get("feature_innovation_scale", {}),
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
