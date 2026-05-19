#!/usr/bin/env python
"""
Predict current Constraints from prior-decade Incentives.

This is a mediation-style diagnostic for the paper's conditional-null Incentives
claim. It is tailored to the already transformed training CSVs written under
ml/output/training, where some original variables have x_log_signed_* names.

Example:
    conda run -n etpe_env python ml/incentive_constraint_mediation_check.py \
        --input-csv ml/output/training/training_df_1773188058.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TRAINING_DIR = Path("ml/output/training")
DEFAULT_OUTPUT_DIR = Path("ml/output/mediation")
DEFAULT_TARGET_COL = "x_log_signed_con_edgar_ghg_mt"
DEFAULT_GROUP_COL = "iso_alpha_3"
DEFAULT_YEAR_COL = "year"

EMISSIONS_TOKENS = ("edgar", "ghg", "co2", "emission", "emissions", "methane", "nitrous")


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    full: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a mediation-style check: how much current Constraints variance "
            "is predicted by prior-decade Incentives?"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Training CSV to use. If omitted, uses the numerically latest "
            "training_df_*.csv in ml/output/training."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-col", default=DEFAULT_GROUP_COL)
    parser.add_argument("--year-col", default=DEFAULT_YEAR_COL)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=10,
        help="Number of prior years used to summarize incentive history.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=10,
        help="Minimum prior years required for each rolling incentive summary.",
    )
    parser.add_argument(
        "--holdout-years",
        type=int,
        default=3,
        help=(
            "Final years reserved for out-of-sample evaluation. Default is 3 "
            "because a full decade lookback leaves 2015-2022 in the common "
            "training exports."
        ),
    )
    parser.add_argument(
        "--year-control",
        choices=["linear", "fixed", "none"],
        default="linear",
        help=(
            "How to control for time. Default 'linear' is compatible with a "
            "future-year holdout. Use 'fixed' for year dummies in descriptive "
            "or non-forward-looking checks."
        ),
    )
    parser.add_argument(
        "--incentive-cols",
        default=None,
        help="Optional comma-separated override for incentive columns.",
    )
    parser.add_argument(
        "--constraint-cols",
        default=None,
        help="Optional comma-separated override for constraint outcome columns.",
    )
    parser.add_argument(
        "--include-emissions-constraints",
        action="store_true",
        help="Include emissions-like con_* variables as constraint outcomes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of largest standardized incentive coefficients to report.",
    )
    return parser.parse_args()


def parse_column_override(value: str | None) -> list[str] | None:
    if value is None:
        return None
    cols = [col.strip() for col in value.split(",") if col.strip()]
    return cols or None


def resolve_input_csv(input_csv: Path | None) -> Path:
    if input_csv is not None:
        return input_csv

    candidates = sorted(DEFAULT_TRAINING_DIR.glob("training_df_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No training_df_*.csv files found in {DEFAULT_TRAINING_DIR}")

    def numeric_suffix(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=numeric_suffix)


def is_base_incentive_col(col: str) -> bool:
    if "_lag_" in col or "_prior" in col:
        return False
    return col.startswith("inc_") or col.startswith("x_log_signed_inc_")


def is_constraint_col(col: str) -> bool:
    if "_lag_" in col or "_prior" in col:
        return False
    return col.startswith("con_") or col.startswith("x_log_signed_con_")


def is_emissions_like(col: str) -> bool:
    lowered = col.lower()
    return any(token in lowered for token in EMISSIONS_TOKENS)


def infer_incentive_cols(df: pd.DataFrame, override: list[str] | None) -> list[str]:
    if override is not None:
        return override
    return [col for col in df.columns if is_base_incentive_col(col)]


def infer_constraint_cols(
    df: pd.DataFrame,
    override: list[str] | None,
    target_col: str,
    include_emissions_constraints: bool,
) -> list[str]:
    if override is not None:
        return override

    cols = []
    for col in df.columns:
        if not is_constraint_col(col):
            continue
        if col == target_col:
            continue
        if not include_emissions_constraints and is_emissions_like(col):
            continue
        cols.append(col)
    return cols


def validate_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def is_binary_series(series: pd.Series) -> bool:
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    return set(values).issubset({0, 1, 0.0, 1.0})


def add_prior_decade_incentive_features(
    df: pd.DataFrame,
    incentive_cols: list[str],
    group_col: str,
    year_col: str,
    lookback_years: int,
    min_history: int,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    out = df.sort_values([group_col, year_col]).reset_index(drop=True).copy()
    feature_cols: list[str] = []
    feature_sources: dict[str, str] = {}

    grouped = out.groupby(group_col, sort=False)

    for col in incentive_cols:
        shifted = grouped[col].shift(1)
        rolling = shifted.groupby(out[group_col], sort=False).rolling(
            window=lookback_years,
            min_periods=min_history,
        )

        lag_col = f"{col}_prior{lookback_years}_latest"
        out[lag_col] = shifted
        feature_cols.append(lag_col)
        feature_sources[lag_col] = col

        if is_binary_series(out[col]):
            ever_active_col = f"{col}_prior{lookback_years}_ever_active"
            out[ever_active_col] = rolling.max().reset_index(level=0, drop=True)
            feature_cols.append(ever_active_col)
            feature_sources[ever_active_col] = col

            years_active_col = f"{col}_prior{lookback_years}_years_active"
            out[years_active_col] = rolling.sum().reset_index(level=0, drop=True)
            feature_cols.append(years_active_col)
            feature_sources[years_active_col] = col
        else:
            mean_col = f"{col}_prior{lookback_years}_mean"
            out[mean_col] = rolling.mean().reset_index(level=0, drop=True)
            feature_cols.append(mean_col)
            feature_sources[mean_col] = col

            max_col = f"{col}_prior{lookback_years}_max"
            out[max_col] = rolling.max().reset_index(level=0, drop=True)
            feature_cols.append(max_col)
            feature_sources[max_col] = col

            lag_start = grouped[col].shift(lookback_years)
            change_col = f"{col}_prior{lookback_years}_change"
            out[change_col] = shifted - lag_start
            feature_cols.append(change_col)
            feature_sources[change_col] = col

    return out, feature_cols, feature_sources


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def make_ridge_pipeline(
    fixed_effect_cols: list[str],
    numeric_feature_cols: list[str],
) -> Pipeline:
    transformers = [
        ("fixed_effects", make_ohe(), fixed_effect_cols),
    ]

    if numeric_feature_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_feature_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    alphas = np.logspace(-4, 4, 60)
    return Pipeline(
        [
            ("pre", preprocessor),
            ("ridge", RidgeCV(alphas=alphas)),
        ]
    )


def split_by_holdout_years(
    df: pd.DataFrame,
    year_col: str,
    holdout_years: int,
) -> SplitData:
    max_year = int(df[year_col].max())
    cutoff = max_year - holdout_years
    train = df.loc[df[year_col] <= cutoff].copy()
    test = df.loc[df[year_col] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Empty train/test split with max_year={max_year} and holdout_years={holdout_years}"
        )
    return SplitData(train=train, test=test, full=df)


def residual_variance_reduction(
    y_true: pd.Series | np.ndarray,
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
) -> float:
    baseline_sse = float(np.sum((np.asarray(y_true) - baseline_pred) ** 2))
    model_sse = float(np.sum((np.asarray(y_true) - model_pred) ** 2))
    if baseline_sse == 0:
        return np.nan
    return 1.0 - (model_sse / baseline_sse)


def classify_constraint(col: str) -> str:
    lowered = col.lower()
    if any(token in lowered for token in ("solar", "wind", "coal", "gas", "hydro", "nuclear", "bioenergy", "fossil_share", "renewables_share", "other_fossil")):
        return "energy_mix"
    if any(token in lowered for token in ("demand", "primary_energy", "energy_per_capita", "electricity_access", "net_elec_imports")):
        return "energy_demand_access"
    if any(token in lowered for token in ("forest", "agricultural_land")):
        return "land_use"
    if "ndgain" in lowered:
        return "adaptation_vulnerability"
    if is_emissions_like(col):
        return "emissions"
    return "other_constraints"


def top_incentive_coefficients(
    pipeline: Pipeline,
    feature_sources: dict[str, str],
    top_k: int,
) -> str:
    preprocessor = pipeline.named_steps["pre"]
    model = pipeline.named_steps["ridge"]

    try:
        names = preprocessor.get_feature_names_out()
    except AttributeError:
        return ""

    rows = []
    for name, coef in zip(names, model.coef_):
        if not name.startswith("numeric__"):
            continue
        feature = name.replace("numeric__", "", 1)
        if feature not in feature_sources:
            continue
        rows.append(
            {
                "feature": feature,
                "source_incentive": feature_sources.get(feature, ""),
                "coef": float(coef),
                "abs_coef": abs(float(coef)),
            }
        )

    rows = sorted(rows, key=lambda row: row["abs_coef"], reverse=True)[:top_k]
    return "; ".join(
        f"{row['feature']}={row['coef']:.4g}" for row in rows
    )


def fit_and_score_constraint(
    df: pd.DataFrame,
    constraint_col: str,
    group_col: str,
    year_col: str,
    holdout_years: int,
    incentive_feature_cols: list[str],
    feature_sources: dict[str, str],
    top_k: int,
    year_control: str,
) -> dict[str, object]:
    model_cols = [group_col, year_col, constraint_col] + incentive_feature_cols
    model_df = df[model_cols].dropna(subset=[constraint_col] + incentive_feature_cols).copy()
    split = split_by_holdout_years(model_df, year_col=year_col, holdout_years=holdout_years)

    if year_control == "fixed":
        fixed_effect_cols = [group_col, year_col]
        baseline_numeric_cols: list[str] = []
        incentive_numeric_cols = incentive_feature_cols
    elif year_control == "linear":
        fixed_effect_cols = [group_col]
        baseline_numeric_cols = [year_col]
        incentive_numeric_cols = [year_col] + incentive_feature_cols
    elif year_control == "none":
        fixed_effect_cols = [group_col]
        baseline_numeric_cols = []
        incentive_numeric_cols = incentive_feature_cols
    else:
        raise ValueError(f"Unknown year_control={year_control}")

    baseline = make_ridge_pipeline(
        fixed_effect_cols=fixed_effect_cols,
        numeric_feature_cols=baseline_numeric_cols,
    )
    incentives = make_ridge_pipeline(
        fixed_effect_cols=fixed_effect_cols,
        numeric_feature_cols=incentive_numeric_cols,
    )

    y_train = split.train[constraint_col]
    y_test = split.test[constraint_col]

    baseline.fit(split.train, y_train)
    incentives.fit(split.train, y_train)

    baseline_train_pred = baseline.predict(split.train)
    baseline_test_pred = baseline.predict(split.test)
    incentives_train_pred = incentives.predict(split.train)
    incentives_test_pred = incentives.predict(split.test)

    baseline_train_r2 = r2_score(y_train, baseline_train_pred)
    incentives_train_r2 = r2_score(y_train, incentives_train_pred)
    baseline_test_r2 = r2_score(y_test, baseline_test_pred)
    incentives_test_r2 = r2_score(y_test, incentives_test_pred)
    baseline_test_mae = mean_absolute_error(y_test, baseline_test_pred)
    incentives_test_mae = mean_absolute_error(y_test, incentives_test_pred)

    return {
        "constraint_variable": constraint_col,
        "constraint_family": classify_constraint(constraint_col),
        "n_obs": int(len(model_df)),
        "n_train": int(len(split.train)),
        "n_test": int(len(split.test)),
        "n_countries": int(model_df[group_col].nunique()),
        "year_min": int(model_df[year_col].min()),
        "year_max": int(model_df[year_col].max()),
        "train_year_min": int(split.train[year_col].min()),
        "train_year_max": int(split.train[year_col].max()),
        "test_year_min": int(split.test[year_col].min()),
        "test_year_max": int(split.test[year_col].max()),
        "n_prior_decade_incentive_features": int(len(incentive_feature_cols)),
        "year_control": year_control,
        "baseline_train_r2": baseline_train_r2,
        "incentives_train_r2": incentives_train_r2,
        "delta_train_r2": incentives_train_r2 - baseline_train_r2,
        "partial_train_r2_vs_fixed_effects": residual_variance_reduction(
            y_train,
            baseline_train_pred,
            incentives_train_pred,
        ),
        "baseline_test_r2": baseline_test_r2,
        "incentives_test_r2": incentives_test_r2,
        "delta_test_r2": incentives_test_r2 - baseline_test_r2,
        "partial_test_r2_vs_fixed_effects": residual_variance_reduction(
            y_test,
            baseline_test_pred,
            incentives_test_pred,
        ),
        "baseline_test_mae": baseline_test_mae,
        "incentives_test_mae": incentives_test_mae,
        "delta_test_mae": incentives_test_mae - baseline_test_mae,
        "top_incentive_predictors": top_incentive_coefficients(
            incentives,
            feature_sources=feature_sources,
            top_k=top_k,
        ),
    }


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rows.append(values)

    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(
        str(header).ljust(widths[idx]) for idx, header in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator] + body)


def write_outputs(
    results: pd.DataFrame,
    metadata: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"incentive_constraint_mediation_{timestamp}.csv"
    md_path = output_dir / f"incentive_constraint_mediation_{timestamp}.md"
    metadata_path = output_dir / f"incentive_constraint_mediation_{timestamp}_metadata.json"

    results.to_csv(csv_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    display_cols = [
        "constraint_variable",
        "constraint_family",
        "baseline_test_r2",
        "incentives_test_r2",
        "delta_test_r2",
        "partial_test_r2_vs_fixed_effects",
        "baseline_test_mae",
        "incentives_test_mae",
        "delta_test_mae",
        "top_incentive_predictors",
    ]
    markdown = dataframe_to_markdown(results[display_cols], floatfmt=".4f")
    md_path.write_text(markdown + "\n", encoding="utf-8")
    return csv_path, md_path, metadata_path


def main() -> None:
    args = parse_args()
    input_csv = resolve_input_csv(args.input_csv)

    df = pd.read_csv(input_csv)
    validate_columns(df, [args.group_col, args.year_col])

    incentive_override = parse_column_override(args.incentive_cols)
    constraint_override = parse_column_override(args.constraint_cols)
    incentive_cols = infer_incentive_cols(df, incentive_override)
    constraint_cols = infer_constraint_cols(
        df,
        constraint_override,
        target_col=args.target_col,
        include_emissions_constraints=args.include_emissions_constraints,
    )

    validate_columns(df, incentive_cols + constraint_cols)
    if not incentive_cols:
        raise ValueError("No incentive columns found. Expected inc_* or x_log_signed_inc_* columns.")
    if not constraint_cols:
        raise ValueError("No constraint columns found. Expected con_* or x_log_signed_con_* columns.")

    df_with_features, incentive_feature_cols, feature_sources = add_prior_decade_incentive_features(
        df,
        incentive_cols=incentive_cols,
        group_col=args.group_col,
        year_col=args.year_col,
        lookback_years=args.lookback_years,
        min_history=args.min_history,
    )

    rows = []
    for constraint_col in constraint_cols:
        rows.append(
            fit_and_score_constraint(
                df_with_features,
                constraint_col=constraint_col,
                group_col=args.group_col,
                year_col=args.year_col,
                holdout_years=args.holdout_years,
                incentive_feature_cols=incentive_feature_cols,
                feature_sources=feature_sources,
                top_k=args.top_k,
                year_control=args.year_control,
            )
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(["partial_test_r2_vs_fixed_effects", "delta_test_r2"], ascending=False)
        .reset_index(drop=True)
    )

    metadata = {
        "input_csv": str(input_csv),
        "n_rows_input": int(len(df)),
        "year_min_input": int(df[args.year_col].min()),
        "year_max_input": int(df[args.year_col].max()),
        "n_countries_input": int(df[args.group_col].nunique()),
        "group_col": args.group_col,
        "year_col": args.year_col,
        "target_col_excluded": args.target_col,
        "lookback_years": args.lookback_years,
        "min_history": args.min_history,
        "holdout_years": args.holdout_years,
        "year_control": args.year_control,
        "include_emissions_constraints": args.include_emissions_constraints,
        "incentive_cols": incentive_cols,
        "constraint_cols": constraint_cols,
        "prior_decade_incentive_feature_cols": incentive_feature_cols,
    }

    csv_path, md_path, metadata_path = write_outputs(results, metadata, args.output_dir)

    print("\nIncentive -> Constraint mediation-style check complete")
    print(f"Input: {input_csv}")
    print(f"Incentives: {len(incentive_cols)} base columns -> {len(incentive_feature_cols)} prior-decade features")
    print(f"Constraints evaluated: {len(constraint_cols)}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved metadata: {metadata_path}\n")

    display_cols = [
        "constraint_variable",
        "constraint_family",
        "baseline_test_r2",
        "incentives_test_r2",
        "delta_test_r2",
        "partial_test_r2_vs_fixed_effects",
        "delta_test_mae",
    ]
    print(results[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
