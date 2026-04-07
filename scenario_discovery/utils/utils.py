import ast
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from threading import Lock
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    PYMOO_OK = True
except Exception:
    PYMOO_OK = False


DEFAULT_NDC_REFERENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed_data" / "ndc_reference.csv"
DEFAULT_INCOME_LEVEL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw_data" / "income_level_data" / "2025_wb_income_level_class.xlsx"
)
NDC_UNCONDITIONAL_COL = "ndc_unconditional"


def slugify_report_col(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_income_level_map(
    income_level_path: str | Path = DEFAULT_INCOME_LEVEL_PATH,
    iso_col: str = "iso_alpha_3",
    income_group_col: str = "income_group",
) -> dict[str, str]:
    income_level_path = Path(income_level_path)
    if not income_level_path.exists():
        raise FileNotFoundError(f"Income level reference file not found: {income_level_path}")

    try:
        income_df = pd.read_excel(income_level_path)
    except ImportError:
        import re
        import zipfile
        import xml.etree.ElementTree as ET

        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with zipfile.ZipFile(income_level_path) as workbook:
            shared_strings = []
            shared_strings_xml = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for string_item in shared_strings_xml.findall("a:si", ns):
                shared_strings.append(
                    "".join(
                        text_node.text or ""
                        for text_node in string_item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    )
                )

            sheet_xml = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
            rows = []
            for row in sheet_xml.findall(".//a:sheetData/a:row", ns):
                values = []
                for cell in row.findall("a:c", ns):
                    cell_ref = cell.get("r", "")
                    col_letters = re.sub(r"\\d", "", cell_ref)
                    col_idx = 0
                    for char in col_letters:
                        col_idx = col_idx * 26 + ord(char.upper()) - ord("A") + 1
                    while len(values) < max(col_idx - 1, 0):
                        values.append(None)

                    value_node = cell.find("a:v", ns)
                    if value_node is None:
                        values.append(None)
                    elif cell.get("t") == "s":
                        values.append(shared_strings[int(value_node.text)])
                    else:
                        values.append(value_node.text)
                rows.append(values)
        income_df = pd.DataFrame(rows[1:], columns=rows[0])

    missing_cols = [col for col in [iso_col, income_group_col] if col not in income_df.columns]
    if missing_cols:
        raise ValueError(f"Income level reference file is missing required columns: {missing_cols}")

    income_df = income_df[[iso_col, income_group_col]].dropna(subset=[iso_col, income_group_col]).copy()
    income_df[iso_col] = income_df[iso_col].astype(str).str.upper()
    income_df[income_group_col] = income_df[income_group_col].astype(str)
    return dict(zip(income_df[iso_col], income_df[income_group_col]))


def parse_feature_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [feature for feature in value.split("|") if feature]
    if isinstance(value, (list, tuple, set)):
        return [str(feature) for feature in value if str(feature)]
    return []


def add_income_group_country_fields(
    row: dict[str, Any],
    country_groups: dict[str, list[str]],
    income_groups: list[str],
) -> dict[str, Any]:
    for income_group in income_groups:
        countries = country_groups.get(income_group, [])
        row[f"{slugify_report_col(income_group)}_countries"] = "|".join(sorted(countries))
    row["unclassified_countries"] = "|".join(sorted(country_groups.get("unclassified", [])))
    return row


def build_top_variable_frequency_report(
    summary_rows: list[dict[str, Any]],
    income_level_path: str | Path = DEFAULT_INCOME_LEVEL_PATH,
) -> pd.DataFrame:
    income_level_by_country = load_income_level_map(income_level_path)
    income_groups = sorted(set(income_level_by_country.values()))
    feature_counter: Counter[str] = Counter()
    feature_income_countries: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for row in summary_rows:
        if row["status"] != "success":
            continue
        country = str(row["country"]).upper()
        income_group = income_level_by_country.get(country, "unclassified")
        for feature in parse_feature_list(row["selected_top_features"]):
            feature_counter[feature] += 1
            feature_income_countries[feature][income_group].append(country)

    report_rows = []
    for feature, count in feature_counter.most_common():
        report_row = {"feature": feature, "count": count}
        report_rows.append(add_income_group_country_fields(report_row, feature_income_countries[feature], income_groups))
    return pd.DataFrame(report_rows)


def build_feature_combination_frequency_report(
    summary_rows: list[dict[str, Any]],
    income_level_path: str | Path = DEFAULT_INCOME_LEVEL_PATH,
) -> pd.DataFrame:
    income_level_by_country = load_income_level_map(income_level_path)
    income_groups = sorted(set(income_level_by_country.values()))
    combination_counter: Counter[str] = Counter()
    combination_income_countries: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for row in summary_rows:
        if row["status"] != "success":
            continue
        country = str(row["country"]).upper()
        income_group = income_level_by_country.get(country, "unclassified")
        selected_top_features = sorted(parse_feature_list(row["selected_top_features"]))
        if not selected_top_features:
            continue
        feature_combination = "|".join(selected_top_features)
        combination_counter[feature_combination] += 1
        combination_income_countries[feature_combination][income_group].append(country)

    report_rows = []
    for feature_combination, count in combination_counter.most_common():
        report_row = {"feature_combination": feature_combination, "count": count}
        report_rows.append(
            add_income_group_country_fields(
                report_row,
                combination_income_countries[feature_combination],
                income_groups,
            )
        )
    return pd.DataFrame(report_rows)


@dataclass
class RandomForestDiscoveryResult:
    rf_models: dict[str, Pipeline]
    training_summary: pd.DataFrame
    feature_importance: pd.DataFrame


class VulnerabilityAnalyzer:
    @staticmethod
    def compute_vulnerability_indicator(
        df: pd.DataFrame,
        year1: str,
        year2: str,
        value_col: str,
        id_cols: tuple[str, str] = ("future_id", "iso_alpha_3"),
        auto_threshold: bool = False,
    ) -> pd.DataFrame:
        df_pivot = df.pivot(index=list(id_cols), columns="year", values=value_col).reset_index()
        df_pivot.columns.name = None
        df_pivot.columns = df_pivot.columns.astype(str)

        if auto_threshold:
            year2_values = pd.to_numeric(df_pivot[year2], errors="coerce")
            valid_values = np.sort(year2_values.dropna().to_numpy())
            if len(valid_values) == 0:
                raise ValueError(f"Column '{year2}' has no valid numeric values for threshold selection.")

            # Pick the split point that minimizes class imbalance.
            # Using a midpoint between adjacent sorted values avoids thresholding exactly on a repeated value.
            best_threshold = float(valid_values[0])
            best_imbalance = len(valid_values)

            if len(valid_values) == 1:
                best_threshold = float(valid_values[0])
            else:
                candidate_thresholds = []
                for idx in range(len(valid_values) - 1):
                    left = float(valid_values[idx])
                    right = float(valid_values[idx + 1])
                    candidate_thresholds.append((left + right) / 2.0)

                # Include the median as a fallback candidate as well.
                candidate_thresholds.append(float(np.quantile(valid_values, 0.5)))

                for threshold in candidate_thresholds:
                    positive_count = int((year2_values > threshold).sum())
                    negative_count = int((year2_values <= threshold).sum())
                    imbalance = abs(positive_count - negative_count)
                    if imbalance < best_imbalance:
                        best_imbalance = imbalance
                        best_threshold = float(threshold)

            df_pivot["vulnerability_threshold"] = best_threshold
            df_pivot["vulnerability_indicator"] = (year2_values > best_threshold).astype(int)
            return df_pivot[[*id_cols, year1, year2, "vulnerability_threshold", "vulnerability_indicator"]]

        df_pivot["vulnerability_indicator"] = (df_pivot[year2] > df_pivot[year1]).astype(int)
        return df_pivot[[*id_cols, year1, year2, "vulnerability_indicator"]]

    @staticmethod
    def compute_emissions_change(
        df: pd.DataFrame,
        year1: str,
        year2: str,
        value_col: str,
        id_cols: tuple[str, str] = ("future_id", "iso_alpha_3"),
        auto_threshold: bool = False,
        ndc_reference_path: str | Path = DEFAULT_NDC_REFERENCE_PATH,
        ndc_iso_col: str = "ISO",
        ndc_value_col: str = "Unconditional",
        ndc_output_col: str = NDC_UNCONDITIONAL_COL,
    ) -> pd.DataFrame:
        emissions_change_df = VulnerabilityAnalyzer.compute_vulnerability_indicator(
            df=df,
            year1=year1,
            year2=year2,
            value_col=value_col,
            id_cols=id_cols,
            auto_threshold=auto_threshold,
        )
        ndc_reference_path = Path(ndc_reference_path)
        if not ndc_reference_path.exists():
            raise FileNotFoundError(f"NDC reference file not found: {ndc_reference_path}")

        ndc_reference = pd.read_csv(ndc_reference_path)
        missing_cols = [col for col in [ndc_iso_col, ndc_value_col] if col not in ndc_reference.columns]
        if missing_cols:
            raise ValueError(f"NDC reference file is missing required columns: {missing_cols}")

        country_col = id_cols[1]
        ndc_targets = (
            ndc_reference[[ndc_iso_col, ndc_value_col]]
            .drop_duplicates(subset=[ndc_iso_col])
            .rename(columns={ndc_iso_col: country_col, ndc_value_col: ndc_output_col})
        )
        ndc_targets[country_col] = ndc_targets[country_col].astype(str).str.upper()
        ndc_targets[ndc_output_col] = pd.to_numeric(ndc_targets[ndc_output_col], errors="coerce")

        out = emissions_change_df.copy()
        out[country_col] = out[country_col].astype(str).str.upper()
        return out.merge(ndc_targets, on=country_col, how="left")

    @staticmethod
    def merge_ensemble_with_vulnerability(
        ensemble_agg_df: pd.DataFrame,
        vulnerability_df: pd.DataFrame,
        on_cols: list[str] | tuple[str, ...] = ("future_id",),
    ) -> pd.DataFrame:
        merged_df = pd.merge(ensemble_agg_df, vulnerability_df, on=list(on_cols), how="left")
        for col in vulnerability_df.columns:
            if "indicator" in col:
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype("Int64")
        return merged_df

    @staticmethod
    def plot_vulnerability_counts(
        df: pd.DataFrame,
        vuln_col: str = "vulnerability_indicator",
        figsize: tuple[int, int] = (6, 4),
        palette: list[str] | dict[int, str] | None = None,
        xtick_labels: list[str] | None = None,
        xlabel: str = "",
        ylabel: str = "Count",
        title: str = "Vulnerability Indicator Counts",
        annotate: bool = True,
        order: list[int] | None = None,
        show: bool = True,
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if vuln_col not in df.columns:
            raise ValueError(f"Column '{vuln_col}' not found in dataframe.")

        palette = palette or ["tab:blue", "tab:orange"]
        order = order or sorted(df[vuln_col].dropna().unique().tolist())

        if xtick_labels is None:
            xtick_labels = (
                ["Not vulnerable (0)", "Vulnerable (1)"]
                if set(order) == {0, 1}
                else [str(v) for v in order]
            )

        plt.figure(figsize=figsize)
        ax_local = plt.gca()
        sns.countplot(x=vuln_col, data=df, palette=palette, order=order, ax=ax_local)
        ax_local.set_xticks(range(len(order)))
        ax_local.set_xticklabels(xtick_labels)
        ax_local.set_ylabel(ylabel)
        ax_local.set_xlabel(xlabel)
        ax_local.set_title(title)

        if annotate:
            for patch in ax_local.patches:
                height = patch.get_height()
                ax_local.annotate(
                    f"{int(height)}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        if show:
            plt.show()
        return ax_local

    @staticmethod
    def plot_xy_by_vulnerability(
        df: pd.DataFrame,
        x: str | int,
        y: str | int,
        vuln_col: str = "vulnerability_indicator",
        labels: dict[int, str] | None = None,
        colors: dict[int, str] | None = None,
        markers: dict[int, str] | None = None,
        s: int = 60,
        alpha: float = 0.8,
        title: str | None = None,
    ):
        import matplotlib.pyplot as plt

        x_col = df.columns[x] if isinstance(x, int) else x
        y_col = df.columns[y] if isinstance(y, int) else y
        data = df[[x_col, y_col, vuln_col]].dropna()

        uniq = np.sort(data[vuln_col].unique())
        if not set(uniq).issubset({0, 1}):
            raise ValueError(f"'{vuln_col}' must be binary {{0,1}}; got {uniq}.")

        labels = labels or {0: "Not vulnerable (0)", 1: "Vulnerable (1)"}
        colors = colors or {0: "tab:blue", 1: "tab:orange"}
        markers = markers or {0: "o", 1: "X"}

        if title is None:
            title = f"Vulnerability Indicator"

        plt.figure(figsize=(8, 6))
        for v in [0, 1]:
            subset = data[data[vuln_col] == v]
            if subset.empty:
                continue
            plt.scatter(
                subset[x_col],
                subset[y_col],
                s=s,
                alpha=alpha,
                edgecolor="k",
                color=colors.get(v, "gray"),
                marker=markers.get(v, "o"),
                label=labels.get(v, str(v)),
            )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend(title=vuln_col)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_future_distribution_with_baseline(
        df_pivot: pd.DataFrame,
        future_col: str = "2030",
        baseline_col: str = "2022",
        reference_cols: tuple[str, ...] = ("2022", NDC_UNCONDITIONAL_COL),
        bins: int = 30,
        figsize: tuple[int, int] = (8, 4),
        color: str = "tab:blue",
        baseline_color: str = "tab:red",
        reference_colors: dict[str, str] | None = None,
        kde: bool = False,
        xlabel: str | None = None,
        ylabel: str = "Count",
        title: str | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
        close: bool = False,
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if future_col not in df_pivot.columns:
            raise ValueError(f"Column '{future_col}' not found in dataframe.")
        if baseline_col not in df_pivot.columns:
            raise ValueError(f"Column '{baseline_col}' not found in dataframe.")

        future_values = pd.to_numeric(df_pivot[future_col], errors="coerce").dropna()

        if future_values.empty:
            raise ValueError(f"Column '{future_col}' has no valid numeric values to plot.")

        def get_unique_reference_value(col: str, required: bool = True) -> float | None:
            values = pd.to_numeric(df_pivot[col], errors="coerce").dropna().unique()
            if len(values) == 0:
                if required:
                    raise ValueError(f"Column '{col}' has no valid numeric values.")
                return None
            if len(values) > 1:
                raise ValueError(
                    f"Column '{col}' contains multiple values; expected a single reference value, "
                    f"got {len(values)} unique values."
                )
            return float(values[0])

        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(future_values, bins=bins, color=color, kde=kde, ax=ax, edgecolor="w")
        baseline_value = get_unique_reference_value(baseline_col)
        ax.axvline(
            baseline_value,
            color=baseline_color,
            linestyle="--",
            linewidth=2,
            label=f"{baseline_col} = {baseline_value:.3g}",
        )
        reference_colors = reference_colors or {
            "2022": "tab:green",
            NDC_UNCONDITIONAL_COL: "tab:purple",
        }
        plotted_reference_cols = {baseline_col}
        for reference_col in reference_cols:
            if reference_col in plotted_reference_cols or reference_col not in df_pivot.columns:
                continue
            reference_value = get_unique_reference_value(reference_col, required=False)
            if reference_value is None:
                continue
            ax.axvline(
                reference_value,
                color=reference_colors.get(reference_col, "black"),
                linestyle="--",
                linewidth=2,
                label=f"{reference_col} = {reference_value:.3g}",
            )
            plotted_reference_cols.add(reference_col)

        ax.set_xlabel(xlabel or future_col)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Distribution of {future_col} with {baseline_col} reference")
        ax.legend(loc="best")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        if close:
            plt.close(fig)
        return ax


class RandomForestDiscovery:
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 500,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced_subsample",
        n_jobs: int = -1,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def _build_model(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        class_weight=self.class_weight,
                        min_samples_leaf=self.min_samples_leaf,
                    ),
                ),
            ]
        )

    def fit(
        self,
        merged_df: pd.DataFrame,
        target_col: str = "vulnerability_indicator",
        non_modeling_cols: list[str] | None = None,
    ) -> RandomForestDiscoveryResult:
        non_modeling_cols = non_modeling_cols or [
            "future_id",
            "iso_alpha_3",
            "year_start",
            "year_end",
            "n_years",
            "2022",
            "2030",
            NDC_UNCONDITIONAL_COL,
        ]

        feature_cols = [col for col in merged_df.columns if col not in non_modeling_cols + [target_col]]
        X_all = merged_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(merged_df[target_col], errors="coerce")
        valid_mask = y.notna()

        X_target = X_all.loc[valid_mask].copy()
        y_target = y.loc[valid_mask].astype(int)
        all_nan_feature_cols = X_target.columns[X_target.notna().sum(axis=0) == 0].tolist()
        if all_nan_feature_cols:
            X_target = X_target.drop(columns=all_nan_feature_cols)
            feature_cols = [col for col in feature_cols if col not in all_nan_feature_cols]

        rf_models: dict[str, Pipeline] = {}
        rf_training_rows: list[dict[str, Any]] = []
        feature_importance_rows: list[dict[str, Any]] = []

        if X_target.empty or y_target.nunique() < 2:
            rf_training_rows.append(
                {
                    "target_col": target_col,
                    "n_rows": int(len(y_target)),
                    "class_0": int((y_target == 0).sum()),
                    "class_1": int((y_target == 1).sum()),
                    "n_features": int(len(feature_cols)),
                    "n_all_nan_features_dropped": int(len(all_nan_feature_cols)),
                    "all_nan_features_dropped": "|".join(all_nan_feature_cols),
                    "status": "skipped_insufficient_rows_or_classes",
                }
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_target,
                y_target,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_target,
            )

            eval_model = self._build_model()
            eval_model.fit(X_train, y_train)
            y_pred = eval_model.predict(X_test)
            y_proba = eval_model.predict_proba(X_test)[:, 1]

            final_model = self._build_model()
            final_model.fit(X_target, y_target)
            rf_models[target_col] = final_model

            rf = final_model.named_steps["rf"]
            importances = rf.feature_importances_
            baseline_accuracy = float(max((y_test == 0).mean(), (y_test == 1).mean()))
            roc_auc = float(roc_auc_score(y_test, y_proba)) if y_test.nunique() == 2 else np.nan

            rf_training_rows.append(
                {
                    "target_col": target_col,
                    "n_rows": int(len(y_target)),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "class_0": int((y_target == 0).sum()),
                    "class_1": int((y_target == 1).sum()),
                    "status": "trained",
                    "n_features": int(len(feature_cols)),
                    "n_all_nan_features_dropped": int(len(all_nan_feature_cols)),
                    "all_nan_features_dropped": "|".join(all_nan_feature_cols),
                    "baseline_accuracy": baseline_accuracy,
                    "test_accuracy": float(accuracy_score(y_test, y_pred)),
                    "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "test_roc_auc": roc_auc,
                    "mean_importance": float(importances.mean()) if importances.size else np.nan,
                    "median_importance": float(np.median(importances)) if importances.size else np.nan,
                    "max_importance": float(importances.max()) if importances.size else np.nan,
                }
            )

            for feature_name, importance in zip(feature_cols, importances):
                feature_importance_rows.append(
                    {
                        "target_col": target_col,
                        "feature": feature_name,
                        "importance": float(importance),
                    }
                )

        training_summary = pd.DataFrame(rf_training_rows).sort_values("target_col").reset_index(drop=True)
        feature_importance = pd.DataFrame(feature_importance_rows)
        return RandomForestDiscoveryResult(
            rf_models=rf_models,
            training_summary=training_summary,
            feature_importance=feature_importance,
        )

    @staticmethod
    def select_top_features(
        feature_importance_df: pd.DataFrame,
        feature_standard_name_col: str = "feature_standard_name",
        top_k: int = 2,
    ) -> list[str]:
        selected_features: list[str] = []
        for _, row in feature_importance_df.iterrows():
            feature_name = row[feature_standard_name_col]
            if feature_name not in selected_features:
                selected_features.append(feature_name)
            if len(selected_features) >= top_k:
                break
        return selected_features
    @staticmethod
    def plot_feature_importance(feature_importance_df: pd.DataFrame, target_col: str = "vulnerability_indicator", top_n: int = 10):
        top_features_df = (
            feature_importance_df[feature_importance_df["target_col"] == target_col]
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        #plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=top_features_df, palette="viridis")
        plt.title(f"Top {top_n} Feature Importances for Predicting {target_col}")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance_histogram(
        df,
        column,
        bins=30,
        figsize=(8, 4),
        color="tab:blue",
        kde=False,
        log_scale=False,
        xlabel=None,
        ylabel="Count",
        title=None,
        annotate_stats=True,
        dropna=True,
        rug=False,
        show=True,
    ):
        """
        Plot a simple histogram for a dataframe column.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the column to plot.
        column : str
            Column name to plot.
        bins : int or sequence
            Number of bins or bin edges for the histogram.
        figsize : tuple
            Figure size.
        color : str
            Bar color.
        kde : bool
            Whether to overlay a KDE (uses seaborn).
        log_scale : bool
            Whether to use a log scale on the x axis.
        xlabel, ylabel, title : str or None
            Axis and title labels.
        annotate_stats : bool
            Annotate mean, median and non-null count on the plot.
        dropna : bool
            Drop NA values before plotting.
        rug : bool
            Add a rug plot (seaborn).
        show : bool
            Whether to call plt.show() before returning the axis.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object of the plot.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        data = df[column]
        if dropna:
            data = data.dropna()

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # use seaborn histogram for consistency; fall back to plt.hist if needed
        try:
            sns.histplot(data, bins=bins, color=color, kde=kde, ax=ax, edgecolor="w", stat="count", rug=rug)
        except Exception:
            ax.hist(data, bins=bins, color=color, edgecolor="w")

        if log_scale:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel or column)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Histogram of {column}")

        if annotate_stats and len(data) > 0:
            mean = float(data.mean())
            median = float(data.median())
            std = float(data.std())
            n = int(data.count())

            # vertical lines
            ax.axvline(mean, color="k", linestyle="--", linewidth=1, label=f"mean={mean:.3g}")
            ax.axvline(median, color="gray", linestyle=":", linewidth=1, label=f"median={median:.3g}")

            # text annotation in upper right
            text = f"n={n}\nmean={mean:.3g}\nmedian={median:.3g}\nstd={std:.3g}"
            ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), fontsize=9)

            ax.legend(loc="upper left", frameon=False)

        plt.tight_layout()
        if show:
            plt.show()

        return ax


class TrajectoryAggregator:
    VALID_AGGREGATION_MODES = {"full", "pct_change"}

    def __init__(
        self,
        rules_path: str | Path,
        aggregation_mode: str = "full",
        start_year: int | str = 2022,
        end_year: int | str = 2030,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.category_map, self.prefix_rules, self.default_category = self.load_projection_rulebook(self.rules_path)
        self.aggregation_mode = self._validate_aggregation_mode(aggregation_mode)
        self.start_year = int(start_year)
        self.end_year = int(end_year)

    @classmethod
    def _validate_aggregation_mode(cls, aggregation_mode: str) -> str:
        if aggregation_mode not in cls.VALID_AGGREGATION_MODES:
            valid_modes = ", ".join(sorted(cls.VALID_AGGREGATION_MODES))
            raise ValueError(f"Unsupported aggregation_mode '{aggregation_mode}'. Expected one of: {valid_modes}.")
        return aggregation_mode

    @staticmethod
    def load_projection_rulebook(rules_path: str | Path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rulebook = json.load(f)

        category_map: dict[str, str] = {}
        for category, columns in rulebook.get("categories", {}).items():
            for col in columns:
                category_map[col] = category

        for col, category in rulebook.get("overrides", {}).items():
            category_map[col] = category

        prefix_rules = rulebook.get("prefix_rules", [])
        default_category = rulebook.get("default_category", "unconstrained")
        return category_map, prefix_rules, default_category

    @staticmethod
    def resolve_category(
        col: str,
        category_map: dict[str, str],
        prefix_rules: list[dict[str, str]],
        default_category: str,
    ) -> str:
        if col in category_map:
            return category_map[col]
        for rule in prefix_rules:
            if col.startswith(rule["prefix"]):
                return rule["category"]
        return default_category

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        if len(values) <= 1:
            return np.nan
        return float(np.std(values, ddof=1))

    @staticmethod
    def _safe_slope(years: np.ndarray, values: np.ndarray) -> float:
        if len(values) <= 1:
            return np.nan
        return float(np.polyfit(years.astype(float), values.astype(float), 1)[0])

    def summarize_trajectory(self, group: pd.DataFrame, value_cols: list[str]) -> dict[str, Any]:
        group = group.sort_values("year")
        row: dict[str, Any] = {
            "future_id": group["future_id"].iloc[0],
            "iso_alpha_3": group["iso_alpha_3"].iloc[0],
            "year_start": int(group["year"].min()),
            "year_end": int(group["year"].max()),
            "n_years": int(group["year"].nunique()),
        }

        for col in value_cols:
            category = self.resolve_category(col, self.category_map, self.prefix_rules, self.default_category)
            valid = group[["year", col]].dropna()
            if valid.empty:
                continue

            years = valid["year"].to_numpy()
            values = valid[col].astype(float).to_numpy()
            prefix = f"{col}__"

            if category == "binary":
                row[prefix + "mean"] = float(values.mean())
                row[prefix + "last"] = float(values[-1])
                row[prefix + "max"] = float(values.max())
                row[prefix + "switches"] = float(np.abs(np.diff(values)).sum()) if len(values) > 1 else 0.0
            elif category == "cumulative_binary":
                row[prefix + "last"] = float(values[-1])
                row[prefix + "max"] = float(values.max())
            elif category == "count":
                row[prefix + "sum"] = float(values.sum())
                row[prefix + "mean"] = float(values.mean())
                row[prefix + "std"] = self._safe_std(values)
                row[prefix + "max"] = float(values.max())
                row[prefix + "last"] = float(values[-1])
                row[prefix + "delta"] = float(values[-1] - values[0])
                row[prefix + "slope"] = self._safe_slope(years, values)
            elif category == "cumulative_count":
                row[prefix + "last"] = float(values[-1])
                row[prefix + "max"] = float(values.max())
                row[prefix + "delta"] = float(values[-1] - values[0])
                row[prefix + "slope"] = self._safe_slope(years, values)
            else:
                row[prefix + "mean"] = float(values.mean())
                row[prefix + "std"] = self._safe_std(values)
                row[prefix + "min"] = float(values.min())
                row[prefix + "max"] = float(values.max())
                row[prefix + "last"] = float(values[-1])
                row[prefix + "delta"] = float(values[-1] - values[0])
                row[prefix + "slope"] = self._safe_slope(years, values)

        return row

    def summarize_pct_change(
        self,
        group: pd.DataFrame,
        value_cols: list[str],
        group_col: str = "future_id",
        keep_country_col: str = "iso_alpha_3",
        year_col: str = "year",
    ) -> dict[str, Any]:
        group = group.sort_values(year_col)
        numeric_years = pd.to_numeric(group[year_col], errors="coerce")
        row: dict[str, Any] = {
            group_col: group[group_col].iloc[0],
            keep_country_col: group[keep_country_col].iloc[0],
            "year_start": self.start_year,
            "year_end": self.end_year,
            "n_years": int(numeric_years.nunique()),
        }

        start_rows = group[numeric_years == self.start_year]
        end_rows = group[numeric_years == self.end_year]
        if start_rows.empty or end_rows.empty:
            return row

        start_row = start_rows.iloc[-1]
        end_row = end_rows.iloc[-1]

        for col in value_cols:
            category = self.resolve_category(col, self.category_map, self.prefix_rules, self.default_category)
            start_value = pd.to_numeric(start_row.get(col), errors="coerce")
            end_value = pd.to_numeric(end_row.get(col), errors="coerce")
            if pd.isna(start_value) or pd.isna(end_value):
                continue

            start_value = float(start_value)
            end_value = float(end_value)
            prefix = f"{col}__"
            row[prefix + "start"] = start_value
            row[prefix + "last"] = end_value
            row[prefix + "delta"] = end_value - start_value

            if category in {"binary", "cumulative_binary"}:
                row[prefix + "change"] = end_value - start_value
            else:
                row[prefix + "pct_change"] = (
                    ((end_value - start_value) / abs(start_value)) * 100.0 if start_value != 0 else np.nan
                )

        return row

    def aggregate(
        self,
        ensemble_df: pd.DataFrame,
        group_col: str = "future_id",
        keep_country_col: str = "iso_alpha_3",
        year_col: str = "year",
        aggregation_mode: str | None = None,
    ) -> pd.DataFrame:
        id_cols = {group_col, keep_country_col, year_col}
        value_cols = [col for col in ensemble_df.columns if col not in id_cols]
        mode = self._validate_aggregation_mode(aggregation_mode or self.aggregation_mode)
        aggregated_rows = []
        for _, group in ensemble_df.groupby(group_col, sort=False):
            if mode == "pct_change":
                aggregated_rows.append(
                    self.summarize_pct_change(
                        group,
                        value_cols,
                        group_col=group_col,
                        keep_country_col=keep_country_col,
                        year_col=year_col,
                    )
                )
            else:
                aggregated_rows.append(self.summarize_trajectory(group, value_cols))
        return pd.DataFrame(aggregated_rows)


class ScenarioDiscoveryOptimizer:
    def __init__(
        self,
        lower: float = 0.1,
        upper: float = 0.9,
        popsize: int = 200,
        generations: int = 200,
        seed: int = 55555,
        target_density: float = 0.85,
        min_density: float = 0.60,
        density_step: float = 0.05,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.popsize = popsize
        self.generations = generations
        self.seed = seed
        self.target_density = target_density
        self.min_density = min_density
        self.density_step = density_step

        if not 0 <= self.min_density <= self.target_density <= 1:
            raise ValueError("Expected 0 <= min_density <= target_density <= 1.")
        if self.density_step <= 0:
            raise ValueError("density_step must be positive.")

    @staticmethod
    def box_stats_multi(
        drivers_thresholds: float | list[float],
        pt: pd.DataFrame,
        vulnerability_indicator: str,
        drivers: list[str],
        cmp: list[str] | str | None = None,
    ) -> tuple[float, float]:
        if np.isscalar(drivers_thresholds):
            drivers_thresholds = [float(drivers_thresholds)] * len(drivers)
        if cmp is None or isinstance(cmp, str):
            cmp = [cmp if isinstance(cmp, str) else "<="] * len(drivers)

        cutoffs = []
        for col, prob in zip(drivers, drivers_thresholds):
            cutoffs.append(np.quantile(pt[col].dropna().to_numpy(), float(prob)))

        meets = pd.Series(True, index=pt.index)
        for col, cutoff, op in zip(drivers, cutoffs, cmp):
            x = pt[col]
            if op == "<":
                meets &= x < cutoff
            elif op == "<=":
                meets &= x <= cutoff
            elif op == ">":
                meets &= x > cutoff
            elif op == ">=":
                meets &= x >= cutoff
            else:
                raise ValueError(f"Unsupported comparator: {op}")
        meets = meets.fillna(False)

        v = pt[vulnerability_indicator].astype(int)
        total_v = int((v == 1).sum())
        in_box_v = v[meets]
        coverage = (int((in_box_v == 1).sum()) / total_v) if total_v > 0 else np.nan
        density = float(in_box_v.mean()) if len(in_box_v) > 0 else np.nan
        return coverage, density

    @staticmethod
    def choose_cmp_by_corr(df: pd.DataFrame, vuln_col: str, drivers: list[str]) -> list[str]:
        cmps = []
        v = df[vuln_col].astype(float).to_numpy()
        for driver in drivers:
            x = df[driver].to_numpy()
            mask = np.isfinite(x) & np.isfinite(v)
            if mask.sum() < 3:
                cmps.append("<=")
                continue
            corr = np.corrcoef(x[mask], v[mask])[0, 1]
            if np.isnan(corr) or abs(corr) < 1e-9:
                cmps.append("<=")
            else:
                cmps.append("<" if corr < 0 else ">")
        return cmps

    def density_thresholds(self) -> list[float]:
        thresholds = []
        threshold = self.target_density
        while threshold >= self.min_density - 1e-12:
            thresholds.append(round(threshold, 10))
            threshold -= self.density_step
        return thresholds

    def select_best_result(self, optimization_results: pd.DataFrame) -> pd.DataFrame:
        if optimization_results.empty:
            return optimization_results.copy()

        results = optimization_results.copy()
        results["selected_density_threshold"] = np.nan
        results["selected_solution"] = False

        selected_idx = None
        selected_threshold = np.nan
        for threshold in self.density_thresholds():
            candidates = results[results["density"] >= threshold]
            if candidates.empty:
                continue
            selected_idx = candidates.sort_values(["coverage", "density"], ascending=False).index[0]
            selected_threshold = threshold
            break

        if selected_idx is None:
            selected_idx = results.sort_values(["density", "coverage"], ascending=False).index[0]

        results.loc[selected_idx, "selected_density_threshold"] = selected_threshold
        results.loc[selected_idx, "selected_solution"] = True

        selected = results.loc[[selected_idx]]
        remaining = results.drop(index=selected_idx).sort_values(["coverage", "density"], ascending=False)
        return pd.concat([selected, remaining], axis=0).reset_index(drop=True)

    def optimize(
        self,
        merged_df: pd.DataFrame,
        selected_features: list[str],
        vuln_col: str = "vulnerability_indicator",
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        if not PYMOO_OK:
            raise RuntimeError("pymoo is required for optimization. Install with `pip install pymoo`.")

        drivers = [f"{feature}__last" for feature in selected_features]
        required_cols = drivers + [vuln_col]
        missing = [col for col in required_cols if col not in merged_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for optimization: {missing}")

        pt = merged_df[required_cols].dropna().copy()
        pt[vuln_col] = pt[vuln_col].astype(int)

        if pt.empty:
            raise ValueError("No rows remain after dropping missing values.")
        if pt[vuln_col].nunique() < 2:
            raise ValueError(f"{vuln_col} must contain both 0 and 1.")

        cmp_selected = self.choose_cmp_by_corr(pt, vuln_col, drivers)
        results = self.nsga2_optimize_nd(pt, vuln_col, drivers, cmp_selected)

        for driver in drivers:
            results[f"{driver}__cutoff"] = results[driver].apply(
                lambda q, col=driver: np.quantile(pt[col].to_numpy(), q)
            )

        results["comparators"] = [cmp_selected] * len(results)
        results = self.select_best_result(results)
        return pt, results, cmp_selected

    def nsga2_optimize_nd(
        self,
        pt: pd.DataFrame,
        vulnerability_indicator: str,
        drivers: list[str],
        cmp: list[str] | None = None,
    ) -> pd.DataFrame:
        n = len(drivers)
        lo = np.array([self.lower] * n, dtype=float)
        hi = np.array([self.upper] * n, dtype=float)

        optimizer = self

        class Problem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=n, n_obj=2, n_constr=0, xl=lo, xu=hi)

            def _evaluate(self, x, out, *args, **kwargs):
                cov, den = optimizer.box_stats_multi(x, pt, vulnerability_indicator, drivers, cmp=cmp)
                out["F"] = np.array([-float(cov), -float(den)])

        algorithm = NSGA2(pop_size=self.popsize, eliminate_duplicates=True)
        termination = get_termination("n_gen", self.generations)
        res = minimize(Problem(), algorithm, termination, seed=self.seed, verbose=False)

        x_res = np.atleast_2d(res.X)
        f_res = np.atleast_2d(res.F)
        out = pd.DataFrame({drivers[i]: x_res[:, i] for i in range(n)})
        out["coverage"] = -f_res[:, 0]
        out["density"] = -f_res[:, 1]
        return out

    @staticmethod
    def plot_pareto_front(
        optimization_results: pd.DataFrame,
        coverage_col: str = "coverage",
        density_col: str = "density",
        highlight_idx: int = 0,
        annotate: bool = True,
        title: str = "Pareto Front: Coverage vs Density",
    ):
        import matplotlib.pyplot as plt

        if optimization_results is None or optimization_results.empty:
            raise ValueError("optimization_results is empty.")

        data = optimization_results[[coverage_col, density_col]].dropna().copy()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            data[density_col],
            data[coverage_col],
            s=70,
            alpha=0.75,
            color="tab:blue",
            edgecolor="k",
            label="Pareto solutions",
        )

        if highlight_idx in data.index:
            ax.scatter(
                data.loc[highlight_idx, density_col],
                data.loc[highlight_idx, coverage_col],
                s=180,
                color="tab:red",
                edgecolor="k",
                marker="*",
                label=f"Highlighted solution (row {highlight_idx})",
            )

        if annotate:
            for idx, row in data.iterrows():
                ax.annotate(
                    str(idx),
                    (row[density_col], row[coverage_col]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_xlabel("Density")
        ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
        return ax

    @staticmethod
    def plot_boxed_scatter_from_optimization_result(
        pt: pd.DataFrame,
        optimization_results: pd.DataFrame,
        row_idx: int = 0,
        vuln_col: str = "vulnerability_indicator",
        x_col: str | None = None,
        y_col: str | None = None,
        pairwise_if_needed: bool = True,
        max_cols: int = 3,
        labels: dict[int, str] | None = None,
        colors: dict[int, str] | None = None,
        markers: dict[int, str] | None = None,
        s: int = 60,
        alpha_pts: float = 0.8,
        box_color: str = "red",
        box_alpha: float = 0.08,
        box_edgecolor: str = "red",
        title: str | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
        close: bool = False,
    ) -> dict[str, Any]:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        result_row = optimization_results.iloc[row_idx]
        driver_cols = [
            col
            for col in optimization_results.columns
            if col in pt.columns and col != vuln_col and f"{col}__cutoff" in optimization_results.columns
        ]
        cmp_raw = result_row.get("comparators", None)
        cmp_values = ast.literal_eval(cmp_raw) if isinstance(cmp_raw, str) else list(cmp_raw)
        comparator_map = dict(zip(driver_cols, cmp_values))

        labels = labels or {0: "Not vulnerable (0)", 1: "Vulnerable (1)"}
        colors = colors or {0: "tab:blue", 1: "tab:orange"}
        markers = markers or {0: "o", 1: "X"}

        def interval(lim_min: float, lim_max: float, thr: float, sign: str) -> tuple[float, float]:
            if sign in ("<", "<="):
                return lim_min, min(thr, lim_max)
            if sign in (">", ">="):
                return max(thr, lim_min), lim_max
            raise ValueError(f"Comparator not supported: {sign}")

        def draw_projection(ax, x_name: str, y_name: str, show_legend: bool = False) -> dict[str, Any]:
            cutoff_x = float(result_row[f"{x_name}__cutoff"])
            cutoff_y = float(result_row[f"{y_name}__cutoff"])
            cmp_x = comparator_map[x_name]
            cmp_y = comparator_map[y_name]

            subset_data = pt[[x_name, y_name, vuln_col]].dropna().copy()
            subset_data[vuln_col] = subset_data[vuln_col].astype(int)

            for value in [0, 1]:
                subset = subset_data[subset_data[vuln_col] == value]
                if subset.empty:
                    continue
                ax.scatter(
                    subset[x_name],
                    subset[y_name],
                    s=s,
                    alpha=alpha_pts,
                    edgecolor="k",
                    color=colors.get(value, "gray"),
                    marker=markers.get(value, "o"),
                    label=labels.get(value, str(value)) if show_legend else None,
                )

            ax.axvline(cutoff_x, ls="--", lw=2, color="black", alpha=0.7)
            ax.axhline(cutoff_y, ls="--", lw=2, color="black", alpha=0.7)

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x0, x1 = interval(xmin, xmax, cutoff_x, cmp_x)
            y0, y1 = interval(ymin, ymax, cutoff_y, cmp_y)
            if x1 > x0 and y1 > y0:
                ax.add_patch(
                    Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=box_edgecolor, linewidth=2)
                )
                ax.add_patch(
                    Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=box_color, alpha=box_alpha, edgecolor="none")
                )

            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.grid(alpha=0.3)
            if show_legend:
                ax.legend(loc="best")

            return {
                "x_col": x_name,
                "y_col": y_name,
                "cutoff_x": cutoff_x,
                "cutoff_y": cutoff_y,
                "comparators": {x_name: cmp_x, y_name: cmp_y},
            }

        if x_col is not None or y_col is not None or len(driver_cols) == 2 or not pairwise_if_needed:
            x_col = x_col or driver_cols[0]
            y_col = y_col or driver_cols[1]
            fig, ax = plt.subplots(figsize=(8, 6))
            info = draw_projection(ax, x_col, y_col, show_legend=True)

            if title is None:
                title = (
                    f"Optimized Box for row {row_idx} | coverage={result_row['coverage']:.3f}, "
                    f"density={result_row['density']:.3f}"
                )
            fig.suptitle(title, y=1.02)
            plt.tight_layout()
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            if close:
                plt.close(fig)
            return {"row_idx": row_idx, "mode": "single_pair", **info}

        pairs = list(combinations(driver_cols, 2))
        n_pairs = len(pairs)
        n_cols = min(max_cols, n_pairs)
        n_rows = int(np.ceil(n_pairs / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()
        pair_info = []

        for ax, (x_name, y_name) in zip(axes_flat, pairs):
            pair_info.append(draw_projection(ax, x_name, y_name, show_legend=(x_name, y_name) == pairs[0]))
        for ax in axes_flat[n_pairs:]:
            ax.set_visible(False)

        if title is None:
            title = (
                f"Pairwise optimized box projections for row {row_idx} | "
                f"coverage={result_row['coverage']:.3f}, density={result_row['density']:.3f}"
            )
        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        if close:
            plt.close(fig)
        return {"row_idx": row_idx, "mode": "pairwise", "driver_cols": driver_cols, "pairs": pair_info}


class ScenarioDiscoveryBatchRunner:
    _plot_lock = Lock()

    def __init__(
        self,
        projected_df: pd.DataFrame,
        ensemble_df: pd.DataFrame,
        rules_path: str | Path,
        value_col: str = "con_edgar_ghg_mt_hp_trend",
        year1: str = "2022",
        year2: str = "2030",
        target_col: str = "vulnerability_indicator",
        auto_threshold: bool = False,
        top_k: int = 2,
        extra_features: list[str] | None = None,
        non_modeling_cols: list[str] | None = None,
        emission_cols: list[str] | None = None,
        trajectory_aggregation_mode: str = "full",
        rf_discovery: RandomForestDiscovery | None = None,
        optimizer: ScenarioDiscoveryOptimizer | None = None,
    ) -> None:
        self.projected_df = projected_df.copy()
        self.ensemble_df = ensemble_df.copy()
        self.rules_path = Path(rules_path)
        self.value_col = value_col
        self.year1 = year1
        self.year2 = year2
        self.target_col = target_col
        self.auto_threshold = auto_threshold
        self.top_k = top_k
        self.extra_features = extra_features or []
        self.non_modeling_cols = non_modeling_cols or [
            "future_id",
            "iso_alpha_3",
            "year_start",
            "year_end",
            "n_years",
            self.year1,
            self.year2,
            NDC_UNCONDITIONAL_COL,
        ]
        if self.auto_threshold and "vulnerability_threshold" not in self.non_modeling_cols:
            self.non_modeling_cols.append("vulnerability_threshold")
        self.emission_cols = emission_cols or [
            "x_log_signed_con_edgar_ghg_mt",
            "emissions_anchor_2022",
            "years_since_2022",
            "trend_year_interaction",
            "em_lag_1y",
            "em_trend_3y",
            "em_trend_5y",
            "em_volatility_5y",
            "em_acceleration",
        ]
        self.rf_discovery = rf_discovery or RandomForestDiscovery()
        self.optimizer = optimizer or ScenarioDiscoveryOptimizer()
        self.trajectory_aggregator = TrajectoryAggregator(
            self.rules_path,
            aggregation_mode=trajectory_aggregation_mode,
            start_year=self.year1,
            end_year=self.year2,
        )

    @staticmethod
    def _prepare_feature_importance(feature_importance_df: pd.DataFrame) -> pd.DataFrame:
        if feature_importance_df.empty:
            return feature_importance_df.copy()
        out = feature_importance_df.copy()
        out["feature_standard_name"] = out["feature"].apply(lambda x: x.split("__")[0] if "__" in x else x)
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    @staticmethod
    def _combine_features(selected_top_features: list[str], extra_features: list[str]) -> list[str]:
        combined: list[str] = []
        for feature in [*selected_top_features, *extra_features]:
            if feature not in combined:
                combined.append(feature)
        return combined

    @staticmethod
    def _single_numeric_value(df: pd.DataFrame, col: str) -> float | None:
        if col not in df.columns:
            return None
        values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        if len(values) == 0:
            return None
        if len(values) > 1 and not np.allclose(values, values[0]):
            return None
        return float(values[0])

    @staticmethod
    def _compare_numeric_values(value: float | None, reference: float | None) -> str | None:
        if value is None or reference is None:
            return None
        if np.isclose(value, reference):
            return "equal"
        return "above" if value > reference else "below"

    def run_country(
        self,
        iso_alpha_3: str,
        output_dir: str | Path,
        extra_features: list[str] | None = None,
        top_k: int | None = None,
        boxed_plot_row_idx: int = 0,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        country_output_dir = output_dir / iso_alpha_3
        country_output_dir.mkdir(parents=True, exist_ok=True)

        country_projected_df = self.projected_df[self.projected_df["iso_alpha_3"] == iso_alpha_3].copy()
        if country_projected_df.empty:
            raise ValueError(f"No projected data found for country '{iso_alpha_3}'.")

        df_pivot = VulnerabilityAnalyzer.compute_emissions_change(
            country_projected_df,
            self.year1,
            self.year2,
            self.value_col,
            auto_threshold=self.auto_threshold,
        )
        vulnerability_threshold = (
            float(df_pivot["vulnerability_threshold"].dropna().iloc[0])
            if "vulnerability_threshold" in df_pivot.columns and not df_pivot["vulnerability_threshold"].dropna().empty
            else None
        )
        year1_value = self._single_numeric_value(df_pivot, self.year1)
        ndc_unconditional_value = self._single_numeric_value(df_pivot, NDC_UNCONDITIONAL_COL)
        vulnerability_threshold_vs_year1 = self._compare_numeric_values(vulnerability_threshold, year1_value)
        vulnerability_threshold_vs_ndc_unconditional = self._compare_numeric_values(
            vulnerability_threshold,
            ndc_unconditional_value,
        )

        country_ensemble_df = self.ensemble_df.drop(columns=self.emission_cols, errors="ignore")
        country_ensemble_df = country_ensemble_df[country_ensemble_df["iso_alpha_3"] == iso_alpha_3].copy()
        if country_ensemble_df.empty:
            raise ValueError(f"No ensemble data found for country '{iso_alpha_3}'.")

        ensemble_agg_df = self.trajectory_aggregator.aggregate(country_ensemble_df)
        merged_df = VulnerabilityAnalyzer.merge_ensemble_with_vulnerability(
            ensemble_agg_df,
            df_pivot,
            on_cols=["future_id", "iso_alpha_3"],
        )

        rf_result = self.rf_discovery.fit(
            merged_df,
            target_col=self.target_col,
            non_modeling_cols=self.non_modeling_cols,
        )
        feature_importance_df = self._prepare_feature_importance(rf_result.feature_importance)
        if feature_importance_df.empty:
            raise ValueError(f"No feature importance results available for country '{iso_alpha_3}'.")

        selected_top_features = self.rf_discovery.select_top_features(
            feature_importance_df,
            feature_standard_name_col="feature_standard_name",
            top_k=top_k or self.top_k,
        )
        features_for_optimization = self._combine_features(
            selected_top_features,
            extra_features if extra_features is not None else self.extra_features,
        )

        pt, optimization_results, cmp_selected = self.optimizer.optimize(
            merged_df,
            features_for_optimization,
            vuln_col=self.target_col,
        )

        optimization_results_path = country_output_dir / f"{iso_alpha_3}_optimization_results.csv"
        plot_path = country_output_dir / f"{iso_alpha_3}_boxed_scatter.png"
        future_distribution_plot_path = country_output_dir / f"{iso_alpha_3}_future_distribution_with_baseline.png"
        feature_importance_path = country_output_dir / f"{iso_alpha_3}_feature_importance.csv"
        training_summary_path = country_output_dir / f"{iso_alpha_3}_rf_training_summary.csv"

        optimization_results.to_csv(optimization_results_path, index=False)
        feature_importance_df.to_csv(feature_importance_path, index=False)
        rf_result.training_summary.to_csv(training_summary_path, index=False)

        with self._plot_lock:
            VulnerabilityAnalyzer.plot_future_distribution_with_baseline(
                df_pivot,
                future_col=self.year2,
                baseline_col="vulnerability_threshold" if self.auto_threshold else self.year1,
                title=(
                    f"{iso_alpha_3}: Distribution of {self.year2} with "
                    f"{'vulnerability_threshold' if self.auto_threshold else self.year1} baseline"
                ),
                save_path=future_distribution_plot_path,
                show=False,
                close=True,
            )

            plot_info = self.optimizer.plot_boxed_scatter_from_optimization_result(
                pt,
                optimization_results,
                row_idx=boxed_plot_row_idx,
                title=f"{iso_alpha_3}: Optimized Box Scenario Discovery",
                save_path=plot_path,
                show=False,
                close=True,
            )
        selected_result_rows = (
            optimization_results[optimization_results["selected_solution"]]
            if "selected_solution" in optimization_results.columns
            else optimization_results.iloc[0:0]
        )
        selected_result = selected_result_rows.iloc[0] if not selected_result_rows.empty else optimization_results.iloc[0]
        selected_density = float(selected_result["density"])
        selected_coverage = float(selected_result["coverage"])

        return {
            "country": iso_alpha_3,
            "status": "success",
            "auto_threshold": self.auto_threshold,
            "vulnerability_threshold": vulnerability_threshold,
            f"vulnerability_threshold_vs_{self.year1}": vulnerability_threshold_vs_year1,
            "vulnerability_threshold_vs_ndc_unconditional": vulnerability_threshold_vs_ndc_unconditional,
            "selected_coverage": selected_coverage,
            "selected_density": selected_density,
            "selected_density_threshold": selected_result.get("selected_density_threshold", np.nan),
            "selected_density_below_min": bool(selected_density < self.optimizer.min_density),
            "selected_top_features": selected_top_features,
            "features_for_optimization": features_for_optimization,
            "cmp_selected": cmp_selected,
            "n_optimization_rows": int(len(optimization_results)),
            "optimization_results_path": str(optimization_results_path),
            "boxed_scatter_path": str(plot_path),
            "future_distribution_plot_path": str(future_distribution_plot_path),
            "feature_importance_path": str(feature_importance_path),
            "rf_training_summary_path": str(training_summary_path),
            "plot_info": plot_info,
        }

    def run_many(
        self,
        countries: list[str],
        output_dir: str | Path,
        extra_features_by_country: dict[str, list[str]] | None = None,
        top_k_by_country: dict[str, int] | None = None,
        boxed_plot_row_idx: int = 0,
        continue_on_error: bool = True,
    ) -> dict[str, pd.DataFrame]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []

        for country in countries:
            try:
                result = self.run_country(
                    country,
                    output_dir=output_dir,
                    extra_features=(extra_features_by_country or {}).get(country),
                    top_k=(top_k_by_country or {}).get(country),
                    boxed_plot_row_idx=boxed_plot_row_idx,
                )
                summary_rows.append(
                    {
                        "country": result["country"],
                        "status": result["status"],
                        "auto_threshold": result["auto_threshold"],
                        "vulnerability_threshold": result["vulnerability_threshold"],
                        f"vulnerability_threshold_vs_{self.year1}": result[f"vulnerability_threshold_vs_{self.year1}"],
                        "vulnerability_threshold_vs_ndc_unconditional": result[
                            "vulnerability_threshold_vs_ndc_unconditional"
                        ],
                        "selected_coverage": result["selected_coverage"],
                        "selected_density": result["selected_density"],
                        "selected_density_threshold": result["selected_density_threshold"],
                        "selected_density_below_min": result["selected_density_below_min"],
                        "selected_top_features": "|".join(result["selected_top_features"]),
                        "features_for_optimization": "|".join(result["features_for_optimization"]),
                        "n_optimization_rows": result["n_optimization_rows"],
                        "optimization_results_path": result["optimization_results_path"],
                        "boxed_scatter_path": result["boxed_scatter_path"],
                        "future_distribution_plot_path": result["future_distribution_plot_path"],
                    }
                )
            except Exception as exc:
                summary_rows.append(
                    {
                        "country": country,
                        "status": "error",
                        "auto_threshold": self.auto_threshold,
                        "vulnerability_threshold": None,
                        f"vulnerability_threshold_vs_{self.year1}": None,
                        "vulnerability_threshold_vs_ndc_unconditional": None,
                        "selected_coverage": None,
                        "selected_density": None,
                        "selected_density_threshold": None,
                        "selected_density_below_min": None,
                        "selected_top_features": "",
                        "features_for_optimization": "",
                        "n_optimization_rows": 0,
                        "optimization_results_path": "",
                        "boxed_scatter_path": "",
                        "future_distribution_plot_path": "",
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    raise

        summary_df = pd.DataFrame(summary_rows)
        feature_frequency_df = build_top_variable_frequency_report(summary_rows)
        feature_combination_frequency_df = build_feature_combination_frequency_report(summary_rows)

        summary_path = output_dir / "country_run_summary.csv"
        feature_frequency_path = output_dir / "top_variable_frequency_report.csv"
        feature_combination_frequency_path = output_dir / "top_variable_combination_frequency_report.csv"
        summary_df.to_csv(summary_path, index=False)
        feature_frequency_df.to_csv(feature_frequency_path, index=False)
        feature_combination_frequency_df.to_csv(feature_combination_frequency_path, index=False)

        return {
            "country_run_summary": summary_df,
            "top_variable_frequency_report": feature_frequency_df,
            "top_variable_combination_frequency_report": feature_combination_frequency_df,
        }
