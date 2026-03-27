import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class ScenarioDiscovery:

    @staticmethod
    def compute_emissions_change(df, year1, year2, value_col):
        df_pivot = df.pivot(index=["future_id", "iso_alpha_3"], columns="year", values=value_col).reset_index()
        df_pivot.columns.name = None

        # ensure all column names are strings
        df_pivot.columns = df_pivot.columns.astype(str)

        # Create vulnerability_indicator field
        df_pivot["vulnerability_indicator"] = (df_pivot[year2] > df_pivot[year1]).astype(int)

        df_pivot = df_pivot[["future_id", "iso_alpha_3", year1, year2, "vulnerability_indicator"]]
        return df_pivot
    

    @staticmethod
    def plot_vulnerability_counts(
        df,
        vuln_col="vulnerability_indicator",
        figsize=(6, 4),
        palette=None,
        xtick_labels=None,
        xlabel="",
        ylabel="Count",
        title="Vulnerability Indicator Counts",
        annotate=True,
        order=None,
        show=True
    ):
        """
        Plot a simple annotated countplot for a binary vulnerability column.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the vulnerability column.
        vuln_col : str
            Column name with binary indicator (expected values like 0 and 1).
        figsize : tuple
            Figure size.
        palette : list or dict
            Colors for bars.
        xtick_labels : list
            Labels for x ticks (same order as `order`).
        xlabel, ylabel, title : str
            Axis and title text.
        annotate : bool
            Whether to annotate bar counts.
        order : list
            Explicit order of categories to plot (e.g., [0, 1]).
        show : bool
            Whether to call plt.show() before returning the axis.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object of the created plot.
        """
        if vuln_col not in df.columns:
            raise ValueError(f"Column '{vuln_col}' not found in dataframe.")

        palette = palette or ["tab:blue", "tab:orange"]

        # determine order of categories
        if order is None:
            order = sorted(df[vuln_col].dropna().unique().tolist())

        # default xtick labels for binary 0/1
        if xtick_labels is None:
            if set(order) == {0, 1}:
                xtick_labels = ["Not vulnerable (0)", "Vulnerable (1)"]
            else:
                xtick_labels = [str(v) for v in order]

        plt.figure(figsize=figsize)
        ax_local = plt.gca()
        sns.countplot(x=vuln_col, data=df, palette=palette, order=order, ax=ax_local)
        ax_local.set_xticks(range(len(order)))
        ax_local.set_xticklabels(xtick_labels)
        ax_local.set_ylabel(ylabel)
        ax_local.set_xlabel(xlabel)
        ax_local.set_title(title)

        if annotate:
            for p in ax_local.patches:
                height = p.get_height()
                ax_local.annotate(
                    f"{int(height)}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha="center",
                    va="bottom"
                )

        plt.tight_layout()
        if show:
            plt.show()
        return ax_local
    
    def merge_ensemble_with_vulnerability(ensemble_agg_df, vulnerability_df, on_cols=["future_id"]):
        merged_df = pd.merge(ensemble_agg_df, vulnerability_df, on=on_cols, how="left")
        # ensure the bin cols are integers (in case they got cast to floats during the merge)
        # use the nullable integer dtype so NaNs are preserved instead of raising on conversion
        for col in vulnerability_df.columns:
            if col.startswith("v"):
                # coerce non-finite values to NaN, then cast to pandas nullable integer dtype
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype("Int64")

        return merged_df
    

class RFUtils:
    #Prepare data
    X_all = merged_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(merged_df[target_col], errors="coerce")
    valid_mask = y.notna()

    X_target = X_all.loc[valid_mask].copy()
    y_target = y.loc[valid_mask].astype(int)

    rf_models = {}
    rf_training_rows = []
    feature_importance_rows = []
    test_size = 0.2  # keep existing value

    # Only one target_col expected; skip training if insufficient data or classes
    if X_target.empty or y_target.nunique() < 2:
        rf_training_rows.append({
            "target_col": target_col,
            "n_rows": int(len(y_target)),
            "class_0": int((y_target == 0).sum()),
            "class_1": int((y_target == 1).sum()),
            "status": "skipped_insufficient_rows_or_classes"
        })
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_target,
            y_target,
            test_size=test_size,
            random_state=42,
            stratify=y_target
        )

        eval_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2
            ))
        ])

        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)
        y_proba = eval_model.predict_proba(X_test)[:, 1]

        final_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2
            ))
        ])
        final_model.fit(X_target, y_target)
        rf_models[target_col] = final_model

        rf = final_model.named_steps["rf"]
        importances = rf.feature_importances_
        max_importance = float(importances.max()) if importances.size else np.nan
        mean_importance = float(importances.mean()) if importances.size else np.nan
        median_importance = float(np.median(importances)) if importances.size else np.nan
        baseline_accuracy = float(max((y_test == 0).mean(), (y_test == 1).mean()))
        roc_auc = float(roc_auc_score(y_test, y_proba)) if y_test.nunique() == 2 else np.nan

        rf_training_rows.append({
            "target_col": target_col,
            "n_rows": int(len(y_target)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "class_0": int((y_target == 0).sum()),
            "class_1": int((y_target == 1).sum()),
            "status": "trained",
            "n_features": int(len(feature_cols)),
            "baseline_accuracy": baseline_accuracy,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc": roc_auc,
            "mean_importance": mean_importance,
            "median_importance": median_importance,
            "max_importance": max_importance,
        })

        for feature_name, importance in zip(feature_cols, importances):
            feature_importance_rows.append({
                "target_col": target_col,
                "feature": feature_name,
                "importance": float(importance),
            })

    rf_training_summary_df = pd.DataFrame(rf_training_rows).sort_values("target_col").reset_index(drop=True)
    feature_importance_df = pd.DataFrame(feature_importance_rows)



class ArimaAggUtils:

    def load_projection_rulebook(rules_path):
        with open(rules_path, 'r') as f:
            rulebook = json.load(f)

        category_map = {}
        for category, columns in rulebook.get('categories', {}).items():
            for col in columns:
                category_map[col] = category

        for col, category in rulebook.get('overrides', {}).items():
            category_map[col] = category

        prefix_rules = rulebook.get('prefix_rules', [])
        default_category = rulebook.get('default_category', 'unconstrained')
        return category_map, prefix_rules, default_category

    def resolve_category(col, category_map, prefix_rules, default_category):
        if col in category_map:
            return category_map[col]
        for rule in prefix_rules:
            if col.startswith(rule['prefix']):
                return rule['category']
        return default_category

    def _safe_std(values):
        if len(values) <= 1:
            return np.nan
        return float(np.std(values, ddof=1))

    def _safe_slope(years, values):
        if len(values) <= 1:
            return np.nan
        return float(np.polyfit(years.astype(float), values.astype(float), 1)[0])

    def summarize_trajectory(group, value_cols, category_map, prefix_rules, default_category):
        group = group.sort_values('year')
        row = {
            'future_id': group['future_id'].iloc[0],
            'iso_alpha_3': group['iso_alpha_3'].iloc[0],
            'year_start': int(group['year'].min()),
            'year_end': int(group['year'].max()),
            'n_years': int(group['year'].nunique()),
        }

        for col in value_cols:
            category = resolve_category(col, category_map, prefix_rules, default_category)
            valid = group[['year', col]].dropna()

            if valid.empty:
                continue

            years = valid['year'].to_numpy()
            values = valid[col].astype(float).to_numpy()
            prefix = f'{col}__'

            if category == 'binary':
                row[prefix + 'mean'] = float(values.mean())
                row[prefix + 'last'] = float(values[-1])
                row[prefix + 'max'] = float(values.max())
                row[prefix + 'switches'] = float(np.abs(np.diff(values)).sum()) if len(values) > 1 else 0.0
            elif category == 'cumulative_binary':
                row[prefix + 'last'] = float(values[-1])
                row[prefix + 'max'] = float(values.max())
            elif category == 'count':
                row[prefix + 'sum'] = float(values.sum())
                row[prefix + 'mean'] = float(values.mean())
                row[prefix + 'std'] = _safe_std(values)
                row[prefix + 'max'] = float(values.max())
                row[prefix + 'last'] = float(values[-1])
                row[prefix + 'delta'] = float(values[-1] - values[0])
                row[prefix + 'slope'] = _safe_slope(years, values)
            elif category == 'cumulative_count':
                row[prefix + 'last'] = float(values[-1])
                row[prefix + 'max'] = float(values.max())
                row[prefix + 'delta'] = float(values[-1] - values[0])
                row[prefix + 'slope'] = _safe_slope(years, values)
            else:
                row[prefix + 'mean'] = float(values.mean())
                row[prefix + 'std'] = _safe_std(values)
                row[prefix + 'min'] = float(values.min())
                row[prefix + 'max'] = float(values.max())
                row[prefix + 'last'] = float(values[-1])
                row[prefix + 'delta'] = float(values[-1] - values[0])
                row[prefix + 'slope'] = _safe_slope(years, values)

        return row

    def aggregate_ensemble_by_future_id(
        ensemble_df,
        rules_path,
        group_col='future_id',
        keep_country_col='iso_alpha_3',
        year_col='year'
    ):
        category_map, prefix_rules, default_category = load_projection_rulebook(rules_path)

        id_cols = {group_col, keep_country_col, year_col}
        value_cols = [col for col in ensemble_df.columns if col not in id_cols]

        aggregated_rows = []
        for _, group in ensemble_df.groupby(group_col, sort=False):
            aggregated_rows.append(
                summarize_trajectory(group, value_cols, category_map, prefix_rules, default_category)
            )

        return pd.DataFrame(aggregated_rows)
