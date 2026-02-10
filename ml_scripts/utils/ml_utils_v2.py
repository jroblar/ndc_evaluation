import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, GridSearchCV, cross_val_score, train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
from sklearn.inspection import permutation_importance

from scipy.stats import qmc
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima import auto_arima
from typing import Dict, Iterable, Optional, Union
from sklearn.impute import SimpleImputer


# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,          
    module=r"sklearn"                # just sklearn
)

class RegressionAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_col: str,
        year_col: str,
        feature_cols: list,
        holdout_years: int = 5,
        scaler_type: str = "standard",
        xgb_params: dict = None,
        rf_params: dict = None,
        enet_params: dict = None,
        include_year: bool = False,
        include_group_enet: bool = True,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.group_col = group_col
        self.year_col = year_col
        self.feature_cols = feature_cols.copy()
        self.include_year = include_year
        self.include_group_enet =include_group_enet

        self.scaler_type = scaler_type.lower()
        self.holdout_years = holdout_years

        self.xgb_params = xgb_params or dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        self.rf_params = rf_params or dict(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

        self.enet_params = enet_params or dict(
            l1_ratio=[0.2, 0.5, 0.8],
            cv=5,
            n_jobs=-1,
            max_iter=10000,
            random_state=42,
        )

        # --- Split train / test by time ---
        cutoff = self.df[self.year_col].max() - self.holdout_years
        train_mask = self.df[self.year_col] <= cutoff

        self.X_train = self.df.loc[train_mask]
        self.X_test  = self.df.loc[~train_mask]
        self.y_train = self.df.loc[train_mask, self.target_col]
        self.y_test  = self.df.loc[~train_mask, self.target_col]
        self.groups_train = self.df.loc[train_mask, self.group_col]

        # --- Build pipelines ---
        self._build_pipelines()

    # -------------------------
    # Utilities
    # -------------------------
    def _get_scaler(self):
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown scaler_type={self.scaler_type}")

    def _make_preprocessor(self, include_group: bool):
        feats = self.feature_cols.copy()
        if self.include_year:
            feats.append(self.year_col)

        if include_group:
            feats = feats + [self.group_col]

        categorical = [c for c in feats if self.df[c].dtype == "object"]
        numeric = [c for c in feats if c not in categorical]

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", self._get_scaler()),
                ]), numeric),

                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ]), categorical),
            ]
        )

    # -------------------------
    # Pipelines
    # -------------------------
    def _build_pipelines(self):
        # ElasticNet → ISO fixed effects
        pre_enet = self._make_preprocessor(include_group=self.include_group_enet)
        self.pipe_enet = Pipeline(
            [
                ("pre", pre_enet),
                ("enet", ElasticNetCV(**self.enet_params)),
            ]
        )

        # Random Forest → no ISO
        pre_rf = self._make_preprocessor(include_group=False)
        self.pipe_rf = Pipeline(
            [
                ("pre", pre_rf),
                ("rf", RandomForestRegressor(**self.rf_params)),
            ]
        )

        # XGBoost → no ISO
        pre_xgb = self._make_preprocessor(include_group=False)
        self.pipe_xgb = Pipeline(
            [
                ("pre", pre_xgb),
                ("xgb", XGBRegressor(**self.xgb_params)),
            ]
        )

        self.pipe_med = Pipeline(
            [("median", DummyRegressor(strategy="median"))]
        )

    # -------------------------
    # Cross-validation
    # -------------------------
    def cross_validate(self):
        tscv = TimeSeriesSplit(n_splits=5)
        gkf = GroupKFold(n_splits=5)

        models = [
            ("ElasticNet", self.pipe_enet, False),  # ISO included → no GroupKFold
            ("RandomForest", self.pipe_rf, True),
            ("XGBoost", self.pipe_xgb, True),
            ("Median", self.pipe_med, True),
        ]

        print("\nCross-validation results:")
        print("-" * 90)
        print(f"{'Model':<15} {'Time MAE':>12} {'Group MAE':>12}")

        for name, pipe, allow_group_cv in models:
            time_mae = -cross_val_score(
                pipe,
                self.X_train,
                self.y_train,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            ).mean()

            if allow_group_cv:
                group_mae = -cross_val_score(
                    pipe,
                    self.X_train,
                    self.y_train,
                    groups=self.groups_train,
                    cv=gkf,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                ).mean()
            else:
                group_mae = np.nan

            print(f"{name:<15} {time_mae:12.4f} {group_mae:12.4f}")

    # -------------------------
    # Fit / evaluate
    # -------------------------
    def fit(self):
        self.pipe_enet.fit(self.X_train, self.y_train)
        self.pipe_rf.fit(self.X_train, self.y_train)
        self.pipe_xgb.fit(self.X_train, self.y_train)
        self.pipe_med.fit(self.X_train, self.y_train)
        return self

    
    def _mae_level_space(self, y_true_log, y_pred_log):
        """
        MAE in original emissions units (level space).
        """
        y_true_level = np.exp(y_true_log)
        y_pred_level = np.exp(y_pred_log)
        return np.mean(np.abs(y_true_level - y_pred_level))


    def _rmse_log_space(self, y_true_log, y_pred_log):
        """
        RMSE in log space.
        """
        return root_mean_squared_error(y_true_log, y_pred_log)

    
    
    def evaluate(self):
        """
        Evaluate train vs test to detect overfitting.
        Reports:
        - MAE (log space)
        - RMSE (log space)
        - MAE (level space)
        """
        evals = {}

        for name, pipe in [
            ('ElasticNet',   self.pipe_enet),
            ('RandomForest', self.pipe_rf),
            ('XGBoost',      self.pipe_xgb),
            ('Median',       self.pipe_med),
        ]:
            # Predictions
            y_train_pred = pipe.predict(self.X_train)
            y_test_pred  = pipe.predict(self.X_test)

            evals[name] = {
                # ---- LOG SPACE ----
                'train_mae_log': mean_absolute_error(self.y_train, y_train_pred),
                'test_mae_log':  mean_absolute_error(self.y_test,  y_test_pred),
                'train_rmse_log': self._rmse_log_space(self.y_train, y_train_pred),
                'test_rmse_log':  self._rmse_log_space(self.y_test,  y_test_pred),

                # ---- LEVEL SPACE ----
                'train_mae_level': self._mae_level_space(self.y_train, y_train_pred),
                'test_mae_level':  self._mae_level_space(self.y_test,  y_test_pred),
            }

        # ---- Pretty print ----
        print("\nHoldout evaluation:")
        print("-" * 120)
        print(
            f"{'Model':<15}"
            f"{'Train MAE (log)':>15}"
            f"{'Test MAE (log)':>15}"
            f"{'Train RMSE (log)':>18}"
            f"{'Test RMSE (log)':>18}"
            f"{'Train MAE (level)':>20}"
            f"{'Test MAE (level)':>20}"
        )
        print("-" * 120)

        for name, m in evals.items():
            print(
                f"{name:<15}"
                f"{m['train_mae_log']:15.4f}"
                f"{m['test_mae_log']:15.4f}"
                f"{m['train_rmse_log']:18.4f}"
                f"{m['test_rmse_log']:18.4f}"
                f"{m['train_mae_level']:20.2f}"
                f"{m['test_mae_level']:20.2f}"
            )

        return evals

    
    def plot_feature_importances(self, model: str = "XGBoost", top_n: int = 15):
        """
        Plot feature importances from RandomForest or XGBoost.
        """
        model = model.lower()

        if model == "xgboost":
            pipe = self.pipe_xgb
            est_name = "xgb"
            title = "XGBoost Feature Importances"
        elif model == "randomforest":
            pipe = self.pipe_rf
            est_name = "rf"
            title = "Random Forest Feature Importances"
        else:
            raise ValueError("model must be 'XGBoost' or 'RandomForest'")

        # ---- Extract fitted components ----
        preprocessor = pipe.named_steps["pre"]
        estimator = pipe.named_steps[est_name]

        if not hasattr(estimator, "feature_importances_"):
            raise RuntimeError(f"{model} does not expose feature_importances_")

        importances = estimator.feature_importances_

        # ---- Get correct feature names AFTER preprocessing ----
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception as e:
            raise RuntimeError(
                "Could not extract feature names from preprocessor. "
                "Make sure the pipeline has been fitted."
            ) from e

        # ---- Sanity check ----
        if len(feature_names) != len(importances):
            raise RuntimeError(
                f"Mismatch: {len(feature_names)} features vs "
                f"{len(importances)} importances"
            )

        # ---- Select top features ----
        idx = np.argsort(importances)[::-1]
        if top_n is not None:
            idx = idx[:top_n]

        top_features = feature_names[idx]
        top_importances = importances[idx]

        # ---- Plot ----
        plt.figure(figsize=(8, max(4, 0.4 * len(idx))))
        plt.barh(range(len(idx)), top_importances[::-1])
        plt.yticks(range(len(idx)), top_features[::-1])
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def top_n_features(
        self,
        model: str = "XGBoost",
        n: int = 10,
        return_importance: bool = False
    ):
        """
        Return the names of the top-n most important features
        according to the fitted model.

        Parameters
        ----------
        model : {'XGBoost','RandomForest'}
            Which model to use.
        n : int
            Number of top features to return.
        return_importance : bool
            If True, return (feature, importance) tuples.

        Returns
        -------
        list
            List of feature names (or tuples if return_importance=True).
        """

        model = model.lower()

        if model == "xgboost":
            pipe = self.pipe_xgb
            est_name = "xgb"
        elif model == "randomforest":
            pipe = self.pipe_rf
            est_name = "rf"
        else:
            raise ValueError("model must be 'XGBoost' or 'RandomForest'")

        # ---- Extract fitted objects ----
        preprocessor = pipe.named_steps["pre"]
        estimator = pipe.named_steps[est_name]

        if not hasattr(estimator, "feature_importances_"):
            raise RuntimeError(f"{model} does not expose feature_importances_")

        importances = estimator.feature_importances_

        # ---- Feature names after preprocessing ----
        feature_names = preprocessor.get_feature_names_out()

        if len(importances) != len(feature_names):
            raise RuntimeError(
                "Mismatch between feature names and importance vector"
            )

        # ---- Rank ----
        order = np.argsort(importances)[::-1][:n]

        if return_importance:
            return [
                (feature_names[i], importances[i])
                for i in order
            ]
        else:
            return [feature_names[i] for i in order]


    def _get_pipe_by_name(self, model: str):
        model = model.strip().lower()
        if model == "randomforest":
            return self.pipe_rf
        elif model == "xgboost":
            return self.pipe_xgb
        elif model == "elasticnet":
            return self.pipe_enet
        elif model == "median":
            return self.pipe_med
        else:
            raise ValueError(
                "model must be one of: "
                "['RandomForest', 'XGBoost', 'ElasticNet', 'Median']"
            )

        
    def per_country_errors(
        self,
        model: str = "XGBoost",
        min_obs: int = 5
    ) -> pd.DataFrame:
        """
        Compute per-country errors on the TEST set.
        Reports:
        - MAE (log space)
        - RMSE (log space)
        - MAE (level space)
        - # observations
        """

        pipe = self._get_pipe_by_name(model)

        # Predictions on test set
        y_true_log = self.y_test.values
        y_pred_log = pipe.predict(self.X_test)

        y_true_lvl = np.exp(y_true_log)
        y_pred_lvl = np.exp(y_pred_log)

        countries = self.df.loc[self.X_test.index, self.group_col].values

        rows = []
        for c in np.unique(countries):
            idx = countries == c
            n = idx.sum()
            if n < min_obs:
                continue

            rows.append({
                "iso_alpha_3": c,
                "n_obs": n,
                "mae_log": np.mean(np.abs(y_true_log[idx] - y_pred_log[idx])),
                "rmse_log": np.sqrt(np.mean((y_true_log[idx] - y_pred_log[idx])**2)),
                "mae_level": np.mean(np.abs(y_true_lvl[idx] - y_pred_lvl[idx]))
            })

        df_err = (
            pd.DataFrame(rows)
            .sort_values("mae_log")
            .reset_index(drop=True)
        )

        return df_err
    
    def plot_per_country_errors(
        self,
        model: str = "XGBoost",
        metric: str = "mae_log",
        top_k: int = 25
    ):
        """
        Bar plot of per-country errors.
        """

        df_err = self.per_country_errors(model=model)

        if metric not in df_err.columns:
            raise ValueError(f"Metric must be one of {list(df_err.columns)}")

        dfp = df_err.sort_values(metric).head(top_k)

        plt.figure(figsize=(10, max(4, 0.35 * len(dfp))))
        plt.barh(dfp["iso_alpha_3"], dfp[metric])
        plt.xlabel(metric.replace("_", " ").upper())
        plt.ylabel("Country")
        plt.title(f"{model}: Per-country {metric}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def _policy_columns(self):
        return [
            c for c in self.feature_cols
            if "policy_flow" in c
        ]
    
    def policy_marginal_effect(
        self,
        model: str = "XGBoost",
        on: str = "test"
    ) -> pd.DataFrame:
        """
        Compute marginal contribution of policy variables:
        prediction(with policy) - prediction(with policy set to zero)
        """

        pipe = self._get_pipe_by_name(model)

        if on == "test":
            X = self.X_test.copy()
            idx = self.X_test.index
        else:
            X = self.X_train.copy()
            idx = self.X_train.index

        policy_cols = self._policy_columns()

        # Predictions with policy
        y_hat_with = pipe.predict(X)

        # Counterfactual: zero-out policy variables
        X_no_policy = X.copy()
        X_no_policy[policy_cols] = 0.0

        y_hat_no = pipe.predict(X_no_policy)

        df_out = self.df.loc[idx, [self.group_col, self.year_col]].copy()
        df_out["delta_log_emissions"] = y_hat_with - y_hat_no
        df_out["delta_emissions"] = (
            np.exp(y_hat_with) - np.exp(y_hat_no)
        )

        return df_out
    
    def policy_effect_by_country(
        self,
        model: str = "XGBoost"
    ) -> pd.DataFrame:
        df = self.policy_marginal_effect(model=model)

        return (
            df.groupby(self.group_col)
            .agg(
                mean_delta_log=("delta_log_emissions", "mean"),
                mean_delta_level=("delta_emissions", "mean"),
                median_delta_level=("delta_emissions", "median"),
                n_obs=("delta_emissions", "size")
            )
            .sort_values("mean_delta_log")
            .reset_index()
        )
    
    def plot_policy_effects(
        self,
        model: str = "XGBoost",
        top_k: int = 25
    ):
        df = self.policy_effect_by_country(model=model)
        dfp = df.sort_values("mean_delta_log").head(top_k)

        plt.figure(figsize=(10, max(4, 0.35 * len(dfp))))
        plt.barh(dfp[self.group_col], dfp["mean_delta_log"])
        plt.xlabel("Mean Δ log emissions (policy effect)")
        plt.ylabel("Country")
        plt.title(f"{model}: Policy marginal contribution")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class FeaturePredictiveEvaluator:
    """
    Evaluate the predictive usefulness of a single feature for emissions models.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "total_emissions",
        country_col: str = "iso_alpha_3",
        year_col: str = "year",
        group_col: str = "iso_alpha_3",
        include_year: bool = True,
        log_target: bool = True,
        n_splits: int = 5,
        random_state: int = 0,
        scaler_type: str = "standard",
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.country_col = country_col
        self.year_col = year_col
        self.group_col = group_col
        self.log_target = log_target
        self.n_splits = n_splits
        self.random_state = random_state
        self.include_year = include_year
        self.scaler_type = scaler_type.lower()

        self._prepare_target()
        self._prepare_baseline()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_target(self):
        y = self.df[self.target_col].values
        if self.log_target:
            y = np.log(y)
        self.y = y

    def _prepare_baseline(self):
        self.baseline_cols = [self.country_col]
        if self.include_year:
            self.baseline_cols.append(self.year_col)

        self.X_base = self.df[self.baseline_cols].copy()
        self.groups = self.df[self.group_col]
        self.cv = GroupKFold(n_splits=self.n_splits)

    def _get_scaler(self):
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown scaler_type={self.scaler_type}")

    def _make_preprocessor(self, features: list[str]) -> ColumnTransformer:
        categorical = [c for c in features if self.df[c].dtype == "object"]
        numeric = [c for c in features if c not in categorical]

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", self._get_scaler()),
                ]), numeric),

                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ]), categorical),
            ]
        )

    def _make_ridge_pipeline(self, features: list[str]) -> Pipeline:
        pre = self._make_preprocessor(features)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 25))
        return Pipeline([("pre", pre), ("ridge", ridge)])

    def _make_xgb_pipeline(self, features: list[str]) -> Pipeline:
        pre = self._make_preprocessor(features)
        xgb_model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return Pipeline([("pre", pre), ("xgb", xgb_model)])

    def _cv_mae_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: pd.Series,
    ) -> float:
        rmses = []
        for train, test in self.cv.split(X, y, groups):
            fold_pipe = clone(pipeline)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Found unknown categories in columns",
                    category=UserWarning,
                    module=r"sklearn\.preprocessing\._encoders",
                )
                fold_pipe.fit(X.iloc[train], y[train])
                preds = fold_pipe.predict(X.iloc[test])
            rmses.append(
                mean_absolute_error(y[test], preds)
            )
        return float(np.mean(rmses))

    def _cv_mae_pipeline_level(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: pd.Series,
    ) -> float:
        maes = []
        for train, test in self.cv.split(X, y, groups):
            fold_pipe = clone(pipeline)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Found unknown categories in columns",
                    category=UserWarning,
                    module=r"sklearn\.preprocessing\._encoders",
                )
                fold_pipe.fit(X.iloc[train], y[train])
                preds = fold_pipe.predict(X.iloc[test])
            if self.log_target:
                y_true = np.exp(y[test])
                y_pred = np.exp(preds)
            else:
                y_true = y[test]
                y_pred = preds
            maes.append(mean_absolute_error(y_true, y_pred))
        return float(np.mean(maes))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def within_group_std(self, feature: str) -> float:
        return (
            self.df
            .groupby(self.group_col)[feature]
            .std()
            .median()
        )

    def evaluate_feature(
        self,
        feature: str,
        ridge_alpha: float = 1.0,
        test_xgboost: bool = True,
        n_perm: int = 20,
    ) -> pd.Series:
        """
        Run full predictive diagnostics for a single feature.
        """

        results = {}

        # 1. Variability check
        results["median_within_group_std"] = self.within_group_std(feature)

        # 2. Ridge regression RMSE
        X_feat = pd.concat([self.X_base, self.df[[feature]]], axis=1)

        ridge_base_pipe = self._make_ridge_pipeline(self.baseline_cols)
        ridge_feat_pipe = self._make_ridge_pipeline(self.baseline_cols + [feature])

        results["mae_base_ridge"] = self._cv_mae_pipeline(
            ridge_base_pipe, self.X_base, self.y, self.groups
        )
        results["mae_with_feature_ridge"] = self._cv_mae_pipeline(
            ridge_feat_pipe, X_feat, self.y, self.groups
        )
        results["mae_delta_ridge"] = (
            results["mae_with_feature_ridge"]
            - results["mae_base_ridge"]
        )
        results["mae_base_ridge_level"] = self._cv_mae_pipeline_level(
            ridge_base_pipe, self.X_base, self.y, self.groups
        )
        results["mae_with_feature_ridge_level"] = self._cv_mae_pipeline_level(
            ridge_feat_pipe, X_feat, self.y, self.groups
        )
        results["mae_delta_ridge_level"] = (
            results["mae_with_feature_ridge_level"]
            - results["mae_base_ridge_level"]
        )

        # 3. Permutation importance (trained on full sample)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Found unknown categories in columns",
                category=UserWarning,
                module=r"sklearn\.preprocessing\._encoders",
            )
            ridge_feat_pipe.fit(X_feat, self.y)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Found unknown categories in columns",
                category=UserWarning,
                module=r"sklearn\.preprocessing\._encoders",
            )
            perm = permutation_importance(
                ridge_feat_pipe,
                X_feat,
                self.y,
                n_repeats=n_perm,
                random_state=self.random_state,
            )

        results["permutation_importance"] = float(
            perm.importances_mean[list(X_feat.columns).index(feature)]
        )

        # 4. Optional nonlinear test
        if test_xgboost and _HAS_XGB:
            xgb_base_pipe = self._make_xgb_pipeline(self.baseline_cols)
            xgb_feat_pipe = self._make_xgb_pipeline(self.baseline_cols + [feature])

            results["mae_base_xgb"] = self._cv_mae_pipeline(
                xgb_base_pipe, self.X_base, self.y, self.groups
            )
            results["mae_with_feature_xgb"] = self._cv_mae_pipeline(
                xgb_feat_pipe, X_feat, self.y, self.groups
            )
            results["mae_delta_xgb"] = (
                results["mae_with_feature_xgb"]
                - results["mae_base_xgb"]
            )
            results["mae_base_xgb_level"] = self._cv_mae_pipeline_level(
                xgb_base_pipe, self.X_base, self.y, self.groups
            )
            results["mae_with_feature_xgb_level"] = self._cv_mae_pipeline_level(
                xgb_feat_pipe, X_feat, self.y, self.groups
            )
            results["mae_delta_xgb_level"] = (
                results["mae_with_feature_xgb_level"]
                - results["mae_base_xgb_level"]
            )
        else:
            results["mae_base_xgb"] = np.nan
            results["mae_with_feature_xgb"] = np.nan
            results["mae_delta_xgb"] = np.nan
            results["mae_base_xgb_level"] = np.nan
            results["mae_with_feature_xgb_level"] = np.nan
            results["mae_delta_xgb_level"] = np.nan

        return pd.Series(results)

class EnsembleProjections:

    @staticmethod
    def generate_ensemble_arima(
        df: pd.DataFrame,
        feature_cols: list[str],
        start_year: int,
        end_year: int,
        n_scenarios: int,
        arima_order: tuple[int, int, int] = (1, 1, 1),
        auto_tune_arima: bool = False,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        random_state: int | None = None
    ) -> pd.DataFrame:
        """
        Generates an ensemble of future time series scenarios for multiple features using ARIMA models only.
        Supports either a fixed ARIMA order or automatic order selection (auto_arima).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing historical data with columns 'iso_alpha_3', 'year', and feature columns.
        feature_cols : list[str]
            List of feature column names to simulate.
        start_year : int
            First year of the simulation horizon (inclusive).
        end_year : int
            Last year of the simulation horizon (inclusive).
        n_scenarios : int
            Number of ensemble scenarios to generate per country.
        arima_order : tuple[int, int, int], default=(1, 1, 1)
            The (p, d, q) order for ARIMA, ignored if auto_tune_arima=True.
        auto_tune_arima : bool, default=False
            If True, selects ARIMA order for each feature using pmdarima.auto_arima.
        max_p, max_d, max_q : int
            Maximum orders to search with auto_arima (if used).
        random_state : int or None, default=None
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame containing simulated future scenarios with columns:
            ['iso_alpha_3', 'future_id', 'year'] + feature_cols.
        """

        rng = np.random.default_rng(random_state)
        years = np.arange(start_year, end_year + 1)
        horizon = len(years)
        rows = []

        for iso, grp in df.groupby("iso_alpha_3"):
            grp = grp.sort_values("year")

            sims_dict = {}
            for feat in feature_cols:
                hist = grp.set_index("year")[feat].copy()
                hist.index = pd.PeriodIndex(hist.index, freq="Y")

                # --- ARIMA order selection ---
                if auto_tune_arima:
                    # Use pmdarima's auto_arima to select order
                    arima_model = auto_arima(
                        hist,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        max_p=max_p, max_d=1, max_q=max_q,
                        error_action='ignore'
                    )
                    order = arima_model.order
                    print("tunned arima order: ", order)
                else:
                    order = arima_order

                # --- Fit ARIMA model ---
                sar = SARIMAX(
                    hist,
                    order=order,
                    enforce_stationarity=False, # por que?
                    enforce_invertibility=False # por que?
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Non-stationary starting autoregressive parameters"
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Non-invertible starting MA parameters" 
                    )
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)

                    fit = sar.fit(
                        disp=False,
                        maxiter=200,
                        method="lbfgs"
                    )

                # --- Simulate n_scenarios future paths ---
                sims = np.vstack([
                    fit.simulate(nsimulations=horizon, anchor="end").values
                    for _ in range(n_scenarios)
                ])
                sims_dict[feat] = sims

            # --- LHS to combine features into joint scenarios ---
            sampler = qmc.LatinHypercube(d=len(feature_cols), seed=random_state)
            u = sampler.random(n=n_scenarios)
            idx = (u * n_scenarios).astype(int)

            # --- Build rows for ensemble ---
            for s in range(n_scenarios):
                fid = f"id_{iso}_{s+1}"
                for t, year in enumerate(years):
                    row = {"iso_alpha_3": iso, "future_id": fid, "year": year}
                    for j, feat in enumerate(feature_cols):
                        sim_i = idx[s, j]
                        row[feat] = sims_dict[feat][sim_i, t] # Seguros de que este es el LHC que queremos? No estoy seguro de que tenga mucha variabilidad, debe tender asintotimante a los intervalos exactos
                    rows.append(row) 

        result = pd.DataFrame(rows)
        return result[["iso_alpha_3", "future_id", "year"] + feature_cols]


    @staticmethod
    def plot_ensemble_time_series(
        df, 
        iso_alpha_3, 
        column, 
        hist_df=None, 
        title=None, 
        ylabel=None, 
        xlabel="Year", 
        figsize=(10, 6)
    ):
        """
        Pony-tail plot: many thin ensemble forecasts + one bold historical.
        """
        # 1. Subset and sort
        ens = (
            df[df["iso_alpha_3"] == iso_alpha_3]
            .sort_values(["future_id", "year"])
        )
        plt.figure(figsize=figsize)

        # 2. Plot ensemble as light gray lines (no labels)
        for _, grp in ens.groupby("future_id"):
            plt.plot(
                grp["year"], 
                grp[column], 
                color="gray", 
                linewidth=1, 
                alpha=0.5
            )

        # 3. Overlay historical series
        if hist_df is not None:
            hist = (
                hist_df[hist_df["iso_alpha_3"] == iso_alpha_3]
                .sort_values("year")
            )
            if not hist.empty:
                plt.plot(
                    hist["year"],
                    hist[column],
                    color="black",
                    linewidth=2.5,
                    marker="o",
                    label="Historical"
                )

        # 4. Labels & legend
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or column)
        plt.title(title or f"{iso_alpha_3} – {column} (Ensemble)")
        if hist_df is not None and not hist.empty:
            plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_ensemble_time_series_grid(
        df,
        panels,
        hist_df=None,
        ncols=2,
        figsize=(12, 8),
        xlabel="Year",
        save_path=None
    ):
        """
        Multi-panel pony-tail plot for ensemble projections.

        Parameters
        ----------
        df : pd.DataFrame
            Ensemble dataframe with columns:
            ['iso_alpha_3', 'future_id', 'year', <vars>]

        panels : list of dict
            Each dict defines one subplot with keys:
            {
                "iso": str,
                "column": str,
                "title": str (optional),
                "ylabel": str (optional)
            }

            Length of panels should be 2 or 4 (or any number).

        hist_df : pd.DataFrame, optional
            Historical dataframe with same structure as df (no future_id).

        ncols : int
            Number of columns in subplot grid (default=2).

        figsize : tuple
            Figure size.

        xlabel : str
            Shared x-axis label.
        """

        n_panels = len(panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes).flatten()

        for ax, panel in zip(axes, panels):
            iso = panel["iso"]
            col = panel["column"]
            title = panel.get("title", f"{iso} – {col}")
            ylabel = panel.get("ylabel", col)

            # --- Subset ensemble ---
            ens = (
                df[df["iso_alpha_3"] == iso]
                .sort_values(["future_id", "year"])
            )

            # --- Plot ensemble members ---
            for _, grp in ens.groupby("future_id"):
                ax.plot(
                    grp["year"],
                    grp[col],
                    color="gray",
                    linewidth=1,
                    alpha=0.4
                )

            # --- Overlay historical series ---
            if hist_df is not None:
                hist = (
                    hist_df[hist_df["iso_alpha_3"] == iso]
                    .sort_values("year")
                )
                if not hist.empty and col in hist.columns:
                    ax.plot(
                        hist["year"],
                        hist[col],
                        color="black",
                        linewidth=2.5,
                        marker="o",
                        label="Historical"
                    )

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)

        # --- Remove unused axes ---
        for ax in axes[n_panels:]:
            ax.axis("off")

        # --- Shared labels ---
        for ax in axes[-ncols:]:
            ax.set_xlabel(xlabel)

        # --- Legend (single) ---
        if hist_df is not None:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2)


        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # ---- SAVE FIGURE (IMPORTANT) ----
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")

        plt.show()
        plt.close(fig)
        
    
    @staticmethod
    def predict_ensemble_emissions(
        ensemble_df: pd.DataFrame,
        model: Any,
        feature_cols: List[str],
        exponentiate: bool = True
    ) -> pd.DataFrame:
        """
        Given an ensemble of future feature vectors and a fitted sklearn‐style model (e.g. Pipeline),
        predict log_total_emissions and (optionally) back‐transform to total_emissions.

        Parameters
        ----------
        ensemble_df : pd.DataFrame
            DataFrame with columns ['iso_alpha_3', 'future_id', 'year'] + feature_cols.
        model : Any
            A fitted model or Pipeline with a .predict(X) method that returns log_total_emissions.
        feature_cols : List[str]
            Column names to use as predictors (in the order the model expects).
        exponentiate : bool, default=True
            If True, creates a 'total_emissions' column = exp(log_total_emissions).

        Returns
        -------
        pd.DataFrame
            A copy of `ensemble_df` with two new columns:
            - 'log_total_emissions': the raw model predictions
            - 'total_emissions'     : exp(log_total_emissions) if exponentiate=True
        """
        # 1) Work on a copy
        df = ensemble_df.copy()

        # 2) Extract the feature matrix
        X = df[feature_cols]

        # 3) Run through the pipeline/model
        df["log_total_emissions"] = model.predict(X)

        # 4) Optional back‐transform
        if exponentiate:
            df["total_emissions"] = np.exp(df["log_total_emissions"])

        return df
    
    @staticmethod
    def calibrate_total_emissions(
        simulated_df: pd.DataFrame,
        initial_emissions_df: pd.DataFrame,
        base_year: int = 2022,
        adjustment_method: str = "additive"  # or "multiplicative"
    ) -> pd.DataFrame:
        """
        Calibrates the 'total_emissions' column in simulated_df using initial emissions values.

        Parameters:
        - simulated_df: DataFrame with time series simulation (must include 'total_emissions').
        - initial_emissions_df: DataFrame with columns ['iso_alpha_3', 'year', 'total_emissions'] for base year.
        - base_year: The year to anchor calibration (default 2022).
        - adjustment_method: Either 'additive' or 'multiplicative'.

        Returns:
        - A copy of simulated_df with calibrated 'total_emissions'.
        """
        # Filter simulation for base year values
        base_sim = simulated_df[simulated_df["year"] == base_year][["future_id", "iso_alpha_3", "total_emissions"]]
        base_sim = base_sim.rename(columns={"total_emissions": "sim_base_emissions"})

        # Merge simulation base emissions into full simulation
        df = simulated_df.merge(base_sim[["future_id", "sim_base_emissions"]], on="future_id", how="left")

        # Prepare initial conditions lookup
        init_emissions = initial_emissions_df.set_index("iso_alpha_3")["total_emissions"].to_dict()

        # Apply calibration
        if adjustment_method == "additive":
            df["total_emissions"] = df.apply(
                lambda row: init_emissions.get(row["iso_alpha_3"], np.nan) +
                            (row["total_emissions"] - row["sim_base_emissions"]),
                axis=1
            )
        elif adjustment_method == "multiplicative":
            df["total_emissions"] = df.apply(
                lambda row: init_emissions.get(row["iso_alpha_3"], np.nan) *
                            (row["total_emissions"] / row["sim_base_emissions"]) if row["sim_base_emissions"] != 0 else np.nan,
                axis=1
            )
        else:
            raise ValueError("adjustment_method must be either 'additive' or 'multiplicative'")

        # Clean up and return
        df.drop(columns=["sim_base_emissions"], inplace=True)
        return df
    
    @staticmethod
    def calibrate_to_initial_conditions(
        simulated_df: pd.DataFrame,
        initial_conditions_df: pd.DataFrame,
        base_year: int = 2022,
        columns: Optional[Iterable[str]] = None,
        col_map: Optional[Dict[str, str]] = None,
        adjustment_method: Union[str, Dict[str, str]] = "multiplicative",
        update_logs: bool = True,
        log_prefix: str = "log_",
        epsilon: float = 1e-12,
    ) -> pd.DataFrame:
        """
        Calibrate selected feature columns in `simulated_df` so that, for each future (future_id),
        the base-year values match those in `initial_conditions_df` (keyed by iso_alpha_3, year=base_year).
        Then propagate the SAME additive or multiplicative deviation across all years.

        Parameters
        ----------
        simulated_df : pd.DataFrame
            Must include ['iso_alpha_3','future_id','year'] and the feature columns to adjust.
        initial_conditions_df : pd.DataFrame
            Must include ['iso_alpha_3','year'] and base-year values for selected features.
        base_year : int
            Anchor year (default 2022).
        columns : iterable[str] | None
            Which *initial* columns to align. Defaults to the intersection of feature columns
            (excluding keys) present in both dataframes.
        col_map : dict[str,str] | None
            Optional mapping {init_col -> sim_col} if column names differ between the two frames.
            By default it assumes the same names in both.
        adjustment_method : {"additive","multiplicative"} or dict[str, {"additive","multiplicative"}]
            Either a single method for all columns or a per-column map.
            Tip: use "additive" for growth rates, "multiplicative" for level variables.
        update_logs : bool
            If True, recompute matching log columns (e.g., 'log_pop_total' from 'pop_total').
            Values <= 0 will produce NaN in the log.
        log_prefix : str
            Prefix used for log columns in `simulated_df`.
        epsilon : float
            Small constant to avoid division by zero in multiplicative adjustment.

        Returns
        -------
        pd.DataFrame
            A copy of `simulated_df` with calibrated columns (and optional log columns) updated.
        """
        required_sim_cols = {"iso_alpha_3", "future_id", "year"}
        if not required_sim_cols.issubset(simulated_df.columns):
            missing = required_sim_cols - set(simulated_df.columns)
            raise ValueError(f"`simulated_df` is missing required columns: {missing}")

        required_init_cols = {"iso_alpha_3", "year"}
        if not required_init_cols.issubset(initial_conditions_df.columns):
            missing = required_init_cols - set(initial_conditions_df.columns)
            raise ValueError(f"`initial_conditions_df` is missing required columns: {missing}")

        # Filter initial conditions to the base year and drop duplicates on iso
        init_base = (
            initial_conditions_df[initial_conditions_df["year"] == base_year]
            .drop_duplicates(subset=["iso_alpha_3"])
            .set_index("iso_alpha_3")
        )

        # Decide which columns to calibrate
        # (exclude key columns from consideration)
        if columns is None:
            exclude = {"iso_alpha_3", "year"}
            candidate_cols = [c for c in init_base.columns if c not in exclude]
            columns = [c for c in candidate_cols if c in simulated_df.columns]

        if len(columns) == 0:
            # Nothing to do
            return simulated_df.copy()

        # Optional mapping: init_col -> sim_col (default: identity)
        col_map = col_map or {c: c for c in columns}

        # Extract base-year simulated values per future_id
        base_sim = (
            simulated_df.loc[simulated_df["year"] == base_year, ["future_id", "iso_alpha_3"] + list(col_map.values())]
            .copy()
        )
        # Rename sim base columns: sim_base_<sim_col>
        base_rename = {sim_col: f"sim_base__{sim_col}" for sim_col in col_map.values()}
        base_sim.rename(columns=base_rename, inplace=True)

        # Merge base sim baselines into all rows by future_id
        df = simulated_df.merge(base_sim[["future_id"] + list(base_rename.values())], on="future_id", how="left")

        def method_for(col: str) -> str:
            if isinstance(adjustment_method, dict):
                return adjustment_method.get(col, "multiplicative")
            return adjustment_method

        # Calibrate per column
        for init_col, sim_col in col_map.items():
            if init_col not in init_base.columns:
                # Skip silently if init col not present in base init df
                continue
            if sim_col not in df.columns:
                continue

            # Lookup initial base value by iso for the whole df (align via index)
            init_vals = df["iso_alpha_3"].map(init_base[init_col])

            sim_base_col = f"sim_base__{sim_col}"
            if sim_base_col not in df.columns:
                # No baseline for this future_id -> cannot calibrate; skip
                continue

            m = method_for(init_col).lower()
            if m not in {"additive", "multiplicative"}:
                raise ValueError(f"Invalid method '{m}' for column '{init_col}'. Use 'additive' or 'multiplicative'.")

            if m == "additive":
                # new = init + (cur - base)
                df[sim_col] = init_vals + (df[sim_col] - df[sim_base_col])
            else:
                # new = init * (cur / base)
                denom = np.where(np.isfinite(df[sim_base_col]) & (np.abs(df[sim_base_col]) > 0), df[sim_base_col], np.nan)
                ratio = df[sim_col] / denom
                df[sim_col] = init_vals * ratio

            # Optional: update corresponding log column if present
            if update_logs:
                log_col = f"{log_prefix}{sim_col}" if not sim_col.startswith(log_prefix) else sim_col
                # Only recompute if a separate log column exists
                if log_col in df.columns and log_col != sim_col:
                    # Guard: log undefined for <= 0 -> set NaN
                    with np.errstate(divide="ignore", invalid="ignore"):
                        df[log_col] = np.where(df[sim_col] > 0, np.log(df[sim_col]), np.nan)

        # Drop baseline helper columns
        df.drop(columns=[c for c in df.columns if c.startswith("sim_base__")], inplace=True)
        return df
