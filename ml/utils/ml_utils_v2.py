import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, GridSearchCV, cross_val_score, train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.decomposition import PCA
from scipy import sparse
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
        use_pca_enet: bool = False,
        enet_pca_n_components: Optional[int] = None,
        enet_pca_random_state: int = 42,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.group_col = group_col
        self.year_col = year_col
        self.feature_cols = feature_cols.copy()
        self.include_year = include_year
        self.include_group_enet =include_group_enet
        self.use_pca_enet = use_pca_enet
        self.enet_pca_n_components = enet_pca_n_components
        self.enet_pca_random_state = enet_pca_random_state

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
        self.enet_params = self._configure_enet_cv(self.enet_params)

        # --- Split train / test by time ---
        cutoff = self.df[self.year_col].max() - self.holdout_years
        train_mask = self.df[self.year_col] <= cutoff

        # Keep temporal order explicit so time-based CV splitters are leak-safe.
        sort_cols = [self.year_col, self.group_col]
        train_df = self.df.loc[train_mask].sort_values(sort_cols).copy()
        test_df = self.df.loc[~train_mask].sort_values(sort_cols).copy()

        self.X_train = train_df
        self.X_test = test_df
        self.y_train = train_df[self.target_col]
        self.y_test = test_df[self.target_col]
        self.groups_train = train_df[self.group_col]

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

    def _configure_enet_cv(self, enet_params: dict) -> dict:
        """
        Ensure ElasticNetCV uses time-aware inner CV to avoid temporal leakage.
        """
        params = enet_params.copy()
        cv_value = params.get("cv", 5)

        if isinstance(cv_value, int):
            if cv_value < 2:
                raise ValueError("ElasticNetCV cv must be >= 2 when provided as an int.")
            params["cv"] = TimeSeriesSplit(n_splits=cv_value)

        return params

    def _make_preprocessor(self, include_group: bool, features: Optional[list] = None):
        feats = self.feature_cols.copy() if features is None else features.copy()
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
    
    def _to_dense(self, X):
        if sparse.issparse(X):
            return X.toarray()
        return X

    def _make_group_dummy_preprocessor(self):
        return ColumnTransformer(
            transformers=[
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ]), [self.group_col]),
            ]
        )

    def _make_enet_pca_pipeline(
        self,
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
        features: Optional[list] = None,
        include_group: Optional[bool] = None,
    ):
        n_components = self.enet_pca_n_components if n_components is None else n_components
        random_state = self.enet_pca_random_state if random_state is None else random_state
        include_group = self.include_group_enet if include_group is None else include_group

        # Keep group dummies out of PCA so fixed effects remain explicit regressors.
        if include_group:
            pca_branch = Pipeline(
                [
                    ("pre_no_group", self._make_preprocessor(include_group=False, features=features)),
                    ("to_dense", FunctionTransformer(self._to_dense, accept_sparse=True)),
                    ("pca", PCA(
                        n_components=n_components,
                        random_state=random_state,
                    )),
                ]
            )
            group_branch = self._make_group_dummy_preprocessor()
            return Pipeline(
                [
                    ("pre_split", FeatureUnion([
                        ("pca_features", pca_branch),
                        ("group_dummies", group_branch),
                    ])),
                    ("to_dense_after_union", FunctionTransformer(self._to_dense, accept_sparse=True)),
                    ("enet", ElasticNetCV(**self.enet_params)),
                ]
            )

        pre = self._make_preprocessor(include_group=False, features=features)
        return Pipeline(
            [
                ("pre", pre),
                ("to_dense", FunctionTransformer(self._to_dense, accept_sparse=True)),
                ("pca", PCA(
                    n_components=n_components,
                    random_state=random_state,
                )),
                ("enet", ElasticNetCV(**self.enet_params)),
            ]
        )

    # -------------------------
    # Pipelines
    # -------------------------
    def _build_pipelines(self):
        # ElasticNet → ISO fixed effects
        pre_enet = self._make_preprocessor(include_group=self.include_group_enet)
        if self.use_pca_enet:
            self.pipe_enet = self._make_enet_pca_pipeline()
        else:
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
            pipe_for_cv = pipe
            if name == "ElasticNet":
                # Avoid nested parallelism: outer CV is parallelized by cross_val_score.
                pipe_for_cv = clone(pipe)
                pipe_for_cv.set_params(enet__n_jobs=1)

            time_mae = -cross_val_score(
                pipe_for_cv,
                self.X_train,
                self.y_train,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            ).mean()

            if allow_group_cv:
                group_mae = -cross_val_score(
                    pipe_for_cv,
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

    def get_model_hyperparameters(
        self,
        model: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return hyperparameters used by fitted model(s).

        Parameters
        ----------
        model : str or None, default=None
            If provided, return only that model ('RandomForest', 'XGBoost',
            'ElasticNet', or 'Median'). If None, return all.

        Returns
        -------
        dict
            Mapping from model name to:
            - 'is_fitted': bool
            - 'hyperparameters': estimator.get_params(deep=False)
            - model-specific fitted selections when available
        """
        model_to_estimator_step = {
            "elasticnet": "enet",
            "randomforest": "rf",
            "xgboost": "xgb",
            "median": "median",
        }
        canonical_names = {
            "elasticnet": "ElasticNet",
            "randomforest": "RandomForest",
            "xgboost": "XGBoost",
            "median": "Median",
        }

        selected_models = (
            [model.strip().lower()] if model is not None else list(model_to_estimator_step)
        )

        output: Dict[str, Dict[str, Any]] = {}
        for model_key in selected_models:
            if model_key not in model_to_estimator_step:
                raise ValueError(
                    "model must be one of: "
                    "['RandomForest', 'XGBoost', 'ElasticNet', 'Median']"
                )

            pipe = self._get_pipe_by_name(model_key)
            est_step = model_to_estimator_step[model_key]
            estimator = pipe.named_steps[est_step]

            is_fitted = (
                hasattr(estimator, "n_features_in_")
                or hasattr(estimator, "coef_")
                or hasattr(estimator, "feature_importances_")
                or hasattr(estimator, "constant_")
                or hasattr(estimator, "alpha_")
            )

            model_info: Dict[str, Any] = {
                "is_fitted": is_fitted,
                "hyperparameters": estimator.get_params(deep=False),
            }

            if model_key == "elasticnet" and is_fitted:
                if hasattr(estimator, "alpha_"):
                    model_info["selected_alpha"] = float(estimator.alpha_)
                if hasattr(estimator, "l1_ratio_"):
                    model_info["selected_l1_ratio"] = float(estimator.l1_ratio_)
                if hasattr(estimator, "n_iter_"):
                    model_info["n_iter"] = estimator.n_iter_

            output[canonical_names[model_key]] = model_info

        return output
    
    def _make_estimator(self, model: str):
        model = model.strip().lower()
        if model == "ridge":
            return "ridge", RidgeCV(alphas=np.logspace(-4, 4, 60))
        if model == "randomforest":
            return "rf", RandomForestRegressor(**self.rf_params)
        if model == "xgboost":
            if not _HAS_XGB:
                raise RuntimeError("XGBoost is not available in this environment.")
            return "xgb", XGBRegressor(**self.xgb_params)
        if model == "elasticnet":
            return "enet", ElasticNetCV(**self.enet_params)
        if model == "median":
            return "median", DummyRegressor(strategy="median")
        raise ValueError(
            "model must be one of: "
            "['Ridge', 'RandomForest', 'XGBoost', 'ElasticNet', 'Median']"
        )

    def _make_baseline_feature_preprocessor(
        self,
        features: Optional[list] = None,
        include_group: bool = True,
        include_year_trend: bool = True,
    ):
        """
        Preprocessor for baseline-plus-feature experiments.

        This intentionally ignores self.include_year because the experiment has a
        fixed reference specification: country fixed effects plus a linear year
        trend, optionally augmented with one or more candidate features.
        """
        feats = [] if features is None else features.copy()

        categorical = []
        numeric = []

        if include_group:
            categorical.append(self.group_col)
        if include_year_trend:
            numeric.append(self.year_col)

        for col in feats:
            if col in [self.target_col, self.group_col, self.year_col]:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric.append(col)
            else:
                categorical.append(col)

        transformers = []
        if numeric:
            transformers.append(
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", self._get_scaler()),
                ]), numeric)
            )

        if categorical:
            transformers.append(
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ]), categorical)
            )

        if not transformers:
            raise ValueError("At least one baseline control or feature is required.")

        return ColumnTransformer(transformers=transformers)

    def _make_baseline_feature_pipeline(
        self,
        model: str,
        features: Optional[list] = None,
        include_group: bool = True,
        include_year_trend: bool = True,
    ) -> Pipeline:
        pre = self._make_baseline_feature_preprocessor(
            features=features,
            include_group=include_group,
            include_year_trend=include_year_trend,
        )
        est_name, est = self._make_estimator(model)
        return Pipeline([("pre", pre), (est_name, est)])

    def _prediction_metrics(self, y_true_log, y_pred_log) -> Dict[str, float]:
        return {
            "mae_log": mean_absolute_error(y_true_log, y_pred_log),
            "rmse_log": self._rmse_log_space(y_true_log, y_pred_log),
            "r2_log": r2_score(y_true_log, y_pred_log),
            "mae_level": self._mae_level_space(y_true_log, y_pred_log),
        }

    def _residual_variance_reduction(
        self,
        y_true_log,
        baseline_pred_log,
        feature_pred_log,
    ) -> float:
        baseline_sse = float(np.sum((np.asarray(y_true_log) - baseline_pred_log) ** 2))
        feature_sse = float(np.sum((np.asarray(y_true_log) - feature_pred_log) ** 2))
        if baseline_sse == 0:
            return np.nan
        return 1.0 - (feature_sse / baseline_sse)

    def _fitted_estimator_summary(self, pipe: Pipeline, model: str) -> Dict[str, Any]:
        model_key = model.strip().lower()
        step_name = {
            "ridge": "ridge",
            "elasticnet": "enet",
            "randomforest": "rf",
            "xgboost": "xgb",
            "median": "median",
        }.get(model_key)
        if step_name is None or step_name not in pipe.named_steps:
            return {}

        est = pipe.named_steps[step_name]
        summary: Dict[str, Any] = {}
        if hasattr(est, "alpha_"):
            summary["selected_alpha"] = float(est.alpha_)
        if hasattr(est, "l1_ratio_"):
            summary["selected_l1_ratio"] = float(est.l1_ratio_)
        if hasattr(est, "n_iter_"):
            summary["n_iter"] = est.n_iter_
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_)
            summary["n_nonzero_coef"] = int(np.sum(np.abs(coef) > 1e-12))
        return summary

    def single_feature_baseline_experiment(
        self,
        features: Optional[Iterable[str]] = None,
        model: str = "Ridge",
        include_group: bool = True,
        include_year_trend: bool = True,
        dropna: bool = True,
        sort_by: str = "test_mae_level_improvement",
        ascending: bool = False,
        plot: bool = False,
        top_n: int = 25,
    ) -> pd.DataFrame:
        """
        Compare a fixed baseline against baseline + one feature at a time.

        Baseline specification:
            target ~ country fixed effects + linear year trend

        Feature specification for each candidate:
            target ~ country fixed effects + linear year trend + feature

        This is useful for screening whether a new feature adds predictive signal
        beyond the panel's country/time structure. It is not a causal test.

        Parameters
        ----------
        features : iterable of str or None
            Candidate features to test. Defaults to self.feature_cols.
        model : str, default="Ridge"
            Estimator to use. Supports "Ridge", "ElasticNet", "RandomForest",
            "XGBoost", and "Median". Ridge is the recommended default for this
            single-feature diagnostic because it is stable and fast.
        include_group : bool, default=True
            Include country fixed effects via one-hot encoded group_col.
        include_year_trend : bool, default=True
            Include year_col as a numeric linear trend.
        dropna : bool, default=True
            If True, fit and score each feature on rows where that feature and
            the target are non-missing. The baseline is refit on the same rows,
            so feature comparisons are sample-consistent.
        sort_by : str, default="test_mae_level_improvement"
            Result column used to sort the returned table.
        ascending : bool, default=False
            Sort direction.
        plot : bool, default=False
            If True, plot the top_n features by sort_by.
        top_n : int, default=25
            Number of rows to show in the optional plot.

        Returns
        -------
        pd.DataFrame
            One row per feature with baseline metrics, baseline+feature metrics,
            deltas, and partial R2 relative to baseline residual variance.
        """
        candidate_features = list(self.feature_cols if features is None else features)
        candidate_features = [
            col for col in candidate_features
            if col not in [self.target_col, self.group_col, self.year_col]
        ]
        if len(candidate_features) == 0:
            raise ValueError("No candidate features provided.")

        rows = []
        for feature in candidate_features:
            if feature not in self.df.columns:
                raise KeyError(f"Feature not found in df: {feature}")

            train_df = self.X_train.copy()
            test_df = self.X_test.copy()

            if dropna:
                required = [self.target_col, feature]
                train_df = train_df.dropna(subset=required)
                test_df = test_df.dropna(subset=required)
            else:
                train_df = train_df.dropna(subset=[self.target_col])
                test_df = test_df.dropna(subset=[self.target_col])

            if train_df.empty or test_df.empty:
                rows.append({
                    "feature": feature,
                    "model": model,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "error": "empty train or test split after filtering",
                })
                continue

            y_train = train_df[self.target_col]
            y_test = test_df[self.target_col]

            baseline_pipe = self._make_baseline_feature_pipeline(
                model=model,
                features=[],
                include_group=include_group,
                include_year_trend=include_year_trend,
            )
            feature_pipe = self._make_baseline_feature_pipeline(
                model=model,
                features=[feature],
                include_group=include_group,
                include_year_trend=include_year_trend,
            )

            baseline_pipe.fit(train_df, y_train)
            feature_pipe.fit(train_df, y_train)

            baseline_train_pred = baseline_pipe.predict(train_df)
            baseline_test_pred = baseline_pipe.predict(test_df)
            feature_train_pred = feature_pipe.predict(train_df)
            feature_test_pred = feature_pipe.predict(test_df)

            baseline_train_metrics = self._prediction_metrics(y_train, baseline_train_pred)
            baseline_test_metrics = self._prediction_metrics(y_test, baseline_test_pred)
            feature_train_metrics = self._prediction_metrics(y_train, feature_train_pred)
            feature_test_metrics = self._prediction_metrics(y_test, feature_test_pred)

            row: Dict[str, Any] = {
                "feature": feature,
                "feature_dtype": str(self.df[feature].dtype),
                "model": model,
                "include_group": include_group,
                "include_year_trend": include_year_trend,
                "dropna": dropna,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "train_year_min": int(train_df[self.year_col].min()),
                "train_year_max": int(train_df[self.year_col].max()),
                "test_year_min": int(test_df[self.year_col].min()),
                "test_year_max": int(test_df[self.year_col].max()),
                "feature_train_missing": int(self.X_train[feature].isna().sum()),
                "feature_test_missing": int(self.X_test[feature].isna().sum()),
                "partial_train_r2_vs_baseline": self._residual_variance_reduction(
                    y_train,
                    baseline_train_pred,
                    feature_train_pred,
                ),
                "partial_test_r2_vs_baseline": self._residual_variance_reduction(
                    y_test,
                    baseline_test_pred,
                    feature_test_pred,
                ),
            }

            for metric_name, metric_value in baseline_train_metrics.items():
                row[f"baseline_train_{metric_name}"] = metric_value
            for metric_name, metric_value in feature_train_metrics.items():
                row[f"feature_train_{metric_name}"] = metric_value
                row[f"delta_train_{metric_name}"] = (
                    metric_value - baseline_train_metrics[metric_name]
                )
            for metric_name, metric_value in baseline_test_metrics.items():
                row[f"baseline_test_{metric_name}"] = metric_value
            for metric_name, metric_value in feature_test_metrics.items():
                row[f"feature_test_{metric_name}"] = metric_value
                row[f"delta_test_{metric_name}"] = (
                    metric_value - baseline_test_metrics[metric_name]
                )

            row["test_mae_level_improvement"] = (
                row["baseline_test_mae_level"] - row["feature_test_mae_level"]
            )
            row["test_mae_log_improvement"] = (
                row["baseline_test_mae_log"] - row["feature_test_mae_log"]
            )
            row["test_r2_log_improvement"] = (
                row["feature_test_r2_log"] - row["baseline_test_r2_log"]
            )
            row.update(self._fitted_estimator_summary(feature_pipe, model))
            rows.append(row)

        df_res = pd.DataFrame(rows)
        if sort_by in df_res.columns:
            df_res = df_res.sort_values(sort_by, ascending=ascending)
        df_res = df_res.reset_index(drop=True)

        if plot:
            if sort_by not in df_res.columns:
                raise ValueError(f"Cannot plot because sort_by={sort_by!r} is not in results.")

            plot_df = df_res.head(top_n).sort_values(sort_by, ascending=True)
            plt.figure(figsize=(10, max(4, 0.35 * len(plot_df))))
            plt.barh(plot_df["feature"], plot_df[sort_by])
            plt.xlabel(sort_by)
            plt.ylabel("Feature")
            plt.title(f"{model}: baseline + one feature vs baseline")
            plt.tight_layout()
            plt.show()

            elbow_df = (
                df_res
                .dropna(subset=[sort_by])
                .sort_values(sort_by, ascending=True)
                .reset_index(drop=True)
            )
            elbow_df["feature_rank"] = np.arange(1, len(elbow_df) + 1)

            plt.figure(figsize=(9, 5))
            plt.plot(
                elbow_df["feature_rank"],
                elbow_df[sort_by],
                marker="o",
                linewidth=1.5,
            )
            if elbow_df[sort_by].min() < 0 < elbow_df[sort_by].max():
                plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.xlabel(f"Feature rank sorted by {sort_by}")
            plt.ylabel(sort_by)
            plt.title(f"{model}: one-feature screen elbow plot")
            plt.tight_layout()
            plt.show()

        return df_res
    
    def pca_experiment(
        self,
        model: str = "ElasticNet",
        n_components: Optional[int] = None,
        plot: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Run PCA ablation: fit with all PCs, then drop one PC at a time
        starting from the lowest-variance component, and track holdout MAE (level).
        """

        model_l = model.strip().lower()
        include_group = self.include_group_enet if model_l == "elasticnet" else False
        pre_tmp = self._make_preprocessor(
            include_group=(include_group and model_l != "elasticnet")
        )
        if model_l == "elasticnet" and include_group:
            pre_tmp = self._make_preprocessor(include_group=False)

        X_train_pre = pre_tmp.fit_transform(self.X_train)
        X_train_pre = self._to_dense(X_train_pre)

        max_components = min(X_train_pre.shape[0], X_train_pre.shape[1])
        if n_components is None:
            n_components = max_components
        else:
            n_components = int(n_components)
            n_components = min(n_components, max_components)
        if n_components < 1:
            raise ValueError("n_components must be >= 1 after preprocessing.")

        results = []
        for k in range(n_components, 0, -1):
            if model_l == "elasticnet":
                pipe = self._make_enet_pca_pipeline(n_components=k, random_state=random_state)
            else:
                pre = self._make_preprocessor(include_group=include_group)
                est_name, est = self._make_estimator(model)
                pipe = Pipeline(
                    [
                        ("pre", pre),
                        ("to_dense", FunctionTransformer(self._to_dense, accept_sparse=True)),
                        ("pca", PCA(n_components=k, random_state=random_state)),
                        (est_name, est),
                    ]
                )

            pipe.fit(self.X_train, self.y_train)
            preds = pipe.predict(self.X_test)
            mae_level = self._mae_level_space(self.y_test, preds)

            results.append(
                {
                    "n_components": k,
                    "removed_components": n_components - k,
                    "test_mae_level": mae_level,
                }
            )

        df_res = (
            pd.DataFrame(results)
            .sort_values("n_components", ascending=False)
            .reset_index(drop=True)
        )

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(
                df_res["removed_components"],
                df_res["test_mae_level"],
                marker="o",
            )
            plt.xlabel("PCs removed (lowest variance first)")
            plt.ylabel("Holdout MAE (level)")
            plt.title(f"{model}: PCA ablation on holdout set")
            plt.tight_layout()
            plt.show()

        return df_res

    def enet_l1_ratio_experiment(
        self,
        l1_ratios: Iterable[float],
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Fit ElasticNet across different l1_ratio values and evaluate holdout MAE (level).

        Parameters
        ----------
        l1_ratios : iterable of float
            l1_ratio values to evaluate.
        plot : bool, default=True
            If True, plot holdout MAE (level) against l1_ratio.

        Returns
        -------
        pd.DataFrame
            Columns:
            - l1_ratio
            - test_mae_level
            - selected_alpha
            - n_iter
        """
        ratios = [float(v) for v in l1_ratios]
        if len(ratios) == 0:
            raise ValueError("l1_ratios must contain at least one value.")

        results = []
        for ratio in ratios:
            pipe = clone(self.pipe_enet)
            pipe.set_params(enet__l1_ratio=ratio)
            pipe.fit(self.X_train, self.y_train)

            preds = pipe.predict(self.X_test)
            mae_level = self._mae_level_space(self.y_test, preds)

            enet_est = pipe.named_steps["enet"]
            results.append(
                {
                    "l1_ratio": ratio,
                    "test_mae_level": mae_level,
                    "selected_alpha": float(enet_est.alpha_) if hasattr(enet_est, "alpha_") else np.nan,
                    "n_iter": enet_est.n_iter_ if hasattr(enet_est, "n_iter_") else np.nan,
                }
            )

        df_res = pd.DataFrame(results).sort_values("l1_ratio").reset_index(drop=True)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(df_res["l1_ratio"], df_res["test_mae_level"], marker="o")
            plt.xlabel("ElasticNet l1_ratio")
            plt.ylabel("Holdout MAE (level)")
            plt.title("ElasticNet l1_ratio sweep on holdout set")
            plt.tight_layout()
            plt.show()

        return df_res

    def feature_ablation_experiment(
        self,
        model: str = "ElasticNet",
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Ablation over numeric feature_cols only:
        - always keeps country dummies (group_col) in the model
        - removes one numeric feature at a time
        - removal order follows feature_cols order from the end to the start
        - reports holdout MAE in level space
        """
        model_l = model.strip().lower()
        base_features = self.feature_cols.copy()
        numeric_removal_order = [
            c for c in base_features if self.df[c].dtype != "object"
        ][::-1]

        removed = []
        results = []

        for step in range(len(numeric_removal_order) + 1):
            current_features = [c for c in base_features if c not in removed]

            if model_l == "elasticnet":
                pipe = Pipeline(
                    [
                        ("pre", self._make_preprocessor(include_group=True, features=current_features)),
                        ("enet", ElasticNetCV(**self.enet_params)),
                    ]
                )
            elif model_l == "median":
                pipe = Pipeline([("median", DummyRegressor(strategy="median"))])
            else:
                est_name, est = self._make_estimator(model)
                pipe = Pipeline(
                    [
                        ("pre", self._make_preprocessor(include_group=True, features=current_features)),
                        (est_name, est),
                    ]
                )

            pipe.fit(self.X_train, self.y_train)
            preds = pipe.predict(self.X_test)
            mae_level = self._mae_level_space(self.y_test, preds)

            results.append(
                {
                    "removed_feature": removed[-1] if removed else None,
                    "n_removed": len(removed),
                    "n_features_remaining": len(current_features),
                    "test_mae_level": mae_level,
                }
            )

            if step < len(numeric_removal_order):
                removed.append(numeric_removal_order[step])

        df_res = pd.DataFrame(results)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(df_res["n_removed"], df_res["test_mae_level"], marker="o")
            plt.xlabel("Numeric features removed (last to first in feature_cols)")
            plt.ylabel("Holdout MAE (level)")
            plt.title(f"{model}: Numeric feature ablation with country dummies retained")
            plt.tight_layout()
            plt.show()

        return df_res

        
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
    
    def get_enet_coef_table(
        self,
        back_project_pca: bool = False,
        only_group_dummies: bool = False,
    ):
        coefs = self.pipe_enet.named_steps["enet"].coef_
        using_pca = self.use_pca_enet and (
            "pca" in self.pipe_enet.named_steps or "pre_split" in self.pipe_enet.named_steps
        )
        using_split_pca = using_pca and "pre_split" in self.pipe_enet.named_steps

        # With PCA enabled, ElasticNet coefficients are on principal components.
        # Optionally back-project to the preprocessed feature space.
        if using_pca and not back_project_pca:
            if using_split_pca:
                pre_split = self.pipe_enet.named_steps["pre_split"]
                pca_branch = dict(pre_split.transformer_list)["pca_features"]
                group_branch = dict(pre_split.transformer_list)["group_dummies"]

                n_pcs = pca_branch.named_steps["pca"].n_components_
                pc_names = np.array([f"pc_{i + 1}" for i in range(n_pcs)])
                group_names = group_branch.get_feature_names_out()
                feature_names = np.concatenate([pc_names, group_names])
            else:
                feature_names = np.array([f"pc_{i + 1}" for i in range(len(coefs))])

            coef_table = (
                pd.DataFrame({
                    "feature": feature_names,
                    "coef": coefs,
                    "abs_coef": np.abs(coefs)
                })
                .sort_values("abs_coef", ascending=False)
            )
            coef_table["type"] = coef_table["feature"].apply(
                lambda x: "pc" if str(x).startswith("pc_")
                else (x.split("__", 1)[0] if "__" in str(x) else "unknown")
            )
            coef_table["clean_feature_name"] = coef_table["feature"].apply(
                lambda x: str(x) if str(x).startswith("pc_")
                else (x.split("__", 1)[1] if "__" in str(x) else str(x))
            )
        else:
            if using_split_pca and back_project_pca:
                pre_split = self.pipe_enet.named_steps["pre_split"]
                pca_branch = dict(pre_split.transformer_list)["pca_features"]
                group_branch = dict(pre_split.transformer_list)["group_dummies"]

                pca = pca_branch.named_steps["pca"]
                pre_no_group = pca_branch.named_steps["pre_no_group"]
                base_feature_names = pre_no_group.get_feature_names_out()
                group_feature_names = group_branch.get_feature_names_out()

                n_pcs = pca.n_components_
                beta_pc = coefs[:n_pcs]
                beta_group = coefs[n_pcs:]
                beta_base = pca.components_.T @ beta_pc

                feature_names = np.concatenate([base_feature_names, group_feature_names])
                coefs = np.concatenate([beta_base, beta_group])
            else:
                feature_names = self.pipe_enet.named_steps["pre"].get_feature_names_out()
                if using_pca and back_project_pca:
                    pca = self.pipe_enet.named_steps["pca"]
                    coefs = pca.components_.T @ coefs

            if len(feature_names) != len(coefs):
                raise RuntimeError(
                    f"Mismatch between feature names ({len(feature_names)}) "
                    f"and ElasticNet coefficients ({len(coefs)})."
                )

            coef_table = (
                pd.DataFrame({
                    "feature": feature_names,
                    "coef": coefs,
                    "abs_coef": np.abs(coefs)
                })
                .sort_values("abs_coef", ascending=False)
            )
            coef_table["type"] = coef_table["feature"].apply(
                lambda x: x.split("__", 1)[0] if "__" in x else "unknown"
            )
            coef_table["clean_feature_name"] = coef_table["feature"].apply(
                lambda x: x.split("__", 1)[1] if "__" in x else x
            )

        if only_group_dummies:
            coef_table = coef_table[
                (coef_table["type"] == "cat")
                & (coef_table["clean_feature_name"].str.startswith(f"{self.group_col}_"))
            ].copy()

        coef_table = coef_table[["clean_feature_name", "coef", "abs_coef", "type"]]
        return coef_table
    

    def plot_top_enet_regressors(self, n=20, figsize=(8, 8)):

        # Get coef table
        coef_table = self.get_enet_coef_table()    
        
        # Filter to only numeric cols
        coef_table_only_num = coef_table[coef_table.type=="num"]

        
        top_n = (
            coef_table_only_num
            .sort_values("abs_coef", ascending=True)
            .tail(n)   # keep only n most important
        )

        plt.figure(figsize=figsize)

        plt.barh(
            top_n["clean_feature_name"],
            top_n["abs_coef"]
        )

        plt.xlabel("Absolute Regression Coefficient")
        plt.ylabel("Feature")

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
