import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, GridSearchCV, cross_val_score, train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

from scipy.stats import qmc
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

class RegressionAnalysis:
    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 group_col: str,
                 year_col: str,
                 feature_cols: list = None,
                 holdout_years: int = 5,
                 tree_based_models_with_scaler: bool = True,
                 rf_params: dict = None,
                 rf_tune: bool = False,
                 rf_tune_params: dict = None,
                 xgb_params: dict = None,
                 xgb_tune: bool = False,
                 xgb_tune_params: dict = None,
                 scaler_type: str = 'standard'):
        """
        Initializes the regression analysis class.
        Adds XGBoost and configurable scaler.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.group_col = group_col
        self.year_col = year_col
        self.holdout_years = holdout_years
        self.rf_tune = rf_tune
        self.xgb_tune = xgb_tune
        self.scaler_type = scaler_type.lower()
        self.rf_params = rf_params or dict(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
        self.rf_tune_params = rf_tune_params or {
            'rf__n_estimators': [100, 200, 500],
            'rf__max_depth': [None, 10, 20]
        }
        self.xgb_params = xgb_params or dict(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        self.xgb_tune_params = xgb_tune_params or {
            'xgb__n_estimators': [100, 200, 500],
            'xgb__max_depth': [4, 6, 10],
            'xgb__learning_rate': [0.05, 0.1, 0.2]
        }
        # Infer features
        non_feats = {self.group_col, self.year_col, self.target_col}
        if feature_cols is None:
            self.feature_cols = [c for c in self.df.columns if c not in non_feats]
        else:
            self.feature_cols = feature_cols
        # Split data
        years = self.df[self.year_col]
        cutoff = years.max() - self.holdout_years
        train_mask = years <= cutoff
        self.X_train = self.df.loc[train_mask, self.feature_cols]
        self.X_test  = self.df.loc[~train_mask, self.feature_cols]
        self.y_train = self.df.loc[train_mask, self.target_col]
        self.y_test  = self.df.loc[~train_mask, self.target_col]
        self.groups_train = self.df.loc[train_mask, self.group_col]
        # Initialize pipelines
        self._build_pipelines(tree_based_models_with_scaler)

    def _get_scaler(self):
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

    def _build_pipelines(self, tree_based_models_with_scaler: bool = True):
        scaler = self._get_scaler()
        # ElasticNet always uses scaler
        self.pipe_enet = Pipeline([
            ('scaler', scaler),
            ('enet', ElasticNetCV(l1_ratio=[.2, .5, .8], cv=5, random_state=42, n_jobs=-1, max_iter=10000))
        ])
        # Random Forest: scaler optional
        if not tree_based_models_with_scaler:
            base_rf = Pipeline([('rf', RandomForestRegressor(**self.rf_params))])
        else:
            base_rf = Pipeline([('scaler', scaler), ('rf', RandomForestRegressor(**self.rf_params))])
        if self.rf_tune:
            self.pipe_rf = GridSearchCV(base_rf,
                                        param_grid=self.rf_tune_params,
                                        cv=GroupKFold(n_splits=5),
                                        scoring='neg_mean_absolute_error',
                                        n_jobs=-1)
        else:
            self.pipe_rf = base_rf
        # XGBoost: scaler optional
        if not tree_based_models_with_scaler:
            base_xgb = Pipeline([('xgb', XGBRegressor(**self.xgb_params))])
        else:
            base_xgb = Pipeline([('scaler', scaler), ('xgb', XGBRegressor(**self.xgb_params))])
        if self.xgb_tune:
            self.pipe_xgb = GridSearchCV(base_xgb,
                                         param_grid=self.xgb_tune_params,
                                         cv=GroupKFold(n_splits=5),
                                         scoring='neg_mean_absolute_error',
                                         n_jobs=-1)
        else:
            self.pipe_xgb = base_xgb
        # Median baseline
        self.pipe_med = Pipeline([('median', DummyRegressor(strategy='median'))])

    def cross_validate(self):
        """
        Perform GroupKFold and TimeSeriesSplit cross-validation on training data.
        Prints CV results in a formatted table, including MAE and R2.
        """
        results = {}
        gkf = GroupKFold(n_splits=5)
        tscv = TimeSeriesSplit(n_splits=5)
        for name, pipe in [('RandomForest', self.pipe_rf),
                           ('XGBoost', self.pipe_xgb),
                           ('ElasticNet', self.pipe_enet),
                           ('Median', self.pipe_med)]:
            scores_group_mae = -cross_val_score(pipe, self.X_train, self.y_train,
                                                groups=self.groups_train,
                                                cv=gkf,
                                                scoring='neg_mean_absolute_error',
                                                n_jobs=-1)
            scores_time_mae = -cross_val_score(pipe, self.X_train, self.y_train,
                                               cv=tscv,
                                               scoring='neg_mean_absolute_error',
                                               n_jobs=-1)
            scores_group_r2 = cross_val_score(pipe, self.X_train, self.y_train,
                                              groups=self.groups_train,
                                              cv=gkf,
                                              scoring='r2',
                                              n_jobs=-1)
            scores_time_r2 = cross_val_score(pipe, self.X_train, self.y_train,
                                             cv=tscv,
                                             scoring='r2',
                                             n_jobs=-1)
            results[name] = {
                'group_mae_mean': scores_group_mae.mean(),
                'group_mae_std': scores_group_mae.std(),
                'time_mae_mean': scores_time_mae.mean(),
                'time_mae_std': scores_time_mae.std(),
                'group_r2_mean': scores_group_r2.mean(),
                'group_r2_std': scores_group_r2.std(),
                'time_r2_mean': scores_time_r2.mean(),
                'time_r2_std': scores_time_r2.std()
            }
        print("\nCross-Validation Results:")
        print("-" * 120)
        print(f"{'Model':<15} {'Group MAE':>12} {'(std)':>8} {'Time MAE':>12} {'(std)':>8} "
              f"{'Group R2':>12} {'(std)':>8} {'Time R2':>12} {'(std)':>8}")
        print("-" * 120)
        for name, res in results.items():
            print(f"{name:<15} {res['group_mae_mean']:12.4f} {res['group_mae_std']:8.4f} "
                  f"{res['time_mae_mean']:12.4f} {res['time_mae_std']:8.4f} "
                  f"{res['group_r2_mean']:12.4f} {res['group_r2_std']:8.4f} "
                  f"{res['time_r2_mean']:12.4f} {res['time_r2_std']:8.4f}")
        return None

    def fit(self):
        """
        Fit all models on the training data.
        """
        self.pipe_rf.fit(self.X_train, self.y_train)
        self.pipe_xgb.fit(self.X_train, self.y_train)
        self.pipe_enet.fit(self.X_train, self.y_train)
        self.pipe_med.fit(self.X_train, self.y_train)
        return self

    def evaluate(self):
        """
        Evaluate train vs test to detect overfitting.
        Prints evaluation metrics in a formatted table.
        """
        evals = {}
        for name, pipe in [('RandomForest', self.pipe_rf),
                           ('XGBoost', self.pipe_xgb),
                           ('ElasticNet', self.pipe_enet),
                           ('Median', self.pipe_med)]:
            y_train_pred = pipe.predict(self.X_train)
            y_test_pred  = pipe.predict(self.X_test)
            evals[name] = {
                'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                'test_mae':  mean_absolute_error(self.y_test,  y_test_pred),
                'train_r2':  r2_score(self.y_train, y_train_pred),
                'test_r2':   r2_score(self.y_test,  y_test_pred)
            }
        print("\nEvaluation Results:")
        print("-" * 80)
        print(f"{'Model':<15} {'Train MAE':>10} {'Test MAE':>10} {'Train R2':>10} {'Test R2':>10}")
        print("-" * 80)
        for name, metrics in evals.items():
            print(f"{name:<15} {metrics['train_mae']:10.4f} {metrics['test_mae']:10.4f} {metrics['train_r2']:10.4f} {metrics['test_r2']:10.4f}")
        return None

    def plot_feature_importances(self, top_n: int = None, model: str = 'RandomForest'):
        """
        Plot feature importances from the Random Forest or XGBoost model.
        Set model='XGBoost' to plot XGBoost importances.
        """
        pipe = None
        if model.lower() == 'xgboost':
            pipe = self.pipe_xgb
            step = 'xgb'
            title = 'XGBoost Feature Importances'
        else:
            pipe = self.pipe_rf
            step = 'rf'
            title = 'Random Forest Feature Importances'
        # Extract model from pipeline
        if isinstance(pipe, GridSearchCV):
            estimator = pipe.best_estimator_.named_steps[step]
        else:
            estimator = pipe.named_steps[step]
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        labels = [self.feature_cols[i] for i in indices]
        if top_n:
            labels = labels[:top_n]
            importances = importances[indices][:top_n]
            indices = indices[:top_n]
        plt.figure(figsize=(8,6))
        plt.barh(range(len(importances)), importances[::-1])
        plt.yticks(range(len(importances)), labels[::-1])
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def run_all(self, plot_importances: bool = True, importances_model: str = 'RandomForest') -> dict:
        """
        Utility to run CV, fit, evaluate, and optionally plot.
        Returns a summary dict.
        """
        self.cross_validate()
        self.fit()
        self.evaluate()
        if plot_importances:
            self.plot_feature_importances(model=importances_model)
        return None

class EnsembleProjections:


    @staticmethod
    def generate_ensemble(
        df: pd.DataFrame,
        feature_cols: list[str],
        start_year: int,
        end_year: int,
        n_scenarios: int
    ) -> pd.DataFrame:
        """
        Build an ensemble of future feature trajectories via Latin Hypercube sampling
        of residuals around a per-country, per-feature linear trend.

        Parameters
        ----------
        df : pd.DataFrame
            Original historical data with columns
            ['iso_alpha_3', 'year'] + feature_cols.
        feature_cols : list[str]
            List of feature column names to simulate.
        start_year : int
            First year of your simulated horizon (e.g. 2023).
        end_year : int
            Last year of your simulated horizon (e.g. 2030).
        n_scenarios : int
            Number of future trajectories to generate per country.

        Returns
        -------
        pd.DataFrame
            Long‐form DataFrame with columns
            ['iso_alpha_3', 'future_id', 'year'] + feature_cols.
        """

        # prepare output accumulator
        rows: list[dict] = []

        # 1) for each country, fit a linear trend + capture residuals
        for iso, grp in df.groupby("iso_alpha_3"):
            # build and store per‐feature models + residuals
            models: dict[str, LinearRegression] = {}
            residuals: dict[str, np.ndarray] = {}

            X = grp["year"].to_numpy().reshape(-1, 1)
            for feat in feature_cols:
                y = grp[feat].to_numpy()
                lr = LinearRegression().fit(X, y)
                models[feat] = lr
                residuals[feat] = y - lr.predict(X)

            # 2) Latin Hypercube draws
            sampler = qmc.LatinHypercube(d=len(feature_cols))
            # draws shape = (n_scenarios, n_features), uniform in [0,1)
            u = sampler.random(n=n_scenarios)

            # 3) for each scenario, map uniform draws → residual quantiles → simulate
            for i in range(n_scenarios):
                future_id = f"id_{iso}_{i+1}"
                # map each feature's u_i → q‐quantile of residuals
                resid_q = {
                    feat: np.quantile(residuals[feat], u[i, j])
                    for j, feat in enumerate(feature_cols)
                }

                # simulate the full time series for this scenario
                for year in range(start_year, end_year + 1):
                    row = {
                        "iso_alpha_3": iso,
                        "future_id": future_id,
                        "year": year,
                    }
                    for feat in feature_cols:
                        base_trend = models[feat].predict([[year]])[0]
                        row[feat] = base_trend + resid_q[feat]
                    rows.append(row)

        # assemble into DataFrame
        ensemble_df = pd.DataFrame(rows)

        # ensure correct column order
        cols = ["iso_alpha_3", "future_id", "year"] + feature_cols
        return ensemble_df[cols]
    
    @staticmethod
    def generate_ensemble_ts_lhs(
        df: pd.DataFrame,
        feature_cols: list[str],
        start_year: int,
        end_year: int,
        n_scenarios: int,
        method: str = "ets",
        arima_order: tuple[int, int, int] = (1, 1, 1),
        random_state: int | None = None
    ) -> pd.DataFrame:
        
        """
        Generates an ensemble of future time series scenarios for multiple features using either
        Exponential Smoothing (ETS) or ARIMA models, and combines them using Latin Hypercube Sampling (LHS).
        For each country (identified by 'iso_alpha_3'), the function fits a time series model to each feature,
        simulates future values, and then uses LHS to combine these simulations into joint scenarios.
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
        method : str, default="ets"
            Time series modeling method to use for simulation. Must be either "ets" (Exponential Smoothing)
            or "arima".
        arima_order : tuple[int, int, int], default=(1, 1, 1)
            The (p, d, q) order of the ARIMA model, used only if method="arima".
        random_state : int or None, default=None
            Random seed for reproducibility.
        Returns
        -------
        pd.DataFrame
            DataFrame containing simulated future scenarios with columns:
            ['iso_alpha_3', 'future_id', 'year'] + feature_cols.
            Each row corresponds to a simulated value for a given country, scenario, year, and feature.
        Raises
        ------
        ValueError
            If `method` is not 'ets' or 'arima'.
        """

        rng = np.random.default_rng(random_state)
        years = np.arange(start_year, end_year + 1)
        horizon = len(years)
        rows = []

        for iso, grp in df.groupby("iso_alpha_3"):
            grp = grp.sort_values("year")

            # 1) Simulate per-feature futures
            sims_dict = {}
            for feat in feature_cols:
                hist = grp.set_index("year")[feat].copy()
                hist.index = pd.PeriodIndex(hist.index, freq="Y")

                if method == "ets":
                    # — ETS branch unchanged —
                    model = ExponentialSmoothing(
                        hist, trend="add", seasonal=None,
                        initialization_method="estimated"
                    ).fit()
                    fitted = model.fittedvalues
                    resid = hist.values - fitted.values
                    base = model.forecast(horizon).values
                    sims = np.vstack([
                        base + rng.choice(resid, size=horizon, replace=True)
                        for _ in range(n_scenarios)
                    ])
                
                elif method == "arima":
                    sar = SARIMAX(
                        hist,
                        order=arima_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
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

                    sims = np.vstack([
                        fit.simulate(nsimulations=horizon, anchor="end").values
                        for _ in range(n_scenarios)
                    ])

                else:
                    raise ValueError("method must be 'ets' or 'arima'")

                sims_dict[feat] = sims

            # 2) LHS to combine feature sims into joint futures
            sampler = qmc.LatinHypercube(d=len(feature_cols), seed=random_state)
            u = sampler.random(n=n_scenarios)
            idx = (u * n_scenarios).astype(int)

            # 3) build ensemble rows
            for s in range(n_scenarios):
                fid = f"id_{iso}_{s+1}"
                for t, year in enumerate(years):
                    row = {"iso_alpha_3": iso, "future_id": fid, "year": year}
                    for j, feat in enumerate(feature_cols):
                        sim_i = idx[s, j]
                        row[feat] = sims_dict[feat][sim_i, t]
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