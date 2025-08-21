import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, GridSearchCV, cross_val_score, train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from scipy.stats import qmc
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima import auto_arima
from typing import Dict, Iterable, Optional, Union

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,          
    module=r"sklearn"                # just sklearn
)

class RegressionAnalysis:
    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 group_col: str,
                 year_col: str,
                 feature_cols: list = None,
                 holdout_years: int = 5,
                 xgb_params: dict = None,
                 rf_params: dict = None,
                 scaler_type: str = 'standard',
                 use_group_feature: bool = False):
        """
        Regression analysis class, now with XGBoost, optional group_col as feature,
        one-hot encoding for all categoricals, and year_col always numeric.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.group_col = group_col
        self.year_col = year_col
        self.holdout_years = holdout_years
        self.scaler_type = scaler_type.lower()
        self.use_group_feature = use_group_feature

        # Random Forest parameters
        self.rf_params = rf_params or dict(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)

        # XGBoost parameters
        self.xgb_params = xgb_params or dict(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    
        # Always treat year_col as numeric feature
        non_feats = {self.group_col, self.year_col, self.target_col}
        if feature_cols is None:
            base_feats = [c for c in self.df.columns if c not in non_feats]
        else:
            base_feats = feature_cols.copy()
        # Ensure year_col included as numeric feature
        if self.year_col not in base_feats:
            base_feats.append(self.year_col)
        # Optionally include group_col as a feature
        if self.use_group_feature and self.group_col not in base_feats:
            base_feats.append(self.group_col)
        self.feature_cols = base_feats

        # Determine which features are categorical (object or category dtype, or forced by OneHotEncoder)
        self.categorical_features = [
            col for col in self.feature_cols
            if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category'
            or (self.use_group_feature and col == self.group_col)
        ]
        # year_col is always numeric (ensure not in categoricals)
        if self.year_col in self.categorical_features:
            self.categorical_features.remove(self.year_col)
        self.numeric_features = [
            col for col in self.feature_cols if col not in self.categorical_features
        ]

        # Split data
        years = self.df[self.year_col]
        cutoff = years.max() - self.holdout_years
        train_mask = years <= cutoff
        self.X_train = self.df.loc[train_mask, self.feature_cols]
        self.X_test  = self.df.loc[~train_mask, self.feature_cols]
        self.y_train = self.df.loc[train_mask, self.target_col]
        self.y_test  = self.df.loc[~train_mask, self.target_col]
        self.groups_train = self.df.loc[train_mask, self.group_col]

        # Build transformers and pipelines
        self._build_transformers()
        self._build_pipelines()

    def _get_scaler(self):
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

    def _build_transformers(self):
        # OneHot for categoricals, scale numeric
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self._get_scaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ]
        )

    def _build_pipelines(self):
        # ElasticNet
        self.pipe_enet = Pipeline([
            ('pre', self.preprocessor),
            ('enet', ElasticNetCV(l1_ratio=[.2, .5, .8], cv=5, random_state=42, n_jobs=-1, max_iter=10000))
        ])
        # Random Forest
        self.pipe_rf = Pipeline([
            ('pre', self.preprocessor),
            ('rf', RandomForestRegressor(**self.rf_params))
        ])
    

        # XGBoost
        self.pipe_xgb = Pipeline([
            ('pre', self.preprocessor),
            ('xgb', XGBRegressor(**self.xgb_params))
        ])

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
        # Extract fitted estimator from pipeline
        if isinstance(pipe, GridSearchCV):
            estimator = pipe.best_estimator_.named_steps[step]
        else:
            estimator = pipe.named_steps[step]
        importances = estimator.feature_importances_
        # After ColumnTransformer and OneHot, feature names change:
        feature_names = []
        if hasattr(self.preprocessor, 'transformers_'):
            # Get feature names after transform
            num_feats = self.numeric_features
            cat_encoder = self.preprocessor.named_transformers_['cat']
            cat_feats = list(cat_encoder.get_feature_names_out(self.categorical_features))
            feature_names = num_feats + cat_feats
        else:
            # fallback
            feature_names = self.numeric_features + self.categorical_features
        indices = np.argsort(importances)[::-1]
        labels = [feature_names[i] for i in indices]
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
            self.plot_feature_importances(model=importances_model, top_n=15)
        return None

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