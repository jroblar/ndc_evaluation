import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.stats import qmc
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


class EmissionsRFModel:
    """
    A reusable Random Forest model for emission forecasting with panel and time-series cross-validation.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed DataFrame containing features, target, group, and year columns.
    feature_cols : list of str
        Names of feature columns to use for training.
    target_col : str, default='log_total_emissions'
        Name of the target column.
    group_col : str, default='iso_alpha_3'
        Name of the column defining panel groups (e.g., country codes).
    year_col : str, default='year'
        Name of the column containing the year for time splits.
    rf_params : dict, optional
        Parameters to pass to RandomForestRegressor.
    """

    def __init__(
        self, df, feature_cols,
        target_col='log_total_emissions',
        group_col='iso_alpha_3',
        year_col='year',
        rf_params=None
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.group_col = group_col
        self.year_col = year_col
        self.rf_params = rf_params or {'n_estimators': 100, 'random_state': 42}
        self.model = None

    def split_data(self, holdout_years=5):
        """
        Split the data into train/test by holding out the last `holdout_years` years.

        Parameters
        ----------
        holdout_years : int
            Number of most recent years to hold out for testing.
        """
        max_year = self.df[self.year_col].max()
        cutoff = max_year - holdout_years

        train_mask = self.df[self.year_col] <= cutoff
        test_mask  = self.df[self.year_col] > cutoff

        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        groups = self.df[self.group_col]

        self.X_train = X[train_mask]
        self.y_train = y[train_mask]
        self.groups_train = groups[train_mask]

        self.X_test = X[test_mask]
        self.y_test = y[test_mask]

        return self.X_train, self.X_test, self.y_train, self.y_test, self.groups_train

    def train(self):
        """
        Train the RandomForestRegressor on the training split.
        """
        self.model = RandomForestRegressor(**self.rf_params)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """
        Evaluate the trained model on the test split.

        Returns
        -------
        dict
            Dictionary containing 'mse' and 'r2' on the test set.
        """
        preds = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)
        r2  = r2_score(self.y_test, preds)
        print(f"Test MSE: {mse:.3f}, R2: {r2:.3f}")
        return {'mse': mse, 'r2': r2}

    def cross_validate_panel(self, n_splits=5):
        """
        Perform panel cross-validation using GroupKFold.

        Parameters
        ----------
        n_splits : int
            Number of folds for GroupKFold.

        Returns
        -------
        dict
            Average 'mse' and 'r2' across folds.
        """
        gkf = GroupKFold(n_splits=n_splits)
        mses, r2s = [], []

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(self.X_train, self.y_train, self.groups_train)
        ):
            X_tr, X_val = (
                self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            )
            y_tr, y_val = (
                self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            )
            model = RandomForestRegressor(**self.rf_params)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            mses.append(mean_squared_error(y_val, preds))
            r2s.append(r2_score(y_val, preds))
            print(
                f"Fold {fold+1} - MSE: {mses[-1]:.3f}, R2: {r2s[-1]:.3f}"
            )

        avg_mse = np.mean(mses)
        avg_r2  = np.mean(r2s)
        print(f"CV avg MSE: {avg_mse:.3f}, R2: {avg_r2:.3f}")
        return {'mse': avg_mse, 'r2': avg_r2}

    def cross_validate_time_series(self, n_splits=5):
        """
        Perform time-series cross-validation using TimeSeriesSplit.

        Parameters
        ----------
        n_splits : int
            Number of folds for TimeSeriesSplit.

        Returns
        -------
        dict
            Average 'mse' and 'r2' across folds.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mses, r2s = [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_tr, X_val = (
                self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            )
            y_tr, y_val = (
                self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            )
            model = RandomForestRegressor(**self.rf_params)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            mses.append(mean_squared_error(y_val, preds))
            r2s.append(r2_score(y_val, preds))
            print(
                f"Time-series CV Fold {fold+1} - MSE: {mses[-1]:.3f}, R2: {r2s[-1]:.3f}"
            )

        avg_mse = np.mean(mses)
        avg_r2  = np.mean(r2s)
        print(f"Time-series CV avg MSE: {avg_mse:.3f}, R2: {avg_r2:.3f}")
        return {'mse': avg_mse, 'r2': avg_r2}

    def plot_feature_importances(self):
        """
        Plot the feature importances of the trained model.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Random Forest Feature Importances")
        plt.bar(
            range(len(self.feature_cols)),
            importances[indices],
            align='center'
        )
        plt.xticks(
            range(len(self.feature_cols)),
            [self.feature_cols[i] for i in indices],
            rotation=90
        )
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()

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
        Generate an ensemble of future feature trajectories by:
        1) simulating `n_scenarios` per‐feature futures via ETS or ARIMA,
        2) combining those feature‐level simulations into joint scenarios
            using Latin Hypercube Sampling (LHS).

        Parameters
        ----------
        df : pd.DataFrame
            Historical data with ['iso_alpha_3','year'] + feature_cols.
        feature_cols : list[str]
            List of feature names to simulate (exclude emissions).
        start_year : int
            First forecast year (e.g. 2023).
        end_year : int
            Last forecast year (e.g. 2030).
        n_scenarios : int
            Number of joint futures to generate per country.
        method : {"ets","arima"}
            ETS = Holt–Winters additive trend; ARIMA = bootstrap via `.simulate()`.
        arima_order : tuple[int,int,int]
            (p,d,q) for ARIMA if method="arima".
        random_state : int or None
            Seed for reproducible sampling.

        Returns
        -------
        pd.DataFrame
            Long‐form DataFrame with columns
            ['iso_alpha_3','future_id','year'] + feature_cols.
        """
        rng = np.random.default_rng(random_state)
        years = np.arange(start_year, end_year + 1)
        horizon = len(years)
        rows: list[dict] = []

        for iso, grp in df.groupby("iso_alpha_3"):
            grp = grp.sort_values("year")

            # 1) Simulate n_scenarios per feature
            sims_dict: dict[str, np.ndarray] = {}
            for feat in feature_cols:
                hist = grp.set_index("year")[feat].copy()
                hist.index = pd.PeriodIndex(hist.index, freq="Y")

                if method == "ets":
                    ets_model = ExponentialSmoothing(
                        hist,
                        trend="add",
                        seasonal=None,
                        initialization_method="estimated"
                    ).fit()
                    fitted = ets_model.fittedvalues
                    resid = hist.values - fitted.values
                    base_forecast = ets_model.forecast(horizon).values

                    # bootstrap residuals → sims shape (n_scenarios, horizon)
                    sims = np.vstack([
                        base_forecast + rng.choice(resid, size=horizon, replace=True)
                        for _ in range(n_scenarios)
                    ])

                elif method == "arima":
                    arima_model = ARIMA(hist, order=arima_order).fit()
                    sims = np.vstack([
                        arima_model.simulate(
                            nsimulations=horizon,
                            anchor="end"
                        ).values
                        for _ in range(n_scenarios)
                    ])

                else:
                    raise ValueError("method must be 'ets' or 'arima'")

                sims_dict[feat] = sims

            # 2) Build an LHS sampler for feature‐combination indices
            sampler = qmc.LatinHypercube(d=len(feature_cols), seed=random_state)
            u = sampler.random(n=n_scenarios)  # shape = (n_scenarios, n_features)
            # convert uniform [0,1) → integer indices [0, n_scenarios-1]
            idx = (u * n_scenarios).astype(int)

            # 3) Assemble joint scenarios
            for s in range(n_scenarios):
                future_id = f"id_{iso}_{s+1}"
                for t, year in enumerate(years):
                    row = {
                        "iso_alpha_3": iso,
                        "future_id": future_id,
                        "year": year
                    }
                    # pick the t-th year value from the selected sim for each feature
                    for j, feat in enumerate(feature_cols):
                        sim_index = idx[s, j]
                        row[feat] = sims_dict[feat][sim_index, t]
                    rows.append(row)

        ensemble_df = pd.DataFrame(rows)
        cols = ["iso_alpha_3", "future_id", "year"] + feature_cols
        return ensemble_df[cols]


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