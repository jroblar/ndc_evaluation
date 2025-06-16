import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import pandas as pd


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

class EmissionsProjectionDataBuilder:
    def __init__(self, df: pd.DataFrame, base_year: int = 2022, projection_years: list = list(range(2023, 2031))):
        self.df = df.copy()
        self.base_year = base_year
        self.projection_years = projection_years
        self.gdp_growth = {}
        self.pop_growth = {}
        self.proj_df = None

    def compute_growth_rates(self, value_col: str) -> dict:
        growth_rates = {}
        for country, group in self.df.groupby('iso_alpha_3'):
            g = group.dropna(subset=[value_col])
            if len(g) >= 3:
                X = g[['year']].values
                y = np.log(g[value_col].values)
                model = LinearRegression().fit(X, y)
                growth_rates[country] = model.coef_[0]  # log-point growth rate
            else:
                growth_rates[country] = 0.0
        return growth_rates

    def estimate_growth(self):
        self.gdp_growth = self.compute_growth_rates("gdp_2015_usd")
        self.pop_growth = self.compute_growth_rates("population")

    def project_variable(self, colname: str, growth_dict: dict) -> pd.DataFrame:
        projections = []
        latest = self.df[self.df["year"] == self.base_year][["iso_alpha_3", colname]]
        for _, row in latest.iterrows():
            iso = row["iso_alpha_3"]
            val = row[colname]
            growth = growth_dict.get(iso, 0)
            for year in self.projection_years:
                years_ahead = year - self.base_year
                projected_val = val * np.exp(growth * years_ahead)
                projections.append({
                    "iso_alpha_3": iso,
                    "year": year,
                    colname: projected_val
                })
        return pd.DataFrame(projections)

    def build_projection_dataset(self) -> pd.DataFrame:
        if not self.gdp_growth or not self.pop_growth:
            self.estimate_growth()

        gdp_proj = self.project_variable("gdp_2015_usd", self.gdp_growth)
        pop_proj = self.project_variable("population", self.pop_growth)

        df_proj = pd.merge(gdp_proj, pop_proj, on=["iso_alpha_3", "year"])

        # Get emission lag1 from base year
        latest_em = self.df[self.df["year"] == self.base_year][["iso_alpha_3", "total_emissions"]].copy()
        latest_em["log_total_emissions_lag1"] = np.log(latest_em["total_emissions"])
        latest_em = latest_em[["iso_alpha_3", "log_total_emissions_lag1"]]

        df_proj = pd.merge(df_proj, latest_em, on="iso_alpha_3", how="left")
        df_proj["log_total_emissions_lag2"] = df_proj["log_total_emissions_lag1"]  # Same for 1st iteration

        self.proj_df = df_proj
        return df_proj
