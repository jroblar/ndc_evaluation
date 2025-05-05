import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import shap
from sklearn.ensemble import RandomForestRegressor

from statsmodels.stats.outliers_influence import variance_inflation_factor


class PolicyEDA:
    """
    Class for performing exploratory data analysis (EDA) on policy data.
    """


    # Function to calculate the correlation, plot the relation, and add a regression line
    @staticmethod
    def plot_correlation(df, col1, col2, iso3=None):
        if iso3:
            df = df[df['iso3'] == iso3]
            
        # Calculate the correlation
        correlation = df[col1].corr(df[col2])
            
        # Plotting the relation
        sns.regplot(x=col1, y=col2, data=df, ci=None, line_kws={"color": "red"})
        plt.title(f"Scatter plot of {col1} vs {col2} for {iso3 if iso3 else 'all countries'}\nCorrelation: {correlation:.2f}")
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
    
    # Function to calculate the correlation, plot the relation, and add regression lines for multiple countries
    @staticmethod
    def plot_correlation_multiple(df, col1, col2, iso3_list):
        n = len(iso3_list)
        ncols = 3  # Number of columns in the subplot grid
        nrows = (n + ncols - 1) // ncols  # Calculate required number of rows

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()  # Flatten in case of a multi-row, multi-column grid

        for i, iso3 in enumerate(iso3_list):
            ax = axes[i]
            df_filtered = df[df['iso_alpha_3'] == iso3]

            if df_filtered.empty:
                ax.set_title(f"No data for {iso3}")
                ax.axis('off')
                continue

            correlation = df_filtered[col1].corr(df_filtered[col2])
            sns.regplot(x=col1, y=col2, data=df_filtered, ci=None, line_kws={"color": "red"}, ax=ax)
            ax.set_title(f"{iso3}\nCorrelation: {correlation:.2f}")
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_matrix(df, iso_code):
        # Filter the dataframe for the given ISO code
        country_df = df[df['iso_alpha_3'] == iso_code]
        
        # Select only numeric columns
        numeric_cols = country_df.select_dtypes(include=['float64', 'int64'])
        
        # Compute the correlation matrix
        correlation_matrix = numeric_cols.corr()
        
        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"Correlation Matrix for {iso_code}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def calculate_corr_coef_for_each_country(df, col1, col2, iso3_list):
        correlation_results = []
        for iso3 in iso3_list:
            df_filtered = df[df['iso_alpha_3'] == iso3]
            if not df_filtered.empty:
                correlation = df_filtered[col1].corr(df_filtered[col2])
            else:
                correlation = None
            correlation_results.append({'iso_alpha_3': iso3, 'correlation': correlation})
        
        return pd.DataFrame(correlation_results)
    
    @staticmethod
    def plot_country_emissions_trajectory(df, iso_code):
        country_df = df[df['iso_alpha_3'] == iso_code]
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=country_df, x='year', y='total_emissions', marker='o')
        plt.title(f"Total Emissions Over Time for {iso_code}")
        plt.xlabel("Year")
        plt.ylabel("Total Emissions")
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_pairplot(df, cols, iso3=None):
        if iso3:
            df = df[df['iso_alpha_3'] == iso3]
        
        # Create a pairplot
        sns.pairplot(df[cols])
        plt.suptitle(f"Pairplot for {iso3 if iso3 else 'all countries'}", y=1.02)
        plt.show()
    

class RegressionUtils:
    """
    Utilities for regression modeling and analysis.
    """

    @staticmethod
    def filter_high_collinear_features(df, features, collinearity_threshold=0.9):
        corr_matrix = df[features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > collinearity_threshold)]
        reduced_features = [f for f in features if f not in to_drop]
        print(f"Features dropped due to collinearity: {to_drop}")
        print(f"Reduced feature set: {reduced_features}")
        return reduced_features

    @staticmethod
    def _build_pipeline(regressor):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])

    @staticmethod
    def _evaluate_model(pipe, X_test, y_test, feature_names):
        y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
        coefs = pd.Series(pipe.named_steps['regressor'].coef_, index=feature_names)
        print("\nCoefficients:")
        print(coefs.sort_values(ascending=False))
        return rmse, r2, coefs

    @staticmethod
    def train_model(df, features, target, regressor, plot_pdp_feature=None, test_size=0.2, random_state=42):
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        pipe = RegressionUtils._build_pipeline(regressor)
        pipe.fit(X_train, y_train)

        print(f"\nModel: {regressor.__class__.__name__}")
        rmse, r2, coefs = RegressionUtils._evaluate_model(pipe, X_test, y_test, features)

        if plot_pdp_feature:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                pipe, X_train, features=[plot_pdp_feature], ax=ax
            )
            plt.title(f"Partial Dependence of {plot_pdp_feature}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return pipe, coefs

class RegressionUtils:

    @classmethod
    def analyze_feature_importance_with_shap(cls, df, features, target, plot_feature=None, n_estimators=100, random_state=42):
        """
        Trains a RandomForestRegressor and visualizes SHAP values for feature importance.

        Parameters:
            df (pd.DataFrame): Dataset
            features (list): List of feature column names
            target (str): Name of target column
            plot_feature (str, optional): If specified, a SHAP scatter plot for this feature will be shown
            n_estimators (int): Number of trees in the forest
            random_state (int): Seed for reproducibility
        """
        X = df[features]
        y = df[target]

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        model.fit(X, y)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        print("SHAP Beeswarm (global feature importance):")
        shap.plots.beeswarm(shap_values)

        if plot_feature:
            print(f"\nSHAP Scatter for feature: {plot_feature}")
            shap.plots.scatter(shap_values[:, plot_feature])