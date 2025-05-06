import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.ensemble import RandomForestRegressor
import logging


# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PolicyEDA:
    """
    Class for performing exploratory data analysis (EDA) on policy data.
    """


    # Function to calculate the correlation, plot the relation, and add a regression line
    @staticmethod
    def plot_correlation(df, col1, col2, iso3=None):
        """
        Plots a scatter plot with a regression line to visualize the correlation 
        between two columns in a DataFrame. Optionally filters the data by a 
        specified ISO3 country code.
        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            col1 (str): The name of the first column to be plotted on the x-axis.
            col2 (str): The name of the second column to be plotted on the y-axis.
            iso3 (str, optional): The ISO3 country code to filter the data. 
                                  If None, data for all countries is used. Defaults to None.
        Returns:
            None: Displays the scatter plot with a regression line and correlation value.
        """
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
        """
        Plots correlation scatter plots for multiple ISO3 country codes.

        This function generates a grid of scatter plots showing the relationship
        between two specified columns (`col1` and `col2`) for each country in the
        provided list of ISO3 country codes (`iso3_list`). Each subplot includes
        a regression line and displays the correlation coefficient.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
                               Must include columns `iso_alpha_3`, `col1`, and `col2`.
            col1 (str): The name of the first column to be used in the scatter plot.
            col2 (str): The name of the second column to be used in the scatter plot.
            iso3_list (list of str): A list of ISO3 country codes to filter the data
                                     and generate individual plots for each.

        Returns:
            None: The function displays the plots but does not return any value.

        Notes:
            - If no data is available for a specific ISO3 code, the corresponding
              subplot will display a message indicating "No data for <ISO3>".
            - Any unused subplots in the grid will be hidden.
            - The function uses Seaborn's `regplot` to generate scatter plots with
              regression lines.

        Example:
            plot_correlation_multiple(df, 'gdp', 'co2_emissions', ['USA', 'CHN', 'IND'])
        """
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
        """
        Plots a correlation matrix heatmap for numeric columns in a DataFrame filtered by a specific ISO country code.
        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            iso_code (str): The ISO Alpha-3 country code to filter the DataFrame.
        Returns:
            None: Displays a heatmap of the correlation matrix for the filtered data.
        Notes:
            - The function filters the DataFrame to include only rows corresponding to the specified ISO Alpha-3 code.
            - Only numeric columns (float64 and int64) are considered for the correlation matrix.
            - The heatmap is displayed using Seaborn with annotations and a "coolwarm" colormap.
        """
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
        """
        Calculate the correlation coefficient between two columns for each country in a given list of ISO3 country codes.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
                Must include a column named 'iso_alpha_3' and the columns specified by `col1` and `col2`.
            col1 (str): The name of the first column to calculate the correlation.
            col2 (str): The name of the second column to calculate the correlation.
            iso3_list (list): A list of ISO3 country codes to filter the data by.
        Returns:
            pd.DataFrame: A DataFrame containing the ISO3 country codes and their corresponding correlation coefficients.
                The DataFrame has two columns:
                    - 'iso_alpha_3': The ISO3 country code.
                    - 'correlation': The correlation coefficient between `col1` and `col2` for the country.
                      If the filtered data for a country is empty, the correlation will be None.
        """
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
        """
        Plots the emissions trajectory of a specific country over time.
        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing emissions data. It must include the columns 
            'iso_alpha_3', 'year', and 'total_emissions'.
        iso_code : str
            The ISO Alpha-3 code of the country whose emissions trajectory is to be plotted.
        Returns:
        --------
        None
            Displays a line plot showing the total emissions of the specified country over time.
        Notes:
        ------
        - The function uses seaborn and matplotlib for visualization.
        - Ensure that the DataFrame contains the required columns and that the `iso_code` exists in the data.
        - The x-axis represents the year, and the y-axis represents the total emissions.
        """
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
        """
        Generates a pairplot for the specified columns in a DataFrame, optionally filtered by a specific ISO Alpha-3 country code.
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame containing the data to be visualized.
        cols : list of str
            A list of column names to include in the pairplot.
        iso3 : str, optional
            An ISO Alpha-3 country code to filter the DataFrame by. If None, the pairplot will include data for all countries.
        Returns:
        --------
        None
            Displays the pairplot using seaborn and matplotlib.
        Notes:
        ------
        - The function filters the DataFrame by the 'iso_alpha_3' column if `iso3` is provided.
        - The plot includes a title indicating whether it is for a specific country or all countries.
        """
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
    def filter_high_collinear_features(df, features, collinearity_threshold=0.9, verbose=False):
        """
        Filters out highly collinear features from a given set of features based on a specified collinearity threshold.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            features (list of str): A list of feature names to evaluate for collinearity.
            collinearity_threshold (float, optional): The threshold above which features are considered collinear. 
                                                      Defaults to 0.9.
            verbose (bool, optional): If True, logs the features dropped and the reduced feature set. Defaults to False.

        Returns:
            list of str: A list of features with high collinearity removed.
        """
        corr_matrix = df[features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > collinearity_threshold)]
        reduced_features = [f for f in features if f not in to_drop]
        if verbose:
            logger.info(f"Features dropped due to collinearity: {to_drop}")
            logger.info(f"Reduced feature set: {reduced_features}")
        return reduced_features

    @staticmethod
    def _build_pipeline(regressor):
        """
        Builds a machine learning pipeline with a standard scaler and a specified regressor.

        Args:
            regressor: An instance of a scikit-learn regressor to be used in the pipeline.

        Returns:
            Pipeline: A scikit-learn Pipeline object with a StandardScaler followed by the specified regressor.
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])

    @staticmethod
    def _evaluate_model(pipe, X_test, y_test, feature_names, verbose=False):
        """
        Evaluate a machine learning model pipeline on test data.

        This function computes the root mean squared error (RMSE), R-squared (R²) score, 
        and feature coefficients for a given model pipeline. Optionally, it logs the 
        evaluation metrics and feature coefficients.

        Args:
            pipe (Pipeline): A scikit-learn pipeline object containing a trained model.
            X_test (pd.DataFrame or np.ndarray): Test feature data.
            y_test (pd.Series or np.ndarray): True target values for the test data.
            feature_names (list of str): List of feature names corresponding to the columns in X_test.
            verbose (bool, optional): If True, logs the RMSE, R² score, and feature coefficients. 
                                      Defaults to False.

        Returns:
            tuple: A tuple containing:
                - rmse (float): Root mean squared error of the model on the test data.
                - r2 (float): R-squared score of the model on the test data.
                - coefs (pd.Series): Feature coefficients of the model, indexed by feature names.
        """
        y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        coefs = pd.Series(pipe.named_steps['regressor'].coef_, index=feature_names)
        if verbose:
            logger.info(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
            logger.info("\nCoefficients:\n%s", coefs.sort_values(ascending=False).to_string())
        return rmse, r2, coefs

    @staticmethod
    def train_model(df, features, target, regressor, plot_pdp_feature=None, test_size=0.2, random_state=42, verbose=False):
        """
        Trains a regression model using the provided dataset and evaluates its performance.

        Args:
            df (pd.DataFrame): The input dataframe containing the dataset.
            features (list): A list of column names to be used as features for training.
            target (str): The column name of the target variable.
            regressor (sklearn.base.RegressorMixin): The regression model to be trained.
            plot_pdp_feature (str, optional): The feature for which to plot the Partial Dependence Plot (PDP). Defaults to None.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.
            verbose (bool, optional): If True, logs additional information during training and evaluation. Defaults to False.

        Returns:
            dict: A dictionary containing the following keys:
                - 'rmse' (float): The Root Mean Squared Error of the model on the test set.
                - 'r2' (float): The R-squared score of the model on the test set.
                - 'coefs' (dict): The coefficients of the trained model (if applicable).
                - 'model' (Pipeline): The trained regression model pipeline.
        """
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        pipe = RegressionUtils._build_pipeline(regressor)
        pipe.fit(X_train, y_train)

        if verbose:
            logger.info(f"\nModel: {regressor.__class__.__name__}")
        rmse, r2, coefs = RegressionUtils._evaluate_model(pipe, X_test, y_test, features, verbose=verbose)

        if plot_pdp_feature:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                pipe, X_train, features=[plot_pdp_feature], ax=ax
            )
            plt.title(f"Partial Dependence of {plot_pdp_feature}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        results = {
            'rmse': rmse,
            'r2': r2,
            'coefs': coefs,
            'model': pipe
        }

        return results

class ShapUtils:

    @classmethod
    def analyze_feature_importance_with_shap(cls, df, features, target, country, plot_feature=None, n_estimators=100, random_state=42):
        """
        Trains a RandomForestRegressor and visualizes SHAP values for feature importance.

        Parameters:
            df (pd.DataFrame): Dataset
            features (list): List of feature column names
            target (str): Name of target column
            country (str): Country name for plot titles
            plot_feature (str, optional): If specified, a SHAP scatter plot for this feature will be shown
            n_estimators (int): Number of trees in the forest
            random_state (int): Seed for reproducibility
        """
        X = df[features]
        y = df[target]

        try:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            model.fit(X, y)

            # SHAP values
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

        except Exception as e:
            logger.error(f"Error for country {country}: {e}")
            return

        # SHAP Beeswarm
        print(f"SHAP Beeswarm (Global Feature Importance) - {country}")
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"SHAP Beeswarm - {country}")
        plt.tight_layout()
        plt.show()

        # Feature Importance Plot
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        importances.plot(kind='barh', title=f"Feature Importances - {country}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

        # SHAP scatter for selected feature
        if plot_feature:
            print(f"\nSHAP Scatter for feature: {plot_feature} - {country}")
            shap.plots.scatter(shap_values[:, plot_feature], show=False)
            plt.title(f"SHAP Scatter for {plot_feature} - {country}")
            plt.tight_layout()
            plt.show()
