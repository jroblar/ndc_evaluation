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
from statsmodels.tsa.stattools import adfuller, kpss


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
    def plot_correlation_matrix(df, iso_code=None):
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
        if iso_code is None:
            country_df = df
            country_name = "All Countries"
        else:
            # Filter the dataframe for the given ISO code
            country_df = df[df['iso_alpha_3'] == iso_code]
            country_name = iso_code
        
        # Select only numeric columns
        numeric_cols = country_df.select_dtypes(include=['float64', 'int64'])
        
        # Compute the correlation matrix
        correlation_matrix = numeric_cols.corr()
        
        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"Correlation Matrix for {country_name}")
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
    
    @staticmethod
    def get_oecd_iso_codes():
        """
        Returns a list of ISO Alpha-3 codes for OECD countries.
        """
        oecd_iso_codes = [
            "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CZE", "DNK", "EST", "FIN",
            "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR",
            "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK",
            "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA"
        ]
        return oecd_iso_codes
    
    @staticmethod
    def find_missing_oecd_countries(input_list, oecd_list=get_oecd_iso_codes()):
        """
        Compares two lists of ISO alpha-3 codes to find missing OECD countries.

        Parameters:
            input_list (list of str): The list of countries you have.
            oecd_list (list of str): The reference list of OECD countries.

        Returns:
            set: A set of ISO alpha-3 codes that are in oecd_list but not in input_list.
        """
        input_set = set(input_list)
        oecd_set = set(oecd_list)
        missing = oecd_set - input_set
        return missing
    @staticmethod
    def plot_column_distributions(df, columns, bins=30):
        """
        Plots histograms for the specified columns in the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            columns (list of str): List of column names to plot.
            bins (int, optional): Number of bins for the histograms. Defaults to 30.

        Returns:
            None: Displays the histograms.
        """
        n = len(columns)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(columns):
            if col in df.columns:
                sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frequency")
            else:
                axes[i].set_visible(False)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_boxplot_by_class(df, continuous_var, class_var, fig_size=(10, 6)):
        """
        Plots boxplots of a continuous variable grouped by a multiclass variable.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            continuous_var (str): The name of the continuous variable (y-axis).
            class_var (str): The name of the multiclass variable (x-axis).

        Returns:
            None: Displays the boxplot.
        """
        plt.figure(figsize=fig_size)
        sns.boxplot(x=class_var, y=continuous_var, data=df)
        plt.title(f"Boxplot of {continuous_var} by {class_var}")
        plt.xlabel(class_var)
        plt.ylabel(continuous_var)
        plt.grid(axis='y')
        plt.tight_layout()
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

        # Plot SHAP Beeswarm and Feature Importance side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # SHAP Beeswarm (plot_size=None disables SHAP's internal resizing)
        plt.sca(axes[0])
        shap.plots.beeswarm(shap_values, show=False, plot_size=None)
        axes[0].set_title(f"SHAP Beeswarm - {country}")

        # Feature Importance Plot
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        importances.plot(kind='barh', ax=axes[1])
        axes[1].set_title(f"Feature Importances - {country}")
        axes[1].set_xlabel("Importance")

        plt.tight_layout()
        plt.show()

        # SHAP scatter for selected feature (separate plot)
        if plot_feature:
            plt.figure(figsize=(8, 6))
            shap.plots.scatter(shap_values[:, plot_feature], show=False)
            plt.title(f"SHAP Scatter for {plot_feature} - {country}")
            plt.tight_layout()
            plt.show()


class EmissionsDataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.country_info_df = None
        self.interpretability_key = {}

    def generate_lags(self, columns, max_lag=3):
        """
        Generates lagged features for specified columns in the DataFrame.

        For each column in `columns`, creates new columns with lagged values up to `max_lag` periods,
        grouped by the 'iso_alpha_3' column. The new columns are named in the format '{column}_lag{lag}'.

        Parameters:
            columns (list of str): List of column names to generate lagged features for.
            max_lag (int, optional): Maximum number of lag periods to generate. Defaults to 3.

        Returns:
            self: Returns the instance with updated DataFrame containing lagged features.
        """
        for col in columns:
            for lag in range(1, max_lag + 1):
                self.df[f"{col}_lag{lag}"] = self.df.groupby("iso_alpha_3")[col].shift(lag)
        return self

    def log_transform(self, columns):
        """
        Applies a natural logarithm transformation to specified columns in the DataFrame.

        For each column in `columns`, creates a new column prefixed with 'log_' containing the natural log of the original values.
        Raises a ValueError if any value in a column is non-positive, as the logarithm is undefined for such values.
        Also updates the `interpretability_key` dictionary to map the new column name to a human-readable description.

        Parameters:
            columns (list of str): List of column names in the DataFrame to apply the log transformation to.

        Returns:
            self: The instance with updated DataFrame and interpretability key.

        Raises:
            ValueError: If any value in the specified columns is less than or equal to zero.
        """
        for col in columns:
            if (self.df[col] <= 0).any():
                raise ValueError(f"Column {col} contains non-positive values, cannot log transform.")
            new_col = f"log_{col}"
            self.df[new_col] = np.log(self.df[col])
            self.interpretability_key[new_col] = f"Natural log of {col}"
        return self

    def difference_columns(self, columns, log_first=False):
        """
        Calculates the year-on-year difference or log-difference for specified columns, grouped by 'iso_alpha_3'.

        Parameters:
            columns (list of str): List of column names to compute differences for.
            log_first (bool, optional): If True, computes the difference of the logarithm of each column (log-difference).
                                        If False, computes the simple difference. Default is False.

        Returns:
            self: Returns the instance with new difference columns added to self.df and interpretability_key updated.

        Notes:
            - For each column, a new column is added to self.df:
                - If log_first is True: 'dlog_{col}' = difference of log values (approximate annual % growth).
                - If log_first is False: 'diff_{col}' = simple year-on-year difference.
            - Updates self.interpretability_key with a human-readable explanation for each new column.
        """
        for col in columns:
            target_col = f"log_{col}" if log_first else col
            new_col = f"{'dlog_' if log_first else 'diff_'}{col}"
            self.df[new_col] = self.df.groupby("iso_alpha_3")[target_col].diff()
            if log_first:
                self.interpretability_key[new_col] = f"Δ ln({col}) ≈ annual % growth"
            else:
                self.interpretability_key[new_col] = f"Δ {col} = year-on-year change"
        return self

    def load_country_fixed_info(self, fixed_info_df: pd.DataFrame):
        self.country_info_df = fixed_info_df.copy()
        return self

    def merge_country_fixed_info(self):
        if self.country_info_df is None:
            raise ValueError("Country-level fixed information not loaded.")
        self.df = pd.merge(self.df, self.country_info_df, on="iso_alpha_3", how="left")
        return self

    def plot_series(self, variable, countries, transformed=True):
        """
        Plots time series data for a specified variable across multiple countries.

        Parameters:
            variable (str): The name of the variable/column to plot.
            countries (list of str): List of country ISO alpha-3 codes to plot data for.
            transformed (bool, optional): If True, plots each transformation of the variable
                ('raw', 'log', 'diff', 'dlog') in its own subplot. If False, plots only the raw variable.
                Defaults to True.

        Returns:
            self: Returns the instance of the class for method chaining.
        """
        for country in countries:
            subset = self.df[self.df["iso_alpha_3"] == country]
            if transformed:
                kinds = ['raw', 'log', 'diff', 'dlog', 'diff_dlog']
                available_cols = []
                for kind in kinds:
                    col = f"{kind}_{variable}" if kind != 'raw' else variable
                    if col in subset.columns:
                        available_cols.append((kind, col))

                n = len(available_cols)
                if n == 0:
                    continue

                fig, axs = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)
                if n == 1:
                    axs = [axs]  # wrap single axis in list

                for ax, (kind, col) in zip(axs, available_cols):
                    ax.plot(subset['year'], subset[col])
                    ax.set_title(f"{kind.upper()} of {variable} - {country}")
                    ax.set_ylabel(kind)
                    ax.grid(True)
                axs[-1].set_xlabel("Year")
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(10, 5))
                plt.plot(subset['year'], subset[variable], label='raw')
                plt.title(f"{variable} - {country}")
                plt.xlabel("Year")
                plt.ylabel(variable)
                plt.grid(True)
                plt.legend()
                plt.show()
        return self

    def adf_kpss_test(self, variable, countries, transformed=False):
        """
        Performs Augmented Dickey-Fuller (ADF) and KPSS stationarity tests on a specified variable,
        optionally across its transformed versions.

        Parameters:
            variable (str): The name of the base column/variable to test for stationarity.
            countries (list of str): List of country ISO alpha-3 codes to include in the test.
            transformed (bool): If True, runs tests on all available transformed versions of the variable.
                                If False, runs only on the raw variable.

        Returns:
            pd.DataFrame: Multi-indexed (country, transformation) DataFrame containing:
                - "ADF Statistic"
                - "ADF p-value"
                - "KPSS Statistic"
                - "KPSS p-value"
                - "passed_ADF"  (True if ADF p < 0.05)
                - "passed_KPSS" (True if KPSS p ≥ 0.05)

        Notes:
            - Skips countries or transformations with entirely NaN data.
            - If KPSS test fails, assigns NaN to its results.
        """
        results = []

        transformations = {
            'raw': variable,
            'log': f'log_{variable}',
            'diff': f'diff_{variable}',
            'dlog': f'dlog_{variable}',
            'diff_dlog': f'diff_dlog_{variable}'
        }

        if not transformed:
            transformations = {'raw': variable}

        for country in countries:
            subset = self.df[self.df["iso_alpha_3"] == country]

            for label, col in transformations.items():
                if col not in subset.columns or subset[col].isna().all():
                    continue

                series = subset[col].dropna()

                try:
                    adf_result = adfuller(series)
                    adf_stat, adf_p = adf_result[0], adf_result[1]
                except Exception:
                    adf_stat, adf_p = np.nan, np.nan

                try:
                    kpss_result = kpss(series, nlags='auto')
                    kpss_stat, kpss_p = kpss_result[0], kpss_result[1]
                except Exception:
                    kpss_stat, kpss_p = np.nan, np.nan

                passed_adf = adf_p < 0.05 if not np.isnan(adf_p) else np.nan
                passed_kpss = kpss_p >= 0.05 if not np.isnan(kpss_p) else np.nan

                results.append({
                    'country': country,
                    'transformation': label,
                    'ADF Statistic': adf_stat,
                    'ADF p-value': adf_p,
                    'KPSS Statistic': kpss_stat,
                    'KPSS p-value': kpss_p,
                    'passed_ADF': passed_adf,
                    'passed_KPSS': passed_kpss
                })

        return pd.DataFrame(results).set_index(['country', 'transformation'])
    
    def adf_kpss_test_summary(self, variable, countries):
        """
        Produces a summary DataFrame of ADF and KPSS p-values and pass/fail flags
        for multiple transformations of a specified variable across countries.

        Parameters:
            variable (str): Base variable name (e.g., 'gdp_2015_usd').
            countries (list of str): List of ISO alpha-3 codes to process.

        Returns:
            pd.DataFrame: One row per country with columns:
                - ADF p-value [raw, log, dlog, diff_dlog]
                - KPSS p-value [raw, log, dlog, diff_dlog]
                - passed_ADF_*
                - passed_KPSS_*
        """
        result_rows = []

        # Define column names for transformations
        col_map = {
            'raw': variable,
            'log': f'log_{variable}',
            'dlog': f'dlog_{variable}',
            'diff': f'diff_{variable}',
            'diff_dlog': f'diff_dlog_{variable}'
        }

        for country in countries:
            row = {'iso_alpha_3': country}
            subset = self.df[self.df["iso_alpha_3"] == country]

            for label, col in col_map.items():
                adf_p, kpss_p = np.nan, np.nan

                if col in subset.columns and not subset[col].isna().all():
                    series = subset[col].dropna()

                    # ADF
                    try:
                        adf_p = adfuller(series)[1]
                    except Exception:
                        pass

                    # KPSS
                    try:
                        kpss_p = kpss(series, nlags='auto')[1]
                    except Exception:
                        pass

                # Add p-values
                row[f'{variable}_adf_p-value {label}'] = adf_p
                row[f'{variable}_kpss_p-value {label}'] = kpss_p

                # Add pass/fail flags
                row[f'{variable}_passed_adf_{label}'] = adf_p < 0.05 if not np.isnan(adf_p) else np.nan
                row[f'{variable}_passed_kpss_{label}'] = kpss_p >= 0.05 if not np.isnan(kpss_p) else np.nan

            result_rows.append(row)

        return pd.DataFrame(result_rows)
    
    def summarize_stationarity_agreement(self, summary_df, variable):
        """
        Computes the percentage of countries where both ADF and KPSS tests
        agree on stationarity for each transformation.

        Parameters:
            summary_df (pd.DataFrame): Output of adf_kpss_test_summary()

        Returns:
            pd.Series: Percentage of agreement per transformation (raw, log, dlog, diff_dlog)
        """
        transformations = ['raw', 'log', 'dlog', 'diff', 'diff_dlog']
        agreement_percentages = {}

        for t in transformations:
            adf_col = f'{variable}_passed_adf_{t}'
            kpss_col = f'{variable}_passed_kpss_{t}'

            valid = summary_df[[adf_col, kpss_col]].dropna()
            if valid.empty:
                agreement_percentages[t] = np.nan
                continue

            agreed = (valid[adf_col] & valid[kpss_col]).sum()
            total = len(valid)
            agreement_percentages[t] = 100 * agreed / total

        return pd.Series(agreement_percentages, name="stationarity_agreement_percent")


    def get_interpretability_key(self):
        return self.interpretability_key.copy()

    def get_processed_data(self):
        return self.df.copy()

class ModelEvaluationUtils:


    @staticmethod
    def plot_actual_vs_predicted_country(test_df, X_test, y_test, best_estimators, country='USA', model_name='rf'):
        """
        Plot actual vs. predicted dlog_total_emissions for a specific country and model.

        Parameters:
            test_df (pd.DataFrame): The test set with 'iso_alpha_3' and 'year' columns.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target values.
            best_estimators (dict): Dict of trained models, e.g. {'rf': ..., 'gb': ..., 'mlp': ...}
            country (str): ISO alpha-3 code for the country to plot.
            model_name (str): Key for the model to use ('rf', 'gb', or 'mlp').
        """
        mask = test_df['iso_alpha_3'] == country
        X_country = X_test[mask]
        years     = test_df.loc[mask, 'year']
        actual    = y_test[mask]
        model = best_estimators[model_name]
        predicted = model.predict(X_country)

        comp = (
            pd.DataFrame({
                'year':      years,
                'actual':    actual,
                'predicted': predicted
            })
            .sort_values('year')
            .reset_index(drop=True)
        )

        print(comp)

        plt.figure(figsize=(8, 4))
        plt.plot(comp['year'], comp['actual'],    marker='o', label='Actual')
        plt.plot(comp['year'], comp['predicted'], marker='x', label='Predicted')
        plt.title(f"dlog_total_emissions: actual vs. predicted for {country}")
        plt.xlabel("Year")
        plt.ylabel("dlog_total_emissions")
        plt.legend()
        plt.tight_layout()
        plt.show()

