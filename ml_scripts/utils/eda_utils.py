import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EDAUtils:
    """
    Class to perform exploratory data analysis (EDA) on a DataFrame.
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
    def plot_correlation_matrix(df, iso_code=None, figsize=(12, 10)):
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
        plt.figure(figsize=figsize)
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
    def create_pairplot(df, iso3=None):
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

        # Select numeric columns for the pairplot
        cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if not cols:
            logger.warning("No numeric columns found for pairplot. Please check the DataFrame.")
            return
        
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

    @staticmethod
    def plot_numeric_fields_distributions(df):
        """
        Plots the distributions of all numeric fields in the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing numeric fields.

        Returns:
            None: Displays the distribution plots for each numeric field.
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns found in the DataFrame.")
            return

        n = len(numeric_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_barplot_for_categorical_field(df, categorical_field):
        """
        Plots a bar plot for the counts of each category in a specified categorical field.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            categorical_field (str): The name of the categorical field to plot.

        Returns:
            None: Displays the bar plot.
        """
        if categorical_field not in df.columns:
            logger.error(f"Column '{categorical_field}' does not exist in the DataFrame.")
            return

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=categorical_field, order=df[categorical_field].value_counts().index)
        plt.title(f"Bar Plot of {categorical_field}")
        plt.xlabel(categorical_field)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    
class DataCleaningUtils:

    @staticmethod
    def fill_numeric_missing_by_group(df, group_cols):
        """
        Forward and backward fills missing values for all numeric columns in the dataframe,
        grouped by the specified columns.

        Parameters:
            df (pd.DataFrame): The dataframe to fill.
            group_cols (list): List of columns to group by.

        Returns:
            pd.DataFrame: DataFrame with missing numeric values filled.
        """
        numeric_cols = df.select_dtypes(include='number').columns.difference(group_cols)
        df = df.sort_values(group_cols)
        df[numeric_cols] = (
            df.groupby(group_cols)[numeric_cols]
            .ffill().bfill()
        )
        return df
    
    @staticmethod
    def remove_high_vif_features(
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: list = ["year"],
        thresh: float = 5.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Removes features with high Variance Inflation Factor (VIF) iteratively,
        keeping only numeric columns, excluding year and the target variable.

        Parameters:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of the target column (excluded from VIF calc).
        exclude_cols (list): List of additional column names to exclude (e.g., ["year"]).
        thresh (float): VIF threshold above which a feature will be removed.
        verbose (bool): If True, prints VIF info at each iteration.

        Returns:
        pd.DataFrame: DataFrame with selected features and original excluded columns.
        """
        df = df.copy()

        # Identify numeric features excluding target and exclude_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = set(exclude_cols + [target_col])
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        vif_df = df[feature_cols].copy()

        while True:
            X = add_constant(vif_df)
            vif = pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns
            ).drop("const")

            max_vif = vif.max()
            if max_vif > thresh:
                max_vif_feature = vif.idxmax()
                if verbose:
                    print(f"Dropping '{max_vif_feature}' with VIF: {max_vif:.2f}")
                vif_df = vif_df.drop(columns=[max_vif_feature])
            else:
                if verbose:
                    print("All VIF values are below threshold.")
                break

        # Return original dataframe with reduced features
        return pd.concat([df[list(exclude_cols)], vif_df], axis=1)
    

class FeatureEngineering:


    @staticmethod
    def generate_lagged_features(df, columns, max_lag=3):
        """
        Generates lagged features for specified columns in the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            columns (list of str): List of column names to generate lagged features for.
            max_lag (int): Maximum number of lags to generate.

        Returns:
            pd.DataFrame: DataFrame with new lagged feature columns added.
        """
        df_with_lags = df.copy()
        
        for col in columns:
            for lag in range(1, max_lag + 1):
                df_with_lags[f"{col}_lag{lag}"] = df_with_lags.groupby("iso_alpha_3")[col].shift(lag)
        return df_with_lags
    
    @staticmethod
    def generate_growth_rate_features(df, columns):
        """
        Generates year-on-year growth rate features for specified columns in the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            columns (list of str): List of column names to generate growth rate features for.

        Returns:
            pd.DataFrame: DataFrame with new growth rate feature columns added.
        """
        df_with_growth_rates = df.copy()
        
        for col in columns:
            df_with_growth_rates[f"{col}_growth_rate"] = df_with_growth_rates.groupby("iso_alpha_3")[col].pct_change()
        return df_with_growth_rates
    

    @staticmethod
    def one_hot_encode_categorical(df, categorical_columns, drop_first=False, prefix_sep='_'):
        """
        Performs one-hot encoding on the specified categorical columns.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            categorical_columns (list of str): List of categorical column names to encode.
            drop_first (bool, optional): Whether to drop the first level to avoid multicollinearity. Defaults to False.
            prefix_sep (str, optional): Separator to use for new column names. Defaults to '_'.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns.
        """
        one_hot_encoded_df = df.copy()
        
        return pd.get_dummies(one_hot_encoded_df, columns=categorical_columns, drop_first=drop_first, prefix_sep=prefix_sep, dtype=int)
    

    @staticmethod
    def log_transform_high_skew(df, columns, skew_threshold=1.0, new_prefix='log_'):
        """
        Applies a log transformation to columns with high skewness and drops the original highly skewed columns.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            columns (list of str): List of column names to check for skewness and transform.
            skew_threshold (float, optional): Absolute skewness above which to apply log transform. Defaults to 1.0.
            new_prefix (str, optional): Prefix for new log-transformed columns. Defaults to 'log_'.

        Returns:
            pd.DataFrame: DataFrame with new log-transformed columns added and highly skewed original columns dropped.
        """
        df_out = df.copy()
        cols_to_drop = []
        for col in columns:
            if col not in df_out.columns:
                continue
            # Only consider columns with all non-negative values for log transform
            if (df_out[col] < 0).any():
                continue
            skewness = df_out[col].skew()
            if abs(skewness) > skew_threshold:
                # Use log1p to handle zeros (log1p(x) = log(1 + x))
                df_out[f"{new_prefix}{col}"] = np.log1p(df_out[col])
                cols_to_drop.append(col)

        print(f"Columns dropped due to high skewness: {cols_to_drop}")        
        if cols_to_drop:
            df_out = df_out.drop(columns=cols_to_drop)
        return df_out

