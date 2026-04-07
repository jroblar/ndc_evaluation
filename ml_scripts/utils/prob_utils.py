from typing import Iterable, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ProbUtils:

    @staticmethod
    def ndc_summary(
        projected_df: pd.DataFrame,
        ndc_df: pd.DataFrame,
        year: int = 2030,
        value_col: str = "con_edgar_ghg_mt_hp_trend",
        cap_cols: Union[str, List[str]] = "Unconditional",
        iso_col: str = "iso_alpha_3",
        scenario_col: str = "future_id",
        year_col: str = "year",
        keep_cap_values: bool = False,
    ) -> pd.DataFrame:
        """
        Merge modeled futures with NDC caps and summarize, by ISO, the share of futures
        that meet each cap in a given target year.

        Parameters
        ----------
        projected_df : pd.DataFrame
            Must include columns: [iso_col, scenario_col, year_col, value_col]
        ndc_df : pd.DataFrame
            Must include columns: [iso_col] + cap_cols
            Where each cap column is a numeric cap/threshold for the given ISO.
        year : int
            Target year to evaluate.
        value_col : str
            Column in projected_df with modeled emissions/value to compare to cap.
        cap_cols : str | list[str]
            One or more cap columns in ndc_df (e.g., ["Unconditional", "Conditional"]).
        iso_col : str
            ISO3 column name in both DataFrames.
        scenario_col : str
            Scenario/future identifier column in projected_df (used to count unique futures).
        year_col : str
            Year column in projected_df.
        keep_cap_values : bool
            If True, also return the cap values per ISO in the output.

        Returns
        -------
        pd.DataFrame
            One row per ISO (and per cap if you keep the long format—here we return wide):
            - iso_col
            - meets_ndc__<cap_col> : share of futures meeting that cap
            - n_futures            : number of unique futures evaluated (same across caps, after filtering)
            Optionally:
            - cap__<cap_col>       : the (deduped) cap value per ISO
        """
        # Normalize cap_cols to list
        if isinstance(cap_cols, str):
            cap_cols = [cap_cols]
        cap_cols = list(cap_cols)

        # Basic column checks (fail fast with helpful message)
        needed_proj = {iso_col, scenario_col, year_col, value_col}
        missing_proj = needed_proj - set(projected_df.columns)
        if missing_proj:
            raise KeyError(f"projected_df missing columns: {sorted(missing_proj)}")

        needed_ndc = {iso_col, *cap_cols}
        missing_ndc = needed_ndc - set(ndc_df.columns)
        if missing_ndc:
            raise KeyError(f"ndc_df missing columns: {sorted(missing_ndc)}")

        # Deduplicate NDC caps per ISO (keep first)
        ndc_min = (
            ndc_df[[iso_col] + cap_cols]
            .drop_duplicates(subset=[iso_col], keep="first")
            .copy()
        )

        # Merge caps onto projections
        full = projected_df.merge(ndc_min, on=iso_col, how="left")

        # Filter to target year and require at least one cap present
        df_y = full.loc[full[year_col].eq(year)].copy()
        df_y[value_col] = pd.to_numeric(df_y[value_col], errors="coerce")

        # Coerce caps to numeric
        for c in cap_cols:
            df_y[c] = pd.to_numeric(df_y[c], errors="coerce")

        # If value is NaN, it can't be evaluated; if all caps NaN, drop row
        df_y = df_y.loc[df_y[value_col].notna()].copy()
        df_y = df_y.loc[df_y[cap_cols].notna().any(axis=1)].copy()

        # Count futures per ISO (after year/value filter)
        base_counts = (
            df_y.groupby(iso_col, as_index=False)
            .agg(n_futures=(scenario_col, "nunique"))
        )

        # Compute "meets" per cap and aggregate mean by ISO
        meets_parts = []
        for c in cap_cols:
            tmp = df_y.loc[df_y[c].notna(), [iso_col, scenario_col, value_col, c]].copy()
            tmp["meets"] = tmp[value_col] < tmp[c]
            agg = (
                tmp.groupby(iso_col, as_index=False)
                .agg(**{f"meets_ndc_{c}": ("meets", "mean")})
            )
            meets_parts.append(agg)

        # Merge all cap results together
        out = base_counts
        for part in meets_parts:
            out = out.merge(part, on=iso_col, how="left")

        # Optionally include cap values (deduped) as columns
        if keep_cap_values:
            cap_wide = ndc_min.rename(columns={c: f"cap_{c}" for c in cap_cols})
            out = out.merge(cap_wide, on=iso_col, how="left")

        return out
    
    @staticmethod
    def plot_ndc_meets_histograms(
        df: pd.DataFrame,
        iso_col: str = "iso_alpha_3",
        unconditional_col: str = "meets_ndc_unconditional_target",
        conditional_col: str = "meets_ndc_conditional_target",
        bins: int = 20,
        figsize=(10, 4),
    ):
        """
        Plot side-by-side histograms of NDC meet shares (unconditional vs conditional).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ISO rows and meets_ndc columns.
        iso_col : str
            ISO column (not used in plotting, just for validation).
        unconditional_col : str
            Column with unconditional meet shares (0–1).
        conditional_col : str
            Column with conditional meet shares (0–1).
        bins : int
            Number of histogram bins.
        figsize : tuple
            Figure size.
        """

        # Basic validation
        for col in [unconditional_col, conditional_col]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        data_uncond = df[unconditional_col].dropna()
        data_cond   = df[conditional_col].dropna()

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # Unconditional
        axes[0].hist(data_uncond, bins=bins)
        axes[0].set_title("Meets NDC – Unconditional")
        axes[0].set_xlabel("Share of futures meeting NDC")
        axes[0].set_ylabel("Number of countries")
        axes[0].set_xlim(0, 1)

        # Conditional
        axes[1].hist(data_cond, bins=bins)
        axes[1].set_title("Meets NDC – Conditional")
        axes[1].set_xlabel("Share of futures meeting NDC")
        axes[1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compute_2030_q1_mean(
        df: pd.DataFrame,
        value_col: str = "con_edgar_ghg_mt_hp_trend",
        year_col: str = "year",
        iso_col: str = "iso_alpha_3",
        target_year: int = 2030,
    ):
        """
        For each country, compute the mean of the lowest quartile (Q1)
        of 2030 emissions across futures.
        """

        # Filter to 2030
        df_2030 = df[df[year_col] == target_year].copy()

        # Ensure numeric
        df_2030[value_col] = pd.to_numeric(df_2030[value_col], errors="coerce")

        def q1_mean(x):
            q1_cutoff = np.quantile(x, 0.25)
            return x[x <= q1_cutoff].mean()

        out = (
            df_2030
            .groupby(iso_col)[value_col]
            .apply(q1_mean)
            .reset_index(name="2030_q1_mean_value")
        )

        return out
    
    @staticmethod
    def plot_share_countries_above_probability(
        df: pd.DataFrame,
        prob_col: str,
        thresholds=None,
        figsize=(6, 4),
    ):
        """
        Plot share of countries whose probability of meeting NDC >= threshold.

        x-axis: probability threshold (0–1)
        y-axis: share of countries meeting or exceeding that probability
        """

        p = df[prob_col].dropna().values
        N = len(p)

        if thresholds is None:
            thresholds = np.linspace(0, 1, 200)

        shares = [(p >= t).sum() / N for t in thresholds]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresholds, shares)

        ax.set_xlabel("Probability threshold to meet NDC")
        ax.set_ylabel("Share of countries ≥ threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.grid()

        plt.tight_layout()
        plt.show()