import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Dict, Union, Iterable, Tuple
from statsmodels.tsa.filters.hp_filter import hpfilter
import math
from joblib import Parallel, delayed

class EnsembleProjections:

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
    def plot_ensemble_time_series_grid(
        df,
        panels,
        hist_df=None,
        ndc_targets_df=None,
        ncols=2,
        figsize=(12, 8),
        xlabel="Year",
        save_path=None,
        ndc_target_display="unconditional",
        ndc_iso_col="iso_alpha_3",
        unconditional_target_col="unconditional_target",
        conditional_target_col="conditional_target",
    ):
        """
        Multi-panel pony-tail plot for ensemble projections.

        Parameters
        ----------
        df : pd.DataFrame
            Ensemble dataframe with columns:
            ['iso_alpha_3', 'future_id', 'year', <vars>]

        panels : list of dict
            Each dict defines one subplot with keys:
            {
                "iso": str,
                "column": str,
                "title": str (optional),
                "ylabel": str (optional)
            }

            Length of panels should be 2 or 4 (or any number).

        hist_df : pd.DataFrame, optional
            Historical dataframe with same structure as df (no future_id).

        ndc_targets_df : pd.DataFrame, optional
            Dataframe with one row per country containing NDC target values to
            draw as dashed horizontal reference lines in each subplot.

        ndc_target_display : str
            Which NDC target reference lines to show. One of
            {"unconditional", "conditional", "both"}.

        ncols : int
            Number of columns in subplot grid (default=2).

        figsize : tuple
            Figure size.

        xlabel : str
            Shared x-axis label.
        """

        n_panels = len(panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes).flatten()
        valid_ndc_target_display = {"unconditional", "conditional", "both"}
        if ndc_target_display not in valid_ndc_target_display:
            raise ValueError(
                f"ndc_target_display must be one of {sorted(valid_ndc_target_display)}. "
                f"Received: {ndc_target_display!r}"
            )

        for ax, panel in zip(axes, panels):
            iso = panel["iso"]
            col = panel["column"]
            title = panel.get("title", f"{iso} – {col}")
            ylabel = panel.get("ylabel", col)

            # --- Subset ensemble ---
            ens = (
                df[df["iso_alpha_3"] == iso]
                .sort_values(["future_id", "year"])
            )

            # print(ens["iso_alpha_3"].unique())

            # --- Plot ensemble members ---
            for _, grp in ens.groupby("future_id"):
                ax.plot(
                    grp["year"],
                    grp[col],
                    color="gray",
                    linewidth=1,
                    alpha=0.4
                )

            # --- Overlay historical series ---
            if hist_df is not None:
                hist = (
                    hist_df[hist_df["iso_alpha_3"] == iso]
                    .sort_values("year")
                )
                if not hist.empty and col in hist.columns:
                    ax.plot(
                        hist["year"],
                        hist[col],
                        color="black",
                        linewidth=2.5,
                        marker="o",
                        label="Historical"
                    )

            if ndc_targets_df is not None and ndc_iso_col in ndc_targets_df.columns:
                ndc_target_row = ndc_targets_df[ndc_targets_df[ndc_iso_col] == iso]
                if not ndc_target_row.empty:
                    unconditional_target = pd.to_numeric(
                        ndc_target_row[unconditional_target_col],
                        errors="coerce",
                    ).iloc[0] if unconditional_target_col in ndc_target_row.columns else np.nan
                    conditional_target = pd.to_numeric(
                        ndc_target_row[conditional_target_col],
                        errors="coerce",
                    ).iloc[0] if conditional_target_col in ndc_target_row.columns else np.nan

                    if ndc_target_display in {"unconditional", "both"} and pd.notna(unconditional_target):
                        ax.axhline(
                            unconditional_target,
                            color="tab:green",
                            linestyle="--",
                            linewidth=1.5,
                            label="NDC unconditional target",
                        )
                    if ndc_target_display in {"conditional", "both"} and pd.notna(conditional_target):
                        ax.axhline(
                            conditional_target,
                            color="tab:orange",
                            linestyle="--",
                            linewidth=1.5,
                            label="NDC conditional target",
                        )

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)

        # --- Remove unused axes ---
        for ax in axes[n_panels:]:
            ax.axis("off")

        # --- Shared labels ---
        for ax in axes[-ncols:]:
            ax.set_xlabel(xlabel)

        # --- Legend (single) ---
        legend_handles = []
        legend_labels = []
        for ax in axes[:n_panels]:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3)


        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # ---- SAVE FIGURE (IMPORTANT) ----
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")

        plt.show()
        plt.close(fig)
        
    
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
            If True, creates a 'total_emissions' column = expm1(log_total_emissions).

        Returns
        -------
        pd.DataFrame
            A copy of `ensemble_df` with two new columns:
            - 'log_total_emissions': the raw model predictions
            - 'total_emissions'     : expm1(log_total_emissions) if exponentiate=True
        """
        # 1) Work on a copy
        df = ensemble_df.copy()

        # 2) Extract the feature matrix
        X = df[feature_cols]

        # 3) Run through the pipeline/model
        df["x_log_signed_con_edgar_ghg_mt"] = model.predict(X)

        # 4) Optional back‐transform
        if exponentiate:
            df["con_edgar_ghg_mt"] = np.expm1(df["x_log_signed_con_edgar_ghg_mt"])

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
    
    @staticmethod
    def hp_filter_panel(
        df: pd.DataFrame,
        id_col: str = "iso_alpha_3",
        time_col: str = "year",
        lambda_: float = 100.0,           # annual data default
        which: str = "trend",             # "trend" or "cycle"
        cols: list[str] | None = None,    # None -> auto numeric except time_col
        suffix: str | None = None,        # None -> "_hp_trend"/"_hp_cycle" (ignored if keep="replace")
        min_len: int = 6,
        interpolate: bool = True,
        keep: str = "both",               # "both" | "filtered_only" | "replace"
        keep_unfiltered_numeric: bool = True
    ) -> pd.DataFrame:
        if which not in {"trend", "cycle"}:
            raise ValueError('`which` must be "trend" or "cycle".')
        if keep not in {"both", "filtered_only", "replace"}:
            raise ValueError('`keep` must be "both", "filtered_only", or "replace".')

        # numeric columns excluding time
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        if cols is None:
            cols = numeric_cols.copy()

        eff_suffix = "" if keep == "replace" else (suffix or ("_hp_trend" if which == "trend" else "_hp_cycle"))

        # base: id/time + non-numeric
        non_numeric_cols = df.columns.difference(df.select_dtypes(include="number").columns).tolist()
        base_cols = []
        for c in [id_col, time_col] + non_numeric_cols:
            if c in df.columns and c not in base_cols:
                base_cols.append(c)
        out = df.loc[:, base_cols].copy()

        # keep other numeric (not filtered), but never the time column
        other_numeric = [c for c in numeric_cols if c not in cols and c != time_col]
        if keep_unfiltered_numeric and other_numeric:
            out = out.join(df.loc[:, other_numeric])

        filtered_blocks = {}

        # apply HP per id
        for _, g in df.groupby(id_col, sort=False):
            g_sorted = g.sort_values(time_col)
            idx = g_sorted.index

            for col in cols:
                s = g_sorted[col]
                if interpolate:
                    s = s.interpolate(limit_direction="both")
                if s.notna().sum() < min_len:
                    filt_vals = np.full(s.shape, np.nan, dtype=float)
                else:
                    # FIX: hpfilter returns (cycle, trend)
                    cycle, trend = hpfilter(s.to_numpy(), lamb=lambda_)
                    filt_vals = trend if which == "trend" else cycle

                out_col = col if keep == "replace" else f"{col}{eff_suffix}"
                if out_col not in filtered_blocks:
                    filtered_blocks[out_col] = pd.Series(index=df.index, dtype=float)
                filtered_blocks[out_col].loc[idx] = filt_vals

        if filtered_blocks:
            out = out.join(pd.DataFrame(filtered_blocks))

        if keep == "both":
            out = out.join(df.loc[:, cols], rsuffix="_orig")
        elif keep == "filtered_only":
            to_drop = [c for c in out.columns if c in cols and c not in base_cols]
            if to_drop:
                out = out.drop(columns=to_drop)

        front = [c for c in [id_col, time_col] if c in out.columns]
        rest = [c for c in out.columns if c not in front]
        return out[front + rest]
    
    @staticmethod
    def plot_iso_numeric_subplots(
        df: pd.DataFrame,
        iso: str,
        id_col: str = "iso_alpha_3",
        time_col: str = "year",
        cols: list[str] | None = None,   # None -> all numeric except time_col
        ncols: int = 3,                   # number of subplot columns
        size_per_plot=(4, 3),             # width, height per subplot (in inches)
    ):
        # subset and sort
        dfi = df[df[id_col] == iso].copy()
        if dfi.empty:
            raise ValueError(f"No rows for {id_col} == {iso!r}.")
        dfi = dfi.sort_values(time_col)

        # choose columns
        numeric_cols = dfi.select_dtypes(include="number").columns.tolist()
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        if cols is None:
            cols_to_plot = numeric_cols
        else:
            cols_to_plot = [c for c in cols if c in numeric_cols]
            if not cols_to_plot:
                raise ValueError("No valid numeric columns to plot in 'cols'.")

        n = len(cols_to_plot)
        nrows = math.ceil(n / ncols)
        figsize = (size_per_plot[0] * ncols, size_per_plot[1] * nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
        # make axes iterable
        axes = np.atleast_1d(axes).ravel()

        x = dfi[time_col].to_numpy()

        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            y = dfi[col].to_numpy(dtype=float)
            ax.plot(x, y)
            ax.set_title(col)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        # turn off any unused axes
        for j in range(len(cols_to_plot), len(axes)):
            axes[j].axis("off")

        # label bottom row x-axis once
        for ax in axes[-ncols:]:
            if ax.has_data():
                ax.set_xlabel(time_col)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def hp_filter_panel_fast(
        df: pd.DataFrame,
        id_col: str = "iso_alpha_3",
        time_col: str = "year",
        lambda_: float = 100.0,           # annual data default
        which: str = "trend",             # "trend" or "cycle"
        cols: list[str] | None = None,    # None -> auto numeric except time_col
        suffix: str | None = None,        # None -> "_hp_trend"/"_hp_cycle" (ignored if keep="replace")
        min_len: int = 6,
        interpolate: bool = True,
        keep: str = "both",               # "both" | "filtered_only" | "replace"
        keep_unfiltered_numeric: bool = True,
        n_jobs: int = 1                   # >1 to parallelize across groups
    ) -> pd.DataFrame:

        if which not in {"trend", "cycle"}:
            raise ValueError('`which` must be "trend" or "cycle".')
        if keep not in {"both", "filtered_only", "replace"}:
            raise ValueError('`keep` must be "both", "filtered_only", or "replace".')

        # Decide columns to filter
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        if cols is None:
            cols = numeric_cols.copy()
        if not cols:
            # nothing to do
            return df.copy()

        eff_suffix = "" if keep == "replace" else (suffix or ("_hp_trend" if which == "trend" else "_hp_cycle"))

        # Base / passthrough columns
        non_numeric_cols = df.columns.difference(df.select_dtypes(include="number").columns).tolist()
        base_cols = []
        for c in [id_col, time_col] + non_numeric_cols:
            if c in df.columns and c not in base_cols:
                base_cols.append(c)

        other_numeric = [c for c in numeric_cols if c not in cols and c != time_col]

        # Work on a sorted view for contiguous groups and stable placement
        w = df[[id_col, time_col] + cols].copy()
        w.sort_values([id_col, time_col], inplace=True)
        # Factorize id for group boundaries
        id_codes, id_uniques = pd.factorize(w[id_col], sort=False)
        # Find group boundaries
        change = np.r_[True, id_codes[1:] != id_codes[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(w)]

        # Prepare storage for results (same row order as sorted `w`)
        out_arr = np.full((len(w), len(cols)), np.nan, dtype=float)
        cols_idx = {c: i for i, c in enumerate(cols)}

        # Helper: process one group slice
        def _process_group(i_start, i_end):
            sl = slice(i_start, i_end)
            X = w.iloc[sl, 2:].to_numpy(dtype=float, copy=False)  # only 'cols'
            # Interpolate once per group across rows (time order already sorted)
            if interpolate:
                # Use pandas for robust edge handling but only once per group
                X = pd.DataFrame(X).interpolate(limit_direction="both").to_numpy(dtype=float, copy=False)

            # Per-column HP filtering (statsmodels not vectorized)
            m = X.shape[0]
            for j in range(X.shape[1]):
                col = X[:, j]
                valid = np.isfinite(col)
                if valid.sum() < min_len:
                    continue
                # statsmodels expects 1D array
                cycle, trend = hpfilter(col[valid], lamb=lambda_)
                out = trend if which == "trend" else cycle
                # place back only on valid rows
                tmp = np.full(m, np.nan, dtype=float)
                tmp[valid] = out
                X[:, j] = tmp
            return (i_start, i_end, X)

        # Run sequentially or in parallel across groups
        if n_jobs == 1:
            for s, e in zip(starts, ends):
                _, _, X = _process_group(s, e)
                out_arr[s:e, :] = X
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_group)(s, e) for s, e in zip(starts, ends)
            )
            for s, e, X in results:
                out_arr[s:e, :] = X

        # Build filtered DataFrame (still sorted)
        filtered_cols = [c if keep == "replace" else f"{c}{eff_suffix}" for c in cols]
        filtered_df_sorted = pd.DataFrame(out_arr, columns=filtered_cols, index=w.index)

        # Assemble output in original row order
        pieces = []

        # base columns from original df (original order)
        out = df.loc[:, base_cols].copy()

        # keep other numeric (not filtered), but never the time column
        if keep_unfiltered_numeric and other_numeric:
            out = out.join(df.loc[:, other_numeric])

        # add filtered columns aligned to original index
        # map from sorted back to original index
        filtered_df = filtered_df_sorted.reindex(df.index)  # reindex aligns by index labels

        out = out.join(filtered_df)

        if keep == "both":
            out = out.join(df.loc[:, cols], rsuffix="_orig")
        elif keep == "filtered_only":
            # if we replaced names, drop the original numeric cols (except base)
            to_drop = [c for c in cols if c in out.columns and c not in base_cols]
            if to_drop:
                out = out.drop(columns=to_drop)

        # Put id/time in front if present
        front = [c for c in [id_col, time_col] if c in out.columns]
        rest = [c for c in out.columns if c not in front]
        return out[front + rest]
    
    @staticmethod
    def remove_timeseries_with_year_outliers_iqr(
        df: pd.DataFrame,
        year: int = 2030,
        value_col: str = "total_emissions_hp_trend",
        country_col: str = "iso_alpha_3",
        id_col: str = "future_id",
        iqr_multiplier: float = 1.5,
        min_group_size: int = 5,
        treat_zero_iqr_as_no_outliers: bool = True,
        return_removed_ids: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
        """
        Identify outliers in `value_col` for a specific `year` per `country_col`
        using the IQR method, then remove the ENTIRE time series for any outlier
        `id_col` (i.e., drop all rows for those future_id values across all years).

        Parameters
        ----------
        df : pd.DataFrame
            Must include columns [id_col, country_col, 'year', value_col].
        year : int
            Target year to detect outliers on (default 2030).
        value_col : str
            Column to analyze for outliers (default 'total_emissions_hp_trend').
        country_col : str
            Country/grouping column (default 'iso_alpha_3').
        id_col : str
            Series identifier (default 'future_id').
        iqr_multiplier : float
            IQR rule multiplier (1.5 is standard; lower is stricter).
        min_group_size : int
            Minimum sample size per country for outlier detection; groups
            smaller than this will not have any outliers removed.
        treat_zero_iqr_as_no_outliers : bool
            If True, when IQR == 0 for a country-year, treat as no outliers.
            If False, bounds collapse to a point and anything != Q1 is outlier.
        return_removed_ids : bool
            If True, also return a list of removed future_id values.

        Returns
        -------
        cleaned_df : pd.DataFrame
            DataFrame with all rows for outlier future_ids removed.
        removed_ids : list[str] (optional)
            Only returned if return_removed_ids=True.
        """
        required = {id_col, country_col, "year", value_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # Only the target year's slice for detecting outliers
        year_df = df[df["year"] == year][[id_col, country_col, value_col]].copy()
        if year_df.empty:
            cleaned = df.copy()
            return (cleaned, []) if return_removed_ids else cleaned

        # Compute Q1, Q3, IQR per country on the target year
        stats = (
            year_df.groupby(country_col)[value_col]
            .agg(q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                n="count")
            .reset_index()
        )
        stats["iqr"] = stats["q3"] - stats["q1"]

        # Optionally ignore countries with IQR == 0
        if treat_zero_iqr_as_no_outliers:
            stats["use_bounds"] = (stats["iqr"] > 0) & (stats["n"] >= min_group_size)
        else:
            stats["use_bounds"] = (stats["n"] >= min_group_size)

        # Lower/upper bounds
        stats["lower"] = stats["q1"] - iqr_multiplier * stats["iqr"]
        stats["upper"] = stats["q3"] + iqr_multiplier * stats["iqr"]

        # Merge bounds onto the target-year rows
        year_with_bounds = year_df.merge(stats[[country_col, "lower", "upper", "use_bounds"]],
                                        on=country_col, how="left")

        # Determine outliers (only where we decide to use bounds)
        mask_use = year_with_bounds["use_bounds"].fillna(False)
        mask_low = year_with_bounds[value_col] < year_with_bounds["lower"]
        mask_high = year_with_bounds[value_col] > year_with_bounds["upper"]
        outlier_mask = mask_use & (mask_low | mask_high)

        removed_ids = year_with_bounds.loc[outlier_mask, id_col].unique().tolist()

        # Drop whole time series for removed IDs
        cleaned_df = df[~df[id_col].isin(removed_ids)].copy()

        return (cleaned_df, removed_ids) if return_removed_ids else cleaned_df
