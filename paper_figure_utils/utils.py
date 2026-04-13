from pathlib import Path
from typing import Any

import pandas as pd


class PaperFiguresUtils:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_PAPER_TABLES_DIR = REPO_ROOT / "paper_tables"
    DEFAULT_SCENARIO_DISCOVERY_REGIONS = ("JPN", "MOZ", "MEX")
    SUPPORTED_SAVE_FORMATS = {"parquet", "csv"}

    @staticmethod
    def save_df_dict(
        data_dict: dict[str, Any],
        base_dir: str | Path,
        save_format: str = "parquet",
        csv_index: bool = False,
        parquet_index: bool = False,
    ) -> Path:
        """Recursively save a nested dictionary of DataFrames."""
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        save_format = save_format.lower()
        if save_format not in PaperFiguresUtils.SUPPORTED_SAVE_FORMATS:
            raise ValueError(
                f"Unsupported save_format '{save_format}'. "
                f"Expected one of {sorted(PaperFiguresUtils.SUPPORTED_SAVE_FORMATS)}."
            )

        for key, value in data_dict.items():
            key_path = base_path / key
            if isinstance(value, dict):
                PaperFiguresUtils.save_df_dict(
                    data_dict=value,
                    base_dir=key_path,
                    save_format=save_format,
                    csv_index=csv_index,
                    parquet_index=parquet_index,
                )
                continue

            if isinstance(value, pd.DataFrame):
                if save_format == "parquet":
                    value.to_parquet(key_path.with_suffix(".parquet"), index=parquet_index)
                else:
                    value.to_csv(key_path.with_suffix(".csv"), index=csv_index)
                continue

            raise TypeError(
                f"Unsupported type for key '{key}': {type(value)}. "
                "Only nested dicts and pandas DataFrames are supported."
            )

        return base_path

    @staticmethod
    def load_df_dict(base_dir: str | Path) -> dict[str, Any]:
        """Recursively load a nested dictionary of DataFrames saved by `save_df_dict`."""
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Saved paper tables directory not found: {base_path}")

        data: dict[str, Any] = {}
        for path in sorted(base_path.iterdir()):
            if path.is_dir():
                data[path.name] = PaperFiguresUtils.load_df_dict(path)
            elif path.suffix == ".parquet":
                data[path.stem] = pd.read_parquet(path)
            elif path.suffix == ".csv":
                data[path.stem] = pd.read_csv(path)
        return data

    @staticmethod
    def _build_source_paths(
        run_id: str | int,
        sd_output_id: str,
        repo_root: str | Path | None = None,
    ) -> dict[str, Path]:
        root = Path(repo_root) if repo_root is not None else PaperFiguresUtils.REPO_ROOT

        return {
            "historical_df": root / "ml" / "output" / "training" / f"training_df_{run_id}.csv",
            "historical_em_df": root
            / "arima"
            / "output"
            / "hp_filtered"
            / f"historical_emissions_hp_trend_{run_id}.parquet",
            "raw_ensemble_df": root / "arima" / "output" / "ensemble" / f"ensemble_arima_{run_id}.parquet",
            "postprocessed_ensemble_df": root
            / "arima"
            / "output"
            / "postprocessed_ensemble"
            / f"postprocessed_ensemble_{run_id}.parquet",
            "ndc_prob_df": root
            / "ndc_probability"
            / "tables"
            / f"ndc_probability_analysis_{run_id}.csv",
            "top_combinations_df": root
            / "scenario_discovery"
            / "output"
            / sd_output_id
            / f"{sd_output_id}_top_variable_combination_frequency_report.csv",
            "sd_output_dir": root / "scenario_discovery" / "output" / sd_output_id,
        }

    @staticmethod
    def _require_existing_file(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return path

    @staticmethod
    def _load_source_tables(
        run_id: str | int,
        sd_output_id: str,
        regions_of_interest: list[str] | tuple[str, ...] | None = None,
        repo_root: str | Path | None = None,
    ) -> dict[str, Any]:
        paths = PaperFiguresUtils._build_source_paths(
            run_id=run_id,
            sd_output_id=sd_output_id,
            repo_root=repo_root,
        )
        sd_output_dir = paths["sd_output_dir"]
        if not sd_output_dir.exists():
            raise FileNotFoundError(f"Scenario discovery output directory not found: {sd_output_dir}")

        if regions_of_interest is None:
            candidate_regions = PaperFiguresUtils.DEFAULT_SCENARIO_DISCOVERY_REGIONS
            if not all((sd_output_dir / region).exists() for region in candidate_regions):
                regions = sorted(path.name for path in sd_output_dir.iterdir() if path.is_dir())
            else:
                regions = list(candidate_regions)
        else:
            regions = list(regions_of_interest)

        data: dict[str, Any] = {
            "historical_df": pd.read_csv(PaperFiguresUtils._require_existing_file(paths["historical_df"])),
            "historical_em_df": pd.read_parquet(
                PaperFiguresUtils._require_existing_file(paths["historical_em_df"])
            ),
            "raw_ensemble_df": pd.read_parquet(PaperFiguresUtils._require_existing_file(paths["raw_ensemble_df"])),
            "postprocessed_ensemble_df": pd.read_parquet(
                PaperFiguresUtils._require_existing_file(paths["postprocessed_ensemble_df"])
            ),
            "ndc_prob_df": pd.read_csv(PaperFiguresUtils._require_existing_file(paths["ndc_prob_df"])),
            "top_combinations_df": pd.read_csv(
                PaperFiguresUtils._require_existing_file(paths["top_combinations_df"])
            ),
            "scenario_discovery": {},
        }

        for region in regions:
            region_dir = sd_output_dir / region
            if not region_dir.exists():
                raise FileNotFoundError(f"Scenario discovery region directory not found: {region_dir}")

            data["scenario_discovery"][region] = {
                "opt_result": pd.read_csv(
                    PaperFiguresUtils._require_existing_file(region_dir / f"{region}_optimization_results.csv")
                ),
                "opt_table_input": pd.read_csv(
                    PaperFiguresUtils._require_existing_file(region_dir / f"{region}_optimization_input_table.csv")
                ),
                "future_dist": pd.read_csv(
                    PaperFiguresUtils._require_existing_file(
                        region_dir / f"{region}_future_distribution_input_table.csv"
                    )
                ),
            }

        return data

    @staticmethod
    def to_notebook_variables(data: dict[str, Any]) -> dict[str, Any]:
        """Return notebook-ready variables derived from the loaded data bundle."""
        scenario_discovery = data.get("scenario_discovery", {})
        future_dist_df_dict = {
            region: region_tables["future_dist"] for region, region_tables in scenario_discovery.items()
        }
        opt_results_df_dict = {
            region: {
                "opt_result": region_tables["opt_result"],
                "opt_table_input": region_tables["opt_table_input"],
                "future_dist": region_tables["future_dist"],
            }
            for region, region_tables in scenario_discovery.items()
        }

        return {
            **data,
            "future_dist_df_dict": future_dist_df_dict,
            "opt_results_df_dict": opt_results_df_dict,
        }

    @staticmethod
    def load_figure_tables(
        run_id: str | int,
        sd_output_id: str,
        save_new_data: bool = False,
        load_saved_data: bool | None = None,
        paper_tables_dir: str | Path | None = None,
        repo_root: str | Path | None = None,
        regions_of_interest: list[str] | tuple[str, ...] | None = None,
        save_format: str = "parquet",
        csv_index: bool = False,
        parquet_index: bool = False,
    ) -> dict[str, Any]:
        """
        Load the tables used by `paper_figures.ipynb`.

        Parameters
        ----------
        run_id
            Run id shared by the ARIMA, ML, and NDC probability outputs.
        sd_output_id
            Scenario discovery output folder name.
        save_new_data
            If True, load from source outputs and persist a parquet cache under `paper_tables_dir`.
        load_saved_data
            If True, force loading from `paper_tables_dir`. If None, defaults to `not save_new_data`.
        paper_tables_dir
            Cache directory for saved parquet files.
        repo_root
            Repository root. Defaults to the project root inferred from this file.
        regions_of_interest
            Scenario discovery regions to include. Defaults to `("JPN", "MOZ", "MEX")`
            when those directories exist, otherwise all available region directories.
        save_format
            Cache format for saved tables. Supported values: `"parquet"` and `"csv"`.
        csv_index
            Whether to preserve the DataFrame index in saved CSV files.
        parquet_index
            Whether to preserve the DataFrame index in saved parquet files.
        """
        output_dir = Path(paper_tables_dir) if paper_tables_dir is not None else PaperFiguresUtils.DEFAULT_PAPER_TABLES_DIR
        if load_saved_data is None:
            load_saved_data = not save_new_data

        if save_new_data:
            data = PaperFiguresUtils._load_source_tables(
                run_id=run_id,
                sd_output_id=sd_output_id,
                regions_of_interest=regions_of_interest,
                repo_root=repo_root,
            )
            PaperFiguresUtils.save_df_dict(
                data,
                output_dir,
                save_format=save_format,
                csv_index=csv_index,
                parquet_index=parquet_index,
            )
            return PaperFiguresUtils.to_notebook_variables(data)

        if load_saved_data:
            data = PaperFiguresUtils.load_df_dict(output_dir)
            return PaperFiguresUtils.to_notebook_variables(data)

        data = PaperFiguresUtils._load_source_tables(
            run_id=run_id,
            sd_output_id=sd_output_id,
            regions_of_interest=regions_of_interest,
            repo_root=repo_root,
        )
        return PaperFiguresUtils.to_notebook_variables(data)
