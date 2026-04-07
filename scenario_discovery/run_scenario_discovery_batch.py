import logging
import os
import time
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from utils.utils import (
    RandomForestDiscovery,
    ScenarioDiscoveryBatchRunner,
    build_feature_combination_frequency_report,
    build_top_variable_frequency_report,
)

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = SCRIPT_DIR.parent
ML_DIR = PARENT_DIR / "ml_scripts"
DATA_DIR = ML_DIR / "output"
ENSEMBLE_DATA_DIR = DATA_DIR / "ensemble"
POST_PROCESSED_DATA_DIR = DATA_DIR / "2030_emissions"
RULES_PATH = ML_DIR / "config" / "variable_projection_rules.json"
OUTPUT_DIR = SCRIPT_DIR / "scenario_discovery_outputs"
LOG_PATH = OUTPUT_DIR / "scenario_discovery_batch.log"
CONFIG_PATH = SCRIPT_DIR / "config" / "config.yaml"


def parse_config_scalar(value: str):
    value = value.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return literal_eval(value)
    if value.startswith("[") and value.endswith("]"):
        return literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def strip_yaml_comment(line: str) -> str:
    quote_char = ""
    for idx, char in enumerate(line):
        if char in {"'", '"'} and (idx == 0 or line[idx - 1] != "\\"):
            quote_char = "" if quote_char == char else char if not quote_char else quote_char
        if char == "#" and not quote_char:
            return line[:idx]
    return line


def load_config(config_path: Path) -> dict:
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except ModuleNotFoundError:
        return load_simple_yaml_config(config_path)


def load_simple_yaml_config(config_path: Path) -> dict:
    """Parse the simple dict/list YAML shape used by this config file."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    lines = config_path.read_text(encoding="utf-8").splitlines()
    idx = 0

    while idx < len(lines):
        raw_line = strip_yaml_comment(lines[idx]).rstrip()
        idx += 1
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            raise ValueError(f"Unexpected list item without a key in {config_path}: {raw_line}")

        key, separator, raw_value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid config line in {config_path}: {raw_line}")
        key = key.strip()
        raw_value = raw_value.strip()

        if raw_value:
            parent[key] = parse_config_scalar(raw_value)
            continue

        list_values: list = []
        child: dict | list
        lookahead = idx
        while lookahead < len(lines):
            next_line = strip_yaml_comment(lines[lookahead]).rstrip()
            if next_line.strip():
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent > indent and next_line.strip().startswith("- "):
                    child = list_values
                    break
                child = {}
                break
            lookahead += 1
        else:
            child = {}

        if isinstance(child, list):
            while idx < len(lines):
                list_line = strip_yaml_comment(lines[idx]).rstrip()
                if not list_line.strip():
                    idx += 1
                    continue
                list_indent = len(list_line) - len(list_line.lstrip(" "))
                if list_indent <= indent:
                    break
                stripped = list_line.strip()
                if not stripped.startswith("- "):
                    raise ValueError(f"Only scalar lists are supported in {config_path}: {list_line}")
                list_values.append(parse_config_scalar(stripped[2:]))
                idx += 1
            parent[key] = list_values
        else:
            parent[key] = child
            stack.append((indent, child))

    return root


def get_required_section(config: dict, section_name: str) -> dict:
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{section_name}' section in {CONFIG_PATH}")
    return section


def resolve_countries(available_countries: list[str], country_config: dict) -> list[str]:
    run_all = country_config.get("run_all", True)
    selected_countries = country_config.get("selected", [])
    if run_all:
        return sorted(available_countries)

    if not isinstance(selected_countries, list) or not selected_countries:
        raise ValueError("Set countries.run_all to true or provide a non-empty countries.selected list.")

    normalized_available = {country.upper(): country for country in available_countries}
    requested = [str(country).upper() for country in selected_countries]
    missing = [country for country in requested if country not in normalized_available]
    if missing:
        raise ValueError(f"Configured countries are not available in the data: {missing}")
    return [normalized_available[country] for country in requested]


def configure_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, mode="a"),
        ],
        force=True,
    )


def run_country_with_logging(
    runner: ScenarioDiscoveryBatchRunner,
    country: str,
    output_dir: Path,
) -> dict:
    started_at = time.perf_counter()
    logging.info("Starting country %s", country)
    result = runner.run_country(
        country,
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - started_at
    logging.info(
        "Completed country %s in %.2fs with %s optimization rows",
        country,
        elapsed,
        result["n_optimization_rows"],
    )
    return result


def build_reports(summary_rows: list[dict], output_dir: Path, run_id: str, n_countries: int) -> dict[str, pd.DataFrame]:
    summary_df = pd.DataFrame(summary_rows).sort_values("country").reset_index(drop=True)
    feature_frequency_df = build_top_variable_frequency_report(summary_rows)
    feature_combination_frequency_df = build_feature_combination_frequency_report(summary_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = f"{run_id}_{n_countries}_{timestamp}"
    summary_path = output_dir / f"{report_prefix}_country_run_summary.csv"
    feature_frequency_path = output_dir / f"{report_prefix}_top_variable_frequency_report.csv"
    feature_combination_frequency_path = output_dir / f"{report_prefix}_top_variable_combination_frequency_report.csv"
    summary_df.to_csv(summary_path, index=False)
    feature_frequency_df.to_csv(feature_frequency_path, index=False)
    feature_combination_frequency_df.to_csv(feature_combination_frequency_path, index=False)
    logging.info("Wrote summary report to %s", summary_path)
    logging.info("Wrote feature frequency report to %s", feature_frequency_path)
    logging.info("Wrote feature combination frequency report to %s", feature_combination_frequency_path)

    return {
        "country_run_summary": summary_df,
        "top_variable_frequency_report": feature_frequency_df,
        "top_variable_combination_frequency_report": feature_combination_frequency_df,
    }

config = load_config(CONFIG_PATH)
scenario_discovery_config = get_required_section(config, "scenario_discovery_config")
rf_discovery_config = config.get("rf_discovery_config", {})
countries_config = config.get("countries", {"run_all": True, "selected": []})
if not isinstance(rf_discovery_config, dict):
    raise ValueError(f"Invalid 'rf_discovery_config' section in {CONFIG_PATH}")
if not isinstance(countries_config, dict):
    raise ValueError(f"Invalid 'countries' section in {CONFIG_PATH}")
rf_discovery_config = {"n_jobs": 1, **rf_discovery_config}

run_id = config.get("run_id")
if run_id is None:
    raise ValueError(f"Missing 'run_id' in {CONFIG_PATH}")
max_workers = int(config.get("max_workers", 2))
rf_n_jobs = int(rf_discovery_config.get("n_jobs", 1))
if max_workers < 1:
    raise ValueError(f"'max_workers' must be at least 1 in {CONFIG_PATH}")

df_all = pd.read_parquet(POST_PROCESSED_DATA_DIR / f"post_processed_projected_emissions_{run_id}.parquet")
ensemble_df_all = pd.read_parquet(ENSEMBLE_DATA_DIR / f"ensemble_arima_{run_id}.parquet")

available_countries = df_all["iso_alpha_3"].dropna().unique().tolist()
countries_to_run = resolve_countries(available_countries, countries_config)

runner = ScenarioDiscoveryBatchRunner(
    projected_df=df_all,
    ensemble_df=ensemble_df_all,
    rules_path=RULES_PATH,
    **scenario_discovery_config,
    rf_discovery=RandomForestDiscovery(**rf_discovery_config),
)

configure_logging()
logging.info("Amount of available countries in the data: %s", len(available_countries))
logging.info("Amount of configured countries to run: %s", len(countries_to_run))
logging.info("Running scenario discovery with max_workers=%s", max_workers)
logging.info("Running random forest with n_jobs=%s per country", rf_n_jobs)

summary_rows: list[dict] = []
completed = 0
total = len(countries_to_run)
overall_started_at = time.perf_counter()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_country = {
        executor.submit(
            run_country_with_logging,
            runner,
            country,
            OUTPUT_DIR,
        ): country
        for country in countries_to_run
    }

    for future in as_completed(future_to_country):
        country = future_to_country[future]
        completed += 1
        try:
            result = future.result()
            summary_rows.append(
                {
                    "country": result["country"],
                    "status": result["status"],
                    "auto_threshold": result["auto_threshold"],
                    "vulnerability_threshold": result["vulnerability_threshold"],
                    f"vulnerability_threshold_vs_{runner.year1}": result[f"vulnerability_threshold_vs_{runner.year1}"],
                    "vulnerability_threshold_vs_ndc_unconditional": result[
                        "vulnerability_threshold_vs_ndc_unconditional"
                    ],
                    "selected_coverage": result["selected_coverage"],
                    "selected_density": result["selected_density"],
                    "selected_density_threshold": result["selected_density_threshold"],
                    "selected_coverage_threshold": result["selected_coverage_threshold"],
                    "selected_density_below_min": result["selected_density_below_min"],
                    "selected_coverage_below_min": result["selected_coverage_below_min"],
                    "selected_top_features": "|".join(result["selected_top_features"]),
                    "features_for_optimization": "|".join(result["features_for_optimization"]),
                    "n_optimization_rows": result["n_optimization_rows"],
                    "optimization_results_path": result["optimization_results_path"],
                    "boxed_scatter_path": result["boxed_scatter_path"],
                    "future_distribution_plot_path": result["future_distribution_plot_path"],
                    "feature_importance_path": result["feature_importance_path"],
                    "rf_training_summary_path": result["rf_training_summary_path"],
                }
            )
            logging.info("Progress: %s/%s countries completed", completed, total)
        except Exception as exc:
            logging.exception("Country %s failed", country)
            summary_rows.append(
                {
                    "country": country,
                    "status": "error",
                    "auto_threshold": runner.auto_threshold,
                    "vulnerability_threshold": None,
                    f"vulnerability_threshold_vs_{runner.year1}": None,
                    "vulnerability_threshold_vs_ndc_unconditional": None,
                    "selected_coverage": None,
                    "selected_density": None,
                    "selected_density_threshold": None,
                    "selected_coverage_threshold": None,
                    "selected_density_below_min": None,
                    "selected_coverage_below_min": None,
                    "selected_top_features": "",
                    "features_for_optimization": "",
                    "n_optimization_rows": 0,
                    "optimization_results_path": "",
                    "boxed_scatter_path": "",
                    "future_distribution_plot_path": "",
                    "feature_importance_path": "",
                    "rf_training_summary_path": "",
                    "error": str(exc),
                }
            )
            logging.info("Progress: %s/%s countries completed", completed, total)

reports = build_reports(summary_rows, OUTPUT_DIR, str(run_id), len(countries_to_run))
successful_runs = sum(row["status"] == "success" for row in summary_rows)
failed_runs = sum(row["status"] == "error" for row in summary_rows)
overall_elapsed = time.perf_counter() - overall_started_at
logging.info(
    "Batch finished in %.2fs. Successful countries: %s. Failed countries: %s. Log file: %s",
    overall_elapsed,
    successful_runs,
    failed_runs,
    LOG_PATH,
)
