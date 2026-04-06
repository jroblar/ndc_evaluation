import os
import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from utils.utils import RandomForestDiscovery, ScenarioDiscoveryBatchRunner

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = SCRIPT_DIR.parent
ML_DIR = PARENT_DIR / "ml_scripts"
DATA_DIR = ML_DIR / "output"
ENSEMBLE_DATA_DIR = DATA_DIR / "ensemble"
POST_PROCESSED_DATA_DIR = DATA_DIR / "2030_emissions"
RULES_PATH = ML_DIR / "config" / "variable_projection_rules.json"
OUTPUT_DIR = SCRIPT_DIR / "scenario_discovery_outputs"
LOG_PATH = OUTPUT_DIR / "scenario_discovery_batch.log"
MAX_WORKERS = int(os.getenv("SCENARIO_DISCOVERY_MAX_WORKERS", "2"))
RF_N_JOBS = int(os.getenv("SCENARIO_DISCOVERY_RF_N_JOBS", "1"))


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


def build_reports(summary_rows: list[dict], output_dir: Path) -> dict[str, pd.DataFrame]:
    summary_df = pd.DataFrame(summary_rows).sort_values("country").reset_index(drop=True)
    feature_counter: Counter[str] = Counter()
    feature_countries: defaultdict[str, list[str]] = defaultdict(list)

    for row in summary_rows:
        if row["status"] != "success":
            continue
        selected_top_features = [feature for feature in row["selected_top_features"].split("|") if feature]
        for feature in selected_top_features:
            feature_counter[feature] += 1
            feature_countries[feature].append(row["country"])

    feature_frequency_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "count": count,
                "countries": "|".join(sorted(feature_countries[feature])),
            }
            for feature, count in feature_counter.most_common()
        ]
    )

    summary_path = output_dir / "country_run_summary.csv"
    feature_frequency_path = output_dir / "top_variable_frequency_report.csv"
    summary_df.to_csv(summary_path, index=False)
    feature_frequency_df.to_csv(feature_frequency_path, index=False)
    logging.info("Wrote summary report to %s", summary_path)
    logging.info("Wrote feature frequency report to %s", feature_frequency_path)

    return {
        "country_run_summary": summary_df,
        "top_variable_frequency_report": feature_frequency_df,
    }

# Load the data
RUN_ID = 1773188058
df_all = pd.read_parquet(POST_PROCESSED_DATA_DIR / f"post_processed_projected_emissions_{RUN_ID}.parquet")
ensemble_df_all = pd.read_parquet(ENSEMBLE_DATA_DIR / f"ensemble_arima_{RUN_ID}.parquet")

available_countries = df_all["iso_alpha_3"].unique()

runner = ScenarioDiscoveryBatchRunner(
    projected_df=df_all,
    ensemble_df=ensemble_df_all,
    rules_path=RULES_PATH,
    top_k=2,
    extra_features=["cap_govt_effectiveness"],
    auto_threshold=True,
    rf_discovery=RandomForestDiscovery(n_jobs=RF_N_JOBS),
)

configure_logging()
logging.info("Amount of available countries in the data: %s", len(available_countries))
logging.info("Running scenario discovery with max_workers=%s", MAX_WORKERS)
logging.info("Running random forest with n_jobs=%s per country", RF_N_JOBS)

summary_rows: list[dict] = []
completed = 0
total = len(available_countries)
overall_started_at = time.perf_counter()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_country = {
        executor.submit(
            run_country_with_logging,
            runner,
            country,
            OUTPUT_DIR,
        ): country
        for country in available_countries
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

reports = build_reports(summary_rows, OUTPUT_DIR)
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
