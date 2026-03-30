from utils.utils import ScenarioDiscoveryBatchRunner
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
ML_DIR = os.path.join(PARENT_DIR, "ml_scripts")
DATA_DIR = os.path.join(ML_DIR, "output")
ENSEMBLE_DATA_DIR = os.path.join(DATA_DIR, "ensemble")
POST_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "2030_emissions")
RULES_PATH = os.path.join(ML_DIR, 'config', 'variable_projection_rules.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "scenario_discovery_outputs")

# Load the data
RUN_ID = 1773188058
df_all = pd.read_parquet(os.path.join(POST_PROCESSED_DATA_DIR, f"post_processed_projected_emissions_{RUN_ID}.parquet"))
ensemble_df_all = pd.read_parquet(os.path.join(ENSEMBLE_DATA_DIR, f"ensemble_arima_{RUN_ID}.parquet"))

available_countries = df_all["iso_alpha_3"].unique()
print("Amount of available countries in the data:", len(available_countries))

runner = ScenarioDiscoveryBatchRunner(
    projected_df=df_all,
    ensemble_df=ensemble_df_all,
    rules_path=RULES_PATH,
    top_k=2,
    extra_features=["cap_govt_effectiveness"],
    auto_threshold=True,
)

reports = runner.run_many(
    countries=available_countries,
    output_dir=OUTPUT_DIR,
)


