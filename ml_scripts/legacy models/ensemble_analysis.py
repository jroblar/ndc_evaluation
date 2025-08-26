import pandas as pd
from utils.ml_utils_v2 import EnsembleProjections
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

%load_ext autoreload
%autoreload 2

def ndc_summary(
    xgb_df: pd.DataFrame,
    ndc_df: pd.DataFrame,
    year: int = 2030,
    value_col: str = "total_emissions",
    cap_col: str = "Unconditional",
    left_iso_col: str = "iso_alpha_3",
    right_iso_col: str = "ISO",
    scenario_col: str = "future_id",
) -> pd.DataFrame:
    """
    Merge modeled futures with NDC caps and summarize, by ISO, the share of futures
    that exceed the (un)conditional cap in a given target year.

    Returns a DataFrame with:
      - left_iso_col
      - proportion_exceeding
      - n_futures
    """
    ndc_min = (ndc_df[[right_iso_col, cap_col]]
               .drop_duplicates(subset=[right_iso_col], keep="first"))

    full = xgb_df.merge(
        ndc_min, left_on=left_iso_col, right_on=right_iso_col, how="left")

    df_y = full.loc[(full["year"] == year) & (full[cap_col].notna())].copy()

    df_y[value_col] = pd.to_numeric(df_y[value_col], errors="coerce")
    df_y[cap_col]   = pd.to_numeric(df_y[cap_col], errors="coerce")

    df_y["meets"] = df_y[value_col] < df_y[cap_col]

    # Summarize by ISO
    out = (df_y.groupby(left_iso_col, as_index=False)
           .agg(meets_ndc=("meets", "mean"),
                n_futures=(scenario_col, "nunique")))

    return out

ep = EnsembleProjections()

# Set up paths
SCRIPT_DIR_PATH = os.getcwd()
ROOT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, "data")
PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "processed_data")
RESULTS_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "results")
OUTPUT_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "output")
ENSEMBLE_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "ensemble")
MODELS_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "models")
TRAINING_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "training")
PROBABILITIES_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "probabilities")

# Ensemble params
n_scenarios = 1000

# Load data
training_df_log_transformed = pd.read_csv(os.path.join(TRAINING_DIR_PATH, "training_df_log_transformed.csv"))
training_df_lags = pd.read_csv(os.path.join(TRAINING_DIR_PATH, "training_df_lags.csv"))
ensemble_arima_df = pd.read_parquet(os.path.join(ENSEMBLE_DIR_PATH, f"ensemble_arima_{n_scenarios}.parquet"))

# Print shapes
print(f"Training df log transformed shape: {training_df_log_transformed.shape}")
print(f"Training df lags shape: {training_df_lags.shape}")
print(f"Ensemble ARIMA df shape: {ensemble_arima_df.shape}") 

# Get the models from the models directory with keys without .pkl extension
models = {}
for model_name in os.listdir(MODELS_DIR_PATH):
    if model_name.endswith(".pkl"):
        model_path = os.path.join(MODELS_DIR_PATH, model_name)
        key_name = model_name[:-4]  # remove .pkl extension
        models[key_name] = joblib.load(model_path)

models.keys()

# Get a df with only iso_alpha_3, income_group and region cols
iso_alpha_3_cols = ["iso_alpha_3", "income_group", "region"]
income_group_region_mapping_df = training_df_log_transformed[iso_alpha_3_cols].drop_duplicates().reset_index(drop=True)
income_group_region_mapping_df.head()

initial_conditions_df = training_df_lags[training_df_lags["year"] == 2022].copy()
initial_conditions_df = initial_conditions_df.reset_index(drop=True)
initial_conditions_df = initial_conditions_df[["iso_alpha_3", "year", "total_emissions"]]
initial_conditions_df

ensemble_arima_df.head()

# Merge the income group and region mapping df with the ensemble df
ensemble_arima_df = pd.merge(ensemble_arima_df, income_group_region_mapping_df,
                             on="iso_alpha_3", how="left")

ensemble_arima_df.head()

ep.plot_ensemble_time_series(
    df=ensemble_arima_df, 
    iso_alpha_3="MEX",
    column="log_pop_total",
    hist_df=training_df_log_transformed)

ep.plot_ensemble_time_series(
    df=ensemble_arima_df, 
    iso_alpha_3="MEX",
    column="log_gdp_2021_ppp_intl_usd",
    hist_df=training_df_log_transformed)

# Get feature cols
feature_cols_no_isos = [c for c in  training_df_log_transformed.columns if c not in ["iso_alpha_3", "log_total_emissions"]]
feature_cols_no_isos

feature_cols_with_isos = [c for c in training_df_log_transformed.columns if c not in ["log_total_emissions"]]
feature_cols_with_isos

xgb_no_isos_df = ep.predict_ensemble_emissions(ensemble_arima_df, models["reg_no_isos_xgb_pipeline"], feature_cols=feature_cols_no_isos)
xgb_no_isos_df.head()

enet_no_isos_df = ep.predict_ensemble_emissions(ensemble_arima_df, models["reg_no_isos_enet_pipeline"], feature_cols=feature_cols_no_isos)
xgb_with_isos_df = ep.predict_ensemble_emissions(ensemble_arima_df, models["reg_with_isos_xgb_pipeline"], feature_cols=feature_cols_with_isos)
enet_with_isos_df = ep.predict_ensemble_emissions(ensemble_arima_df, models["reg_with_isos_enet_pipeline"], feature_cols=feature_cols_with_isos)

ep.plot_ensemble_time_series(
    df=xgb_no_isos_df, 
    iso_alpha_3="USA",
    column="total_emissions",
    hist_df=training_df_lags)

ep.plot_ensemble_time_series(
    df=xgb_with_isos_df, 
    iso_alpha_3="USA",
    column="total_emissions",
    hist_df=training_df_lags)

ep.plot_ensemble_time_series(
    df=enet_no_isos_df, 
    iso_alpha_3="USA",
    column="total_emissions",
    hist_df=training_df_lags)

ep.plot_ensemble_time_series(
    df=enet_with_isos_df, 
    iso_alpha_3="USA",
    column="total_emissions",
    hist_df=training_df_lags)

xgb_no_isos_df_calibrated = ep.calibrate_total_emissions(
    simulated_df=xgb_no_isos_df,
    initial_emissions_df=initial_conditions_df,
    adjustment_method="additive"
)

enet_with_isos_df_calibrated = ep.calibrate_total_emissions(
    simulated_df=enet_with_isos_df,
    initial_emissions_df=initial_conditions_df,
    adjustment_method="additive"
)

ep.plot_ensemble_time_series(
    df=xgb_no_isos_df_calibrated, 
    iso_alpha_3="MEX",
    column="total_emissions",
    hist_df=training_df_lags)

ep.plot_ensemble_time_series(
    df=enet_with_isos_df_calibrated, 
    iso_alpha_3="MEX",
    column="total_emissions",
    hist_df=training_df_lags)

### Compare with NDC goals in 2030

ndc_ref = pd.read_csv(os.path.join(PROCESSED_DATA_DIR_PATH, "ndc_reference.csv"))
ndc_ref.head()
ndc_ref_base = ndc_ref[["ISO", "Unconditional", "Conditional", "Unconditional_b20", "Conditional_b20"]].copy()

base_df = pd.read_csv(os.path.join(TRAINING_DIR_PATH, "training_df_lags.csv"))
base_df = base_df[base_df["year"] == 2020][["iso_alpha_3", "total_emissions"]].\
    rename(columns={"total_emissions": "2020_emissions"})

ndc_ref_base = ndc_ref.merge(base_df, left_on="ISO", right_on="iso_alpha_3", how="left")
ndc_ref_base["Unconditional_etpe"] = (1 + ndc_ref_base["Unconditional_b20"]) * ndc_ref_base["2020_emissions"]
ndc_ref_base["Conditional_etpe"] = (1 + ndc_ref_base["Conditional_b20"]) * ndc_ref_base["2020_emissions"]

### Evaluating the NDC achievement probabilities

test_1 = ndc_summary(
    xgb_df=xgb_no_isos_df_calibrated,
    ndc_df=ndc_ref_base,
    cap_col = "Unconditional_etpe",
    year=2030
)

test_2 = ndc_summary(
    xgb_df=xgb_no_isos_df_calibrated,
    ndc_df=ndc_ref_base,
    cap_col="Conditional_etpe",
    year=2030
)


test_3 = ndc_summary(
    xgb_df=enet_with_isos_df_calibrated,
    ndc_df=ndc_ref_base,
    cap_col = "Unconditional_etpe",
    year=2030
)

test_4 = ndc_summary(
    xgb_df=enet_with_isos_df_calibrated,
    ndc_df=ndc_ref_base,
    cap_col="Conditional_etpe",
    year=2030
)

### Saving Results

test_1.to_csv(os.path.join(PROBABILITIES_DIR_PATH, "xgb_unconditional_ndc_achievement_2030.csv"), index=False)
test_2.to_csv(os.path.join(PROBABILITIES_DIR_PATH, "xgb_conditional_ndc_achievement_2030.csv"), index=False)
test_3.to_csv(os.path.join(PROBABILITIES_DIR_PATH, "enet_unconditional_ndc_achievement_2030.csv"), index=False)
test_4.to_csv(os.path.join(PROBABILITIES_DIR_PATH, "enet_conditional_ndc_achievement_2030.csv"), index=False)

### Plotting the results

#XGBoost
plt.hist(test_1["meets_ndc"], bins=20, alpha=0.5, density=True, label="Unconditional")
plt.hist(test_2["meets_ndc"], bins=20, alpha=0.5, density=True, label="Conditional")

plt.xlabel("Probability of Reaching NDC")
plt.ylabel("Count")
plt.title("Distribution of NDC Achievement using XGBoost")
plt.legend()
plt.show()

#ElasticNet
plt.hist(test_3["meets_ndc"], bins=20, alpha=0.5, density=True, label="Unconditional")
plt.hist(test_4["meets_ndc"], bins=20, alpha=0.5, density=True, label="Conditional")

plt.xlabel("Probability of Reaching NDC")
plt.ylabel("Count")
plt.title("Distribution of NDC Achievement using ElasticNet")
plt.legend()
plt.show()