import os
import numpy as np
import pandas as pd

from scipy.stats import qmc
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

import joblib
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def simulate_country(
    df, iso, feature_cols, years, horizon,
    n_scenarios, arima_order, auto_tune_arima,
    max_p, max_d, max_q, rand_seed
):
    """Simulations per country"""

    rng = np.random.default_rng(rand_seed)
    sims_dict = {}

    # Arima
    for feat in feature_cols:
        hist = df.set_index("year")[feat].copy()
        hist.index = pd.PeriodIndex(hist.index, freq="Y")

        if auto_tune_arima:
            order = auto_arima(
                hist, seasonal=False, stepwise=True, suppress_warnings=True,
                max_p=max_p, max_d=max_d, max_q=max_q,
                error_action='ignore'
            ).order
        else:
            order = arima_order

        fit = SARIMAX(hist, order=order,
                      enforce_stationarity = True, # Series muy cortas
                      enforce_invertibility = True).fit(disp=False) # Mejor que este polinomio sea invertible

        sims = np.vstack([
            fit.simulate(nsimulations=horizon, anchor="end").values
            for _ in range(n_scenarios)
        ])
        sims_dict[feat] = sims

    # LHCS
    sampler = qmc.LatinHypercube(d=len(feature_cols), seed=rand_seed)
    idx = (sampler.random(n_scenarios) * n_scenarios).astype(int)

    # Output
    rows = []
    for s in range(n_scenarios):
        fid = f"id_{iso}_{s+1}"
        for t, year in enumerate(years):
            row = {"iso_alpha_3": iso, "future_id": fid, "year": year}
            for j, feat in enumerate(feature_cols):
                sim_i = idx[s, j]
                row[feat] = sims_dict[feat][sim_i, t]
            rows.append(row)

    return pa.Table.from_pandas(pd.DataFrame(rows))

def generate_ensemble_arima_parallel(
    df: pd.DataFrame,
    feature_cols: list[str],
    start_year: int,
    end_year: int,
    n_scenarios: int,
    out_path: str,
    arima_order=(1,1,1),
    auto_tune_arima=False,
    max_p=3, max_d=1, max_q=3,
    n_jobs: int = -1,
    random_state: int | None = None
):
    """Parallelized Version of Arima Model and Output Writing"""
    years   = np.arange(start_year, end_year + 1)
    horizon = len(years)

    # Split df by ISO 
    iso_groups = {iso: grp.copy() for iso, grp in df.groupby("iso_alpha_3")}

    # Partial for joblib
    worker = partial(
        simulate_country,
        feature_cols=feature_cols,
        years=years,
        horizon=horizon,
        n_scenarios=n_scenarios,
        arima_order=arima_order,
        auto_tune_arima=auto_tune_arima,
        max_p=max_p, max_d=max_d, max_q=max_q
    )

    # Seeding the simulations
    iso_list = sorted(iso_groups)
    seeds    = {iso: (hash((iso, random_state)) % 2**32) for iso in iso_list}

    tables = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
        joblib.delayed(worker)(df=iso_groups[iso], iso=iso, rand_seed=seeds[iso])
        for iso in iso_list
    )

    # Output
    schema   = tables[0].schema
    writer   = pq.ParquetWriter(os.path.join(ENSEMBLE_DIR_PATH, f"ensemble_arima_{n_scenarios}.parquet"), 
                                schema, compression="snappy")
    for tbl in tables:
        writer.write_table(tbl)
    writer.close()

    print(f"Saved {len(iso_list)} countries × {n_scenarios} scenarios "
          f"to {out_path}")


### Running the script

SCRIPT_DIR_PATH = os.getcwd()
ROOT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, "data")
PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "processed_data")
RESULTS_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "results")
OUTPUT_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "output")
ENSEMBLE_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "ensemble")
MODELS_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "models")
TRAINING_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "training")

# Make sure the output directories exist
os.makedirs(ENSEMBLE_DIR_PATH, exist_ok=True)
os.makedirs(MODELS_DIR_PATH, exist_ok=True)
os.makedirs(TRAINING_DIR_PATH, exist_ok=True)

training_df_log_transformed = pd.read_csv(
    os.path.join(TRAINING_DIR_PATH, "training_df_log_transformed.csv"))
training_df_log_transformed.head()

numeric_cols = training_df_log_transformed.select_dtypes(include=["float64", "int64"]).columns.tolist()
numeric_cols_to_drop = ["year", "log_total_emissions"]
numeric_cols = [col for col in numeric_cols if col not in numeric_cols_to_drop]
numeric_cols

generate_ensemble_arima_parallel(
    df=training_df_log_transformed,
    feature_cols=numeric_cols,
    start_year=2022,
    end_year=2030,
    n_scenarios=1000,                          
    out_path="ensemble_arima_1000.parquet",
    arima_order=(1, 1, 1),                     
    auto_tune_arima= True,                      
    max_p=3, max_d=1, max_q=3,                 
    n_jobs=-1,                                 
    random_state=42                            
)

### Checking the output

ensemble_df = pd.read_parquet(os.path.join(ENSEMBLE_DIR_PATH, "ensemble_arima_1000.parquet"))
print("Shape:", ensemble_df.shape)
ensemble_df.head(20)