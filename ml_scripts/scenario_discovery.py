from ema_workbench.analysis import prim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def filter_years(df, years, y):
    """
    Filter df for specific years AND only rows whose iso_alpha_3 exist in y.
    """
    isos = set(y["iso_alpha_3"].unique())
    return {
        year: df[(df["year"] == year) & (df["iso_alpha_3"].isin(isos))].\
            select_dtypes(include="number")\
                .copy()
        for year in years
    }


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

# Data

y = pd.read_csv(os.path.join(PROBABILITIES_DIR_PATH, "xgb_unconditional_ndc_achievement_2030.csv"))
X = pd.read_csv(os.path.join(TRAINING_DIR_PATH, "training_df_lags.csv"))
X_ref = filter_years(X, [2010, 2015, 2020], y)

X_prim = X_ref[2020]
success = (y["meets_ndc"] > 0.80).values
fail = (y["meets_ndc"] < 0.80).values

success.mean()
fail.mean()

prim_alg = prim.Prim(X_prim, fail, threshold=0.8)
box = prim_alg.find_box()

box.show_tradeoff()

box_list = list(range(1, len(box.box_lims), 2))
for b in box_list:
    box.inspect(b)
    box.inspect(b, style="graph")

box.inspect(29)
box.inspect(3, style="graph")