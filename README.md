# Emissions Target Probability Explorer (ETPE)

## 🎯 Objectives

* Develop a framework to estimate the likelihood that a country will meet its 2030 emissions target.
* Generate probabilistic emissions projections ("pony-tails") for each country to create a distribution of possible emission values by 2030.
* Assess each country’s 2030 emissions target by calculating the probability of target achievement according to the generated emissions distribution.

## 🚀 Getting Started

**Run:**

```bash
conda env create -f environment.yml
conda activate etpe_env
```

## 📁 Important Files and Structure

### `data/processed_data/`

Contains cleaned and processed datasets:

* **`IEA_policies_clean.csv`** – Cleaned IEA climate policy dataset.
* **`IEA_scored_cpsi.csv`** – Climate Policy Stringency Index (CPSI). Run `cpsi_index_final.ipynb` to generate this file (too large for GitHub).
* **`total_emissions.csv`** – Historical emissions time series per country.
* **`wb_indicators.csv`** – Historical time series of socio-economic indicators per country.

### `data/data_processing_scripts/`

Notebooks used to process raw datasets from the World Bank and IEA:

* **`IEA_eda.ipynb`** – Performs EDA and cleans the IEA policy dataset. Produces `IEA_policies_clean.csv`.
* **`wb_data_etl.ipynb`** – Fetches and processes World Bank indicators. Produces `wb_indicators.csv`.
* **`emissions_data_etl.ipynb`** – Processes EDGAR emissions data. Produces `total_emissions.csv`.

### `index/`

Files for constructing the policy index:

* **`add_index_to_policies.ipynb`** – Builds and explores multiple climate policy index variants. Review for methodological notes and scoring approaches. Produces `IEA_scored_index.csv`.
* **`documentation/index_report.pdf`** – Documentation on how the index was built.

### `ml_scripts/`

Notebooks for merging socio-economic and emissions data (We removed the policy index out of the predictors), performing exploratory data analysis, and training machine learning models to estimate emissions trajectories and the likelihood of target achievement.

* **`model_v7.3.ipynb`** – Trains multiple models to predict emissions using World Bank indicators and a Climate Policy Index. Saves the best-performing pipelines as `.pkl` files in [`ml_scripts/output/models`](ml_scripts/output/models).
* **`parallel_arima.py`** – Legacy ARIMA ensemble generator for future predictors.
  * **Inputs:** [`ml_scripts/output/training/training_df_top15_preds.csv`](ml_scripts/output/training/training_df_top15_preds.csv).
  * **Outputs:** [`ml_scripts/output/ensemble/ensemble_arima_<n_scenarios>.parquet`](ml_scripts/output/ensemble).
* **`parallel_arima_v2.py`** – Refactored ARIMA ensemble generator with level-space simulation and bounded-variable handling.
  * **Inputs:** [`ml_scripts/output/training/training_df_top15_preds.csv`](ml_scripts/output/training/training_df_top15_preds.csv).
  * **Outputs:** [`ml_scripts/output/ensemble/ensemble_arima_<n_scenarios>.parquet`](ml_scripts/output/ensemble).
* **`postprocess_arima_projections_v3.ipynb`** – Uses trained models and ARIMA ensemble features to forecast future emissions, with outlier removal and HP filtering.
  * **Inputs:** [`ml_scripts/output/training/training_df_top15_preds.csv`](ml_scripts/output/training/training_df_top15_preds.csv), [`ml_scripts/output/ensemble/ensemble_arima_<n_scenarios>.parquet`](ml_scripts/output/ensemble), and model `.pkl` files in [`ml_scripts/output/models`](ml_scripts/output/models).
  * **Outputs:** [`ml_scripts/output/2030_emissions/post_processed_projected_emissions.csv`](ml_scripts/output/2030_emissions).
* **`postprocess_arima_projections_v4.ipynb`** – Updated postprocessing notebook with the same ARIMA ensemble inputs and emissions forecasting pipeline.
  * **Inputs:** [`ml_scripts/output/training/training_df_top15_preds.csv`](ml_scripts/output/training/training_df_top15_preds.csv), [`ml_scripts/output/ensemble/ensemble_arima_<n_scenarios>.parquet`](ml_scripts/output/ensemble), and model `.pkl` files in [`ml_scripts/output/models`](ml_scripts/output/models).
  * **Outputs:** [`ml_scripts/output/2030_emissions/post_processed_projected_emissions.csv`](ml_scripts/output/2030_emissions).
