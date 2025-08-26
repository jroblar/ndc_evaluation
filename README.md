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
* **`parallel_arima.py`** – Loads training data and runs an ensemble of projected predictors using an ARIMA model. Saves the resulting ensemble CSV files in [`ml_scripts/output/ensemble`](ml_scripts/output/ensemble).
* **`postprocess_arima_projections_v3.ipynb`** – Uses the trained model and ARIMA ensemble results to forecast future emissions trends for all countries based on the ARIMA-projected features. This notebook also performs outlier removal and HP filtering. The output file is saved in [`ml_scripts/output/2030_emissions`](ml_scripts/output/2030_emissions).
