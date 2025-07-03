# Emissions Target Probability Explorer (ETPE)

## 🎯 Objectives

* Develop a framework to estimate the likelihood that a country will meet its 2030 emissions target, based on historical emissions trends and socio-economic indicators.
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

Notebooks used to process raw datasets from World Bank and IEA:

* **`IEA_eda.ipynb`** – Performs EDA and cleans the IEA policy dataset. Produces `IEA_policies_clean.csv`.
* **`wb_data_etl.ipynb`** – Fetches and processes World Bank indicators. Produces `wb_indicators.csv`.
* **`emissions_data_etl.ipynb`** – Processes EDGAR emissions data. Produces `total_emissions.csv`.

### `index/`

Files for constructing the Climate Policy Stringency Index (CPSI):

* **`cpsi_index_final.ipynb`** – Builds and explores multiple CPSI index variants. Review for methodological notes and scoring approaches. Produces `IEA_scored_cpsi.csv`.
* **`documentation/index_report.pdf`** – Documentation on how the index was built.

### `ml_scripts/`

Notebook(s) for merging socio-economic and emissions data, perform exploratory data analysis  and for training machine learning models to estimate emissions trajectories and likelihood of target achievement.

* **`model_v6.ipynb`** – Trains an RF model to predict emissions using World Bank indicators. In addition it creates an ensemble of projected predictors using an ARIMA model and then predicts future emissions based on this new trajectories.

