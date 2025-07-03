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

Files for constructing the Climate Policy Stringency Index (CPSI):

* **`cpsi_index_final.ipynb`** – Builds and explores multiple CPSI index variants. Review for methodological notes and scoring approaches. Produces `IEA_scored_cpsi.csv`.
* **`documentation/index_report.pdf`** – Documentation on how the index was built.

### `ml_scripts/`

Notebooks for merging socio-economic and emissions data (We removed the policy index out of the predictors), performing exploratory data analysis, and training machine learning models to estimate emissions trajectories and the likelihood of target achievement.

* **`model_v7.1.ipynb`** – Trains several models to predict emissions using World Bank indicators. Saves the best pipelines as `.pkl` files under [`ml_scripts/output/models`](ml_scripts/output/models).
* **`run_ensemble.ipynb`** – Loads training data and runs an ensemble of projected predictors using an ARIMA model. Saves the ensemble CSV files under [`ml_scripts/output/ensemble`](ml_scripts/output/ensemble). Since these files can be too large, we are not tracking them in GitHub. In addition, this notebook allows you to run ARIMA with an auto-tuning feature. This may take longer to run, but it might be worth exploring the results.
* **`ensemble_analysis.ipynb`** – Once we have a trained model to predict emissions with historical WB indicator data, and CSV files with ensemble experiments representing multiple future trajectories for our predictors, we can finally predict emissions for each of those futures with our model and obtain the distribution of emissions all the way to 2030.

## TODO:

* Run the ARIMA projections with the auto-tuning feature and see how much it improves compared to the baseline.
* Run a larger experiment, around 100 to 1,000 futures.
* Finalize the analysis in `ensemble_analysis.ipynb` or create a new notebook to obtain the distribution of predicted emissions in 2030 for each country. Then, calculate the probability of a country reaching its mitigation goal. We still need to collect the targets for each country.
