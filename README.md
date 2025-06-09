# Climate Policy Impact Analyzer (CPIA)

## Objectives

* Build a causal inference model to explore the impact of policies on CO₂e emissions mitigation.
* Develop a predictive model to estimate how likely a country is to meet its target emissions in a specific future year, based on its policies, historical emission trends, and other socio-economic indicators.

## Getting Started

### Create the environment:

```bash
conda create -n cpia_env python=3.11
conda activate cpia_env
```

### Install dependencies:

Inside the project folder, run:

```bash
pip install -r requirements.txt
```

## Important Files

* The `data/processed_data` folder contains processed datasets for:

  * Policies and the policy index
  * Historical emissions data
  * World Bank socio-economic indicators

  **Files:**

  * `IEA_policies_clean.csv`
  * `IEA_scored_cpsi.csv`
  * `total_emissions.csv`
  * `wb_control_vars.csv`

* The `data/data_processing_scripts` folder contains notebooks used to generate the World Bank control datasets and the IEA policy dataset.

  **Files:**

  * `IEA_eda.ipynb`
  * `wb_controls_v2.ipynb`

* The `index/` folder contains files to create the Climate Policy Stringency Index (CPSI) out of `IEA_policies_clean.csv`.

  **Files:**

  * `cpsi_index_final.ipynb`: Make sure to review this notebook since it has some modifications on how to create an explore different types of index.

* The `ml_scripts/` folder contains a notebook where the three datasets (Index, WB Controls and Emissions) are merged to explore the relationship of the index and emissions totals as well as the impact of the index as a feature to predict total emissions.