# Emissions Target Probability Explorer (ETPE)

This repository contains the workflow used to estimate the probability that a country meets its 2030 emissions target, generate the scenario-discovery diagnostics, and reproduce the figures assembled in [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb).

The repository already includes the main artifacts used by the current paper workflow, so a new user can either:

1. reproduce the published figures from the committed outputs, or
2. rerun the full pipeline and generate a new run ID.

For the current paper run, you can start from the ARIMA stage because the training data and fitted model are already available in the repository.

## Current paper run IDs

The current experiment run ID used by the main workflow is `1773188058`.

It is stored in:

- [`arima/config/arima_projections_with_lags_config.yaml`](/Users/tony/Documents/research_project/etpe_project/arima/config/arima_projections_with_lags_config.yaml)
- [`scenario_discovery/config/config.yaml`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/config/config.yaml)

The current scenario-discovery batch output referenced in [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb) is:

- `1773188058_124_20260408_145246`

There is also an older ARIMA config snapshot with a different run ID:

- [`arima/config/arima_projections_config.yaml`](/Users/tony/Documents/research_project/etpe_project/arima/config/arima_projections_config.yaml) uses `1771445441`

For the paper workflow, use `1773188058`.

## Environment setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate etpe_env
```

Then start Jupyter from the repository root when working with the notebooks:

```bash
jupyter notebook
```

## Repository workflow

The pipeline is organized in five stages.

### 1. Training data and model fitting

- Input panel: [`data/country_panel_data/panel_2002_2022_136countries_68vars.parquet`](/Users/tony/Documents/research_project/etpe_project/data/country_panel_data/panel_2002_2022_136countries_68vars.parquet)
- Data dictionary: [`data/country_panel_data/DATA_DICTIONARY.md`](/Users/tony/Documents/research_project/etpe_project/data/country_panel_data/DATA_DICTIONARY.md)
- Training notebook: [`ml/ml_model_training.ipynb`](/Users/tony/Documents/research_project/etpe_project/ml/ml_model_training.ipynb)
- Saved outputs:
  - training table in [`ml/output/training`](/Users/tony/Documents/research_project/etpe_project/ml/output/training)
  - fitted model in [`ml/output/models`](/Users/tony/Documents/research_project/etpe_project/ml/output/models)

For the current paper workflow, the relevant committed artifacts are:

- [`ml/output/training/training_df_1773188058.csv`](/Users/tony/Documents/research_project/etpe_project/ml/output/training/training_df_1773188058.csv)
- [`ml/output/models/enet_pipeline_1773188058.pkl`](/Users/tony/Documents/research_project/etpe_project/ml/output/models/enet_pipeline_1773188058.pkl)

Important: this notebook creates a timestamp-based run ID when it saves outputs. If you retrain from scratch, you will generate a new run ID and must update all downstream references.

### 2. Future driver simulation with ARIMA/SARIMAX

- Main script: [`arima/parallel_arima_v5.py`](/Users/tony/Documents/research_project/etpe_project/arima/parallel_arima_v5.py)
- Main config for the paper run: [`arima/config/arima_projections_with_lags_config.yaml`](/Users/tony/Documents/research_project/etpe_project/arima/config/arima_projections_with_lags_config.yaml)
- Variable rules: [`arima/config/variable_projection_rules.json`](/Users/tony/Documents/research_project/etpe_project/arima/config/variable_projection_rules.json)
- Output:
  - [`arima/output/ensemble/ensemble_arima_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/ensemble/ensemble_arima_1773188058.parquet)

Run:

```bash
python arima/parallel_arima_v5.py
```

### 3. Postprocess ensemble emissions

- Notebook: [`arima/postprocess_arima_projections_v5.ipynb`](/Users/tony/Documents/research_project/etpe_project/arima/postprocess_arima_projections_v5.ipynb)
- Inputs:
  - training table from stage 1
  - ARIMA ensemble from stage 2
  - trained elastic-net pipeline from stage 1
- Outputs:
  - [`arima/output/hp_filtered/historical_emissions_hp_trend_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/hp_filtered/historical_emissions_hp_trend_1773188058.parquet)
  - [`arima/output/hp_filtered/predicted_hp_filtered_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/hp_filtered/predicted_hp_filtered_1773188058.parquet)
  - [`arima/output/postprocessed_ensemble/postprocessed_ensemble_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/postprocessed_ensemble/postprocessed_ensemble_1773188058.parquet)

This notebook currently uses `run_id = 1773188058`.

### 4. NDC probability analysis

- Probability notebook: [`ndc_probability/ndc_probability_analysis.ipynb`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/ndc_probability_analysis.ipynb)
- Utility module used by the notebook: [`ndc_probability/utils/prob_utils.py`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/utils/prob_utils.py)
- Main output table for the paper run:
  - [`ndc_probability/tables/ndc_probability_analysis_1773188058.csv`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/tables/ndc_probability_analysis_1773188058.csv)

The current workflow treats the table in [`ndc_probability/tables`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/tables) as the canonical downstream artifact. For a given run, the expected filename is:

- `ndc_probability/tables/ndc_probability_analysis_<RUN_ID>.csv`

For the paper run, keep `RUN_ID = 1773188058` in [`ndc_probability/ndc_probability_analysis.ipynb`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/ndc_probability_analysis.ipynb) and execute the notebook to regenerate the table if needed.

Important: scenario discovery now reads the NDC probability table matching the configured `run_id` from [`scenario_discovery/config/config.yaml`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/config/config.yaml). It no longer depends on the deprecated `data/ndc_data/ndc_reference.csv` file.

### 5. Scenario discovery and paper figures

- Scenario discovery batch runner: [`scenario_discovery/run_scenario_discovery_batch.py`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/run_scenario_discovery_batch.py)
- Config: [`scenario_discovery/config/config.yaml`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/config/config.yaml)
- Current batch output used in the paper notebook:
  - [`scenario_discovery/output/1773188058_124_20260408_145246`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/output/1773188058_124_20260408_145246)
- Final figure notebook: [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb)
- Exported figure files: [`paper_figures`](/Users/tony/Documents/research_project/etpe_project/paper_figures)

Run scenario discovery:

```bash
python scenario_discovery/run_scenario_discovery_batch.py
```

This script reads:

- [`arima/output/postprocessed_ensemble/postprocessed_ensemble_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/postprocessed_ensemble/postprocessed_ensemble_1773188058.parquet)
- [`arima/output/ensemble/ensemble_arima_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/ensemble/ensemble_arima_1773188058.parquet)
- [`ndc_probability/tables/ndc_probability_analysis_1773188058.csv`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/tables/ndc_probability_analysis_1773188058.csv)

and writes a timestamped batch directory under [`scenario_discovery/output`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/output).

For the current paper notebook, set:

- `RUN_ID = 1773188058`
- `SD_OUTPUT_ID = "1773188058_124_20260408_145246"`

inside [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb).

## Fastest way to reproduce the current paper figures

If your goal is to reproduce the results already represented in [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb), you do not need to rerun the entire pipeline because the required artifacts are already committed.

For run `1773188058`, start from the ARIMA stage, not from model training. The training table and fitted elastic-net model are already available in the repository.

Use the repository as-is and verify that these files exist:

- [`ml/output/training/training_df_1773188058.csv`](/Users/tony/Documents/research_project/etpe_project/ml/output/training/training_df_1773188058.csv)
- [`ml/output/models/enet_pipeline_1773188058.pkl`](/Users/tony/Documents/research_project/etpe_project/ml/output/models/enet_pipeline_1773188058.pkl)
- [`arima/output/ensemble/ensemble_arima_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/ensemble/ensemble_arima_1773188058.parquet)
- [`arima/output/postprocessed_ensemble/postprocessed_ensemble_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/postprocessed_ensemble/postprocessed_ensemble_1773188058.parquet)
- [`arima/output/hp_filtered/historical_emissions_hp_trend_1773188058.parquet`](/Users/tony/Documents/research_project/etpe_project/arima/output/hp_filtered/historical_emissions_hp_trend_1773188058.parquet)
- [`ndc_probability/tables/ndc_probability_analysis_1773188058.csv`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/tables/ndc_probability_analysis_1773188058.csv)
- [`scenario_discovery/output/1773188058_124_20260408_145246`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/output/1773188058_124_20260408_145246)

To reproduce the current run outputs from the earliest necessary stage:

1. Run [`arima/parallel_arima_v5.py`](/Users/tony/Documents/research_project/etpe_project/arima/parallel_arima_v5.py).
2. Execute [`arima/postprocess_arima_projections_v5.ipynb`](/Users/tony/Documents/research_project/etpe_project/arima/postprocess_arima_projections_v5.ipynb).
3. Execute [`ndc_probability/ndc_probability_analysis.ipynb`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/ndc_probability_analysis.ipynb).
4. Run [`scenario_discovery/run_scenario_discovery_batch.py`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/run_scenario_discovery_batch.py) if you want to rebuild the scenario-discovery outputs.
5. Open and run [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb).

## Full rerun workflow

If you want to regenerate the full experiment pipeline instead of reusing the committed outputs, run the stages in this order:

1. Execute [`ml/ml_model_training.ipynb`](/Users/tony/Documents/research_project/etpe_project/ml/ml_model_training.ipynb) and record the new timestamp-based run ID.
2. Update the downstream run ID references in:
   - [`arima/config/arima_projections_with_lags_config.yaml`](/Users/tony/Documents/research_project/etpe_project/arima/config/arima_projections_with_lags_config.yaml)
   - [`arima/postprocess_arima_projections_v5.ipynb`](/Users/tony/Documents/research_project/etpe_project/arima/postprocess_arima_projections_v5.ipynb)
   - [`ndc_probability/ndc_probability_analysis.ipynb`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/ndc_probability_analysis.ipynb)
   - [`scenario_discovery/config/config.yaml`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/config/config.yaml)
   - [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb)
3. Run [`arima/parallel_arima_v5.py`](/Users/tony/Documents/research_project/etpe_project/arima/parallel_arima_v5.py).
4. Execute [`arima/postprocess_arima_projections_v5.ipynb`](/Users/tony/Documents/research_project/etpe_project/arima/postprocess_arima_projections_v5.ipynb).
5. Execute [`ndc_probability/ndc_probability_analysis.ipynb`](/Users/tony/Documents/research_project/etpe_project/ndc_probability/ndc_probability_analysis.ipynb) to produce `ndc_probability/tables/ndc_probability_analysis_<RUN_ID>.csv`.
6. Run [`scenario_discovery/run_scenario_discovery_batch.py`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery/run_scenario_discovery_batch.py) and record the new timestamped batch directory.
7. Update `SD_OUTPUT_ID` in [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb) to that new batch directory.
8. Execute [`paper_figures.ipynb`](/Users/tony/Documents/research_project/etpe_project/paper_figures.ipynb).

## Main folders

- [`data`](/Users/tony/Documents/research_project/etpe_project/data): raw and processed data used by the pipeline
- [`ml`](/Users/tony/Documents/research_project/etpe_project/ml): model training notebook, utilities, and saved model artifacts
- [`arima`](/Users/tony/Documents/research_project/etpe_project/arima): future covariate simulation and emissions postprocessing
- [`ndc_probability`](/Users/tony/Documents/research_project/etpe_project/ndc_probability): NDC probability analysis and output tables used downstream by scenario discovery
- [`scenario_discovery`](/Users/tony/Documents/research_project/etpe_project/scenario_discovery): batch scenario discovery outputs and reports
- [`paper_figures`](/Users/tony/Documents/research_project/etpe_project/paper_figures): exported paper-ready figures

## Notes for paper sharing

- The paper-facing workflow currently depends on committed intermediate artifacts for run `1773188058`.
- Some notebooks use hard-coded `run_id` values rather than parameterized inputs.
- The machine-learning training notebook generates a fresh timestamp each time it saves outputs, so exact filename reproduction requires carrying that new run ID through the downstream pipeline.
