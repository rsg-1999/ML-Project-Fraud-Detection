# Fraud Detection in Electricity Billing

## Overview
This project focuses on detecting fraudulent activities in electricity billing using machine learning techniques. The goal is to minimize financial losses while ensuring customer satisfaction. We developed and evaluated multiple models to classify fraudulent and non-fraudulent clients based on their billing and client data.

## Motivation
Electricity fraud has resulted in significant financial losses for the company, amounting to €600,000. By building a robust fraud detection model, this project aims to:

- Reduce financial losses by identifying fraudulent clients.
- Balance operational costs and customer satisfaction.
- Improve the company’s overall revenue protection strategy.

Please refer to the Python notebook for a detailed walkthrough of our workflow.

More information about the challenge can be found here: https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge

## Set up your Environment



### **`macOS`** type the following commands : 



- For installing the virtual environment and the required package you can either follow the commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
Or ....
-  use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```

### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


   
## Databricks pipeline (Free Edition)

Everything lives in [`fraud_pipeline/`](fraud_pipeline). The notebooks are thin; the logic is in
`settings.py` (all knobs), `lakehouse.py` (bronze/silver/gold) and `modeling.py` (train, registry, monitoring).

```
source CSVs ──01──> landing/ (initial + 3 arrival batches + score feed)
                      │ 02
                      ▼
              bronze (raw as-is, Delta) ─03─> silver (typed, deduped) ─> gold (1 row/client + split)
                                                                             │ 04
              MLflow experiment <── runs, metrics ── train / val / frozen test ┘
              Unity Catalog model: aliases `challenger` / `champion`
                      │ 05                         │ 06 (per labeled batch)
          score unlabeled test data         score batch with champion, check drift + PR-AUC,
          (label-free monitoring)           retrain if triggered, promote only if it beats champion
```

**Splits.** Labeled clients are hashed into buckets: 60% train, 15% val, 10% frozen test, and 15% held back as three
arrival batches. The frozen test is never used for fitting or tuning. 30% of every arrived batch is a `recent` holdout
that no model ever trains on; the rest of the batch is added to the training data when a retrain happens.
`data/test/` has no labels, so it is only scored (05) for monitoring.

**Drift and retraining.** `batch_2` and `batch_3` are drifted on purpose (01): invoice readings are scaled
(covariate drift) and the top 5% consumers are also labeled fraud (concept drift). For each arriving batch (06):
the champion scores it, the check logs PR-AUC and PSI per feature to MLflow, and a retrain happens if PR-AUC fell more
than 25% or any PSI > 0.25. A retrained model becomes `champion` only if it beats the champion on `recent` data
by `MIN_GAIN` without losing more than `MAX_TEST_REGRESSION` on the frozen test (all in `settings.py`).

**Run it**
1. Add this repo as a Git folder in your workspace.
2. Run `00_setup`, then upload the four CSVs from `data/` to the volume path it prints.
3. Run `01` to `05` in order (serverless compute), or deploy the `fraud_initial_load` job.
4. Run `06_process_new_batch` with `batch = next` three times (or schedule the `fraud_next_batch` job); each run picks the
   next batch that has not arrived yet.
5. Watch the experiment in the MLflow UI: `check_*`, `score_*` and `train_*` runs and the registered model versions.
6. `99_reset` wipes tables, landing data and checkpoints (not the uploaded CSVs) for a clean start.

**Jobs as code.** [`databricks.yml`](databricks.yml) defines both jobs (they run the notebooks from this Git branch):
```bash
databricks auth login --host <your workspace url>
databricks bundle validate && databricks bundle deploy
```
Ingestion uses Auto Loader (`settings.AUTOLOADER`); if you drop a bronze table, also delete its folder under
`_checkpoints/` (or just run `99_reset`).

**Local check** (synthetic data, local Spark + MLflow, no Databricks needed):
```bash
pytest tests/test_smoke.py -s
```

## Notebook

[`Fraud_Detection_workflow.ipynb`](Fraud_Detection_workflow.ipynb) is the original exploratory analysis. The pipeline
fixes some issues it has: SMOTE inside cross-validation instead of before it, thresholds chosen on a validation set
instead of the test set, ROC AUC computed from probabilities, and imputation fitted on train only.
