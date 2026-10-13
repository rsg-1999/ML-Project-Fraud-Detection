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
arrival batches. The frozen test is never used for fitting or tuning, so every model version is comparable on it.
Retraining uses train + the batches that have arrived; val picks the config and threshold.
`data/test/` has no labels, so it is only scored (05) for monitoring.

**Run it**
1. Add this repo as a Git folder in your workspace (or import the `fraud_pipeline` folder).
2. Run `00_setup`, then upload the four CSVs from `data/` to the volume path it prints.
3. Run `01` → `02` → `03` → `04` → `05` in order (serverless compute).
4. Run `06_process_new_batch` for `batch_1`, then `batch_2` (drifted on purpose, should trigger a retrain), then `batch_3`.
5. Watch the experiment in the MLflow UI: compare `check_*` runs, `score_*` runs and the registered model versions.

`batch_1`/`batch_3` come from the same distribution, so they usually won't trigger a retrain. The drift in `batch_2`
is injected by `01` (`DRIFT_BATCH`, `DRIFT_FACTOR` in `settings.py`).

**Local check** (synthetic data, local Spark + MLflow, no Databricks needed):
```bash
pytest tests/test_smoke.py -s
```

## Notebook

[`Fraud_Detection_workflow.ipynb`](Fraud_Detection_workflow.ipynb) is the original exploratory analysis. The pipeline
fixes some issues it has: SMOTE inside cross-validation instead of before it, thresholds chosen on a validation set
instead of the test set, ROC AUC computed from probabilities, and imputation fitted on train only.
