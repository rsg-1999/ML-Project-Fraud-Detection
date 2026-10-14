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

Everything lives in [`fraud_pipeline/`](fraud_pipeline) as plain Python scripts, deployed with the Databricks CLI.
Shared code: `settings.py` (all knobs), `transforms.py` (bronze/silver/gold logic), `lakehouse.py` (simulation, arrivals),
`modeling.py` (train, registry, monitoring).

```
data CSVs ─upload─> volume/source ─simulate_arrivals─> landing/ (initial)   pending/ (batch_1..3)
                                                          │                      │ arrive_next_batch (one per run)
                                                          ▼                      ▼
                     ETL pipeline (pipeline_*.py):  landing/ ──Auto Loader──> BRONZE ─> SILVER ─> GOLD
                                                                                                 │
        MLflow experiment + UC model (champion / challenger) <── train.py / check_batch.py ─────┘
                                      score.py: score the unlabeled test data, label-free monitoring
```

| Script | Does |
|---|---|
| `setup_workspace.py` | creates schema and volume |
| `simulate_arrivals.py` | splits the labeled CSVs: initial data to `landing/`, three batches wait in `pending/` |
| `pipeline_bronze/silver/gold.py` | the ETL pipeline: Auto Loader into bronze, typed and de-duplicated silver with data-quality rules, gold features and splits |
| `train.py` | trains 3 configs, registers the best as `challenger`, `champion` if none exists |
| `score.py` | scores the unlabeled clients with the champion, logs score/PSI monitoring to MLflow |
| `arrive_next_batch.py` | moves the next batch from `pending/` to `landing/` and logs it |
| `check_batch.py` | checks the oldest unchecked batch with the champion, retrains if triggered |
| `reset.py` | wipes landing/pending/predictions (and the model if asked); keeps the uploaded CSVs |

**Splits.** Labeled clients are hashed into buckets: 60% train, 15% val, 10% frozen test, and 15% held back as three
arrival batches. The frozen test is never used for fitting or tuning. 30% of every arrived batch is a `recent` holdout
that no model ever trains on; the rest of the batch is added to the training data when a retrain happens.
`data/test/` has no labels, so it is only scored (05) for monitoring.

**Drift and retraining.** `batch_2` and `batch_3` are drifted on purpose (01): invoice readings are scaled
(covariate drift) and the top 5% consumers are also labeled fraud (concept drift). For each arriving batch (06):
the champion scores it, the check logs PR-AUC and PSI per feature to MLflow, and a retrain happens if PR-AUC fell more
than 25% or any PSI > 0.25. A retrained model becomes `champion` only if it beats the champion on `recent` data
by `MIN_GAIN` without losing more than `MAX_TEST_REGRESSION` on the frozen test (all in `settings.py`).

**Run it** (Databricks CLI, `-p` is your profile name from `databricks auth profiles`)
```bash
databricks bundle validate -p <profile> && databricks bundle deploy -p <profile>
databricks bundle run fraud_reset -p <profile> --params delete_model=yes   # only if you have old tables/models
databricks bundle run fraud_initial_load -p <profile>    # setup, simulate, pipeline (full refresh), train, score
# upload the 4 CSVs to /Volumes/workspace/fraud/raw/source/ before simulate_arrivals runs, e.g.
#   databricks fs cp data/train/client_train.csv dbfs:/Volumes/workspace/fraud/raw/source/ -p <profile>
databricks bundle run fraud_next_batch -p <profile>      # x3: arrive, pipeline, check/retrain, score
```
The first `fraud_initial_load` fails at `simulate_arrivals` if the CSVs are not uploaded yet: upload them, run it again.
`fraud_next_batch` has a daily schedule (paused). With nothing left to arrive it exits quietly.

**Local check** (synthetic data, local Spark + MLflow, no Databricks needed):
```bash
pytest tests/test_smoke.py -s
```

## Notebook

[`Fraud_Detection_workflow.ipynb`](Fraud_Detection_workflow.ipynb) is the original exploratory analysis. The pipeline
fixes some issues it has: SMOTE inside cross-validation instead of before it, thresholds chosen on a validation set
instead of the test set, ROC AUC computed from probabilities, and imputation fitted on train only.
