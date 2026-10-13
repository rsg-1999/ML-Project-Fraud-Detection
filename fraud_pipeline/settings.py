"""Shared constants. Change CATALOG if your Free Edition catalog isn't called `workspace`."""
CATALOG, SCHEMA, VOLUME = "workspace", "fraud", "raw"
ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"  # upload the 4 source CSVs to ROOT/source/
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_model"
EXPERIMENT_NAME = "fraud-detection"  # created under /Users/<you>/
SEED = 101

# client_id hashes to a bucket 0..99, so a client always lands in the same split on every rerun.
SPLITS = {"train": (0, 60), "val": (60, 75), "test": (75, 85),
          "batch_1": (85, 90), "batch_2": (90, 95), "batch_3": (95, 100)}
BATCHES = ["batch_1", "batch_2", "batch_3"]
LANDING = {"initial": (0, 85), **{b: SPLITS[b] for b in BATCHES}}  # what 01_simulate_arrivals writes

# simulated drift in these batches: (1) invoice readings are scaled (covariate drift) and
# (2) a new kind of fraud appears: the DRIFT_NEW_FRAUD highest consumers are relabeled as fraud (concept drift)
DRIFT_BATCHES, DRIFT_FACTOR, DRIFT_NEW_FRAUD = ["batch_2", "batch_3"], 3.0, 0.05

# ingestion: Auto Loader on Databricks; False falls back to plain Spark (used by the local test)
AUTOLOADER = True

# retrain triggers (see modeling.check_batch)
PR_AUC_DROP = 0.25  # relative drop of batch PR-AUC vs the champion's frozen-test PR-AUC
PSI_MAX = 0.25      # >0.25 on any numeric feature or on the score = significant shift
# A challenger is promoted if it beats the champion on `recent` data (a held-out 30% of every arrived batch)
# by MIN_GAIN PR-AUC without losing more than MAX_TEST_REGRESSION on the frozen test. Before any batch has
# arrived there is no recent data, so it must beat the champion's frozen-test PR-AUC by MIN_GAIN.
MIN_GAIN, MAX_TEST_REGRESSION, RECENT_SHARE = 0.005, 0.02, 0.3

CAT = ["disrict", "client_catg", "region", "tarif_type_mode", "counter_coefficient_mode",
       "counter_type_mode", "reading_remarque_min", "reading_remarque_mode"]
CONS = [f"consommation_level_{i}" for i in range(1, 5)]
NUM = (["creation_day", "creation_month", "creation_year", "bill_count",
        "avg_frequency_of_bills_in_days", "single_bill_client"]
       + [f"{c}_{s}" for c in CONS + ["diff_index"] for s in ("mean", "std")])

CONFIGS = [  # each becomes an MLflow child run; best validation PR-AUC wins
    {"name": "xgb_weighted", "imbalance": "weight", "max_depth": 4, "n_estimators": 300, "learning_rate": 0.05},
    {"name": "xgb_weighted_deep", "imbalance": "weight", "max_depth": 6, "n_estimators": 300, "learning_rate": 0.05},
    {"name": "xgb_smote", "imbalance": "smote", "max_depth": 4, "n_estimators": 300, "learning_rate": 0.05},
]
