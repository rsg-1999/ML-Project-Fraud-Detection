"""DESTRUCTIVE reset: deletes landing/ and pending/, the predictions and batch_log tables, any old plain bronze/silver/gold
tables left from the notebook version, and (if asked) the model. The uploaded source CSVs are kept.
Pipeline-owned tables are rebuilt by the initial-load job (it runs the pipeline with a full refresh).
    python reset.py --delete-model yes|no"""
import argparse
import shutil

import settings as S
from lakehouse import spark, t

ap = argparse.ArgumentParser()
ap.add_argument("--delete-model", default="no")
delete_model = ap.parse_args().delete_model == "yes"

sp = spark()
legacy = [f"{layer}_{feed}_{e}" for layer in ("bronze", "silver") for feed in ("labeled", "score") for e in ("client", "invoice")]
legacy += ["gold_labeled", "gold_score"]
for name in ["predictions", "batch_log"]:
    sp.sql(f"DROP TABLE IF EXISTS {t(name)}")
for name in legacy:  # plain tables from the notebook version; pipeline-owned ones refuse to drop and are left alone
    try:
        sp.sql(f"DROP TABLE IF EXISTS {t(name)}")
    except Exception as e:
        print(f"kept {name}: {str(e).splitlines()[0][:100]}")
for folder in ("landing", "pending", "_checkpoints"):
    shutil.rmtree(f"{S.ROOT}/{folder}", ignore_errors=True)

if delete_model:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{S.MODEL_NAME}'"):
        client.delete_model_version(S.MODEL_NAME, mv.version)
    client.delete_registered_model(S.MODEL_NAME)
print("reset done; model deleted" if delete_model else "reset done")
