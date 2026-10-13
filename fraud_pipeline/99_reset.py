# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Reset (destructive)
# MAGIC Drops every bronze/silver/gold/predictions table, deletes `landing/` and the Auto Loader checkpoints, and
# MAGIC optionally deletes the registered model. **The uploaded `source/` CSVs are kept.** After this, rerun 01 to 05.
# MAGIC Use it whenever you change the simulation or the features and want a clean start.

# COMMAND ----------
dbutils.widgets.dropdown("delete_model", "no", ["no", "yes"])

# COMMAND ----------
import settings as S

for feed in ("labeled", "score"):
    for layer in ("bronze", "silver"):
        for entity in ("client", "invoice"):
            spark.sql(f"DROP TABLE IF EXISTS {S.CATALOG}.{S.SCHEMA}.{layer}_{feed}_{entity}")
    spark.sql(f"DROP TABLE IF EXISTS {S.CATALOG}.{S.SCHEMA}.gold_{feed}")
spark.sql(f"DROP TABLE IF EXISTS {S.CATALOG}.{S.SCHEMA}.predictions")
for folder in ("landing", "_checkpoints"):
    dbutils.fs.rm(f"{S.ROOT}/{folder}", True)

# COMMAND ----------
if dbutils.widgets.get("delete_model") == "yes":
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{S.MODEL_NAME}'"):
        client.delete_model_version(S.MODEL_NAME, mv.version)
    client.delete_registered_model(S.MODEL_NAME)
