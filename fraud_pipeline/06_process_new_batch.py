# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · New labeled batch arrives
# MAGIC `batch = next` picks the first arrival batch that has not been ingested yet (and exits quietly if there is none),
# MAGIC so this notebook can run on a schedule. It ingests the batch, rebuilds the tables, scores it with the champion,
# MAGIC logs the check to MLflow, and retrains if PR-AUC fell too far or any feature drifted (thresholds in `settings.py`).
# MAGIC A retrained model is promoted only if it beats the champion on `recent` data without regressing on the frozen test.

# COMMAND ----------
# MAGIC %pip install -q xgboost imbalanced-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.dropdown("batch", "next", ["next", "batch_1", "batch_2", "batch_3"])

# COMMAND ----------
import lakehouse
import modeling

batch = dbutils.widgets.get("batch")
if batch == "next":
    batch = lakehouse.next_batch()
if batch is None:
    dbutils.notebook.exit("no new batch")

modeling.setup_mlflow()
modeling.check_batch(batch)
