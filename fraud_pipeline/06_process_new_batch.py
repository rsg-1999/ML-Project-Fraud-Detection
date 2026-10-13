# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · New labeled batch arrives
# MAGIC Ingest it, rebuild the tables, score it with the champion, log the check to MLflow, and retrain if
# MAGIC PR-AUC fell too far or any feature drifted (thresholds in `settings.py`). A retrained model is only promoted
# MAGIC if it beats the champion on the frozen test split. Run for batch_1, then batch_2 (drifted), then batch_3.

# COMMAND ----------
# MAGIC %pip install -q xgboost imbalanced-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.dropdown("batch", "batch_1", ["batch_1", "batch_2", "batch_3"])

# COMMAND ----------
import modeling

modeling.setup_mlflow()
modeling.check_batch(dbutils.widgets.get("batch"))
