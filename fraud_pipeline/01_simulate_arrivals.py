# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Simulate data arrivals
# MAGIC Splits the labeled source files by client into `initial` (85%) and three arrival batches (5% each),
# MAGIC and copies the unlabeled test files to the `score` feed. Raw values are untouched, except
# MAGIC `settings.DRIFT_BATCHES`: there the invoice readings are scaled (covariate drift) and the top consumers are also
# MAGIC relabeled as fraud (concept drift), so there is something to detect and something new for a retrain to learn.
# MAGIC Run once.

# COMMAND ----------
import lakehouse

lakehouse.simulate_arrivals()

# COMMAND ----------
import settings as S

display(dbutils.fs.ls(f"{S.ROOT}/landing/labeled"))
