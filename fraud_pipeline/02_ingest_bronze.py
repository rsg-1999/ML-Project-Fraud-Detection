# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Bronze: raw data as-is
# MAGIC Appends the landing files to Delta tables, every column still a string, plus `_ingest_ts` and `_source_file`.
# MAGIC Files already loaded are skipped, so re-running is safe.

# COMMAND ----------
import lakehouse

lakehouse.ingest_bronze("labeled", "initial")
lakehouse.ingest_bronze("score", "initial")

# COMMAND ----------
for name in ("bronze_labeled_client", "bronze_labeled_invoice", "bronze_score_client", "bronze_score_invoice"):
    print(name, spark.table(lakehouse.t(name)).count())
