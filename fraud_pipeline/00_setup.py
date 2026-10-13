# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the schema and the volume, then tells you where to upload the four source CSVs.

# COMMAND ----------
import settings as S

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {S.CATALOG}.{S.SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {S.CATALOG}.{S.SCHEMA}.{S.VOLUME}")
dbutils.fs.mkdirs(f"{S.ROOT}/source")
print(f"Upload client_train.csv, invoice_train.csv, client_test.csv, invoice_test.csv to:\n  {S.ROOT}/source/")
print("(Catalog -> workspace -> fraud -> raw -> source -> Upload to this volume)")
