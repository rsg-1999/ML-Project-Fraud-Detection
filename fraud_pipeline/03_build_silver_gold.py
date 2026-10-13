# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Silver (typed, deduplicated) and gold (one row per client, with the train/val/test split)

# COMMAND ----------
import lakehouse

for feed in ("labeled", "score"):
    lakehouse.build_silver(feed)
    lakehouse.build_gold(feed)

# COMMAND ----------
display(spark.table(lakehouse.t("gold_labeled")).groupBy("split").agg({"client_id": "count", "target": "avg"}))

# COMMAND ----------
display(spark.table(lakehouse.t("gold_labeled")).limit(20))
