# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Train, evaluate, register
# MAGIC Each config in `settings.CONFIGS` is an MLflow child run scored on **val**. The best one is scored once on the
# MAGIC frozen **test** split, registered in Unity Catalog as `challenger`, and becomes `champion` if there is none yet.

# COMMAND ----------
# MAGIC %pip install -q xgboost imbalanced-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import modeling

modeling.setup_mlflow()
modeling.train_and_register()
