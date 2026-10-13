# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Score the unlabeled test data with the champion
# MAGIC No labels, so no accuracy. The MLflow run logs what we *can* watch: score distribution, flag rate and PSI per feature
# MAGIC against the training baseline. Predictions are appended to the `predictions` table.

# COMMAND ----------
# MAGIC %pip install -q xgboost imbalanced-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import modeling

modeling.setup_mlflow()
preds = modeling.score_unlabeled()
print(preds.flag.mean(), len(preds))
