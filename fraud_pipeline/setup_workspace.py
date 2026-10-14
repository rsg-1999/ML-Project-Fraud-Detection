"""Create the schema and volume. Then upload the four source CSVs, e.g. with the CLI:
    databricks fs cp data/train/client_train.csv dbfs:/Volumes/workspace/fraud/raw/source/ -p <profile>   (x4)"""
import os

import settings as S
from lakehouse import spark

sp = spark()
sp.sql(f"CREATE SCHEMA IF NOT EXISTS {S.CATALOG}.{S.SCHEMA}")
sp.sql(f"CREATE VOLUME IF NOT EXISTS {S.CATALOG}.{S.SCHEMA}.{S.VOLUME}")
os.makedirs(f"{S.ROOT}/source", exist_ok=True)
print(f"ready: {S.ROOT}/source/ has", os.listdir(f"{S.ROOT}/source"))
