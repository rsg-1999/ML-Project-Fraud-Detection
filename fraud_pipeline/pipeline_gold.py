"""ETL pipeline, gold: one row per client with model-ready features and the train/val/test/batch split."""
from pyspark import pipelines as dp
from pyspark.sql import SparkSession

import transforms as X

spark = SparkSession.builder.getOrCreate()


def gold(feed):
    @dp.table(name=f"gold_{feed}", comment=f"{feed} clients, one row each, with features")
    def _gold():
        return X.gold(spark.read.table(f"silver_{feed}_client"), spark.read.table(f"silver_{feed}_invoice"), feed)


for feed in ("labeled", "score"):
    gold(feed)
