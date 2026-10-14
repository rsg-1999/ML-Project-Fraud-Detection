"""ETL pipeline, bronze: raw CSV rows as-is (every column a string) + ingest metadata, loaded by Auto Loader.
Auto Loader watches landing/ and loads each new file exactly once; a batch 'arrives' when its folder appears there."""
from pyspark import pipelines as dp
from pyspark.sql import SparkSession

import settings as S
import transforms as X

spark = SparkSession.builder.getOrCreate()


def bronze(feed, entity):
    @dp.table(name=f"bronze_{feed}_{entity}", comment=f"raw {feed} {entity} files, untouched")
    def _bronze():
        return X.stamp(spark.readStream.format("cloudFiles")
                       .option("cloudFiles.format", "csv").option("header", True)
                       .load(f"{S.ROOT}/landing/{feed}/*/{entity}"))


for feed in ("labeled", "score"):
    for entity in ("client", "invoice"):
        bronze(feed, entity)
