"""ETL pipeline, silver: typed and de-duplicated, with data-quality rules (rows that fail are dropped and counted)."""
from pyspark import pipelines as dp
from pyspark.sql import SparkSession

import transforms as X

spark = SparkSession.builder.getOrCreate()


def silver_client(feed):
    rules = {"client_id_present": "client_id IS NOT NULL", "creation_date_valid": "creation_date IS NOT NULL"}
    if feed == "labeled":
        rules["target_is_0_or_1"] = "target IN (0.0, 1.0)"

    @dp.table(name=f"silver_{feed}_client", comment="clients: typed, de-duplicated")
    @dp.expect_all_or_drop(rules)
    def _client():
        return X.silver_client(spark.read.table(f"bronze_{feed}_client"))


def silver_invoice(feed):
    @dp.table(name=f"silver_{feed}_invoice", comment="invoices: typed, de-duplicated")
    @dp.expect_all_or_drop({"client_id_present": "client_id IS NOT NULL", "invoice_date_valid": "invoice_date IS NOT NULL"})
    def _invoice():
        return X.silver_invoice(spark.read.table(f"bronze_{feed}_invoice"))


for feed in ("labeled", "score"):
    silver_client(feed)
    silver_invoice(feed)
