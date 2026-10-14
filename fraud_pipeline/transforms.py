"""Pure DataFrame -> DataFrame steps. Used by the ETL pipeline files and by the local test."""
from pyspark.sql import Window, functions as F

import settings as S


def bucket():
    return F.expr("pmod(xxhash64(client_id), 100)")


def in_bucket(lo, hi):
    return bucket().between(lo, hi - 1)


def stamp(df):
    """Bronze: keep the raw columns as they are, add where/when each row came from."""
    return df.select("*", F.current_timestamp().alias("_ingest_ts"), F.col("_metadata.file_path").alias("_source_file"))


def silver_client(df):
    target = F.col("target").cast("double") if "target" in df.columns else F.lit(None).cast("double")
    return df.select("client_id", F.col("disrict").cast("int"), F.col("client_catg").cast("int"),
                     F.col("region").cast("int"), F.to_date("creation_date", "dd/MM/yyyy").alias("creation_date"),
                     target.alias("target")).dropDuplicates()


def silver_invoice(df):
    return df.select("client_id", F.to_date("invoice_date").alias("invoice_date"),
                     F.col("tarif_type").cast("int"), F.col("counter_number").cast("long"), "counter_statue",
                     F.col("counter_code").cast("int"), F.col("reading_remarque").cast("int"),
                     F.col("counter_coefficient").cast("int"),
                     *[F.col(x).cast("double") for x in S.CONS + ["old_index", "new_index"]],
                     F.col("months_number").cast("int"), "counter_type").dropDuplicates()


def gold(client, inv, feed):
    """One row per client, model-ready. Only stateless features here; imputing/encoding is fit in the model."""
    inv = inv.withColumn("diff_index", F.col("new_index") - F.col("old_index"))
    w = Window.partitionBy("client_id").orderBy("invoice_date")
    inv = inv.withColumn("gap", F.datediff("invoice_date", F.lag("invoice_date").over(w)))
    num_cols = S.CONS + ["diff_index"]
    agg = inv.groupBy("client_id").agg(
        F.count("*").alias("bill_count"),
        F.avg("gap").alias("avg_frequency_of_bills_in_days"),
        F.mode("tarif_type").alias("tarif_type_mode"),
        F.mode("counter_coefficient").alias("counter_coefficient_mode"),
        F.mode("counter_type").alias("counter_type_mode"),
        F.min("reading_remarque").alias("reading_remarque_min"),
        F.mode("reading_remarque").alias("reading_remarque_mode"),
        *[F.avg(c).alias(f"{c}_mean") for c in num_cols],
        *[F.stddev(c).alias(f"{c}_std") for c in num_cols])
    g = (client.join(agg, "client_id", "left")
         .withColumn("creation_day", F.dayofmonth("creation_date"))
         .withColumn("creation_month", F.month("creation_date"))
         .withColumn("creation_year", F.year("creation_date"))
         .withColumn("single_bill_client", (F.col("bill_count") == 1).cast("int")))
    for c in num_cols:  # std of a single bill is undefined -> 0
        g = g.withColumn(f"{c}_std", F.coalesce(F.col(f"{c}_std"), F.lit(0.0)))
    if feed == "labeled":
        split = F.lit("unassigned")
        for name, (lo, hi) in reversed(list(S.SPLITS.items())):
            split = F.when(in_bucket(lo, hi), name).otherwise(split)
    else:
        split = F.lit("score")
    recent = F.expr("pmod(xxhash64(client_id, 'recent'), 100)") < int(S.RECENT_SHARE * 100)
    return g.select("client_id", "target", split.alias("split"),
                    (split.startswith("batch_") & recent).alias("recent_eval"),
                    *[F.col(c).cast("string") for c in S.CAT], *[F.col(c).cast("double") for c in S.NUM])
