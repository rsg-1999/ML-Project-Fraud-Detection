"""Bronze -> silver -> gold tables (PySpark). Notebooks 01-03 and 06 call these."""
from pyspark.sql import SparkSession, Window, functions as F

import settings as S


def spark():
    return SparkSession.builder.getOrCreate()


def t(name):
    return f"{S.CATALOG}.{S.SCHEMA}.{name}"


def bucket():
    return F.expr("pmod(xxhash64(client_id), 100)")


def _in(lo, hi):
    return bucket().between(lo, hi - 1)


def _read_raw(path):
    return spark().read.option("header", True).csv(path)  # no inferSchema: every column stays a string


def _write_raw(df, path):
    df.write.mode("overwrite").option("header", True).csv(path)


def _drift(inv):
    f = S.DRIFT_FACTOR
    for c in S.CONS:
        inv = inv.withColumn(c, (F.col(c).cast("double") * f).cast("long").cast("string"))
    o, n = F.col("old_index").cast("double"), F.col("new_index").cast("double")
    return inv.withColumn("new_index", (o + f * (n - o)).cast("long").cast("string"))


def simulate_arrivals():
    """Split the labeled source files into `initial` + arrival batches (by client bucket) and
    copy the unlabeled test files to the `score` feed. Raw values are written back untouched,
    except in S.DRIFT_BATCH where invoice readings are scaled on purpose."""
    src = f"{S.ROOT}/source"
    clients, inv = _read_raw(f"{src}/client_train.csv"), _read_raw(f"{src}/invoice_train.csv")
    for name, (lo, hi) in S.LANDING.items():
        i = inv.filter(_in(lo, hi))
        _write_raw(clients.filter(_in(lo, hi)), f"{S.ROOT}/landing/labeled/{name}/client")
        _write_raw(_drift(i) if name == S.DRIFT_BATCH else i, f"{S.ROOT}/landing/labeled/{name}/invoice")
    for entity in ("client", "invoice"):
        _write_raw(_read_raw(f"{src}/{entity}_test.csv"), f"{S.ROOT}/landing/score/initial/{entity}")


# ---- bronze: raw as-is + ingest metadata, append-only, safe to rerun ----
def ingest_bronze(feed, batch):
    sp = spark()
    for entity in ("client", "invoice"):
        table = t(f"bronze_{feed}_{entity}")
        df = _read_raw(f"{S.ROOT}/landing/{feed}/{batch}/{entity}").select(
            "*", F.current_timestamp().alias("_ingest_ts"), F.col("_metadata.file_path").alias("_source_file"))
        if sp.catalog.tableExists(table):  # skip files already loaded
            done = [r[0] for r in sp.table(table).select("_source_file").distinct().collect()]
            df = df.filter(~F.col("_source_file").isin(done))
        df.write.mode("append").option("mergeSchema", True).saveAsTable(table)


# ---- silver: typed + deduplicated ----
def build_silver(feed):
    sp = spark()
    c = sp.table(t(f"bronze_{feed}_client"))
    target = F.col("target").cast("double") if "target" in c.columns else F.lit(None).cast("double")
    c = c.select("client_id", F.col("disrict").cast("int"), F.col("client_catg").cast("int"),
                 F.col("region").cast("int"), F.to_date("creation_date", "dd/MM/yyyy").alias("creation_date"),
                 target.alias("target")).dropDuplicates()
    assert c.count() == c.select("client_id").distinct().count(), "client_id is not unique"
    c.write.mode("overwrite").saveAsTable(t(f"silver_{feed}_client"))

    i = sp.table(t(f"bronze_{feed}_invoice"))
    i = i.select("client_id", F.to_date("invoice_date").alias("invoice_date"),
                 F.col("tarif_type").cast("int"), F.col("counter_number").cast("long"), "counter_statue",
                 F.col("counter_code").cast("int"), F.col("reading_remarque").cast("int"),
                 F.col("counter_coefficient").cast("int"),
                 *[F.col(x).cast("double") for x in S.CONS + ["old_index", "new_index"]],
                 F.col("months_number").cast("int"), "counter_type").dropDuplicates()
    i.write.mode("overwrite").saveAsTable(t(f"silver_{feed}_invoice"))


# ---- gold: one row per client, model-ready (stateless features only; imputing/encoding is fit in the model) ----
def build_gold(feed):
    sp = spark()
    client, inv = sp.table(t(f"silver_{feed}_client")), sp.table(t(f"silver_{feed}_invoice"))
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
            split = F.when(_in(lo, hi), name).otherwise(split)
    else:
        split = F.lit("score")
    g = g.select("client_id", "target", split.alias("split"),
                 *[F.col(c).cast("string") for c in S.CAT], *[F.col(c).cast("double") for c in S.NUM])
    g.write.mode("overwrite").saveAsTable(t(f"gold_{feed}"))
