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


def _drift(clients, inv):
    """Covariate drift: scale readings. Concept drift: the top DRIFT_NEW_FRAUD consumers become fraud too."""
    f = S.DRIFT_FACTOR
    for c in S.CONS:
        inv = inv.withColumn(c, (F.col(c).cast("double") * f).cast("long").cast("string"))
    o, n = F.col("old_index").cast("double"), F.col("new_index").cast("double")
    inv = inv.withColumn("new_index", (o + f * (n - o)).cast("long").cast("string"))
    use = inv.groupBy("client_id").agg(F.avg(F.col("consommation_level_1").cast("double")).alias("use"))
    rank = use.withColumn("r", F.percent_rank().over(Window.orderBy(F.col("use").desc())))
    new_fraud = rank.filter(F.col("r") < S.DRIFT_NEW_FRAUD).select("client_id", F.lit(True).alias("_new"))
    clients = (clients.join(new_fraud, "client_id", "left")
               .withColumn("target", F.when(F.col("_new"), F.lit("1.0")).otherwise(F.col("target"))).drop("_new"))
    return clients, inv


def simulate_arrivals():
    """Split the labeled source files into `initial` + arrival batches (by client bucket) and
    copy the unlabeled test files to the `score` feed. Raw values are written back untouched,
    except in S.DRIFT_BATCHES (see _drift)."""
    src = f"{S.ROOT}/source"
    clients, inv = _read_raw(f"{src}/client_train.csv"), _read_raw(f"{src}/invoice_train.csv")
    for name, (lo, hi) in S.LANDING.items():
        c, i = clients.filter(_in(lo, hi)), inv.filter(_in(lo, hi))
        if name in S.DRIFT_BATCHES:
            c, i = _drift(c, i)
        _write_raw(c, f"{S.ROOT}/landing/labeled/{name}/client")
        _write_raw(i, f"{S.ROOT}/landing/labeled/{name}/invoice")
    for entity in ("client", "invoice"):
        _write_raw(_read_raw(f"{src}/{entity}_test.csv"), f"{S.ROOT}/landing/score/initial/{entity}")


# ---- bronze: raw as-is + ingest metadata, append-only, safe to rerun ----
def _stamp(df):
    return df.select("*", F.current_timestamp().alias("_ingest_ts"), F.col("_metadata.file_path").alias("_source_file"))


def _autoload(path, table, ckpt):
    """Auto Loader remembers which files it has seen in the checkpoint, so each file is loaded exactly once.
    Delete the checkpoint folder if you drop the table, otherwise nothing is reloaded."""
    (_stamp(spark().readStream.format("cloudFiles").option("cloudFiles.format", "csv").option("header", True)
            .option("cloudFiles.schemaLocation", ckpt).load(path))  # no type inference: strings as-is
     .writeStream.option("checkpointLocation", ckpt).option("mergeSchema", True)
     .trigger(availableNow=True).toTable(table).awaitTermination())


def _append_new_files(path, table):
    """Plain-Spark fallback (local test): skip files already present in the table."""
    sp = spark()
    df = _stamp(_read_raw(path))
    if sp.catalog.tableExists(table):
        done = [r[0] for r in sp.table(table).select("_source_file").distinct().collect()]
        df = df.filter(~F.col("_source_file").isin(done))
    df.write.mode("append").option("mergeSchema", True).saveAsTable(table)


def ingest_bronze(feed, batch):
    for entity in ("client", "invoice"):
        path, table = f"{S.ROOT}/landing/{feed}/{batch}/{entity}", t(f"bronze_{feed}_{entity}")
        if S.AUTOLOADER:
            _autoload(path, table, f"{S.ROOT}/_checkpoints/{feed}_{batch}_{entity}")
        else:
            _append_new_files(path, table)


def next_batch():
    """First arrival batch whose files are not in bronze yet, or None when everything has arrived."""
    sp = spark()
    table = t("bronze_labeled_client")
    loaded = [r[0] for r in sp.table(table).select("_source_file").distinct().collect()] if sp.catalog.tableExists(table) else []
    return next((b for b in S.BATCHES if not any(f"/landing/labeled/{b}/" in f for f in loaded)), None)


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
    recent = F.expr("pmod(xxhash64(client_id, 'recent'), 100)") < int(S.RECENT_SHARE * 100)
    g = g.select("client_id", "target", split.alias("split"),
                 (split.startswith("batch_") & recent).alias("recent_eval"),
                 *[F.col(c).cast("string") for c in S.CAT], *[F.col(c).cast("double") for c in S.NUM])
    g.write.mode("overwrite").saveAsTable(t(f"gold_{feed}"))
