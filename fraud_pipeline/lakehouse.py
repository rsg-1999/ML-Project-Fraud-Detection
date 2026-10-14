"""Files and bookkeeping around the pipeline: simulating arrivals, moving batches into landing, the arrival log."""
import os
import shutil

from pyspark.sql import SparkSession, Window, functions as F

import settings as S
from transforms import in_bucket


def spark():
    return SparkSession.builder.getOrCreate()


def t(name):
    return f"{S.CATALOG}.{S.SCHEMA}.{name}"


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
    """Split the labeled source files by client: the initial buckets go to landing/, each arrival batch waits in
    pending/. The unlabeled test files go to landing/score. Raw values are written back untouched, except in
    S.DRIFT_BATCHES (see _drift)."""
    src = f"{S.ROOT}/source"
    clients, inv = _read_raw(f"{src}/client_train.csv"), _read_raw(f"{src}/invoice_train.csv")
    parts = {"landing/labeled/initial": S.INITIAL, **{f"pending/labeled/{b}": S.SPLITS[b] for b in S.BATCHES}}
    for folder, (lo, hi) in parts.items():
        c, i = clients.filter(in_bucket(lo, hi)), inv.filter(in_bucket(lo, hi))
        if folder.split("/")[-1] in S.DRIFT_BATCHES:
            c, i = _drift(c, i)
        # same column order in every file: CSV readers that glob several folders match columns by position
        _write_raw(c.select(clients.columns), f"{S.ROOT}/{folder}/client")
        _write_raw(i.select(inv.columns), f"{S.ROOT}/{folder}/invoice")
    for entity in ("client", "invoice"):
        _write_raw(_read_raw(f"{src}/{entity}_test.csv"), f"{S.ROOT}/landing/score/initial/{entity}")


# ---- arrivals: pending/ -> landing/, plus an append-only log of what arrived and what has been checked ----
def _log(batch, event):
    (spark().createDataFrame([(batch, event)], "batch string, event string")
     .withColumn("at", F.current_timestamp()).write.mode("append").saveAsTable(t("batch_log")))


def arrive_next_batch():
    """Move the next waiting batch into landing/ (this is the 'new files appear' moment). None if nothing waits."""
    pending = f"{S.ROOT}/pending/labeled"
    waiting = [b for b in S.BATCHES if os.path.isdir(f"{pending}/{b}")]
    if not waiting:
        return None
    b = waiting[0]
    shutil.copytree(f"{pending}/{b}", f"{S.ROOT}/landing/labeled/{b}")
    shutil.rmtree(f"{pending}/{b}")
    _log(b, "arrived")
    return b


def next_unchecked():
    """Oldest arrived batch that has not been checked yet, or None."""
    sp = spark()
    if not sp.catalog.tableExists(t("batch_log")):
        return None
    log = sp.table(t("batch_log"))
    checked = {r.batch for r in log.filter("event = 'checked'").collect()}
    arrived = [r.batch for r in log.filter("event = 'arrived'").orderBy("at").collect()]
    return next((b for b in arrived if b not in checked), None)


def mark_checked(batch):
    _log(batch, "checked")
