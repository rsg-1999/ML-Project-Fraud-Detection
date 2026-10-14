"""End-to-end check on synthetic data, local Spark + local MLflow (no Databricks needed):
    pytest tests/test_smoke.py -s
Needs: pyspark, mlflow, xgboost, imbalanced-learn, pandas, scikit-learn (see requirements.txt)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fraud_pipeline"))


def _fake_source(src, n_clients, rng, labeled):
    pre = "train" if labeled else "test"
    ids = [f"{pre}_Client_{i}" for i in range(n_clients)]
    fraud = (rng.random(n_clients) < 0.06) if labeled else np.zeros(n_clients, bool)
    c = pd.DataFrame({"disrict": rng.choice([60, 62, 63, 69], n_clients), "client_id": ids,
                      "client_catg": rng.choice([11, 12, 51], n_clients), "region": rng.choice([101, 103, 301], n_clients),
                      "creation_date": [f"{d:02d}/0{m}/{y}" for d, m, y in zip(rng.integers(1, 28, n_clients), rng.integers(1, 9, n_clients), rng.integers(1990, 2015, n_clients))]})
    rows = []
    for cid, f in zip(ids, fraud):
        for k in range(rng.integers(1, 7)):
            old = int(rng.integers(1000, 5000)); use = int(rng.gamma(2, 60 if f else 150))
            rows.append({"client_id": cid, "invoice_date": f"{2012 + k}-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}", "tarif_type": 11,
                         "counter_number": int(rng.integers(1e5, 1e6)), "counter_statue": 0, "counter_code": 203,
                         "reading_remarque": int(rng.choice([6, 8, 9])), "counter_coefficient": 1,
                         "consommation_level_1": use, "consommation_level_2": 0, "consommation_level_3": 0,
                         "consommation_level_4": 0, "old_index": old, "new_index": old + use, "months_number": 4,
                         "counter_type": rng.choice(["ELEC", "GAZ"])})
    inv = pd.DataFrame(rows)
    inv = pd.concat([inv, inv.head(5)])  # a few exact duplicates for silver to remove
    if labeled:
        c["target"] = fraud.astype(float)
    c.to_csv(f"{src}/client_{pre}.csv", index=False)
    inv.to_csv(f"{src}/invoice_{pre}.csv", index=False)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import mlflow
    from pyspark.sql import SparkSession
    import settings as S

    tmp = tmp_path_factory.mktemp("fraud")
    S.ROOT, S.CATALOG, S.SCHEMA, S.MODEL_NAME = str(tmp / "vol"), "spark_catalog", "fraud", "fraud_model"
    os.makedirs(f"{S.ROOT}/source")
    rng = np.random.default_rng(0)
    _fake_source(f"{S.ROOT}/source", 12000, rng, True)
    _fake_source(f"{S.ROOT}/source", 2000, rng, False)
    sp = (SparkSession.builder.master("local[2]").config("spark.sql.warehouse.dir", str(tmp / "wh"))
          .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    sp.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.fraud")
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{tmp}/mlflow.db")
    mlflow.set_experiment("smoke")
    return S


def build_tables(sp):
    """What the ETL pipeline does on Databricks (Auto Loader -> bronze -> silver -> gold), recomputed locally."""
    import lakehouse
    import transforms as X
    import settings as S

    for feed in ("labeled", "score"):
        for entity in ("client", "invoice"):
            raw = sp.read.option("header", True).csv(f"{S.ROOT}/landing/{feed}/*/{entity}")
            X.stamp(raw).write.mode("overwrite").saveAsTable(lakehouse.t(f"bronze_{feed}_{entity}"))
        c = X.silver_client(sp.table(lakehouse.t(f"bronze_{feed}_client")))
        i = X.silver_invoice(sp.table(lakehouse.t(f"bronze_{feed}_invoice")))
        assert c.count() == c.select("client_id").distinct().count(), "client_id is not unique"
        c.write.mode("overwrite").saveAsTable(lakehouse.t(f"silver_{feed}_client"))
        i.write.mode("overwrite").saveAsTable(lakehouse.t(f"silver_{feed}_invoice"))
        X.gold(sp.table(lakehouse.t(f"silver_{feed}_client")), sp.table(lakehouse.t(f"silver_{feed}_invoice")),
               feed).write.mode("overwrite").saveAsTable(lakehouse.t(f"gold_{feed}"))


def test_pipeline(env):
    import os
    import lakehouse, modeling
    from mlflow import MlflowClient

    S = env
    lakehouse.simulate_arrivals()
    assert sorted(os.listdir(f"{S.ROOT}/pending/labeled")) == S.BATCHES  # batches wait, only `initial` has arrived
    assert os.listdir(f"{S.ROOT}/landing/labeled") == ["initial"]
    sp = lakehouse.spark()
    build_tables(sp)
    assert sp.table(lakehouse.t("bronze_labeled_invoice")).count() > sp.table(lakehouse.t("silver_labeled_invoice")).count()
    splits = {r["split"]: r["n"] for r in sp.table(lakehouse.t("gold_labeled")).groupBy("split").count().withColumnRenamed("count", "n").collect()}
    assert set(splits) == {"train", "val", "test"} and abs(splits["train"] / sum(splits.values()) - 60 / 85) < 0.03, splits

    first = modeling.train_and_register()
    assert first["promoted"] and first["test_pr_auc"] > 0.10  # base rate is 0.06

    preds = modeling.score_unlabeled()
    assert len(preds) == 2000 and preds.score.between(0, 1).all()

    assert lakehouse.next_unchecked() is None
    results = {}
    while (b := lakehouse.arrive_next_batch()):  # what the scheduled job does on each run
        build_tables(sp)
        assert lakehouse.next_unchecked() == b
        results[b] = modeling.check_batch(b)
        lakehouse.mark_checked(b)
    assert list(results) == S.BATCHES and lakehouse.next_unchecked() is None
    assert lakehouse.arrive_next_batch() is None and not os.listdir(f"{S.ROOT}/pending/labeled")
    assert results["batch_2"] is not None and results["batch_2"]["version"] != first["version"]  # drift -> retrain
    runs = MlflowClient().search_runs([mlflow_exp_id()], "tags.run_type = 'batch_check'")
    flags = {r.data.tags["batch"]: r.data.tags["retrain_triggered"] for r in runs}
    assert flags["batch_2"] == "True", flags
    gold = sp.table(lakehouse.t("gold_labeled"))
    assert gold.filter("recent_eval").count() > 0 and gold.filter("split in ('train','val','test') and recent_eval").count() == 0


def mlflow_exp_id():
    import mlflow
    return mlflow.get_experiment_by_name("smoke").experiment_id
