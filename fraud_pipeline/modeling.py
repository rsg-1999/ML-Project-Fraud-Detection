"""Training, registry, scoring and monitoring (sklearn + XGBoost + MLflow)."""
import json

import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import lakehouse
import settings as S


def setup_mlflow():
    mlflow.set_registry_uri("databricks-uc")
    user = lakehouse.spark().sql("select current_user()").first()[0]
    mlflow.set_experiment(f"/Users/{user}/{S.EXPERIMENT_NAME}")


# ---- data ----
def load(splits, feed="labeled", recent=None):
    """Gold rows in `splits` -> (X, y, client_ids). Gold already has typed columns (CAT=string, NUM=double).
    recent=True/False keeps only the held-out / training part of arrival batches."""
    df = lakehouse.spark().table(lakehouse.t(f"gold_{feed}")).filter(F.col("split").isin(list(splits)))
    if recent is not None:
        df = df.filter(F.col("recent_eval") == recent)
    df = df.toPandas()
    X = df[S.CAT + S.NUM].copy()
    X[S.CAT] = X[S.CAT].fillna("missing")
    return X, df["target"], df["client_id"]


# ---- model ----
def build_pipeline(cfg, pos_weight):
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), S.NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), S.CAT)])
    steps = [("prep", prep)]
    if cfg["imbalance"] == "smote":  # inside the pipeline: only ever applied to training rows
        steps.append(("smote", SMOTE(random_state=S.SEED)))
    steps.append(("clf", XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr", n_jobs=-1,
        scale_pos_weight=pos_weight if cfg["imbalance"] == "weight" else 1.0, random_state=S.SEED)))
    return Pipeline(steps)


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def metrics(y, p, thr, prefix):
    flag = p >= thr
    return {f"{prefix}_pr_auc": average_precision_score(y, p), f"{prefix}_roc_auc": roc_auc_score(y, p),
            f"{prefix}_recall": recall_score(y, flag), f"{prefix}_precision": precision_score(y, flag, zero_division=0),
            f"{prefix}_flag_rate": float(flag.mean())}


# ---- drift (PSI against the training baseline stored with the model) ----
def _frac(v, edges):
    return np.bincount(np.searchsorted(edges, v), minlength=len(edges) + 1) / max(len(v), 1)


def psi_baseline(X, scores):
    cols = {c: X[c].dropna().to_numpy() for c in S.NUM} | {"__score__": np.asarray(scores)}
    out = {}
    for c, v in cols.items():
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        out[c] = {"edges": edges.tolist(), "frac": _frac(v, edges).tolist()}
    return out


def psi(base, v):
    p, q = np.clip(base["frac"], 1e-4, None), np.clip(_frac(v, np.array(base["edges"])), 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def monitor(X, scores, thr, base):
    """Label-free signals: score distribution, flag rate and PSI per numeric feature."""
    out = {"mean_score": float(np.mean(scores)), "flag_rate": float((scores >= thr).mean()),
           "psi_score": psi(base["__score__"], np.asarray(scores))}
    out |= {f"psi_{c}": psi(base[c], X[c].dropna().to_numpy()) for c in S.NUM}
    out["psi_max"] = max(v for k, v in out.items() if k.startswith("psi_"))
    return out


# ---- train + register ----
def _champion_or_none():
    try:
        return MlflowClient().get_model_version_by_alias(S.MODEL_NAME, "champion")
    except MlflowException:
        return None


def train_and_register(batches=()):
    """Fit every config on train (+ the training part of arrived batches), pick the best on val and score it on the
    frozen test and on `recent` (the held-out part of the arrived batches). Register it as `challenger`;
    it becomes `champion` if it passes the gate described in settings.py."""
    Xtr, ytr, _ = load(["train"])
    if batches:
        Xb, yb, _ = load(batches, recent=False)
        Xtr, ytr = pd.concat([Xtr, Xb], ignore_index=True), pd.concat([ytr, yb], ignore_index=True)
    Xva, yva, _ = load(["val"])
    Xte, yte, _ = load(["test"])
    Xrec, yrec, _ = load(batches, recent=True) if batches else (None, None, None)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    client = MlflowClient()

    with mlflow.start_run(run_name=f"train_{len(batches)}_batches"):
        best = None
        for cfg in S.CONFIGS:
            with mlflow.start_run(run_name=cfg["name"], nested=True):
                pipe = build_pipeline(cfg, pos_weight).fit(Xtr, ytr)
                p = pipe.predict_proba(Xva)[:, 1]
                thr = best_threshold(yva, p)
                m = metrics(yva, p, thr, "val")
                mlflow.log_params(cfg)
                mlflow.log_metrics(m)
                if best is None or m["val_pr_auc"] > best["m"]["val_pr_auc"]:
                    best = {"cfg": cfg, "pipe": pipe, "thr": thr, "m": m, "p_val": p}

        pipe, thr = best["pipe"], best["thr"]
        m_test = metrics(yte, pipe.predict_proba(Xte)[:, 1], thr, "test")  # frozen split: reporting only
        logged = best["m"] | m_test
        champ = _champion_or_none()
        if champ is None:
            promoted = True
        else:
            champ_model = mlflow.sklearn.load_model(f"models:/{S.MODEL_NAME}@champion")
            champ_test = average_precision_score(yte, champ_model.predict_proba(Xte)[:, 1])
            if batches:
                new_rec = average_precision_score(yrec, pipe.predict_proba(Xrec)[:, 1])
                champ_rec = average_precision_score(yrec, champ_model.predict_proba(Xrec)[:, 1])
                logged |= {"recent_pr_auc": new_rec, "champion_recent_pr_auc": champ_rec, "n_recent": len(Xrec)}
                promoted = (new_rec >= champ_rec + S.MIN_GAIN
                            and m_test["test_pr_auc"] >= champ_test - S.MAX_TEST_REGRESSION)
            else:
                promoted = m_test["test_pr_auc"] >= champ_test + S.MIN_GAIN
            logged["champion_test_pr_auc"] = champ_test
        mlflow.log_params({"best_config": best["cfg"]["name"], "threshold": thr, "n_train": len(Xtr),
                           "train_splits": ",".join(["train", *batches]), "promoted": promoted})
        mlflow.log_metrics(logged)
        mlflow.log_dict(psi_baseline(Xtr, best["p_val"]), "baseline.json")
        sample = Xte.head(100)
        mlflow.sklearn.log_model(pipe, name="model", signature=infer_signature(sample, pipe.predict(sample)),
                                 input_example=sample.head(5), registered_model_name=S.MODEL_NAME,
                                 serialization_format="cloudpickle")  # skops (new default) rejects xgboost/imblearn types

    mv = max(client.search_model_versions(f"name='{S.MODEL_NAME}'"), key=lambda v: int(v.version))
    client.set_model_version_tag(S.MODEL_NAME, mv.version, "threshold", str(thr))
    client.set_registered_model_alias(S.MODEL_NAME, "challenger", mv.version)
    if promoted:
        client.set_registered_model_alias(S.MODEL_NAME, "champion", mv.version)
    print(f"v{mv.version}: " + ", ".join(f"{k} {v:.4f}" for k, v in logged.items()
          if k in ("test_pr_auc", "champion_test_pr_auc", "recent_pr_auc", "champion_recent_pr_auc")) + f" -> promoted={promoted}")
    return {"version": mv.version, "promoted": promoted, "test_pr_auc": m_test["test_pr_auc"]}


# ---- champion, scoring, monitoring ----
def load_champion():
    mv = MlflowClient().get_model_version_by_alias(S.MODEL_NAME, "champion")
    model = mlflow.sklearn.load_model(f"models:/{S.MODEL_NAME}@champion")
    base = json.load(open(mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/baseline.json")))
    return mv, model, float(mv.tags["threshold"]), base


def score_unlabeled():
    """Score gold_score with the champion; save predictions; log label-free monitoring metrics."""
    mv, model, thr, base = load_champion()
    X, _, ids = load(["score"], feed="score")
    p = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"client_id": ids, "score": p, "flag": p >= thr, "model_version": int(mv.version)})
    out["scored_at"] = pd.Timestamp.now()
    lakehouse.spark().createDataFrame(out).write.mode("append").saveAsTable(lakehouse.t("predictions"))
    with mlflow.start_run(run_name=f"score_v{mv.version}"):
        mlflow.set_tags({"run_type": "score", "model_version": mv.version})
        mlflow.log_metrics(monitor(X, p, thr, base))
    return out


def check_batch(batch):
    """New labeled batch (already in gold): score with champion -> compare -> retrain if triggered."""
    mv, model, thr, base = load_champion()
    X, y, _ = load([batch])
    p = model.predict_proba(X)[:, 1]
    m = metrics(y, p, thr, "batch")
    mon = monitor(X, p, thr, base)
    ref = MlflowClient().get_run(mv.run_id).data.metrics["test_pr_auc"]
    drop = 1 - m["batch_pr_auc"] / ref
    drifted = max((k for k in mon if k.startswith("psi_") and k != "psi_max"), key=mon.get)
    retrain = drop > S.PR_AUC_DROP or mon["psi_max"] > S.PSI_MAX
    with mlflow.start_run(run_name=f"check_{batch}_v{mv.version}"):
        mlflow.set_tags({"run_type": "batch_check", "batch": batch, "model_version": mv.version,
                         "retrain_triggered": str(retrain), "most_drifted": drifted})
        mlflow.log_metrics(m | mon | {"pr_auc_rel_drop": drop, "n_rows": len(X)})
    print(f"{batch}: PR-AUC {m['batch_pr_auc']:.4f} (ref {ref:.4f}, drop {drop:.1%}), "
          f"psi_max {mon['psi_max']:.3f} ({drifted}) -> retrain={retrain}")
    if retrain:
        return train_and_register(batches=S.BATCHES[: S.BATCHES.index(batch) + 1])
