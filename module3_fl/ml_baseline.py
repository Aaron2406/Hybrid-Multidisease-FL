"""
Module 3A - ML Baseline Classifiers  (v3 - Unified Multi-Disease)
Merges all disease quantum CSVs and runs baseline on unified dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = r"C:\Users\DELL\Hybrid-Multidisease-FL"
DATA_FOLDER  = os.path.join(PROJECT_ROOT, "data")

DISEASE_LABEL = {"diabetes": 0, "kidney": 1, "heart": 2, "liver": 3}
DISEASE_NAMES = {v: k for k, v in DISEASE_LABEL.items()}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":               round(float(f1_score(y_true, y_pred,
                                average="weighted", zero_division=0)), 4),
        "precision":        round(float(precision_score(y_true, y_pred,
                                average="weighted", zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, y_pred,
                                average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def merge_for_baseline(train_paths, test_paths, emit=None):
    """Merge all disease CSVs adding disease_label column."""
    def log(msg, level="info"):
        if emit: emit.log(msg, module="M3A", level=level)
        else:    print(f"  {msg}")

    train_frames, test_frames = [], []

    for disease in train_paths:
        df_train = pd.read_csv(train_paths[disease])
        df_test  = pd.read_csv(test_paths[disease])
        q_cols   = [c for c in df_train.columns if c.startswith("Q_feature")]

        df_tr = df_train[q_cols].copy()
        df_te = df_test[q_cols].copy()

        label = DISEASE_LABEL.get(disease, len(train_frames))
        df_tr["disease_label"] = label
        df_te["disease_label"] = label

        train_frames.append(df_tr)
        test_frames.append(df_te)
        log(f"Loaded {disease}: train={len(df_tr)} test={len(df_te)}")

    df_train_all = pd.concat(train_frames, ignore_index=True)
    df_test_all  = pd.concat(test_frames,  ignore_index=True)

    q_cols  = [c for c in df_train_all.columns if c.startswith("Q_feature")]
    X_train = df_train_all[q_cols].values.astype(np.float64)
    y_train = df_train_all["disease_label"].values.astype(np.int32)
    X_test  = df_test_all[q_cols].values.astype(np.float64)
    y_test  = df_test_all["disease_label"].values.astype(np.int32)

    log(f"Merged: train={X_train.shape} test={X_test.shape} "
        f"classes={np.unique(y_train).tolist()}", level="success")
    return X_train, X_test, y_train, y_test, q_cols


def run_baseline_unified(train_paths, test_paths, emit=None):
    def log(msg, level="info"):
        if emit: emit.log(msg, module="M3A", level=level)
        else:    print(f"  {msg}")

    def section(t):
        if emit: emit.section(t, module="M3A")
        else:    print(f"\n{'='*55}\n  {t}\n{'='*55}")

    section("3A — ML Baseline (Unified Multi-Disease)")

    X_train, X_test, y_train, y_test, feature_cols = merge_for_baseline(
        train_paths, test_paths, emit
    )

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────
    log("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["Random Forest"] = compute_metrics(y_test, rf.predict(X_test))
    log(f"RF: acc={results['Random Forest']['accuracy']}  "
        f"f1={results['Random Forest']['f1']}", level="success")

    # ── MLP ───────────────────────────────────────────────────────────────
    log("Training MLP ...")
    sc2  = StandardScaler()
    Xtr2 = sc2.fit_transform(X_train)
    Xte2 = sc2.transform(X_test)
    mlp  = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp.fit(Xtr2, y_train)
    results["MLP"] = compute_metrics(y_test, mlp.predict(Xte2))
    log(f"MLP: acc={results['MLP']['accuracy']}  "
        f"f1={results['MLP']['f1']}", level="success")

    # ── Logistic Regression ───────────────────────────────────────────────
    log("Training Logistic Regression ...")
    sc3  = StandardScaler()
    Xtr3 = sc3.fit_transform(X_train)
    Xte3 = sc3.transform(X_test)
    lr   = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        C=1.0, max_iter=500, random_state=42
    )
    lr.fit(Xtr3, y_train)
    results["Logistic Regression"] = compute_metrics(y_test, lr.predict(Xte3))
    log(f"LR: acc={results['Logistic Regression']['accuracy']}  "
        f"f1={results['Logistic Regression']['f1']}", level="success")

    best = max(results, key=lambda k: results[k]["accuracy"])
    log(f"Best baseline: {best}  acc={results[best]['accuracy']}", level="success")
    section("3A COMPLETE")

    return (
        {
            "models":        results,
            "best_model":    best,
            "feature_cols":  feature_cols,
            "train_samples": int(X_train.shape[0]),
            "test_samples":  int(X_test.shape[0]),
            "n_features":    int(X_train.shape[1]),
            "n_classes":     int(len(np.unique(y_train))),
            "class_names":   {str(k): v for k, v in DISEASE_NAMES.items()
                              if k < len(np.unique(y_train))},
        },
        X_train, X_test, y_train, y_test, sc3
    )