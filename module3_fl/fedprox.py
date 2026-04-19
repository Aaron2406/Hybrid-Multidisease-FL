"""
Module 3B - FedProx Federated Learning  (v3 - Unified Multi-Disease)
Merges all disease quantum CSVs and trains ONE unified model.
Saves global_unified_model.pkl to data folder.
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = r"C:\Users\DELL\Hybrid-Multidisease-FL"
DATA_FOLDER  = os.path.join(PROJECT_ROOT, "data")
MODULE3_PATH = os.path.join(PROJECT_ROOT, "module3_fl")
os.makedirs(DATA_FOLDER, exist_ok=True)

N_CLIENTS = 3
N_ROUNDS  = 5
MU        = 0.01
SEED      = 42

# ── Disease label mapping ─────────────────────────────────────────────────────
DISEASE_LABEL = {
    "diabetes": 0,
    "kidney":   1,
    "heart":    2,
    "liver":    3,
}
DISEASE_NAMES = {v: k for k, v in DISEASE_LABEL.items()}

# ── Unified output paths ──────────────────────────────────────────────────────
UNIFIED_MODEL_PATH   = os.path.join(DATA_FOLDER, "global_unified_model.pkl")
UNIFIED_SCALER_PATH  = os.path.join(DATA_FOLDER, "global_unified_scaler.pkl")
UNIFIED_RESULTS_PATH = os.path.join(DATA_FOLDER, "fl_unified_results.json")
MERGED_TRAIN_PATH    = os.path.join(DATA_FOLDER, "unified_train.csv")
MERGED_TEST_PATH     = os.path.join(DATA_FOLDER, "unified_test.csv")


# ── Merge disease CSVs ────────────────────────────────────────────────────────

def merge_disease_csvs(paths_dict, emit=None):
    """
    Merge quantum CSVs from multiple diseases into one DataFrame.
    Adds a 'disease_label' column: 0=diabetes, 1=kidney, 2=heart, 3=liver.
    All Q_feature columns are kept as-is (same 8 quantum features).
    """
    def log(msg, level="info"):
        if emit: emit.log(msg, module="M3B", level=level)
        else:    print(f"  {msg}")

    frames = []
    for disease, path in paths_dict.items():
        df = pd.read_csv(path)

        # Keep only Q_feature columns
        q_cols = [c for c in df.columns if c.startswith("Q_feature")]

        # Add disease label
        df_merged = df[q_cols].copy()
        df_merged["disease_label"] = DISEASE_LABEL.get(disease, len(frames))

        frames.append(df_merged)
        log(f"Loaded {len(df_merged):,} rows from {disease}", level="success")

    combined = pd.concat(frames, ignore_index=True)
    log(f"Merged total: {len(combined):,} rows | "
        f"{len([c for c in combined.columns if c.startswith('Q_')])} Q_features | "
        f"{combined['disease_label'].nunique()} diseases", level="success")
    return combined


# ── Client split ──────────────────────────────────────────────────────────────

def split_into_clients(X, y, n_clients=N_CLIENTS):
    """Stratified split ensuring each client has all disease classes."""
    np.random.seed(SEED)

    unique_classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        splits = np.array_split(cls_idx, n_clients)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    clients = []
    for idx in client_indices:
        idx = np.array(idx)
        np.random.shuffle(idx)
        clients.append((X[idx], y[idx]))
    return clients


# ── Global model ──────────────────────────────────────────────────────────────

class GlobalModel:
    def __init__(self, n_features, n_classes):
        self.n_classes_  = n_classes
        self.coef_       = np.zeros((n_classes, n_features))
        self.intercept_  = np.zeros(n_classes)

    def get_weights(self):
        return self.coef_.copy(), self.intercept_.copy()

    def set_weights(self, coef, intercept):
        self.coef_      = coef.copy()
        self.intercept_ = intercept.copy()

    def to_sklearn(self, classes=None):
        lr            = LogisticRegression(multi_class="multinomial",
                                           solver="lbfgs", max_iter=500)
        lr.coef_      = self.coef_
        lr.intercept_ = self.intercept_
        lr.classes_   = np.array(classes if classes is not None
                                 else list(range(self.n_classes_)))
        return lr


# ── FedProx client update ─────────────────────────────────────────────────────

def fedprox_client_update(X, y, global_coef, global_intercept, mu=MU):
    """Local FedProx update — handles multi-class and single-class clients."""
    unique_classes = np.unique(y)

    # Guard: skip if only one class
    if len(unique_classes) < 2:
        metrics = {
            "accuracy":   0.0,
            "f1":         0.0,
            "n_samples":  int(len(y)),
            "class_dist": {str(int(c)): int(np.sum(y == c)) for c in unique_classes},
            "skipped":    True,
        }
        return global_coef.copy(), global_intercept.copy(), metrics

    lr = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        C=1.0, max_iter=300, random_state=SEED
    )
    lr.fit(X, y)

    local_coef = lr.coef_

    # Align shapes for proximal term
    if local_coef.shape == global_coef.shape:
        g_coef = global_coef
    else:
        g_coef = global_coef[:local_coef.shape[0], :]

    prox_coef      = local_coef    - mu * (local_coef    - g_coef)
    prox_intercept = lr.intercept_ - mu * (lr.intercept_ - global_intercept[:len(lr.intercept_)])

    y_pred  = lr.predict(X)
    metrics = {
        "accuracy":   round(float(accuracy_score(y, y_pred)), 4),
        "f1":         round(float(f1_score(y, y_pred,
                              average="weighted", zero_division=0)), 4),
        "n_samples":  int(len(y)),
        "class_dist": {str(int(c)): int(np.sum(y == c)) for c in unique_classes},
    }
    return prox_coef, prox_intercept, metrics


# ── FedProx aggregation ───────────────────────────────────────────────────────

def fedprox_aggregate(updates, total, n_features, n_classes):
    agg_coef = np.zeros((n_classes, n_features))
    agg_int  = np.zeros(n_classes)

    for coef, intercept, m in updates:
        if m.get("skipped"):
            continue
        w = m["n_samples"] / total

        # Pad if needed
        rows = min(coef.shape[0], n_classes)
        agg_coef[:rows, :] += w * coef[:rows, :]
        agg_int[:len(intercept)] += w * intercept

    return agg_coef, agg_int


# ── Main unified FedProx ──────────────────────────────────────────────────────

def run_fedprox_unified(train_paths, emit=None, n_rounds=N_ROUNDS, mu=MU):
    """
    Run FedProx on merged data from all diseases.

    Parameters
    ----------
    train_paths : dict  {disease: csv_path}
    emit        : LogEmitter
    n_rounds    : int
    mu          : float

    Returns
    -------
    gm         : GlobalModel
    scaler     : StandardScaler
    fl_results : dict
    """
    def log(msg, level="info"):
        if emit: emit.log(msg, module="M3B", level=level)
        else:    print(f"  {msg}")

    def section(t):
        if emit: emit.section(t, module="M3B")
        else:    print(f"\n{'='*55}\n  {t}\n{'='*55}")

    def progress(cur, tot, label):
        if emit: emit.progress(cur, tot, label=label, module="M3B")

    section("3B — Unified FedProx Multi-Disease Training")
    log(f"Diseases: {list(train_paths.keys())}")
    log(f"Config: {N_CLIENTS} clients | {n_rounds} rounds | mu={mu}")

    # ── Merge all disease CSVs ────────────────────────────────────────────
    section("Merging disease quantum features")
    df_merged = merge_disease_csvs(train_paths, emit)
    df_merged.to_csv(MERGED_TRAIN_PATH, index=False)
    log(f"Merged train saved: {MERGED_TRAIN_PATH}", level="success")

    q_cols    = [c for c in df_merged.columns if c.startswith("Q_feature")]
    X_all     = df_merged[q_cols].values.astype(np.float64)
    y_all     = df_merged["disease_label"].values.astype(np.int32)
    n_classes = len(np.unique(y_all))
    n_features = X_all.shape[1]

    log(f"Total samples: {len(X_all):,} | Features: {n_features} | Classes: {n_classes}")
    for cls in np.unique(y_all):
        log(f"  Class {cls} ({DISEASE_NAMES.get(cls, cls)}): "
            f"{np.sum(y_all == cls):,} samples")

    scaler = StandardScaler()
    X_all  = scaler.fit_transform(X_all)

    # ── Split into clients (stratified) ───────────────────────────────────
    clients       = split_into_clients(X_all, y_all, N_CLIENTS)
    total_samples = sum(len(c[1]) for c in clients)

    section("Client profiles (non-IID stratified)")
    client_profiles = []
    for i, (Xc, yc) in enumerate(clients):
        dist = {str(int(c)): int(np.sum(yc == c)) for c in np.unique(yc)}
        log(f"Hospital {i+1}: {len(yc):,} samples  dist={dist}", level="success")
        client_profiles.append({
            "client_id":  i,
            "name":       f"Hospital {i+1}",
            "n_samples":  int(len(yc)),
            "class_dist": dist,
        })

    # ── FedProx training ──────────────────────────────────────────────────
    gm = GlobalModel(n_features, n_classes)
    g_coef, g_int = gm.get_weights()

    fl_results = {
        "mode":    "unified",
        "diseases": list(train_paths.keys()),
        "config": {
            "n_clients":   N_CLIENTS,
            "n_rounds":    n_rounds,
            "mu":          mu,
            "algorithm":   "FedProx",
            "local_model": "Logistic Regression (multinomial)",
            "n_classes":   n_classes,
            "class_names": {str(k): v for k, v in DISEASE_NAMES.items()
                            if k < n_classes},
        },
        "rounds":          [],
        "client_profiles": client_profiles,
    }

    section(f"FedProx unified training ({n_rounds} rounds)")
    classes = sorted(np.unique(y_all).tolist())

    for rnd in range(1, n_rounds + 1):
        log(f"Round {rnd}/{n_rounds}")
        progress(rnd, n_rounds, label=f"Unified FL Round {rnd}/{n_rounds}")

        updates  = []
        round_cm = []
        for i, (Xc, yc) in enumerate(clients):
            coef, intercept, m = fedprox_client_update(
                Xc, yc, g_coef, g_int, mu
            )
            m["client_id"] = i
            updates.append((coef, intercept, m))
            round_cm.append(m)
            status = "SKIPPED" if m.get("skipped") else f"acc={m['accuracy']}"
            log(f"  Hospital {i+1}: {status}  n={m['n_samples']}")

        g_coef, g_int = fedprox_aggregate(
            updates, total_samples, n_features, n_classes
        )
        gm.set_weights(g_coef, g_int)

        lr_eval = gm.to_sklearn(classes=classes)
        g_acc   = round(float(accuracy_score(
            y_all, lr_eval.predict(X_all))), 4)
        g_f1    = round(float(f1_score(
            y_all, lr_eval.predict(X_all),
            average="weighted", zero_division=0)), 4)

        log(f"Global: acc={g_acc}  f1={g_f1}", level="success")

        fl_results["rounds"].append({
            "round":          rnd,
            "global_acc":     g_acc,
            "global_f1":      g_f1,
            "client_metrics": round_cm,
            "weight_norm":    round(float(np.linalg.norm(g_coef)), 4),
        })

    # ── Save unified model ────────────────────────────────────────────────
    joblib.dump(gm,     UNIFIED_MODEL_PATH)
    joblib.dump(scaler, UNIFIED_SCALER_PATH)
    with open(UNIFIED_RESULTS_PATH, "w") as f:
        json.dump(fl_results, f, indent=2)

    section("3B UNIFIED COMPLETE")
    final = fl_results["rounds"][-1]
    log(f"Final: acc={final['global_acc']}  f1={final['global_f1']}", level="success")
    log(f"Model: {UNIFIED_MODEL_PATH}", level="success")

    return gm, scaler, fl_results


if __name__ == "__main__":
    train_paths = {
        d: os.path.join(DATA_FOLDER, f"quantum_{d}_train.csv")
        for d in ["diabetes", "kidney", "heart", "liver"]
        if os.path.exists(os.path.join(DATA_FOLDER, f"quantum_{d}_train.csv"))
    }
    print(f"Found: {list(train_paths.keys())}")
    gm, scaler, results = run_fedprox_unified(train_paths)
    for r in results["rounds"]:
        print(f"  Round {r['round']}: acc={r['global_acc']}  f1={r['global_f1']}")