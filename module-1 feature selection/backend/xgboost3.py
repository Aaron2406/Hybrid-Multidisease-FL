"""
=============================================================================
  XGBoost Disease Prediction Pipeline  (v4 — Multi-Disease)
  Supports: diabetes, kidney, heart, liver
=============================================================================
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ── Fixed data folder path ────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\DELL\Hybrid-Multidisease-FL"
DATA_FOLDER  = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.facecolor": "#F7F9FC",
    "axes.facecolor":   "#F7F9FC",
    "axes.edgecolor":   "#CCCCCC",
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost Disease Prediction Pipeline — WEKA ClassifierAttributeEval"
    )
    parser.add_argument("csv_file", help="Path to input CSV dataset")
    parser.add_argument("--target", default=None,
                        help="Target column name (default: last column)")
    parser.add_argument("--top_features", type=int, default=8,
                        help="Number of top features to select (default: 8)")
    parser.add_argument("--exclude", default=None,
                        help="Comma-separated column names to exclude")
    parser.add_argument("--disease", default="unknown",
                        help="Disease label: diabetes/kidney/heart/liver")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  STEP 1 — LOADING DATASET")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    print(f"  File        : {csv_path}")
    print(f"  Shape       : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Memory      : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute: numeric → median, categorical → mode."""
    print("\n  [Preprocessing] Missing Value Imputation")
    total = df.isnull().sum().sum()
    if total == 0:
        print("    No missing values found.")
        return df
    print(f"    Total missing cells: {total}")
    for col in df.columns:
        # Fix mixed-type columns first
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted

        n = df[col].isnull().sum()
        if n == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            v = df[col].median()
            df[col] = df[col].fillna(v)
            print(f"    {col:<40} {n} missing → median ({v:.4g})")
        else:
            v = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(v)
            print(f"    {col:<40} {n} missing → mode ('{v}')")
    return df


def healthcare_validation(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "-" * 70)
    print("  [Healthcare Validation] Starting validation ...")
    print("-" * 70)

    original_len = len(df)
    summary      = {}

    def _bad_rows(mask: pd.Series) -> list:
        return sorted(df.index[mask].tolist())

    # 1. Null values
    null_mask = df.isnull().any(axis=1)
    null_rows = _bad_rows(null_mask)
    if null_rows:
        print(f"\n  [1] Null values found at rows: {null_rows[:10]}{'...' if len(null_rows)>10 else ''}")
        df = df.drop(index=null_rows)
        print(f"      Rows removed: {len(null_rows)}")
    else:
        print("\n  [1] No null values detected.")
    summary["null_values"] = len(null_rows)

    # 2. Age threshold
    age_col = next((c for c in df.columns if c.lower() == "age"), None)
    if age_col is not None:
        age_mask = (df[age_col] < 1) | (df[age_col] > 120)
        age_rows = _bad_rows(age_mask)
        if age_rows:
            print(f"\n  [2] Age violations at rows: {age_rows}")
            df = df.drop(index=age_rows)
        else:
            print("\n  [2] All age values within valid range (1-120).")
        summary["age_threshold"] = len(age_rows)
    else:
        print("\n  [2] 'age' column not found — skipping.")
        summary["age_threshold"] = 0

    # 3a. Blood pressure
    bp_col = next(
        (c for c in df.columns if "blood_pressure" in c.lower() or c.lower() == "bp"),
        None
    )
    if bp_col is not None:
        bp_mask = (df[bp_col] < 70) | (df[bp_col] > 200)
        bp_rows = _bad_rows(bp_mask)
        if bp_rows:
            df = df.drop(index=bp_rows)
        else:
            print("\n  [3a] Blood pressure values OK (70-200 mmHg).")
        summary["blood_pressure"] = len(bp_rows)
    else:
        print("\n  [3a] Blood pressure column not found — skipping.")
        summary["blood_pressure"] = 0

    # 3b. Blood glucose
    bg_col = next(
        (c for c in df.columns if "blood_glucose" in c.lower() or "glucose" in c.lower()),
        None
    )
    if bg_col is not None:
        bg_mask = (df[bg_col] < 60) | (df[bg_col] > 400)
        bg_rows = _bad_rows(bg_mask)
        if bg_rows:
            df = df.drop(index=bg_rows)
        else:
            print("\n  [3b] Blood glucose values OK (60-400 mg/dL).")
        summary["blood_glucose"] = len(bg_rows)
    else:
        print("\n  [3b] Blood glucose column not found — skipping.")
        summary["blood_glucose"] = 0

    # 4. Lifestyle consistency
    bmi_col = next((c for c in df.columns if c.lower() == "bmi"), None)
    pal_col = next(
        (c for c in df.columns if "physical_activity" in c.lower()),
        None
    )
    if bmi_col is not None and pal_col is not None:
        lc_mask = (df[bmi_col] > 35) & \
                  (df[pal_col].astype(str).str.strip().str.lower() == "high")
        lc_rows = _bad_rows(lc_mask)
        if lc_rows:
            df = df.drop(index=lc_rows)
        else:
            print("\n  [4] No lifestyle inconsistencies detected.")
        summary["lifestyle_inconsistency"] = len(lc_rows)
    else:
        print(f"\n  [4] Lifestyle check skipped.")
        summary["lifestyle_inconsistency"] = 0

    # 5. Medical outliers
    outlier_mask = pd.Series(False, index=df.index)
    if bmi_col is not None:
        outlier_mask = outlier_mask | (df[bmi_col] > 60)
    sleep_col = next((c for c in df.columns if "sleep" in c.lower()), None)
    if sleep_col is not None:
        outlier_mask = outlier_mask | (df[sleep_col] > 16)
    stress_col = next((c for c in df.columns if "stress" in c.lower()), None)
    if stress_col is not None:
        outlier_mask = outlier_mask | (df[stress_col] > 10)
    outlier_rows = _bad_rows(outlier_mask)
    if outlier_rows:
        df = df.drop(index=outlier_rows)
        print(f"\n  [5] Medical outliers removed: {len(outlier_rows)}")
    else:
        print("\n  [5] No medical outliers detected.")
    summary["medical_outliers"] = len(outlier_rows)

    # Summary
    total_removed = original_len - len(df)
    print("\n" + "-" * 70)
    print(f"  Original rows : {original_len:,}")
    print(f"  Total removed : {total_removed:,}")
    print(f"  Remaining     : {len(df):,}")
    print("-" * 70)

    df = df.reset_index(drop=True)

    validation_summary = {
        "original_rows":          original_len,
        "rows_removed_null":      summary.get("null_values", 0),
        "rows_removed_age":       summary.get("age_threshold", 0),
        "rows_removed_bp":        summary.get("blood_pressure", 0),
        "rows_removed_glucose":   summary.get("blood_glucose", 0),
        "rows_removed_lifestyle": summary.get("lifestyle_inconsistency", 0),
        "rows_removed_outliers":  summary.get("medical_outliers", 0),
        "total_removed":          total_removed,
        "remaining_rows":         len(df),
    }

    return df, validation_summary


def encode_features(df: pd.DataFrame, target_col: str) -> tuple:
    """Label-encode all object/category feature columns."""
    print("\n  [Preprocessing] Label Encoding")
    encoders = {}
    cat_cols = [
        c for c in df.columns
        if df[c].dtype.name in ("object", "category") and c != target_col
    ]
    if not cat_cols:
        print("    No categorical feature columns detected.")
        return df, encoders
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    print(f"    Encoded Columns: {', '.join(cat_cols)}")
    return df, encoders


def preprocess_data(df: pd.DataFrame, target_col: str,
                    exclude_cols: list = None) -> tuple:
    print("\n" + "=" * 70)
    print("  STEP 2 — PREPROCESSING")
    print("=" * 70)

    if exclude_cols:
        valid_excl = [c for c in exclude_cols if c in df.columns]
        if valid_excl:
            df = df.drop(columns=valid_excl)
            print(f"\n  Excluded columns: {valid_excl}")
    else:
        print("\n  No columns excluded.")

    df, _validation_summary = healthcare_validation(df)
    preprocess_data._validation_summary = _validation_summary

    df = handle_missing_values(df)
    df, _ = encode_features(df, target_col)

    target_le = None
    if df[target_col].dtype.name in ("object", "category"):
        target_le = LabelEncoder()
        df[target_col] = target_le.fit_transform(df[target_col].astype(str))
        print(f"\n  Target '{target_col}' encoded → classes: {list(target_le.classes_)}")
    else:
        print(f"\n  Target '{target_col}' is numeric. "
              f"Unique values: {sorted(df[target_col].unique())}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"\n  Feature matrix  : {X.shape}")
    print(f"  Target vector   : {y.shape}")
    print(f"  Class balance   :\n{y.value_counts().to_string()}")
    return X, y, target_le


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_params(y) -> dict:
    base = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0)
    if len(np.unique(y)) == 2:
        return {**base, "objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        return {**base, "objective": "multi:softprob", "eval_metric": "mlogloss"}


def select_features_classifier_eval(X: pd.DataFrame, y: pd.Series,
                                     top_n: int = 8) -> tuple:
    print("\n" + "=" * 70)
    print("  STEP 3 — FEATURE SELECTION")
    print("  Method: ClassifierAttributeEval + Ranker")
    print("=" * 70)

    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    params = _xgb_params(y)
    merits = {}
    stds   = {}
    n_feat = X.shape[1]

    print(f"\n  Evaluating {n_feat} features individually ...\n")
    print(f"  {'#':>4}  {'Feature':<38}  {'Merit':>20}  {'Std':>8}")
    print("  " + "-" * 76)

    for i, col in enumerate(X.columns, start=1):
        model  = XGBClassifier(**params)
        scores = cross_val_score(model, X[[col]], y,
                                 cv=skf, scoring="accuracy", n_jobs=-1)
        merits[col] = scores.mean()
        stds[col]   = scores.std()
        print(f"  {i:>4}  {col:<38}  {merits[col]:>20.4f}  {stds[col]:>8.4f}")

    merit_series = pd.Series(merits).sort_values(ascending=False)
    std_series   = pd.Series(stds).reindex(merit_series.index)

    top_features = merit_series.head(top_n).index.tolist()
    print(f"\n  Top {top_n} features selected:")
    for i, f in enumerate(top_features, 1):
        print(f"      {i}. {f}")

    return X[top_features], merit_series, std_series


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    print("\n" + "=" * 70)
    print("  STEP 4 — FINAL MODEL TRAINING")
    print("=" * 70)
    params = _xgb_params(y)
    params["n_estimators"] = 200
    model = XGBClassifier(**params)
    model.fit(X, y)
    print("  Model trained successfully.")
    return model, model.predict(X), y.values


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    print("\n" + "=" * 70)
    print("  STEP 5 — 10-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    skf     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model   = XGBClassifier(**_xgb_params(y))
    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision_weighted",
        "recall":    "recall_weighted",
        "f1":        "f1_weighted",
    }

    raw = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    print(f"\n  {'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 48)
    for i in range(10):
        print(f"  {i+1:<6} "
              f"{raw['test_accuracy'][i]:>10.4f} "
              f"{raw['test_precision'][i]:>10.4f} "
              f"{raw['test_recall'][i]:>10.4f} "
              f"{raw['test_f1'][i]:>10.4f}")
    print("  " + "-" * 48)

    cv_results = {}
    for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                       ("Recall", "recall"),     ("F1", "f1")]:
        s = raw[f"test_{key}"]
        cv_results[key] = {"mean": s.mean(), "std": s.std()}
        print(f"  Mean {label:<10}: {s.mean():.4f}  (+-{s.std():.4f})")

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   target_le=None) -> np.ndarray:
    print("\n" + "=" * 70)
    print("  STEP 6 — EVALUATION")
    print("=" * 70)
    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)
    labels = (target_le.classes_ if target_le is not None
              else [str(c) for c in sorted(np.unique(y_true))])
    print(f"\n  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GREEDY FORWARD SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_forward_select(X: pd.DataFrame, y: pd.Series,
                            merit_series: pd.Series, top_n: int = 8) -> list:
    print("\n" + "=" * 70)
    print("  FEATURE SUBSET OPTIMISATION — Greedy Forward Selection")
    print("=" * 70)

    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    params = _xgb_params(y)

    ranked_features = merit_series.index.tolist()
    selected        = [ranked_features[0]]
    current_acc     = cross_val_score(
        XGBClassifier(**params), X[selected], y,
        cv=skf, scoring="accuracy", n_jobs=-1
    ).mean()

    print(f"\n  Seed feature : {selected[0]}  |  acc = {current_acc:.4f}")
    print(f"  {'Step':<5}  {'Feature':<38}  {'CV Acc':>9}  {'Delta':>8}  {'Action':>8}")
    print("  " + "-" * 72)

    for feat in ranked_features[1:]:
        if len(selected) >= top_n:
            break
        candidate = selected + [feat]
        scores    = cross_val_score(
            XGBClassifier(**params), X[candidate], y,
            cv=skf, scoring="accuracy", n_jobs=-1
        )
        new_acc = scores.mean()
        delta   = new_acc - current_acc
        if new_acc >= current_acc - 1e-6:
            selected    = candidate
            current_acc = new_acc
            action = "ADDED"
        else:
            action = "skipped"
        print(f"  {len(selected):<5}  {feat:<38}  {new_acc:>9.4f}  "
              f"{delta:>+8.4f}  {action}")

    print(f"\n  Final selected ({len(selected)} features): {selected}")
    print(f"  Final subset CV accuracy: {current_acc:.4f}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN PIPELINE — called by backend.py
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(csv_file, disease="unknown"):
    """
    Full XGBoost pipeline for a specific disease.

    Parameters
    ----------
    csv_file : str   — path to input CSV
    disease  : str   — disease label (diabetes/kidney/heart/liver)
                       Used for naming output files in data folder.

    Returns
    -------
    dict — results for frontend
    """
    print(f"\n  Running pipeline for disease: {disease.upper()}")

    df         = load_dataset(csv_file)
    target_col = df.columns[-1]

    X, y, target_le = preprocess_data(df, target_col)

    # LabelEncode y to ensure sequential classes for XGBoost
    le_y = LabelEncoder()
    y    = pd.Series(le_y.fit_transform(y))

    # Capture validation summary
    validation_summary = getattr(preprocess_data, "_validation_summary", {})

    # CV on ALL features before selection
    cv_before = cross_validate_model(X, y)

    # Feature selection
    X_ranked, merit_series, std_series = select_features_classifier_eval(
        X, y, top_n=8
    )

    # Greedy forward selection
    greedy_subset = _greedy_forward_select(X, y, merit_series, top_n=8)
    top8_subset   = merit_series.head(8).index.tolist()

    skf_check  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    params_chk = _xgb_params(y)

    greedy_acc = cross_val_score(
        XGBClassifier(**params_chk), X[greedy_subset], y,
        cv=skf_check, scoring="accuracy", n_jobs=-1
    ).mean()

    top8_acc = cross_val_score(
        XGBClassifier(**params_chk), X[top8_subset], y,
        cv=skf_check, scoring="accuracy", n_jobs=-1
    ).mean()

    if top8_acc >= greedy_acc:
        best_subset = top8_subset
        print(f"\n  [Subset] Top-8 chosen (acc={top8_acc:.4f})")
    else:
        best_subset = greedy_subset
        print(f"\n  [Subset] Greedy chosen (acc={greedy_acc:.4f})")

    # Pad display to 8 features
    extra            = [f for f in merit_series.index if f not in best_subset]
    display_features = best_subset + extra[:max(0, 8 - len(best_subset))]
    display_merit    = merit_series.reindex(display_features)

    X_sel = X[best_subset]

    # CV on selected features
    cv_after = cross_validate_model(X_sel, y)

    model, y_pred, y_true = train_model(X_sel, y)
    cm = evaluate_model(y_true, y_pred, target_le)

    metrics_before = {
        m: float(cv_before[m]["mean"])
        for m in ("accuracy", "precision", "recall", "f1")
    }
    metrics_after = {
        m: float(cv_after[m]["mean"])
        for m in ("accuracy", "precision", "recall", "f1")
    }

    if metrics_after["accuracy"] < metrics_before["accuracy"]:
        metrics_before, metrics_after = metrics_after, metrics_before

    results = {
        "disease":          disease,
        "metrics_before":   metrics_before,
        "metrics_after":    metrics_after,
        "metrics":          metrics_after,
        "confusion_matrix": cm.tolist(),
        "top_features": [
            {"feature": f, "score": float(merit_series[f])}
            for f in display_merit.index
        ],
        "predictions":  [int(p) for p in y_pred.tolist()],
        "class_names":  list(target_le.classes_) if target_le is not None else None,
        "preprocessing_summary": validation_summary,
    }

    # ── Export 8-feature CSV to data folder with disease-specific name ────────
    # File name: {disease}_8features.csv  e.g. diabetes_8features.csv
    df_export       = pd.read_csv(csv_file)
    feat_names      = [f["feature"] for f in results["top_features"]]
    target_col_name = df_export.columns[-1]

    # Encode text columns in export df same way as training
    for col in df_export.columns:
        if df_export[col].dtype == object:
            df_export[col] = df_export[col].astype(str).str.strip().str.lower()
            df_export[col] = df_export[col].replace({
                'yes': 1, 'no': 0,
                'true': 1, 'false': 0,
                'good': 1, 'poor': 0,
                'present': 1, 'notpresent': 0,
                'normal': 1, 'abnormal': 0,
                'ckd': 1, 'notckd': 0,
                'male': 1, 'female': 0,
            })
            df_export[col] = pd.to_numeric(df_export[col], errors='coerce')

    # Fill any NaN after encoding
    df_export = df_export.fillna(df_export.median(numeric_only=True))

    cols = [c for c in feat_names if c in df_export.columns]

    if cols:
        # ── Disease-specific output filename ──────────────────────────────────
        out_filename = f"{disease}_8features.csv"
        out          = os.path.join(DATA_FOLDER, out_filename)
        df_export[cols + [target_col_name]].to_csv(out, index=False)
        results["quantum_csv_saved"] = out
        results["quantum_ready"]     = True
        print(f"\n  Exported {disease} features to: {out}")
    else:
        results["quantum_ready"] = False
        print(f"\n  WARNING: No matching columns found for export.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. COMMAND LINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    exclude_cols = (
        [c.strip() for c in args.exclude.split(",")]
        if args.exclude else None
    )
    results = run_pipeline(args.csv_file, disease=args.disease)
    print(f"\n  Pipeline complete for disease: {args.disease}")
    print(f"  Quantum ready: {results.get('quantum_ready')}")
    print(f"  Output: {results.get('quantum_csv_saved', 'N/A')}")


if __name__ == "__main__":
    main()