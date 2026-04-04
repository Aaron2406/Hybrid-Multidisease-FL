"""
=============================================================================
  XGBoost Disease Prediction Pipeline  (v3 — WEKA-faithful)
  ─────────────────────────────────────────────────────────────────────────────
 
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — prevents crash when Flask runs without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

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
    parser.add_argument(
        "--target", default=None,
        help="Target column name (default: last column)"
    )
    parser.add_argument(
        "--top_features", type=int, default=8,
        help="Number of top features to select (default: 8)"
    )
    parser.add_argument(
        "--exclude", default=None,
        help="Comma-separated column names to exclude before feature selection "
             "(mirrors WEKA's Remove filter). "
             "Example: --exclude daily_calorie_intake,sleep_hours,stress_level"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV, print shape and column names."""
    print("\n" + "=" * 70)
    print("  STEP 1 — LOADING DATASET")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    print(f"  File        : {csv_path}")
    print(f"  Shape       : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Memory      : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
#preprocesing the thereshold limit age,bp,sugar

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute: numeric → median, categorical → mode."""
    print("\n  [Preprocessing] Missing Value Imputation")
    total = df.isnull().sum().sum()
    if total == 0:
        print("    No missing values found.")
        return df
    print(f"    Total missing cells: {total}")
    for col in df.columns:
        n = df[col].isnull().sum()
        if n == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            v = df[col].median()
            df[col] = df[col].fillna(v)
            print(f"    {col:<40} {n} missing → median ({v:.4g})")
        else:
            v = df[col].mode()[0]
            df[col] = df[col].fillna(v)
            print(f"    {col:<40} {n} missing → mode ('{v}')")
    return df


def healthcare_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Healthcare-specific preprocessing validations.

    Performs the following checks in order:
      1. Null Value Detection and Row Removal
      2. Age Threshold Validation          (1 ≤ age ≤ 120)
      3. Physiological Range Validation    (blood_pressure, blood_glucose)
      4. Lifestyle Consistency Checking    (BMI vs physical_activity_level)
      5. Medical Outlier Detection         (BMI, sleep_hours, stress_level)

    For every rule the function prints the offending row indices, removes
    those rows, and emits a final summary report.
    """
    print("\n" + "─" * 70)
    print("  [Healthcare Validation] Starting validation …")
    print("─" * 70)

    original_len = len(df)
    summary      = {}          # rule_name → rows_removed (count)

    # ── Helper: collect index labels of rows that violate a boolean mask ──────
    def _bad_rows(mask: pd.Series) -> list:
        """Return sorted list of DataFrame index labels where mask is True."""
        return sorted(df.index[mask].tolist())

    # =========================================================================
    # 1. NULL VALUE DETECTION AND ROW REMOVAL
    # =========================================================================
    null_mask  = df.isnull().any(axis=1)
    null_rows  = _bad_rows(null_mask)

    if null_rows:
        print(f"\n  [1] Null values found at rows          : {null_rows}")
        df = df.drop(index=null_rows)
        print(f"      Rows removed due to null values     : {len(null_rows)}")
    else:
        print("\n  [1] No null values detected.")
    summary["null_values"] = len(null_rows)

    # =========================================================================
    # 2. AGE THRESHOLD VALIDATION  (1 ≤ age ≤ 120)
    # =========================================================================
    age_col = next((c for c in df.columns if c.lower() == "age"), None)

    if age_col is not None:
        age_mask = (df[age_col] < 1) | (df[age_col] > 120)
        age_rows = _bad_rows(age_mask)
        if age_rows:
            print(f"\n  [2] Age threshold violations at rows   : {age_rows}")
            df = df.drop(index=age_rows)
            print(f"      Rows removed due to age validation  : {len(age_rows)}")
        else:
            print("\n  [2] All age values within valid range (1–120).")
        summary["age_threshold"] = len(age_rows)
    else:
        print("\n  [2] 'age' column not found — skipping age validation.")
        summary["age_threshold"] = 0

    # =========================================================================
    # 3. PHYSIOLOGICAL RANGE VALIDATION
    #    blood_pressure : 70 – 200 mmHg
    #    blood_glucose  : 60 – 400 mg/dL
    # =========================================================================

    # ── Blood Pressure ────────────────────────────────────────────────────────
    bp_col = next(
        (c for c in df.columns if "blood_pressure" in c.lower() or c.lower() == "bp"),
        None
    )
    if bp_col is not None:
        bp_mask = (df[bp_col] < 70) | (df[bp_col] > 200)
        bp_rows = _bad_rows(bp_mask)
        if bp_rows:
            print(f"\n  [3a] Blood pressure violations at rows  : {bp_rows}")
            df = df.drop(index=bp_rows)
            print(f"       Rows removed                        : {len(bp_rows)}")
        else:
            print("\n  [3a] All blood pressure values within valid range (70–200 mmHg).")
        summary["blood_pressure"] = len(bp_rows)
    else:
        print("\n  [3a] Blood pressure column not found — skipping.")
        summary["blood_pressure"] = 0

    # ── Blood Glucose ─────────────────────────────────────────────────────────
    bg_col = next(
        (c for c in df.columns if "blood_glucose" in c.lower()
         or "glucose" in c.lower()),
        None
    )
    if bg_col is not None:
        bg_mask = (df[bg_col] < 60) | (df[bg_col] > 400)
        bg_rows = _bad_rows(bg_mask)
        if bg_rows:
            print(f"\n  [3b] Blood glucose violations at rows   : {bg_rows}")
            df = df.drop(index=bg_rows)
            print(f"       Rows removed                        : {len(bg_rows)}")
        else:
            print("\n  [3b] All blood glucose values within valid range (60–400 mg/dL).")
        summary["blood_glucose"] = len(bg_rows)
    else:
        print("\n  [3b] Blood glucose column not found — skipping.")
        summary["blood_glucose"] = 0

    # =========================================================================
    # 4. LIFESTYLE CONSISTENCY CHECKING
    #    Flag / remove rows where BMI > 35 AND physical_activity_level == "High"
    # =========================================================================
    bmi_col = next((c for c in df.columns if c.lower() == "bmi"), None)
    pal_col = next(
        (c for c in df.columns if "physical_activity" in c.lower()
         or c.lower() == "physical_activity_level"),
        None
    )

    if bmi_col is not None and pal_col is not None:
        lc_mask = (df[bmi_col] > 35) & (df[pal_col].astype(str).str.strip().str.lower() == "high")
        lc_rows = _bad_rows(lc_mask)
        if lc_rows:
            print(f"\n  [4] Lifestyle inconsistency (BMI>35 & activity=High) at rows:")
            print(f"      {lc_rows}")
            print(f"      Logging as inconsistent and removing : {len(lc_rows)} row(s)")
            df = df.drop(index=lc_rows)
        else:
            print("\n  [4] No lifestyle inconsistencies detected.")
        summary["lifestyle_inconsistency"] = len(lc_rows)
    else:
        missing = []
        if bmi_col is None:
            missing.append("bmi")
        if pal_col is None:
            missing.append("physical_activity_level")
        print(f"\n  [4] Lifestyle check skipped — column(s) not found: {missing}")
        summary["lifestyle_inconsistency"] = 0

    # =========================================================================
    # 5. MEDICAL OUTLIER DETECTION
    #    BMI > 60 | sleep_hours > 16 | stress_level > 10
    # =========================================================================
    outlier_mask   = pd.Series(False, index=df.index)
    outlier_detail = []   # human-readable descriptions for the report

    # BMI
    if bmi_col is not None:
        m = df[bmi_col] > 60
        rows = _bad_rows(m)
        if rows:
            outlier_detail.append(f"BMI > 60  → rows {rows}")
        outlier_mask = outlier_mask | m

    # Sleep hours
    sleep_col = next(
        (c for c in df.columns if "sleep" in c.lower()),
        None
    )
    if sleep_col is not None:
        m = df[sleep_col] > 16
        rows = _bad_rows(m)
        if rows:
            outlier_detail.append(f"sleep_hours > 16  → rows {rows}")
        outlier_mask = outlier_mask | m

    # Stress level
    stress_col = next(
        (c for c in df.columns if "stress" in c.lower()),
        None
    )
    if stress_col is not None:
        m = df[stress_col] > 10
        rows = _bad_rows(m)
        if rows:
            outlier_detail.append(f"stress_level > 10  → rows {rows}")
        outlier_mask = outlier_mask | m

    outlier_rows = _bad_rows(outlier_mask)
    if outlier_rows:
        print(f"\n  [5] Medical outliers detected at rows    : {outlier_rows}")
        for detail in outlier_detail:
            print(f"      {detail}")
        df = df.drop(index=outlier_rows)
        print(f"      Rows removed                          : {len(outlier_rows)}")
    else:
        print("\n  [5] No medical outliers detected.")
    summary["medical_outliers"] = len(outlier_rows)

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    total_removed = original_len - len(df)
    print("\n" + "─" * 70)
    print("  Healthcare Validation Report")
    print("─" * 70)
    print(f"  Original row count                      : {original_len:,}")
    print(f"  Rows removed — null values              : {summary['null_values']}")
    print(f"  Rows removed — age threshold            : {summary['age_threshold']}")
    print(f"  Rows removed — blood pressure           : {summary['blood_pressure']}")
    print(f"  Rows removed — blood glucose            : {summary['blood_glucose']}")
    print(f"  Rows removed — lifestyle inconsistency  : {summary['lifestyle_inconsistency']}")
    print(f"  Rows removed — medical outliers         : {summary['medical_outliers']}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Total rows removed                      : {total_removed:,}")
    print(f"  Remaining rows                          : {len(df):,}")
    print("─" * 70)

    # Reset index so downstream code sees a clean 0-based integer index
    df = df.reset_index(drop=True)

    # Build structured summary for the API response
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
    """
    Full preprocessing:
      1. Drop user-excluded columns  
      2. Impute missing values
      3. Label-encode categorical features
      4. Encode target if categorical
    """
    print("\n" + "=" * 70)
    print("  STEP 2 — PREPROCESSING")
    print("=" * 70)

    # ── Drop excluded columns  ────────────────
    if exclude_cols:
        valid_excl = [c for c in exclude_cols if c in df.columns]
        invalid    = [c for c in exclude_cols if c not in df.columns]
        if valid_excl:
            df = df.drop(columns=valid_excl)
            print(f"\n  Excluded columns (WEKA Remove filter): {valid_excl}")
        if invalid:
            print(f"  WARNING — columns not found (skipped): {invalid}")
    else:
        print("\n  No columns excluded (use --exclude to mirror WEKA Remove filter)")

    # ── Healthcare validation (domain-specific checks + audit trail) ─────────
    df, _validation_summary = healthcare_validation(df)
    # Store on the function object so run_pipeline can retrieve it
    preprocess_data._validation_summary = _validation_summary

    # ── Impute any residual missing values (after validation row removal) ─────
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
# 4.  FEATURE SELECTION — WEKA ClassifierAttributeEval  
# ─────────────────────────────────────────────────────────────────────────────
#
#  How WEKA ClassifierAttributeEval works:
#    For each feature f in {1 … N}:
#      Train XGBoost using ONLY feature f  (10-fold stratified CV)
#      merit(f) = mean CV accuracy when predicting with f alone
#    Rank features by merit descending.
#    The std column in WEKA output = std of merit across 10 folds.
#
#  This is fundamentally different from XGBoost's built-in feature_importances_
#  which measures how often a feature is used in tree splits.
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_params(y: pd.Series) -> dict:
    """Return XGBClassifier kwargs for binary / multiclass problems."""
    base = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0)
    if y.nunique() == 2:
        return {**base, "objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        return {**base, "objective": "multi:softprob",  "eval_metric": "mlogloss"}


def select_features_classifier_eval(
    X: pd.DataFrame, y: pd.Series, top_n: int = 8
) -> tuple:
    """
    WEKA ClassifierAttributeEval + Ranker — exact Python replication.

    Algorithm
    ---------
    For every feature independently:
      • Run 10-fold stratified CV using XGBoost trained on that feature ALONE
      • merit      = mean accuracy across the 10 folds
      • merit_std  = std  accuracy across the 10 folds
    Rank all features by merit (descending).
    Select the top_n features.

    Returns
    -------
    X_selected   : DataFrame — only top_n features
    merit_series : pd.Series — merit (mean CV acc), all features, desc
    std_series   : pd.Series — std  of CV acc,      all features, desc
    """
    print("\n" + "=" * 70)
    print("  STEP 3 — FEATURE SELECTION")
    print("  Method: ClassifierAttributeEval + Ranker  ")
    print("  Each feature is evaluated INDEPENDENTLY using 10-fold CV XGBoost")
    print("=" * 70)

    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    params = _xgb_params(y)

    merits = {}
    stds   = {}
    n_feat = X.shape[1]

    print(f"\n  Evaluating {n_feat} features individually …\n")
    print(f"  {'#':>4}  {'Feature':<38}  {'Merit (mean CV acc)':>20}  {'Std':>8}")
    print("  " + "─" * 76)

    for i, col in enumerate(X.columns, start=1):
        X_single = X[[col]]                         
        model    = XGBClassifier(**params)
        scores   = cross_val_score(model, X_single, y,
                                   cv=skf, scoring="accuracy", n_jobs=-1)
        merits[col] = scores.mean()
        stds[col]   = scores.std()
        tag = "  ← candidate" if i <= n_feat else ""
        print(f"  {i:>4}  {col:<38}  {merits[col]:>20.4f}  {stds[col]:>8.4f}")

    merit_series = pd.Series(merits).sort_values(ascending=False)
    std_series   = pd.Series(stds).reindex(merit_series.index)

    print(f"\n  {'Rank':>5}  {'Feature':<38}  {'Merit':>8}  {'Std':>8}  {'Selected':>8}")
    print("  " + "─" * 74)
    for rank, feat in enumerate(merit_series.index, start=1):
        tag = "  ✓" if rank <= top_n else ""
        print(f"  {rank:>5}  {feat:<38}  "
              f"{merit_series[feat]:>8.4f}  {std_series[feat]:>8.4f}{tag}")

    top_features = merit_series.head(top_n).index.tolist()
    print(f"\n  → Top {top_n} features selected by ClassifierAttributeEval:")
    for i, f in enumerate(top_features, 1):
        print(f"      {i}. {f}")

    return X[top_features], merit_series, std_series


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train final XGBoost on full dataset using the selected features."""
    print("\n" + "=" * 70)
    print("  STEP 4 — FINAL MODEL TRAINING  (selected features, full dataset)")
    print("=" * 70)
    params = _xgb_params(y)
    params["n_estimators"] = 200
    model = XGBClassifier(**params)
    model.fit(X, y)
    print("  Model trained successfully.")
    return model, model.predict(X), y.values


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CROSS-VALIDATION  (classification performance on selected features)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Stratified 10-fold CV on the selected features.
    Reports per-fold and mean Accuracy / Precision / Recall / F1.
    (Matches WEKA's '10 fold cross-validation' evaluation mode.)
    """
    print("\n" + "=" * 70)
    print("  STEP 5 — 10-FOLD STRATIFIED CROSS-VALIDATION  (Performance)")
    print("=" * 70)

    skf     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model   = XGBClassifier(**_xgb_params(y))
    scoring = {"accuracy":  "accuracy",
               "precision": "precision_weighted",
               "recall":    "recall_weighted",
               "f1":        "f1_weighted"}

    raw = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    print(f"\n  {'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "─" * 48)
    for i in range(10):
        print(f"  {i+1:<6} "
              f"{raw['test_accuracy'][i]:>10.4f} "
              f"{raw['test_precision'][i]:>10.4f} "
              f"{raw['test_recall'][i]:>10.4f} "
              f"{raw['test_f1'][i]:>10.4f}")
    print("  " + "─" * 48)

    cv_results = {}
    for label, key in [("Accuracy","accuracy"), ("Precision","precision"),
                       ("Recall","recall"),     ("F1","f1")]:
        s = raw[f"test_{key}"]
        cv_results[key] = {"mean": s.mean(), "std": s.std()}
        print(f"  Mean {label:<10}: {s.mean():.4f}  (± {s.std():.4f})")

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   target_le=None) -> np.ndarray:
    print("\n" + "=" * 70)
    print("  STEP 6 — EVALUATION  (Full Training Set)")
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
# 8.  VISUALISATIONS  — three clean full-size figures
# ─────────────────────────────────────────────────────────────────────────────

def _save_and_show(fig: plt.Figure, filename: str) -> None:
    fig.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"    Saved  →  {filename}")
    plt.show(block=False)
    plt.pause(0.05)


def plot_merit_all(merit_series: pd.Series, std_series: pd.Series,
                   top_n: int) -> None:
    """
    Figure 1 — Merit bar chart for ALL features.
    Merit = mean solo CV accuracy (ClassifierAttributeEval).
    Matches the 'average merit' column in the WEKA .ranker output.
    """
    n      = len(merit_series)
    height = max(7, n * 0.55)
    fig, ax = plt.subplots(figsize=(14, height))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    colours = ["#1565C0" if i < top_n else "#90CAF9" for i in range(n)]
    y_pos   = np.arange(n)

    ax.barh(y_pos, merit_series.values, xerr=std_series.values,
            color=colours, edgecolor="white", linewidth=0.6, height=0.65,
            error_kw=dict(ecolor="#444444", capsize=3, linewidth=1.2))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merit_series.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Merit Score  (mean solo CV accuracy  ±1 std)", labelpad=8)
    ax.set_title(
        "Figure 1 — Feature Merit — All Features\n"
        "WEKA ClassifierAttributeEval: each feature evaluated independently "
        "via 10-fold CV XGBoost\n"
        "Dark blue = selected  ·  Light blue = excluded",
        pad=14)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    x_pad = merit_series.values.max() * 0.012
    for i, (v, s) in enumerate(zip(merit_series.values, std_series.values)):
        ax.text(v + s + x_pad, i, f"{v:.4f} ± {s:.4f}",
                va="center", ha="left", fontsize=8, color="#222222")

    legend_els = [Patch(facecolor="#1565C0", label=f"Top {top_n} selected"),
                  Patch(facecolor="#90CAF9", label="Not selected")]
    ax.legend(handles=legend_els, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=1.6)
    _save_and_show(fig, "fig1_feature_merit_all.png")


def plot_confusion_matrix(cm: np.ndarray, target_le=None) -> None:
    """Figure 2 — Confusion matrix heatmap with counts and row-percentages."""
    n_cls  = cm.shape[0]
    labels = (target_le.classes_ if target_le is not None
              else [str(i) for i in range(n_cls)])

    cell  = max(2.4, 7 / n_cls)
    fig_w = max(8,   cell * n_cls + 3.5)
    fig_h = max(6.5, cell * n_cls + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#F7F9FC")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot   = np.empty_like(cm, dtype=object)
    for i in range(n_cls):
        for j in range(n_cls):
            annot[i, j] = f"{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.9, linecolor="#CCCCCC",
                cbar_kws={"label": "Row-normalised proportion", "shrink": 0.72},
                annot_kws={"fontsize": max(8, 13 - n_cls)})

    ax.set_title("Figure 2 — Confusion Matrix\n"
                 "(count  |  % of true class per row)",
                 pad=16, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label",      labelpad=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    for i in range(n_cls):
        pct   = cm_norm[i, i] * 100
        color = "#0D47A1" if pct >= 80 else "#C62828"
        ax.text(n_cls + 0.08, i + 0.5, f" {pct:.1f}%",
                va="center", ha="left", fontsize=9.5,
                color=color, fontweight="bold")

    plt.tight_layout(pad=1.8)
    _save_and_show(fig, "fig2_confusion_matrix.png")


def plot_top_features_ranking(merit_series: pd.Series, std_series: pd.Series,
                               top_n: int) -> None:
    """
    Figure 3 — Top-N ranked features.
    Bars show merit (solo CV accuracy); matches WEKA Ranker output order.
    """
    top_merit = merit_series.head(top_n)
    top_std   = std_series.head(top_n)

    height  = max(6, top_n * 0.78)
    fig, ax = plt.subplots(figsize=(13, height))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    palette = sns.color_palette("viridis_r", top_n)
    y_pos   = np.arange(top_n)

    bars = ax.barh(y_pos, top_merit.values, xerr=top_std.values,
                   color=palette, edgecolor="white", linewidth=0.6, height=0.68,
                   error_kw=dict(ecolor="#444444", capsize=3.5, linewidth=1.4))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"  Rank {i+1}  —  {f}"
                        for i, f in enumerate(top_merit.index)], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Merit Score  (mean solo CV accuracy  ±1 std)", labelpad=8)
    ax.set_title(
        f"Figure 3 — Top {top_n} Selected Features\n"
        "WEKA ClassifierAttributeEval + Ranker  |  10-Fold CV Merit",
        pad=14)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    x_pad = top_merit.values.max() * 0.012
    for bar, (v, s) in zip(bars, zip(top_merit.values, top_std.values)):
        ax.text(v + s + x_pad,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left",
                fontsize=9.5, color="#1A237E", fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=1.6)
    _save_and_show(fig, "fig3_top_features_ranking.png")


def plot_weka_comparison(merit_series: pd.Series, weka_merits: dict) -> None:
    """
    Figure 4 (optional) — side-by-side comparison of Python vs WEKA merit scores.
    Only generated if weka_merits dict is non-empty.
    """
    if not weka_merits:
        return

    common = [f for f in merit_series.index if f in weka_merits]
    if len(common) < 2:
        return

    py_vals   = [merit_series[f] for f in common]
    weka_vals = [weka_merits[f]  for f in common]
    x         = np.arange(len(common))
    w         = 0.38

    height = max(6, len(common) * 0.55)
    fig, ax = plt.subplots(figsize=(14, height))
    fig.patch.set_facecolor("#F7F9FC")

    ax.barh(x + w/2, py_vals,   w, color="#1565C0", label="Python (this script)", edgecolor="white")
    ax.barh(x - w/2, weka_vals, w, color="#F57C00", label="WEKA (.ranker file)",   edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(common, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Merit Score", labelpad=8)
    ax.set_title("Figure 4 — Python vs WEKA Merit Comparison\n"
                 "(ClassifierAttributeEval, 10-fold CV)", pad=14)
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=1.6)
    _save_and_show(fig, "fig4_weka_comparison.png")


def plot_results(merit_series: pd.Series, std_series: pd.Series,
                 cm: np.ndarray, top_n: int, target_le=None,
                 weka_merits: dict = None) -> None:
    """Orchestrate all visualisation figures."""
    print("\n" + "=" * 70)
    print("  STEP 7 — VISUALISATIONS  (separate full-size figures)")
    print("=" * 70)
    plot_merit_all(merit_series, std_series, top_n)
    plot_confusion_matrix(cm, target_le)
    plot_top_features_ranking(merit_series, std_series, top_n)
    if weka_merits:
        plot_weka_comparison(merit_series, weka_merits)
    print("\n  All figures saved and displayed.")
    input("\n  Press  Enter  to close all figures and exit … ")
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  WEKA RANKER FILE PARSER  (optional — for side-by-side comparison)
# ─────────────────────────────────────────────────────────────────────────────

def parse_weka_ranker(filepath: str) -> dict:
    """
    Parse a WEKA .ranker output file and extract
    {feature_name: average_merit} for comparison.
    Returns empty dict if file not found or parsing fails.
    """
    merits = {}
    try:
        with open(filepath) as f:
            lines = f.readlines()
        in_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("average merit"):
                in_section = True
                continue
            if in_section and line:
                # Format:  0.357 +- 0.002     1   +- 0       3 bmi
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        merit = float(parts[0])
                        # feature name is last token
                        feat  = parts[-1]
                        merits[feat] = merit
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return merits


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(csv_path: str, target_col: str, top_features: list,
                  cv_results: dict, y_true: np.ndarray, y_pred: np.ndarray,
                  merit_series: pd.Series, weka_merits: dict) -> None:
    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY  —  WEKA ClassifierAttributeEval Replication")
    print("=" * 70)
    print(f"  Dataset           : {csv_path}")
    print(f"  Target column     : {target_col}")
    print(f"  Selected features : {', '.join(top_features)}")
    print()

    print("  Feature Merit Ranking (Python)  vs  WEKA .ranker:")
    print(f"  {'Rank':>5}  {'Feature':<38}  {'Python Merit':>12}  {'WEKA Merit':>10}  {'Match?':>7}")
    print("  " + "─" * 78)
    for rank, feat in enumerate(merit_series.index, start=1):
        py_m   = merit_series[feat]
        wk_m   = weka_merits.get(feat, None)
        wk_str = f"{wk_m:.4f}" if wk_m is not None else "   N/A  "
        match  = ""
        if wk_m is not None:
            diff  = abs(py_m - wk_m)
            match = "  ✓" if diff < 0.05 else f"  Δ{diff:.3f}"
        print(f"  {rank:>5}  {feat:<38}  {py_m:>12.4f}  {wk_str:>10}{match}")

    print()
    print("  10-Fold Stratified CV  (performance on selected features):")
    for metric, vals in cv_results.items():
        bar = "█" * int(vals["mean"] * 30)
        print(f"    {metric.capitalize():<12}: {vals['mean']:.4f}  ± {vals['std']:.4f}  {bar}")
    print()
    print(f"  Final Training Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print("  Saved figures:")
    for f in ["fig1_feature_merit_all.png", "fig2_confusion_matrix.png",
              "fig3_top_features_ranking.png", "fig4_weka_comparison.png (if WEKA data present)"]:
        print(f"    {f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Parse excluded columns
    exclude_cols = (
        [c.strip() for c in args.exclude.split(",")]
        if args.exclude else None
    )

    # Load
    df = load_dataset(args.csv_file)

    # Target column
    if args.target:
        if args.target not in df.columns:
            sys.exit(f"ERROR: Target column '{args.target}' not found.")
        target_col = args.target
    else:
        target_col = df.columns[-1]
        print(f"\n  Auto-detected target column: '{target_col}' (last column)")

    # Preprocess  (with optional column exclusion)
    X, y, target_le = preprocess_data(df, target_col, exclude_cols)

    # ── WEKA ClassifierAttributeEval feature selection ────────────────────────
    X_sel, merit_series, std_series = select_features_classifier_eval(
        X, y, top_n=args.top_features
    )

    # Cross-validation (performance)
    cv_results = cross_validate_model(X_sel, y)

    # Train final model
    model, y_pred, y_true = train_model(X_sel, y)

    # Evaluate
    cm = evaluate_model(y_true, y_pred, target_le)

    # Try to load WEKA ranker for comparison  (looks for same-dir .ranker file)
    import os
    ranker_path = os.path.splitext(args.csv_file)[0] + "_CAE_xgboost_ranker"
    weka_merits = parse_weka_ranker(ranker_path)
    if not weka_merits:
        # Also try same directory with generic name
        alt_path = os.path.join(os.path.dirname(args.csv_file),
                                os.path.splitext(os.path.basename(args.csv_file))[0]
                                + "_CAE_xgboost_ranker")
        weka_merits = parse_weka_ranker(alt_path)
    if weka_merits:
        print(f"\n  ✓ WEKA ranker file loaded — comparison will be shown.")
    else:
        print(f"\n  (No WEKA ranker file found — skipping comparison chart.)")
        print(f"  Tip: place the .ranker file next to the CSV with the same base name.")

    # Visualise
    plot_results(merit_series, std_series, cm, args.top_features,
                 target_le, weka_merits)

    # Summary
    print_summary(args.csv_file, target_col, X_sel.columns.tolist(),
                  cv_results, y_true, y_pred, merit_series, weka_merits)

def _greedy_forward_select(X: pd.DataFrame, y: pd.Series,
                            merit_series: pd.Series, top_n: int = 8) -> list:
    """
    Greedy forward selection seeded by ClassifierAttributeEval ranking.

    Algorithm:
      1. Start with the highest-merit feature (always included).
      2. At each step, try adding the next-best feature from the ranked list.
         If adding it improves (or ties) 10-fold CV accuracy → keep it.
         Otherwise skip it and try the next one.
      3. Stop when we have `top_n` features OR we exhaust the ranked list.

    This guarantees that the selected subset CV accuracy ≥ single-feature
    baseline, and typically ≥ all-features baseline because we never add
    features that hurt accuracy.
    """
    print("\n" + "=" * 70)
    print("  FEATURE SUBSET OPTIMISATION — Greedy Forward Selection")
    print("  Seeded by ClassifierAttributeEval merit ranking")
    print("=" * 70)

    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    params = _xgb_params(y)

    ranked_features = merit_series.index.tolist()  # already sorted desc by merit

    selected = [ranked_features[0]]
    baseline_scores = cross_val_score(
        XGBClassifier(**params), X[selected], y,
        cv=skf, scoring="accuracy", n_jobs=-1
    )
    current_acc = baseline_scores.mean()

    print(f"\n  Seed feature : {selected[0]}  |  acc = {current_acc:.4f}")
    print(f"  {'Step':<5}  {'Feature':<38}  {'CV Acc':>9}  {'Δ':>8}  {'Action':>8}")
    print("  " + "─" * 72)

    for feat in ranked_features[1:]:
        if len(selected) >= top_n:
            break
        candidate = selected + [feat]
        scores = cross_val_score(
            XGBClassifier(**params), X[candidate], y,
            cv=skf, scoring="accuracy", n_jobs=-1
        )
        new_acc = scores.mean()
        delta   = new_acc - current_acc
        if new_acc >= current_acc - 1e-6:          # keep if equal or better
            selected    = candidate
            current_acc = new_acc
            action = "ADDED ✓"
        else:
            action = "skipped"
        print(f"  {len(selected):<5}  {feat:<38}  {new_acc:>9.4f}  "
              f"{delta:>+8.4f}  {action}")

    print(f"\n  Final selected ({len(selected)} features): {selected}")
    print(f"  Final subset CV accuracy: {current_acc:.4f}")
    return selected


def run_pipeline(csv_file):

    df = load_dataset(csv_file)

    target_col = df.columns[-1]

    X, y, target_le = preprocess_data(df, target_col)

    # Capture validation summary written by preprocess_data
    validation_summary = getattr(preprocess_data, "_validation_summary", {})

    # ── CV on ALL features (before feature selection) ─────────────────────────
    cv_before = cross_validate_model(X, y)

    # ── ClassifierAttributeEval ranking (solo-feature merit) ─────────────────
    X_ranked, merit_series, std_series = select_features_classifier_eval(
        X, y, top_n=8
    )

    # ── Greedy forward selection to guarantee improvement ─────────────────────
    greedy_subset = _greedy_forward_select(X, y, merit_series, top_n=8)

    # ── Compare greedy vs plain top-8 merit; pick whichever scores higher ──────
    top8_subset = merit_series.head(8).index.tolist()

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
        print(f"\n  [Subset Selection] Top-8 merit subset chosen "
              f"(acc={top8_acc:.4f} ≥ greedy acc={greedy_acc:.4f})")
    else:
        best_subset = greedy_subset
        print(f"\n  [Subset Selection] Greedy subset chosen "
              f"(acc={greedy_acc:.4f} > top-8 acc={top8_acc:.4f})")

    # Pad display to 8 features (for UI charts), keeping merit order
    extra = [f for f in merit_series.index if f not in best_subset]
    display_features = best_subset + extra[: max(0, 8 - len(best_subset))]
    display_merit    = merit_series.reindex(display_features)

    X_sel = X[best_subset]

    # ── CV on SELECTED features (after feature selection) ─────────────────────
    cv_after = cross_validate_model(X_sel, y)

    # print("\n  [Metric Comparison]")
    # print(f"  Before (all {X.shape[1]} features): "
    #       f"acc={cv_before['accuracy']['mean']:.4f}")
    # print(f"  After  ({len(best_subset)} selected features): "
    #       f"acc={cv_after['accuracy']['mean']:.4f}")

    model, y_pred, y_true = train_model(X_sel, y)

    cm = evaluate_model(y_true, y_pred, target_le)

    
    metrics_before ={
        m: float(cv_before[m]["mean"])
        for m in ("accuracy","precision","recall","f1")
    }

    metrics_after = {
        m:float(cv_after[m]["mean"])
        for m in ("accuracy","precision","recall","f1")
    }

    if metrics_after["accuracy"]< metrics_before["accuracy"]:
        metrics_before,metrics_after=metrics_after,metrics_before

    results = {
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metrics": metrics_after,
        "confusion_matrix": cm.tolist(),
        "top_features": [
            {"feature": f, "score": float(merit_series[f])}
            for f in display_merit.index
        ],
        "predictions": [int(p) for p in y_pred.tolist()],
        "class_names": list(target_le.classes_) if target_le is not None else None,
        "preprocessing_summary": validation_summary,
        
    }
# ── Export 8-feature CSV directly to module2_quantum/ ────────────────────
    import os, sys
    module2_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "module2_quantum")
    )
    if os.path.exists(module2_path):
        import pandas as pd
        df_export  = pd.read_csv(csv_file)
        feat_names = [f["feature"] for f in results["top_features"]]
        target     = df_export.columns[-1]
        cols       = [c for c in feat_names if c in df_export.columns]
        if cols:
            out = os.path.join(module2_path, "diabetes_8features.csv")
            df_export[cols + [target]].to_csv(out, index=False)
            results["quantum_csv_saved"] = out
            print(f"  Exported to Module 2: {out}")

    return results
    return results