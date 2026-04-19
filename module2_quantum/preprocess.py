"""
Module 2 - Quantum Feature Encoding
preprocess.py  (fixed - dynamic feature reading)
---------------------------------------------------------------------
ROOT CAUSE OF BUG:
  SELECTED_FEATURES was hardcoded as a fixed list.
  Module 1 picks features dynamically via greedy selection —
  the actual 8 features change every run depending on the dataset.
  So hardcoded list and actual CSV columns never match reliably.

FIX:
  Read feature columns directly from the CSV at load time.
  SELECTED_FEATURES is derived from the file, not hardcoded.
  N_QUBITS is set to however many features the CSV actually has.
---------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

TARGET_COL   = "Diabetes"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# These are set dynamically when load_data() runs
# Do NOT hardcode these — they come from the CSV
SELECTED_FEATURES = []
N_QUBITS          = 8   # default, overwritten by load_data()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD CSV  (derives feature list from actual file)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the Module 1 output CSV and derive SELECTED_FEATURES
    dynamically from whatever columns are present.

    Works with any CSV that has:
      - N feature columns  (any names)
      - 'Diabetes' as the last or named target column
    """
    global SELECTED_FEATURES, N_QUBITS

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[preprocess] File not found: {csv_path}\n"
            f"  Run Module 1 first to generate diabetes_8features.csv"
        )

    df = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print("  MODULE 2 - PREPROCESSING")
    print("=" * 60)
    print(f"  File loaded : {csv_path}")
    print(f"  Shape       : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"  Columns     : {list(df.columns)}")

    # Identify target column — last column or named 'Diabetes'
    if TARGET_COL in df.columns:
        target = TARGET_COL
    else:
        target = df.columns[-1]
        print(f"  Note: 'Diabetes' not found, using last column '{target}'")

    # Derive feature columns = all columns except target
    SELECTED_FEATURES = [c for c in df.columns if c != target]
    N_QUBITS          = len(SELECTED_FEATURES)

    print(f"  Features    : {SELECTED_FEATURES}")
    print(f"  Target      : {target}")
    print(f"  N_QUBITS    : {N_QUBITS}  (one qubit per feature)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SEPARATE X AND y
# ─────────────────────────────────────────────────────────────────────────────

def split_features_target(df: pd.DataFrame):
    """Split into feature matrix X and target vector y."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer

    target = TARGET_COL if TARGET_COL in df.columns else df.columns[-1]

    # ── Step 1: Copy feature columns ─────────────────────────
    df_encoded = df[SELECTED_FEATURES].copy()

    # ── Step 2: Encode ALL columns (text and numeric both) ───
    for col in df_encoded.columns:
        # Replace any 'yes'/'no' style values first
        df_encoded[col] = df_encoded[col].astype(str).str.strip().str.lower()
        df_encoded[col] = df_encoded[col].replace({
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0,
            'male': 1, 'female': 0,
            'positive': 1, 'negative': 0,
        })
        # If still non-numeric use LabelEncoder
        try:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='raise')
        except Exception:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # ── Step 3: Convert to float ──────────────────────────────
    X_raw = df_encoded.values.astype(np.float64)

    # ── Step 4: Impute any remaining NaN ─────────────────────
    if np.isnan(X_raw).any():
        print("\n  ⚠ NaN values found — applying median imputation")
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_raw)
        print(f"  ✅ NaN values fixed")
    else:
        X = X_raw

    # ── Step 5: Encode target if text ────────────────────────
    y_raw = df[target].astype(str).str.strip().str.lower()
    y_raw = y_raw.replace({'yes': 1, 'no': 0,
                           'true': 1, 'false': 0,
                           'positive': 1, 'negative': 0})
    try:
        y = pd.to_numeric(y_raw, errors='raise').values.astype(np.int32)
    except Exception:
        le = LabelEncoder()
        y  = le.fit_transform(y_raw.astype(str)).astype(np.int32)

    print(f"\n  Feature matrix X : shape={X.shape}  dtype={X.dtype}")
    print(f"  Target vector  y : shape={y.shape}   dtype={y.dtype}")

    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls} : {cnt:,}  ({cnt/len(y)*100:.1f}%)")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test(X: np.ndarray, y: np.ndarray):
    """Stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    print(f"\n  Train/Test split  : 80% / 20%")
    print(f"  X_train shape     : {X_train.shape}")
    print(f"  X_test  shape     : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — NORMALIZE TO [0, π]
# ─────────────────────────────────────────────────────────────────────────────

def normalize_for_quantum(X_train: np.ndarray, X_test: np.ndarray):
    """
    Scale features to [0, pi] for RY angle encoding.
    Scaler fitted on train only — avoids data leakage.
    Test values clamped to [0, pi] in quantum_circuit.py.
    """
    scaler         = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\n  Normalization   : MinMaxScaler -> [0, pi]")
    print(f"  X_train range   : [{X_train_scaled.min():.4f},  {X_train_scaled.max():.4f}]")
    print(f"  X_test  range   : [{X_test_scaled.min():.4f},  {X_test_scaled.max():.4f}]")

    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Apply SMOTE on training data only if classes are imbalanced."""
    from sklearn.impute import SimpleImputer

    # ── Fix NaN values before SMOTE ──────────────────────────
    if np.isnan(X_train).any():
        print("\n  ⚠ NaN values detected — applying median imputation")
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        print(f"  ✅ NaN values fixed")

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Class balance before SMOTE:")
    for cls, cnt in zip(unique, counts):
        print(f"    Class {int(cls)} : {int(cnt)}")

    # ── Check minimum samples per class ──────────────────────
    min_samples = min(counts)
    ratio       = max(counts) / min_samples

    if min_samples < 6:
        print(f"\n  ⚠ Not enough samples for SMOTE "
              f"(min class has {min_samples} samples, need 6)")
        print(f"  SMOTE skipped — returning original data")
        return X_train, y_train

    if ratio > 1.5:
        # Set k_neighbors safely
        k = min(5, min_samples - 1)
        smote              = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_res, y_res       = smote.fit_resample(X_train, y_train)
        u2, c2             = np.unique(y_res, return_counts=True)
        print(f"  Class balance after SMOTE:")
        for cls, cnt in zip(u2, c2):
            print(f"    Class {int(cls)} : {int(cnt)}")
        return X_res, y_res
    else:
        print(f"  Classes already balanced (ratio={ratio:.2f}) - SMOTE skipped.")
        return X_train, y_train

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(csv_path: str = "diabetes_8features.csv"):
    """
    Full preprocessing pipeline for Module 2.

    Parameters
    ----------
    csv_path : str
        Path to the CSV saved by Module 1.
        Features are read dynamically — no hardcoded list needed.

    Returns
    -------
    X_train  : np.ndarray  shape (N_train, N_features)  in [0, pi]
    X_test   : np.ndarray  shape (N_test,  N_features)  in [0, pi]
    y_train  : np.ndarray  shape (N_train,)
    y_test   : np.ndarray  shape (N_test,)
    scaler   : MinMaxScaler  (fitted on train only)
    """
    # Step 1 — load and derive features dynamically
    df = load_data(csv_path)

    # Step 2 — separate X and y
    X, y = split_features_target(df)

    # Step 3 — split first (before normalization and SMOTE)
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(X, y)

    # Step 4 — normalize to [0, pi]
    X_train_scaled, X_test_scaled, scaler = normalize_for_quantum(
        X_train_raw, X_test_raw
    )

    # Step 5 — SMOTE on normalized train only
    X_train_final, y_train_final = apply_smote(X_train_scaled, y_train)

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  X_train (after SMOTE) : {X_train_final.shape}")
    print(f"  X_test                : {X_test_scaled.shape}")
    print(f"  y_train               : {y_train_final.shape}")
    print(f"  y_test                : {y_test.shape}")
    print(f"  Features              : {SELECTED_FEATURES}")
    print(f"  Qubits                : {N_QUBITS}")
    print(f"  Ready for quantum circuit encoding")
    print("=" * 60)

    return X_train_final, X_test_scaled, y_train_final, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(
        csv_path="diabetes_8features_sample.csv"
    )

    print(f"\n  Quick sanity checks:")
    print(f"  Features detected : {SELECTED_FEATURES}")
    print(f"  N_QUBITS          : {N_QUBITS}")
    print(f"  First sample      : {np.round(X_train[0], 4)}")
    print(f"  All values [0,pi] : {bool(X_train.min() >= 0 and X_train.max() <= np.pi)}")
    print(f"  Unique labels     : {np.unique(y_train)}")
    print(f"\n  preprocess.py passed all checks")