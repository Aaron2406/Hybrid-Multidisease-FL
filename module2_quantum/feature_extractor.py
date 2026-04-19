"""
Module 2 - Quantum Feature Encoding
feature_extractor.py  (v2 - Multi-Disease)
---------------------------------------------------------------------
Flow:
  {disease}_8features.csv
        -> preprocess.py
        -> quantum_circuit.py
        -> feature_extractor.py  (this file)
        -> quantum_{disease}_train.csv   (for Module 3)
        -> quantum_{disease}_test.csv    (for Module 3)
        -> quantum_{disease}_weights.npy (saved weights)
"""

import os
import numpy as np
import pandas as pd
from quantum_circuit import run_quantum_encoding, N_QUBITS, N_LAYERS

# ── Fixed data folder path ────────────────────────────────────────────────────
DATA_FOLDER = r"C:\Users\DELL\Hybrid-Multidisease-FL\data"
os.makedirs(DATA_FOLDER, exist_ok=True)


# ── Helper: build disease-specific output paths ───────────────────────────────

def _get_output_paths(disease="unknown"):
    """
    Returns disease-specific output file paths.

    diabetes → quantum_diabetes_train.csv
    kidney   → quantum_kidney_train.csv
    heart    → quantum_heart_train.csv
    liver    → quantum_liver_train.csv
    """
    return {
        "train": os.path.join(DATA_FOLDER, f"quantum_{disease}_train.csv"),
        "test":  os.path.join(DATA_FOLDER, f"quantum_{disease}_test.csv"),
        "npy":   os.path.join(DATA_FOLDER, f"quantum_{disease}_weights.npy"),
    }


# ── Save functions ────────────────────────────────────────────────────────────

def save_quantum_features(Q, y, filepath):
    """Save quantum feature matrix + labels to CSV."""
    col_names = [f"Q_feature_{i}" for i in range(N_QUBITS)]
    df        = pd.DataFrame(Q, columns=col_names)

    # Use generic label column name — works for all diseases
    df["label"] = y.astype(int)

    df.to_csv(filepath, index=False)
    print(f"\n  Saved : {filepath}")
    print(f"  Shape : {df.shape}  ({df.shape[0]} samples x {df.shape[1]} columns)")
    return df


def save_weights(weights, filepath):
    """Save circuit weights as .npy for reuse."""
    np.save(filepath, weights)
    print(f"\n  Saved : {filepath}")
    print(f"  Shape : {weights.shape}")


def print_summary_report(df_train, df_test, weights,
                          y_train, y_test, disease="unknown"):
    """Print full summary of Module 2 outputs."""
    paths  = _get_output_paths(disease)
    q_cols = [c for c in df_train.columns if c.startswith("Q_feature")]

    print("\n" + "=" * 60)
    print(f"  MODULE 2 - FINAL SUMMARY REPORT  [{disease.upper()}]")
    print("=" * 60)

    train_cls, train_cnt = np.unique(y_train, return_counts=True)
    test_cls,  test_cnt  = np.unique(y_test,  return_counts=True)

    train_dist = "  ".join(
        [f"class {int(c)}={int(n)}" for c, n in zip(train_cls, train_cnt)]
    )
    test_dist = "  ".join(
        [f"class {int(c)}={int(n)}" for c, n in zip(test_cls, test_cnt)]
    )

    print(f"\n  Disease : {disease}")
    print(f"\n  Dataset sizes:")
    print(f"  Train : {len(df_train)} samples | {len(q_cols)} features | {train_dist}")
    print(f"  Test  : {len(df_test)}  samples | {len(q_cols)} features | {test_dist}")

    print(f"\n  Quantum Feature Statistics (train set):")
    print(f"  {'Feature':<15} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-' * 50}")
    for col in q_cols:
        vals = df_train[col]
        print(f"  {col:<15} {vals.min():>8.4f} {vals.max():>8.4f} "
              f"{vals.mean():>8.4f} {vals.std():>8.4f}")

    train_vals = df_train[q_cols].values
    spread     = float(train_vals.max() - train_vals.min())
    print(f"\n  Overall range  : [{float(train_vals.min()):.4f}, {float(train_vals.max()):.4f}]")
    print(f"  Overall spread : {spread:.4f}  {'(good)' if spread >= 0.5 else '(narrow)'}")

    print(f"\n  Circuit configuration:")
    print(f"    Qubits             : {N_QUBITS}")
    print(f"    Variational layers : {N_LAYERS}")
    print(f"    Total parameters   : {N_LAYERS * N_QUBITS * 2}")
    print(f"    Encoding           : angle (RY gates)")
    print(f"    Measurement        : PauliZ expectation")

    print(f"\n  Output files:")
    print(f"    {paths['train']}")
    print(f"    {paths['test']}")
    print(f"    {paths['npy']}")

    print("\n" + "=" * 60)
    print(f"  MODULE 2 COMPLETE [{disease.upper()}] - ready for Module 3")
    print("=" * 60)


# ── Main extraction function ──────────────────────────────────────────────────

def run_feature_extraction(csv_path, disease="unknown"):
    """
    Full Module 2 feature extraction pipeline.

    Parameters
    ----------
    csv_path : str
        Path to {disease}_8features.csv from Module 1 output.
    disease  : str
        Disease label: diabetes / kidney / heart / liver
        Used to name output files in data folder.

    Returns
    -------
    df_train : pd.DataFrame  quantum features + labels
    df_test  : pd.DataFrame  quantum features + labels
    weights  : np.ndarray    circuit weights shape (N_LAYERS, N_QUBITS, 2)
    """
    print("\n" + "=" * 60)
    print(f"  FEATURE EXTRACTOR - Module 2  [{disease.upper()}]")
    print("=" * 60)
    print(f"  Disease   : {disease}")
    print(f"  Input CSV : {csv_path}")

    # Get disease-specific output paths
    paths = _get_output_paths(disease)

    # Step 1 - Run quantum encoding pipeline
    Q_train, Q_test, y_train, y_test, weights = run_quantum_encoding(csv_path)

    # Step 2 - Save quantum features to disease-specific CSVs
    print("\n" + "-" * 60)
    print(f"  SAVING OUTPUTS  [{disease.upper()}]")
    print("-" * 60)

    df_train = save_quantum_features(Q_train, y_train, paths["train"])
    df_test  = save_quantum_features(Q_test,  y_test,  paths["test"])

    # Step 3 - Save weights to disease-specific .npy
    save_weights(weights, paths["npy"])

    # Step 4 - Print summary
    print_summary_report(df_train, df_test, weights,
                          y_train, y_test, disease=disease)

    return df_train, df_test, weights


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Usage: python feature_extractor.py diabetes
    disease  = sys.argv[1] if len(sys.argv) > 1 else "diabetes"
    csv_path = os.path.join(DATA_FOLDER, f"{disease}_8features.csv")

    print(f"\n  Running standalone test for disease: {disease}")
    print(f"  Input: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"\n  ERROR: {csv_path} not found.")
        print(f"  Run Module 1 first: python xgboost3.py <dataset.csv> --disease {disease}")
        sys.exit(1)

    df_train, df_test, weights = run_feature_extraction(
        csv_path=csv_path,
        disease=disease,
    )

    paths = _get_output_paths(disease)

    print("\n  Verifying output files ...")
    for name, path in paths.items():
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        status = f"OK  ({size:,} bytes)" if exists else "NOT FOUND"
        print(f"  {name:<8} {path:<60} {status}")

    print(f"\n  feature_extractor.py [{disease}] passed all checks")
    print(f"  Next step -> Module 3: python stream_m3.py {disease}")