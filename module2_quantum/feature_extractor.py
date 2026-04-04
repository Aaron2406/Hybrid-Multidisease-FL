"""
Module 2 - Quantum Feature Encoding
feature_extractor.py
---------------------------------------------------------------------
Final step of Module 2.

Flow:
  diabetes_8features_sample.csv
        -> preprocess.py
        -> quantum_circuit.py
        -> feature_extractor.py  (this file)
        -> quantum_train_features.csv   (for Module 3)
        -> quantum_test_features.csv    (for Module 3)
        -> quantum_weights.npy          (saved weights)
"""

import os
import numpy as np
import pandas as pd
from quantum_circuit import run_quantum_encoding, N_QUBITS, N_LAYERS

OUT_TRAIN_CSV   = "quantum_train_features.csv"
OUT_TEST_CSV    = "quantum_test_features.csv"
OUT_WEIGHTS_NPY = "quantum_weights.npy"


def save_quantum_features(Q, y, filepath):
    """Save quantum feature matrix + labels to CSV."""
    col_names = [f"Q_feature_{i}" for i in range(N_QUBITS)]
    df = pd.DataFrame(Q, columns=col_names)
    df["Diabetes"] = y.astype(int)
    df.to_csv(filepath, index=False)
    print(f"\n  Saved : {filepath}")
    print(f"  Shape : {df.shape}  ({df.shape[0]} samples x {df.shape[1]} columns)")
    return df


def save_weights(weights, filepath):
    """Save circuit weights as .npy for reuse."""
    np.save(filepath, weights)
    print(f"\n  Saved : {filepath}")
    print(f"  Shape : {weights.shape}")
    print(f"  Reload: weights = np.load('{filepath}')")


def print_summary_report(df_train, df_test, weights, y_train, y_test):
    """Print full summary of Module 2 outputs."""

    q_cols = [c for c in df_train.columns if c.startswith("Q_feature")]

    print("\n" + "=" * 60)
    print("  MODULE 2 - FINAL SUMMARY REPORT")
    print("=" * 60)

    # Dataset sizes - build label distribution as plain strings
    train_cls, train_cnt = np.unique(y_train, return_counts=True)
    test_cls,  test_cnt  = np.unique(y_test,  return_counts=True)
    train_dist = "  ".join(
        [f"class {int(c)}={int(n)}" for c, n in zip(train_cls, train_cnt)]
    )
    test_dist = "  ".join(
        [f"class {int(c)}={int(n)}" for c, n in zip(test_cls, test_cnt)]
    )

    print(f"\n  Dataset sizes:")
    print(f"  Train : {len(df_train)} samples | {len(q_cols)} features | {train_dist}")
    print(f"  Test  : {len(df_test)} samples | {len(q_cols)} features | {test_dist}")

    # Per-feature statistics
    print(f"\n  Quantum Feature Statistics (train set):")
    print(f"  {'Feature':<15} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-' * 50}")
    for col in q_cols:
        vals = df_train[col]
        print(f"  {col:<15} {vals.min():>8.4f} {vals.max():>8.4f} "
              f"{vals.mean():>8.4f} {vals.std():>8.4f}")

    # Overall range
    train_vals = df_train[q_cols].values
    spread = float(train_vals.max() - train_vals.min())
    print(f"\n  Overall range  : [{float(train_vals.min()):.4f},  {float(train_vals.max()):.4f}]")
    print(f"  Overall spread : {spread:.4f}  {'(good)' if spread >= 0.5 else '(narrow)'}")

    # Circuit info
    print(f"\n  Circuit configuration:")
    print(f"    Qubits             : {N_QUBITS}")
    print(f"    Variational layers : {N_LAYERS}")
    print(f"    Entanglement       : linear chain")
    print(f"    Total parameters   : {N_LAYERS * N_QUBITS * 2}")
    print(f"    Encoding           : angle (RY gates)")
    print(f"    Measurement        : PauliZ expectation")

    # Output files
    print(f"\n  Output files:")
    print(f"    {OUT_TRAIN_CSV}  -> Module 3 (train)")
    print(f"    {OUT_TEST_CSV}  -> Module 3 (test)")
    print(f"    {OUT_WEIGHTS_NPY}  -> reuse weights")

    print("\n" + "=" * 60)
    print("  MODULE 2 COMPLETE - ready for Module 3")
    print("=" * 60)


def run_feature_extraction(csv_path="diabetes_8features_sample.csv"):
    """
    Full Module 2 feature extraction pipeline.

    Parameters
    ----------
    csv_path : str
        Use 'diabetes_8features_sample.csv' for testing (fast).
        Use 'diabetes_8features.csv' for production.

    Returns
    -------
    df_train : pd.DataFrame  quantum features + labels
    df_test  : pd.DataFrame  quantum features + labels
    weights  : np.ndarray    circuit weights shape (N_LAYERS, N_QUBITS, 2)
    """
    print("\n" + "=" * 60)
    print("  FEATURE EXTRACTOR - Module 2 Final Step")
    print("=" * 60)
    print(f"  Input CSV : {csv_path}")

    # Step 1 - Run full quantum encoding pipeline
    Q_train, Q_test, y_train, y_test, weights = run_quantum_encoding(csv_path)

    # Step 2 - Save quantum features as CSVs
    print("\n" + "-" * 60)
    print("  SAVING OUTPUTS")
    print("-" * 60)

    df_train = save_quantum_features(Q_train, y_train, OUT_TRAIN_CSV)
    df_test  = save_quantum_features(Q_test,  y_test,  OUT_TEST_CSV)

    # Step 3 - Save weights
    save_weights(weights, OUT_WEIGHTS_NPY)

    # Step 4 - Print summary
    print_summary_report(df_train, df_test, weights, y_train, y_test)

    return df_train, df_test, weights


if __name__ == "__main__":

    df_train, df_test, weights = run_feature_extraction(
        csv_path="diabetes_8features_sample.csv"
    )

    # Verify output files were created
    print("\n  Verifying output files ...")
    for fname in [OUT_TRAIN_CSV, OUT_TEST_CSV, OUT_WEIGHTS_NPY]:
        exists = os.path.exists(fname)
        size   = os.path.getsize(fname) if exists else 0
        status = f"OK  ({size:,} bytes)" if exists else "NOT FOUND"
        print(f"  {fname:<42} {status}")

    # Quick data checks
    print("\n  Quick data checks:")
    q_cols = [c for c in df_train.columns if c.startswith("Q_")]
    assert len(q_cols) == N_QUBITS,             "Wrong number of Q features"
    assert "Diabetes" in df_train.columns,      "Label column missing"
    assert df_train["Diabetes"].nunique() == 2,  "Should have 2 classes"
    reloaded = np.load(OUT_WEIGHTS_NPY)
    assert reloaded.shape == weights.shape,      "Weights shape mismatch"

    print(f"  {N_QUBITS} quantum features present")
    print(f"  Diabetes label column present")
    print(f"  Both classes (0 and 1) in train set")
    print(f"  Weights reload verified  shape={reloaded.shape}")
    print(f"\n  feature_extractor.py passed all checks")
    print(f"\n  Next step -> Module 3")
    print(f"  Files ready: quantum_train_features.csv + quantum_test_features.csv")