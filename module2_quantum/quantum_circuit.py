"""
=============================================================================
  Module 2 — Quantum Feature Encoding
  quantum_circuit.py  (v3 — spread fix)
  -----------------------------------------------------------------------------
  Three targeted fixes over v2 to get spread above 0.5:

  FIX 1 — Entanglement: circular → linear
    Circular CNOT (ring) with 8 qubits creates symmetric cancellation.
    All 8 qubits influence each other equally → measurements average out.
    Linear CNOT (chain) breaks the symmetry → each qubit has a unique
    neighbourhood → PauliZ measurements spread across [-1, +1].

  FIX 2 — Layers: 2 → 3
    More variational layers = more expressive circuit.
    3 layers × 8 qubits × 2 params = 48 total parameters.

  FIX 3 — Weight range: [-π/2, π/2] → [-π/4, π/4]
    Smaller initial rotations → less mutual cancellation between
    RZ and RY gates → expectation values differentiate better.

  Circuit architecture (per sample):
  ┌──────────────────────────────────────────────────────┐
  │  Layer 0 : Hadamard        push qubits to equator    │
  │  Layer 1 : Angle Encoding  RY(θᵢ) per feature       │
  │  Layer 2 : Linear CNOT     q0→q1→q2→...→q7 chain    │
  │  Layer 3 : Variational     RZ + RY  (trainable)      │
  │  Layer 4 : Linear CNOT     chain again               │
  │  Layer 5 : Variational     RZ + RY  (trainable)      │
  │  Layer 6 : Linear CNOT     chain again               │
  │  Layer 7 : Variational     RZ + RY  (trainable)      │
  │  Layer 8 : Measurement     PauliZ on all 8 qubits    │
  └──────────────────────────────────────────────────────┘

  Input  : 8 features ∈ [0, π]
  Output : 8 expectation values ∈ [−1, +1]
=============================================================================
"""

import numpy as np
import pennylane as qml
from preprocess import run_preprocessing, N_QUBITS

# ── Circuit configuration ─────────────────────────────────────────────────────
N_LAYERS     = 3                        # increased from 2 → 3
WEIGHT_LOW   = -np.pi / 4              # tighter range → less cancellation
WEIGHT_HIGH  =  np.pi / 4
RANDOM_STATE = 42

# ── Device ────────────────────────────────────────────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

@qml.qnode(dev)
def quantum_circuit(features: np.ndarray, weights: np.ndarray) -> list:
    """
    8-qubit circuit — v3.

    Key change from v2: LINEAR entanglement replaces CIRCULAR.

    Circular (v2):  0→1→2→3→4→5→6→7→0  (ring — symmetric cancellation)
    Linear   (v3):  0→1→2→3→4→5→6→7    (chain — asymmetric, unique outputs)

    Parameters
    ----------
    features : np.ndarray  shape (8,)   values in [0, π]
    weights  : np.ndarray  shape (N_LAYERS, N_QUBITS, 2)

    Returns
    -------
    list of 8 floats in [−1, +1]
    """

    # Layer 0 — Hadamard: move all qubits to equator of Bloch sphere
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)

    # Layer 1 — Angle Encoding: each feature rotates its qubit
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)

    # Variational + Linear Entanglement repeated N_LAYERS times
    for layer in range(N_LAYERS):

        # Linear CNOT chain: 0→1, 1→2, 2→3, ..., 6→7
        # NO wrap-around (unlike circular) — breaks symmetry
        # Each qubit only influences its right neighbour
        # → qubit 0 and qubit 7 are NOT directly connected
        # → asymmetric influence → diverse PauliZ readouts
        for i in range(N_QUBITS - 1):                # 0,1,2,3,4,5,6 (not 7)
            qml.CNOT(wires=[i, i + 1])

        # Variational rotations — trainable weights
        for i in range(N_QUBITS):
            qml.RZ(weights[layer][i][0], wires=i)    # phase shift
            qml.RY(weights[layer][i][1], wires=i)    # amplitude shift

    # Measurement — PauliZ on all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_weights(n_layers: int = N_LAYERS,
                       n_qubits: int = N_QUBITS) -> np.ndarray:
    """
    Initialize in [-π/4, +π/4].

    Comparison of ranges:
      [0,   2π]   → std=1.81  → large rotations → heavy cancellation → near 0
      [-π/2, π/2] → std=0.91  → medium rotations → some cancellation
      [-π/4, π/4] → std=0.45  → small rotations → minimal cancellation ✓

    Shape  : (n_layers, n_qubits, 2)
    Total  : 3 × 8 × 2 = 48 parameters
    """
    np.random.seed(RANDOM_STATE)
    weights = np.random.uniform(
        low  = WEIGHT_LOW,
        high = WEIGHT_HIGH,
        size = (n_layers, n_qubits, 2)
    )
    print(f"\n  Weights initialized : shape={weights.shape}")
    print(f"  Range               : [-π/4, +π/4]  =  "
          f"[{WEIGHT_LOW:.4f}, {WEIGHT_HIGH:.4f}]")
    print(f"  Total parameters    : {n_layers} layers × {n_qubits} qubits "
          f"× 2 = {n_layers * n_qubits * 2}")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_quantum_features(X: np.ndarray,
                             weights: np.ndarray,
                             label: str = "data") -> np.ndarray:
    """
    Run the circuit on every row of X.

    Parameters
    ----------
    X       : np.ndarray  shape (N, 8)
    weights : np.ndarray  shape (N_LAYERS, N_QUBITS, 2)
    label   : str

    Returns
    -------
    np.ndarray  shape (N, 8)  values in [−1, +1]
    """
    X_clamped = np.clip(X, 0, np.pi)
    n_samples  = X_clamped.shape[0]
    q_features = np.zeros((n_samples, N_QUBITS))

    print(f"\n  Extracting quantum features ({label}) ...")
    print(f"  Samples      : {n_samples}")
    print(f"  Qubits       : {N_QUBITS}")
    print(f"  Entanglement : linear chain  (v3 fix)")
    print(f"  Layers       : Hadamard + encoding + {N_LAYERS} variational")

    for idx in range(n_samples):
        q_features[idx] = quantum_circuit(X_clamped[idx], weights)
        if (idx + 1) % 20 == 0 or (idx + 1) == n_samples:
            pct = (idx + 1) / n_samples * 100
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"  [{bar}] {idx+1}/{n_samples}  ({pct:.0f}%)", end="\r")

    print()
    spread = q_features.max() - q_features.min()
    print(f"  Done ✓  shape  : {q_features.shape}")
    print(f"  Range          : [{q_features.min():.4f},  {q_features.max():.4f}]")

    if spread >= 0.8:
        print(f"  ✓ Excellent spread  (spread={spread:.4f})")
    elif spread >= 0.5:
        print(f"  ✓ Good spread  (spread={spread:.4f})")
    else:
        print(f"  ⚠ Still narrow  (spread={spread:.4f})")

    return q_features


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

def print_circuit_diagram(weights: np.ndarray) -> None:
    print("\n" + "=" * 60)
    print("  QUANTUM CIRCUIT DIAGRAM  (v3 — linear entanglement)")
    print("=" * 60)
    dummy = np.zeros(N_QUBITS)
    print(qml.draw(quantum_circuit)(dummy, weights))
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_quantum_encoding(csv_path: str = "diabetes_8features_sample.csv"):
    """
    Full Module 2 pipeline.

    Returns
    -------
    Q_train  : np.ndarray  (N_train, 8)
    Q_test   : np.ndarray  (N_test,  8)
    y_train  : np.ndarray
    y_test   : np.ndarray
    weights  : np.ndarray  (N_LAYERS, N_QUBITS, 2)
    """
    print("\n" + "=" * 60)
    print("  MODULE 2 — QUANTUM FEATURE ENCODING  (v3)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler = run_preprocessing(csv_path)
    weights = initialize_weights()
    print_circuit_diagram(weights)

    print("\n" + "─" * 60)
    print("  QUANTUM FEATURE EXTRACTION")
    print("─" * 60)

    Q_train = extract_quantum_features(X_train, weights, label="train")
    Q_test  = extract_quantum_features(X_test,  weights, label="test")

    print("\n" + "=" * 60)
    print("  MODULE 2 COMPLETE")
    print("=" * 60)
    print(f"  Q_train : {Q_train.shape}  "
          f"range=[{Q_train.min():.4f}, {Q_train.max():.4f}]")
    print(f"  Q_test  : {Q_test.shape}   "
          f"range=[{Q_test.min():.4f}, {Q_test.max():.4f}]")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")
    print(f"\n  First quantum feature vector (train[0]):")
    print(f"  {np.round(Q_train[0], 4)}")
    print(f"\n  Ready to pass to Module 3 (classifier) ✓")
    print("=" * 60)

    return Q_train, Q_test, y_train, y_test, weights


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n  Running quantum circuit v3 (linear entanglement fix) ...")

    Q_train, Q_test, y_train, y_test, weights = run_quantum_encoding(
        csv_path="diabetes_8features_sample.csv"
    )

    print("\n  Sanity checks:")
    assert Q_train.shape[1] == N_QUBITS,    "❌ Wrong number of quantum features"
    assert Q_train.min()    >= -1.01,        "❌ Values below -1"
    assert Q_train.max()    <=  1.01,        "❌ Values above +1"
    assert len(y_train)     == len(Q_train), "❌ Label/feature count mismatch"

    spread = Q_train.max() - Q_train.min()
    status = "✓ good" if spread >= 0.5 else "⚠ still narrow"
    print(f"  ✓ Output shape correct")
    print(f"  ✓ Values within [-1, +1]")
    print(f"  ✓ Labels match feature count")
    print(f"  Spread = {spread:.4f}  ({status})")
    print(f"\n  quantum_circuit.py v3 passed all checks ✓")