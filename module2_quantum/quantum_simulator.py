"""
Module 2 - Quantum Feature Encoding
quantum_simulator.py  (fixed - runtime feature reading)
---------------------------------------------------------------------
ROOT CAUSE OF EMPTY Raw features bug:
  from preprocess import SELECTED_FEATURES
  copies the empty list [] at import time, BEFORE
  run_preprocessing() / load_data() fills it in.

FIX:
  import preprocess as _pre  (import the module, not the variable)
  then read preprocess.SELECTED_FEATURES at runtime inside functions
  so it always reflects the current populated value.
"""

import os
import json
import numpy as np
import pandas as pd
import pennylane as qml

# Import the MODULE, not the variables — this is the key fix
import preprocess as _pre
from preprocess import run_preprocessing

N_LAYERS    = 3
REPORT_FILE = "simulation_report.txt"
BLOCH_FILE  = "bloch_states.json"


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME ACCESSORS  (read from module at call time, not import time)
# ─────────────────────────────────────────────────────────────────────────────

def get_features():
    """Always returns the current SELECTED_FEATURES list."""
    return _pre.SELECTED_FEATURES


def get_n_qubits():
    """Always returns current N_QUBITS value."""
    return _pre.N_QUBITS


# ─────────────────────────────────────────────────────────────────────────────
# JSON SANITIZER
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_for_json(obj):
    """
    Recursively convert numpy types to plain Python so Flask
    jsonify never raises TypeError.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def qubit_state_label(z):
    z = float(z)
    if z > 0.9:   return "|0> (no disease signal)"
    if z < -0.9:  return "|1> (strong disease signal)"
    if z > 0.3:   return "mostly |0> (weak signal)"
    if z < -0.3:  return "mostly |1> (moderate signal)"
    return "superposition (uncertain)"


def make_z_step(z_vals):
    """Build the two keys every step must always have."""
    clean = [round(float(v), 4) for v in z_vals]
    return {
        "qubit_z_values": clean,
        "qubit_labels":   [qubit_state_label(v) for v in clean],
    }


def run_qnode(dev, circuit_fn):
    """Run a QNode and return plain Python floats immediately."""
    node   = qml.QNode(circuit_fn, dev)
    result = node()
    return [float(v) for v in result]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER BY LAYER TRACE
# ─────────────────────────────────────────────────────────────────────────────

def simulate_layer_by_layer(features, weights):
    """
    Trace one patient sample through every circuit layer.
    Reads SELECTED_FEATURES at runtime via get_features().
    """
    # Read at runtime — populated by run_preprocessing() before this runs
    FEATURES = get_features()
    N        = get_n_qubits()
    dev      = qml.device("default.qubit", wires=N)
    feats    = [float(f) for f in features]
    steps    = []

    print(f"  Simulating with features: {FEATURES}")
    print(f"  N_QUBITS: {N}")

    # ── Step 0: Initial state ─────────────────────────────────────────────────
    steps.append({
        "step":        0,
        "layer_name":  "Initial state",
        "gate":        "none",
        "description": (
            "All qubits start at |0> (north pole of Bloch sphere). "
            "No patient data encoded yet."
        ),
        **make_z_step([1.0] * N),
    })

    # ── Step 1: Hadamard ──────────────────────────────────────────────────────
    def h_circuit():
        for i in range(N):
            qml.Hadamard(wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    z = run_qnode(dev, h_circuit)
    steps.append({
        "step":        1,
        "layer_name":  "Hadamard layer",
        "gate":        "H",
        "description": (
            "Hadamard gate pushes every qubit to the equator "
            "of the Bloch sphere. Maximum superposition before encoding."
        ),
        **make_z_step(z),
    })

    # ── Step 2: Angle encoding ────────────────────────────────────────────────
    def enc_circuit():
        for i in range(N):
            qml.Hadamard(wires=i)
        for i in range(N):
            qml.RY(feats[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    z = run_qnode(dev, enc_circuit)
    z_clean = [round(v, 4) for v in z]
    steps.append({
        "step":        2,
        "layer_name":  "Angle encoding layer",
        "gate":        "RY",
        "description": (
            "RY(theta) rotates each qubit by its feature value. "
            "High value pushes qubit toward |1>. "
            "Low value keeps qubit near |0>."
        ),
        "feature_to_qubit_mapping": {
            FEATURES[i]: {
                "angle_radians":  round(feats[i], 4),
                "z_value":        z_clean[i],
                "interpretation": qubit_state_label(z_clean[i]),
            }
            for i in range(N)
        },
        **make_z_step(z),
    })

    # ── Step 3: Entanglement ──────────────────────────────────────────────────
    def cnot_circuit():
        for i in range(N):
            qml.Hadamard(wires=i)
        for i in range(N):
            qml.RY(feats[i], wires=i)
        for i in range(N - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    z = run_qnode(dev, cnot_circuit)
    steps.append({
        "step":        3,
        "layer_name":  "Entanglement layer (CNOT chain)",
        "gate":        "CNOT",
        "description": (
            "CNOT gates link qubits in a chain. "
            "Creates quantum correlations between neighbouring features."
        ),
        "entanglement_pairs": [
            f"q{i} -> q{i+1}  ({FEATURES[i]} + {FEATURES[i+1]})"
            for i in range(N - 1)
        ],
        **make_z_step(z),
    })

    # ── Steps 4+: Variational layers ──────────────────────────────────────────
    for layer in range(N_LAYERS):

        def make_var(ll):
            def var_circuit():
                for i in range(N):
                    qml.Hadamard(wires=i)
                for i in range(N):
                    qml.RY(feats[i], wires=i)
                for i in range(N - 1):
                    qml.CNOT(wires=[i, i + 1])
                for l2 in range(ll + 1):
                    for i in range(N - 1):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(N):
                        qml.RZ(float(weights[l2][i][0]), wires=i)
                        qml.RY(float(weights[l2][i][1]), wires=i)
                return [qml.expval(qml.PauliZ(i)) for i in range(N)]
            return var_circuit

        z = run_qnode(dev, make_var(layer))
        steps.append({
            "step":        4 + layer,
            "layer_name":  f"Variational layer {layer + 1} of {N_LAYERS}",
            "gate":        "RZ + RY",
            "description": (
                f"Variational layer {layer + 1}: trainable RZ and RY rotations. "
                "Learnable weights transform the quantum state to separate "
                "diabetic from non-diabetic patients."
            ),
            "weight_sample": {
                f"qubit_{i}": {
                    "RZ": round(float(weights[layer][i][0]), 4),
                    "RY": round(float(weights[layer][i][1]), 4),
                }
                for i in range(N)
            },
            **make_z_step(z),
        })

    # ── Final measurement step ────────────────────────────────────────────────
    z_clean = [round(float(v), 4) for v in z]
    steps.append({
        "step":        len(steps),
        "layer_name":  "Measurement (PauliZ)",
        "gate":        "PauliZ",
        "description": (
            "PauliZ measured on all qubits. "
            "<Z>=+1 means |0> (low signal). "
            "<Z>=-1 means |1> (high signal). "
            "These values form the quantum feature vector for Module 3."
        ),
        "final_quantum_vector": z_clean,
        "qubit_to_feature": {
            f"q{i}": {
                "feature": FEATURES[i],
                "q_value": z_clean[i],
                "meaning": qubit_state_label(z_clean[i]),
            }
            for i in range(N)
        },
        **make_z_step(z),
    })

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# CLASSICAL vs QUANTUM COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_classical_quantum(features_raw, features_normalized, quantum_vector):
    FEATURES = get_features()
    N        = get_n_qubits()
    return [
        {
            "feature":              FEATURES[i],
            "classical_raw":        round(float(features_raw[i]), 4),
            "classical_normalized": round(float(features_normalized[i]), 4),
            "quantum_encoded":      round(float(quantum_vector[i]), 4),
            "qubit":                f"q{i}",
        }
        for i in range(N)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_simulation_report(sim, filepath=REPORT_FILE):
    FEATURES = get_features()
    N        = get_n_qubits()
    lines    = [
        "=" * 65,
        "  QUANTUM CIRCUIT SIMULATION REPORT",
        "  Module 2 - Quantum-Assisted Feature Representation",
        "=" * 65,
        "",
        f"  Features    : {FEATURES}",
        f"  Qubits      : {N}",
        f"  Layers      : {N_LAYERS} variational + Hadamard + encoding",
        "",
        "  FEATURE TO QUBIT MAPPING",
        "  " + "-" * 50,
    ]
    for i, feat in enumerate(FEATURES):
        lines.append(f"  qubit {i}  <->  {feat}")

    lines += ["", "  LAYER BY LAYER TRACE", "  " + "-" * 50]
    for step in sim["layer_trace"]:
        lines.append(f"\n  Step {step['step']}: {step['layer_name']}")
        lines.append(f"  Gate : {step['gate']}")
        lines.append(f"  Z    : {step['qubit_z_values']}")

    lines += ["", "  CLASSICAL vs QUANTUM", "  " + "-" * 50]
    for row in sim["classical_vs_quantum"]:
        lines.append(
            f"  {row['feature']:<25} "
            f"raw={row['classical_raw']:>6}  "
            f"norm={row['classical_normalized']:>7.4f}  "
            f"quantum={row['quantum_encoded']:>8.4f}"
        )
    lines += ["", "=" * 65, "  END OF REPORT", "=" * 65]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(csv_path, weights):
    """
    Full simulation for one patient sample.
    Guaranteed JSON-serializable output.

    Parameters
    ----------
    csv_path : str
    weights  : np.ndarray  shape (N_LAYERS, N_QUBITS, 2)

    Returns
    -------
    dict — full simulation result
    """
    print("\n" + "=" * 60)
    print("  QUANTUM SIMULATOR")
    print("=" * 60)

    # run_preprocessing populates _pre.SELECTED_FEATURES and _pre.N_QUBITS
    X_train, _, y_train, _, scaler = run_preprocessing(csv_path)

    # NOW read features — they are populated after run_preprocessing()
    FEATURES = get_features()
    N        = get_n_qubits()

    features_normalized = np.clip(X_train[0], 0, np.pi)

    # Get raw values from CSV using the NOW-populated FEATURES list
    df           = pd.read_csv(csv_path)
    # ── Encode text values before extracting raw features ──
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({
                'yes': 1, 'no': 0,
                'true': 1, 'false': 0,
                'male': 1, 'female': 0,
                'positive': 1, 'negative': 0,
            })
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    features_raw = df[FEATURES].iloc[0].values

    print(f"  Features         : {FEATURES}")
    print(f"  Raw features     : {features_raw}")
    print(f"  Normalized [0,pi]: {np.round(features_normalized, 4)}")
    print(f"  Running layer-by-layer trace ...")

    layer_trace    = simulate_layer_by_layer(features_normalized, weights)
    quantum_vector = layer_trace[-1]["qubit_z_values"]

    comparison = compare_classical_quantum(
        features_raw, features_normalized, quantum_vector
    )

    bloch = [
        {
            "qubit":   i,
            "feature": FEATURES[i],
            "z":       layer_trace[2]["qubit_z_values"][i],
            "label":   layer_trace[2]["qubit_labels"][i],
        }
        for i in range(N)
    ]

    sim = {
        "sample_index":         0,
        "features_used":        FEATURES,
        "n_qubits":             N,
        "features_raw":         [float(v) for v in features_raw],
        "features_normalized":  [float(v) for v in features_normalized],
        "quantum_vector":       quantum_vector,
        "layer_trace":          layer_trace,
        "classical_vs_quantum": comparison,
        "bloch_after_encoding": bloch,
        "total_steps":          len(layer_trace),
        "summary":              (
            f"Traced 1 patient sample through {len(layer_trace)} steps. "
            f"{N} features: {FEATURES}"
        ),
    }

    # Final sanitize — converts any remaining numpy types
    sim = sanitize_for_json(sim)

    save_simulation_report(sim)
    with open(BLOCH_FILE, "w", encoding="utf-8") as f:
        json.dump(bloch, f, indent=2)

    print(f"  Traced {len(layer_trace)} steps  OK")
    print(f"  Quantum vector : {quantum_vector}")
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from quantum_circuit import initialize_weights

    weights = initialize_weights()
    sim     = run_simulation("diabetes_8features_sample.csv", weights)

    print(f"\n  Verifying steps...")
    for step in sim["layer_trace"]:
        assert "qubit_z_values" in step, f"Step {step['step']} missing qubit_z_values"
        assert "qubit_labels"   in step, f"Step {step['step']} missing qubit_labels"
        for v in step["qubit_z_values"]:
            assert isinstance(v, float), f"Step {step['step']}: {v} is {type(v)}"

    print(f"  All {len(sim['layer_trace'])} steps OK")
    print(f"  Features used: {sim['features_used']}")

    import json
    try:
        json.dumps(sim)
        print(f"  JSON serialization OK")
    except TypeError as e:
        print(f"  JSON FAILED: {e}")

    print(f"\n  quantum_simulator.py passed all checks")