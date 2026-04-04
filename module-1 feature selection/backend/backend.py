"""
backend.py  (updated - Module 1 + Module 2)
---------------------------------------------------------------------
Endpoints:
  GET  /status        health check
  POST /run-model     Module 1 - XGBoost feature selection
  POST /run-quantum   Module 2 - Quantum feature encoding
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import sys

from xgboost3 import run_pipeline

# Add module2_quantum folder to Python path
MODULE2_PATH = os.path.normpath(
    os.path.join(
        r"C:\Users\DELL\OneDrive\Desktop\Hybrid-Multidisease-FL\Hybrid-Multidisease-FL",
        "module2_quantum"
    )
)
if os.path.exists(MODULE2_PATH) and MODULE2_PATH not in sys.path:
    sys.path.insert(0, MODULE2_PATH)

app = Flask(__name__)
CORS(app)

# Shared state - stores the 8-feature CSV path after /run-model runs
_quantum_csv_path = None


def _save_upload(file):
    suffix = os.path.splitext(file.filename)[1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    file.save(tmp_path)
    tmp.close()
    return tmp_path


def _cleanup(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


# ── GET /status ───────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":            "running",
        "module1":           "ready",
        "module2":           "ready" if os.path.exists(MODULE2_PATH) else "path not found",
        "module2_path":      MODULE2_PATH,
        "quantum_csv_ready": bool(_quantum_csv_path and os.path.exists(_quantum_csv_path)),
        "endpoints":         ["/status", "/run-model", "/run-quantum"],
    })


# ── POST /run-model  (Module 1) ───────────────────────────────────────────────

@app.route("/run-model", methods=["POST"])
def run_model():
    """
    Module 1 - XGBoost feature selection pipeline.
    Unchanged behaviour + exports 8features CSV for Module 2.
    """
    global _quantum_csv_path

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    tmp_path = None
    try:
        tmp_path = _save_upload(file)
        results  = run_pipeline(tmp_path)

        # Export 8-feature CSV so /run-quantum can use it automatically
        if os.path.exists(MODULE2_PATH):
            import pandas as pd
            selected = [f["feature"] for f in results.get("top_features", [])]
            target   = "Diabetes"
            df       = pd.read_csv(tmp_path)
            cols     = [c for c in selected if c in df.columns]
            if cols and target in df.columns:
                out = os.path.join(MODULE2_PATH, "diabetes_8features.csv")
                df[cols + [target]].to_csv(out, index=False)
                _quantum_csv_path        = out
                results["quantum_ready"] = True
            else:
                results["quantum_ready"] = False
        else:
            results["quantum_ready"] = False

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        _cleanup(tmp_path)

    return jsonify(results)


# ── POST /run-quantum  (Module 2) ─────────────────────────────────────────────

@app.route("/run-quantum", methods=["POST"])
def run_quantum():
    """
    Module 2 - Quantum feature encoding.

    Call after /run-model (no file needed - uses saved CSV automatically).
    Or upload a fresh CSV directly.

    Returns:
      circuit_info       - qubits, layers, gates, feature mapping
      quantum_features   - sample encoded vectors
      spread             - value distribution across [-1, +1]
      simulation         - layer-by-layer trace for reviewer/guide
      output_files       - list of files saved
      pipeline_status    - M1 + M2 status
    """
    global _quantum_csv_path

    tmp_path = None
    csv_path = None

    try:
        # Decide which CSV to use
        if "file" in request.files and request.files["file"].filename != "":
            tmp_path = _save_upload(request.files["file"])
            csv_path = tmp_path
        elif _quantum_csv_path and os.path.exists(_quantum_csv_path):
            csv_path = _quantum_csv_path
        else:
            sample = os.path.join(MODULE2_PATH, "diabetes_8features_sample.csv")
            if os.path.exists(sample):
                csv_path = sample
            else:
                return jsonify({
                    "error": "No CSV available. Call /run-model first or upload a file."
                }), 400

        # Import Module 2
        try:
            from feature_extractor import run_feature_extraction
            from quantum_simulator import run_simulation
        except ImportError as ie:
            return jsonify({
                "error": f"Module 2 import failed: {str(ie)}. "
                         f"Check module2_quantum/ exists at: {MODULE2_PATH}"
            }), 500

        # Run Module 2
        df_train, df_test, weights = run_feature_extraction(csv_path)
        simulation                 = run_simulation(csv_path, weights)

        # Build response
        q_cols     = [c for c in df_train.columns if c.startswith("Q_feature")]
        vectors    = df_train[q_cols].head(5).round(4).values.tolist()
        labels     = df_train["Diabetes"].head(5).tolist()
        train_vals = df_train[q_cols].values
        test_vals  = df_test[q_cols].values

        results = {
            "quantum_features_sample": [
                {"label": int(labels[i]), "vector": vectors[i]}
                for i in range(len(vectors))
            ],
            "q_train_shape": list(df_train.shape),
            "q_test_shape":  list(df_test.shape),
            "spread": {
                "train": round(float(train_vals.max() - train_vals.min()), 4),
                "test":  round(float(test_vals.max()  - test_vals.min()),  4),
            },
            "circuit_info": {
                "n_qubits":     8,
                "n_layers":     3,
                "entanglement": "linear chain",
                "encoding":     "angle encoding (RY gates)",
                "measurement":  "PauliZ expectation value",
                "total_params": 48,
                "weight_range": "[-pi/4, +pi/4]",
                "gates_used":   ["Hadamard", "RY", "CNOT", "RZ", "PauliZ"],
                "feature_map": {
                    "qubit_0": "HighBP",
                    "qubit_1": "GenHlth",
                    "qubit_2": "HighChol",
                    "qubit_3": "BMI",
                    "qubit_4": "DifficultyWalk",
                    "qubit_5": "Age",
                    "qubit_6": "PhysHlth",
                    "qubit_7": "HeartDiseaseorAttack",
                },
            },
            "simulation": simulation,
            "output_files": [
                "module2_quantum/quantum_train_features.csv",
                "module2_quantum/quantum_test_features.csv",
                "module2_quantum/quantum_weights.npy",
                "module2_quantum/simulation_report.txt",
                "module2_quantum/bloch_states.json",
            ],
            "pipeline_status": {
                "module1": "complete",
                "module2": "complete",
                "csv_used": os.path.basename(csv_path),
            },
        }

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        _cleanup(tmp_path)

    return jsonify(results)


if __name__ == "__main__":
    print("\n  Flask server starting ...")
    print(f"  Module 2 path : {MODULE2_PATH}")
    print(f"  Endpoints     : /status  /run-model  /run-quantum")
    print(f"  URL           : http://localhost:5000\n")
    app.run(debug=True, port=5000)