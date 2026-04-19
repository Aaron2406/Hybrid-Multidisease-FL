# backend.py - Multi-Disease Integration (Module 1 + 2 + 3)
# Project root: C:/Users/DELL/Hybrid-Multidisease-FL
# Data folder:  C:/Users/DELL/Hybrid-Multidisease-FL/data
#
# Endpoints:
#   GET  /status
#   GET  /pipeline-status
#   POST /upload-disease/<disease>
#   POST /run-model/<disease>
#   POST /run-quantum/<disease>
#   POST /run-m3-stream/<disease>
#   GET  /metabolic-report

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import tempfile, os, sys, json, queue, threading

from xgboost3 import run_pipeline

# ── Fixed paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\DELL\Hybrid-Multidisease-FL"
DATA_FOLDER  = os.path.join(PROJECT_ROOT, "data")
MODULE2_PATH = os.path.join(PROJECT_ROOT, "module2_quantum")
MODULE3_PATH = os.path.join(PROJECT_ROOT, "module3_fl")

os.makedirs(DATA_FOLDER, exist_ok=True)

for p in [MODULE2_PATH, MODULE3_PATH]:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# ── Disease configuration ─────────────────────────────────────────────────────
# Each disease has:
#   target_col : the column name used as label in that dataset
#   display    : human readable name shown in frontend
#   color      : color used in frontend UI
DISEASES = {
    "diabetes": {
        "target_col": "Diabetes",
        "display":    "Diabetes",
        "color":      "#2563eb",
        "icon":       "🩺",
    },
    "kidney": {
        "target_col": "classification",
        "display":    "Kidney Disease",
        "color":      "#7c3aed",
        "icon":       "🫘",
    },
    "heart": {
        "target_col": "condition",
        "display":    "Heart Disease",
        "color":      "#dc2626",
        "icon":       "❤️",
    },
    "liver": {
        "target_col": "is_patient",
        "display":    "Liver Disease",
        "color":      "#059669",
        "icon":       "🫀",
    },
}

print(f"\n  Backend starting — Multi-Disease Mode")
print(f"  Data folder : {DATA_FOLDER}")
print(f"  Module 2    : {MODULE2_PATH}")
print(f"  Module 3    : {MODULE3_PATH}")
print(f"  Diseases    : {list(DISEASES.keys())}")

app = Flask(__name__)
CORS(app)


# ── Path helpers ──────────────────────────────────────────────────────────────

def _m1_path(disease):
    return os.path.join(DATA_FOLDER, f"{disease}_8features.csv")

def _q_train_path(disease):
    return os.path.join(DATA_FOLDER, f"quantum_{disease}_train.csv")

def _q_test_path(disease):
    return os.path.join(DATA_FOLDER, f"quantum_{disease}_test.csv")

def _model_path(disease):
    return os.path.join(DATA_FOLDER, f"global_{disease}_model.pkl")

def _m1_ready(disease):
    return os.path.exists(_m1_path(disease))

def _m2_ready(disease):
    return os.path.exists(_q_train_path(disease)) and \
           os.path.exists(_q_test_path(disease))

def _m3_ready(disease):
    return os.path.exists(_model_path(disease))


# ── Upload / cleanup helpers ──────────────────────────────────────────────────

def _save_upload(file):
    suffix = os.path.splitext(file.filename)[1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    tmp.close()
    return tmp.name

def _cleanup(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass

def _validate_disease(disease):
    if disease not in DISEASES:
        return jsonify({
            "error": f"Unknown disease '{disease}'. "
                     f"Valid options: {list(DISEASES.keys())}"
        }), 400
    return None


# ── GET /status ───────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":      "running",
        "mode":        "multi-disease",
        "data_folder": DATA_FOLDER,
        "diseases":    list(DISEASES.keys()),
        "endpoints": [
            "GET  /status",
            "GET  /pipeline-status",
            "POST /upload-disease/<disease>",
            "POST /run-model/<disease>",
            "POST /run-quantum/<disease>",
            "POST /run-m3-stream/<disease>",
            "GET  /metabolic-report",
        ],
    })


# ── GET /pipeline-status ──────────────────────────────────────────────────────

@app.route("/pipeline-status", methods=["GET"])
def pipeline_status():
    """
    Returns which pipeline stages are complete for each disease.
    Frontend uses this to show progress indicators.
    """
    status_map = {}
    for disease, cfg in DISEASES.items():
        status_map[disease] = {
            "display":  cfg["display"],
            "color":    cfg["color"],
            "icon":     cfg["icon"],
            "module1":  _m1_ready(disease),
            "module2":  _m2_ready(disease),
            "module3":  _m3_ready(disease),
            "complete": _m3_ready(disease),
        }
    all_complete = all(v["complete"] for v in status_map.values())
    return jsonify({
        "diseases":     status_map,
        "all_complete": all_complete,
    })


# ── POST /upload-disease/<disease> ────────────────────────────────────────────

@app.route("/upload-disease/<disease>", methods=["POST"])
def upload_disease(disease):
    """
    Upload a CSV for a specific disease.
    Saves it directly to data folder as {disease}_raw.csv
    so it can be used by run-model later.
    """
    err = _validate_disease(disease)
    if err:
        return err

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save raw uploaded file to data folder
    raw_path = os.path.join(DATA_FOLDER, f"{disease}_raw.csv")
    file.save(raw_path)

    return jsonify({
        "status":   "uploaded",
        "disease":  disease,
        "display":  DISEASES[disease]["display"],
        "path":     raw_path,
        "message":  f"{DISEASES[disease]['display']} dataset uploaded successfully.",
    })


# ── POST /run-model/<disease>  (Module 1) ─────────────────────────────────────

@app.route("/run-model/<disease>", methods=["POST"])
def run_model(disease):
    """
    Module 1 - XGBoost feature selection for a specific disease.
    Reads uploaded CSV, runs pipeline, saves {disease}_8features.csv
    """
    err = _validate_disease(disease)
    if err:
        return err

    # Get CSV — either uploaded now or previously saved
    tmp = None
    if "file" in request.files and request.files["file"].filename:
        tmp      = _save_upload(request.files["file"])
        csv_path = tmp
    else:
        csv_path = os.path.join(DATA_FOLDER, f"{disease}_raw.csv")
        if not os.path.exists(csv_path):
            return jsonify({
                "error": f"No CSV found for {disease}. "
                         f"Upload via /upload-disease/{disease} first."
            }), 400

    try:
        results = run_pipeline(csv_path, disease=disease)
        results["disease"]        = disease
        results["display"]        = DISEASES[disease]["display"]
        results["quantum_ready"]  = _m1_ready(disease)
        results["m1_output_path"] = _m1_path(disease)
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500
    finally:
        _cleanup(tmp)

    return jsonify(results)


# ── POST /run-quantum/<disease>  (Module 2) ───────────────────────────────────

@app.route("/run-quantum/<disease>", methods=["POST"])
def run_quantum(disease):
    """
    Module 2 - Quantum encoding for a specific disease.
    Reads {disease}_8features.csv from data folder.
    Saves quantum_{disease}_train/test.csv to data folder.
    """
    err = _validate_disease(disease)
    if err:
        return err

    tmp = None
    try:
        # Use disease-specific CSV from Module 1
        csv_path = _m1_path(disease)
        if not os.path.exists(csv_path):
            return jsonify({
                "error": f"{disease}_8features.csv not found. "
                         f"Run /run-model/{disease} first."
            }), 400

        try:
            from feature_extractor import run_feature_extraction
            from quantum_simulator import run_simulation
        except ImportError as ie:
            return jsonify({
                "error": f"Module 2 import failed: {str(ie)}"
            }), 500

        # Run Module 2 with disease-specific output paths
        df_train, df_test, weights = run_feature_extraction(
            csv_path, disease=disease
        )
        simulation = run_simulation(csv_path, weights)

        q_cols     = [c for c in df_train.columns if c.startswith("Q_feature")]
        vectors    = df_train[q_cols].head(5).round(4).values.tolist()
        label_col  = df_train.columns[-1]
        labels     = df_train[label_col].head(5).tolist()
        train_vals = df_train[q_cols].values
        test_vals  = df_test[q_cols].values

        return jsonify({
            "disease":  disease,
            "display":  DISEASES[disease]["display"],
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
                "n_qubits":     len(q_cols),
                "n_layers":     3,
                "entanglement": "linear chain",
                "encoding":     "angle encoding (RY gates)",
                "measurement":  "PauliZ expectation value",
                "total_params": len(q_cols) * 3 * 2,
                "weight_range": "[-pi/4, +pi/4]",
                "gates_used":   ["Hadamard", "RY", "CNOT", "RZ", "PauliZ"],
                "feature_map":  {
                    f"qubit_{i}": q_cols[i]
                    for i in range(len(q_cols))
                },
            },
            "simulation":      simulation,
            "pipeline_status": {
                "module1": "complete",
                "module2": "complete",
            },
            "m3_ready": _m2_ready(disease),
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error":     str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        _cleanup(tmp)


# ── POST /run-m3-stream/<disease>  (Module 3 SSE) ────────────────────────────

@app.route("/run-m3-unified", methods=["POST"])
def run_m3_unified():
    if not os.path.exists(MODULE3_PATH):
        return jsonify({"error": f"module3_fl not found at: {MODULE3_PATH}"}), 500

    log_queue = queue.Queue()

    try:
        from stream_m3 import run_module3_streaming
    except ImportError as ie:
        return jsonify({"error": f"stream_m3 import failed: {str(ie)}"}), 500

    threading.Thread(
        target=run_module3_streaming,
        args=(log_queue, "unified"),
        daemon=True
    ).start()

    def generate():
        while True:
            try:
                ev = log_queue.get(timeout=300)
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                continue
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev.get("type") in ("done", "error"):
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        }
    )

# ── GET /metabolic-report ─────────────────────────────────────────────────────

@app.route("/metabolic-report", methods=["GET"])
def metabolic_report():
    """
    Final combined metabolic syndrome risk report.
    Reads results from all disease models and combines into
    a single Metabolic Syndrome Risk Profile.
    Only available after all 4 diseases complete Module 3.
    """
    report = {}
    all_ready = True

    for disease, cfg in DISEASES.items():
        model_path = _model_path(disease)
        if not os.path.exists(model_path):
            all_ready = False
            report[disease] = {
                "display":  cfg["display"],
                "color":    cfg["color"],
                "icon":     cfg["icon"],
                "status":   "not_ready",
                "message":  f"Run Module 3 for {cfg['display']} first.",
            }
            continue

        try:
            import joblib
            import numpy as np
            import pandas as pd

            model     = joblib.load(model_path)
            test_path = _q_test_path(disease)
            df_test   = pd.read_csv(test_path)
            q_cols    = [c for c in df_test.columns if c.startswith("Q_feature")]
            label_col = df_test.columns[-1]

            X_test = df_test[q_cols].values
            y_test = df_test[label_col].values

            # Get prediction probability for patient #1
            sample     = X_test[0:1]
            prob       = float(model.predict_proba(sample)[0][1])
            risk_level = "High" if prob > 0.65 else \
                         "Medium" if prob > 0.35 else "Low"

            report[disease] = {
                "display":    cfg["display"],
                "color":      cfg["color"],
                "icon":       cfg["icon"],
                "status":     "complete",
                "risk_level": risk_level,
                "probability": round(prob * 100, 1),
            }

        except Exception as e:
            all_ready = False
            report[disease] = {
                "display": cfg["display"],
                "color":   cfg["color"],
                "icon":    cfg["icon"],
                "status":  "error",
                "message": str(e),
            }

    # Overall metabolic syndrome risk
    # High if 2 or more diseases are high risk
    if all_ready:
        high_count = sum(
            1 for d in report.values()
            if d.get("risk_level") == "High"
        )
        medium_count = sum(
            1 for d in report.values()
            if d.get("risk_level") == "Medium"
        )
        overall = "High"   if high_count >= 2 else \
                  "Medium" if high_count == 1 or medium_count >= 2 else \
                  "Low"

        # Average probability across all diseases
        avg_prob = round(
            sum(d.get("probability", 0) for d in report.values()) / len(report),
            1
        )
    else:
        overall  = "Incomplete"
        avg_prob = 0

    return jsonify({
        "diseases":         report,
        "all_complete":     all_ready,
        "overall_risk":     overall,
        "average_risk_pct": avg_prob,
        "summary": (
            f"Metabolic Syndrome Risk: {overall}. "
            f"Average disease risk: {avg_prob}%."
        ),
    })


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  Multi-Disease Endpoints:")
    print(f"    GET  http://localhost:5000/status")
    print(f"    GET  http://localhost:5000/pipeline-status")
    print(f"    POST http://localhost:5000/upload-disease/<disease>")
    print(f"    POST http://localhost:5000/run-model/<disease>")
    print(f"    POST http://localhost:5000/run-quantum/<disease>")
    print(f"    POST http://localhost:5000/run-m3-stream/<disease>")
    print(f"    GET  http://localhost:5000/metabolic-report")
    print(f"\n  Valid diseases: {list(DISEASES.keys())}\n")
    app.run(debug=True, port=5000)