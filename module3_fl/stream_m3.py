"""
Module 3 - stream_m3.py  (v3 - Unified Multi-Disease)
Reads quantum_{disease}_train/test.csv from data folder.
Merges all 4 disease datasets and trains ONE unified model.
Called by backend.py POST /run-m3-unified
"""

import os, sys

PROJECT_ROOT = r"C:\Users\DELL\Hybrid-Multidisease-FL"
DATA_FOLDER  = os.path.join(PROJECT_ROOT, "data")
MODULE3_PATH = os.path.join(PROJECT_ROOT, "module3_fl")
MODULE2_PATH = os.path.join(PROJECT_ROOT, "module2_quantum")

for p in [MODULE3_PATH, MODULE2_PATH]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

ALL_DISEASES = ["diabetes", "kidney", "heart", "liver"]


class LogEmitter:
    def __init__(self, q):
        self.q = q

    def log(self, msg, module="M3", level="info"):
        self.q.put({"type": "log", "module": module, "level": level, "msg": str(msg)})

    def section(self, title, module="M3"):
        self.q.put({"type": "section", "module": module, "msg": title})

    def progress(self, current, total, label="", module="M3"):
        self.q.put({"type": "progress", "module": module, "current": current,
                    "total": total, "pct": round(current / total * 100), "label": label})

    def success(self, msg, module="M3"):
        self.log(msg, module=module, level="success")

    def warn(self, msg, module="M3"):
        self.log(msg, module=module, level="warn")

    def result(self, data):
        self.q.put({"type": "result", "data": data})

    def done(self):
        self.q.put({"type": "done"})

    def error(self, msg):
        self.q.put({"type": "error", "msg": str(msg)})


def get_ready_diseases(emit):
    """Check which diseases have quantum CSVs ready in data folder."""
    ready = []
    for disease in ALL_DISEASES:
        train = os.path.join(DATA_FOLDER, f"quantum_{disease}_train.csv")
        test  = os.path.join(DATA_FOLDER, f"quantum_{disease}_test.csv")
        if os.path.exists(train) and os.path.exists(test):
            ready.append(disease)
            emit.log(f"{disease}: ready", module="M3", level="success")
        else:
            emit.log(f"{disease}: not ready — skipping", module="M3", level="warn")
    return ready


def run_module3_streaming(result_queue, disease="unified"):
    """
    Run the unified Module 3 pipeline.
    Merges all available disease quantum CSVs — trains ONE unified model.
    """
    emit = LogEmitter(result_queue)

    try:
        emit.section("MODULE 3 — Unified Multi-Disease FedProx", module="M3")
        emit.log("Scanning data folder for quantum CSVs...", module="M3")

        # ── Find ready diseases ───────────────────────────────────────────
        ready_diseases = get_ready_diseases(emit)

        if len(ready_diseases) < 2:
            emit.error(
                f"Need at least 2 diseases ready. Found: {ready_diseases}. "
                f"Run Module 1 + Module 2 for more diseases first."
            )
            return

        emit.log(f"Training unified model on: {ready_diseases}", module="M3", level="success")

        train_paths = {d: os.path.join(DATA_FOLDER, f"quantum_{d}_train.csv")
                       for d in ready_diseases}
        test_paths  = {d: os.path.join(DATA_FOLDER, f"quantum_{d}_test.csv")
                       for d in ready_diseases}

        # ── 3A: ML Baseline ───────────────────────────────────────────────
        from ml_baseline import run_baseline_unified
        baseline_results, X_train, X_test, y_train, y_test, lr_scaler = \
            run_baseline_unified(train_paths, test_paths, emit)

        # ── 3B: Unified FedProx ───────────────────────────────────────────
        from fedprox import run_fedprox_unified
        global_model, scaler, fl_results = run_fedprox_unified(
            train_paths=train_paths, emit=emit
        )

        # ── 3C: FL Simulation ─────────────────────────────────────────────
        from fl_simulator import run_simulation
        sim_results = run_simulation(fl_results, emit)

        # ── 3D: Unified Predictor ─────────────────────────────────────────
        from predictor import run_predictor_unified
        pred_results = run_predictor_unified(
            test_paths=test_paths,
            ready_diseases=ready_diseases,
            emit=emit,
        )

        emit.success("MODULE 3 UNIFIED COMPLETE", module="M3")
        emit.result({
            "mode":          "unified",
            "diseases_used": ready_diseases,
            "baseline":      baseline_results,
            "fl":            fl_results,
            "simulation":    sim_results,
            "prediction":    pred_results,
        })

    except Exception as e:
        import traceback
        emit.error(f"{str(e)}\n{traceback.format_exc()}")
    finally:
        emit.done()