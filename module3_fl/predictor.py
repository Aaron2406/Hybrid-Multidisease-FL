"""
Module 3D - Final Prediction + SHAP  (v3 - Unified Multi-Disease)
Reads unified model and predicts which disease(s) are detected.
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)

PROJECT_ROOT = r"C:\Users\DELL\Hybrid-Multidisease-FL"
DATA_FOLDER  = os.path.join(PROJECT_ROOT, "data")
MODULE3_PATH = os.path.join(PROJECT_ROOT, "module3_fl")

UNIFIED_MODEL_PATH  = os.path.join(DATA_FOLDER, "global_unified_model.pkl")
UNIFIED_SCALER_PATH = os.path.join(DATA_FOLDER, "global_unified_scaler.pkl")
MERGED_TEST_PATH    = os.path.join(DATA_FOLDER, "unified_test.csv")

DISEASE_LABEL = {"diabetes": 0, "kidney": 1, "heart": 2, "liver": 3}
DISEASE_NAMES = {v: k for k, v in DISEASE_LABEL.items()}

DISEASE_DISPLAY = {
    "diabetes": {"label": "Diabetic",      "neg": "No Diabetes",      "icon": "🩺"},
    "kidney":   {"label": "CKD Detected",  "neg": "No Kidney Disease", "icon": "🫘"},
    "heart":    {"label": "Heart Disease", "neg": "No Heart Disease",  "icon": "❤️"},
    "liver":    {"label": "Liver Disease", "neg": "No Liver Disease",  "icon": "🫀"},
}


def load_unified_model():
    if not os.path.exists(UNIFIED_MODEL_PATH):
        raise FileNotFoundError(
            f"Unified model not found at {UNIFIED_MODEL_PATH}. "
            f"Run unified Module 3 first."
        )
    return joblib.load(UNIFIED_MODEL_PATH), joblib.load(UNIFIED_SCALER_PATH)


def merge_test_data(test_paths, ready_diseases, emit=None):
    """Merge all disease test CSVs for evaluation."""
    def log(msg, level="info"):
        if emit: emit.log(msg, module="M3D", level=level)
        else:    print(f"  {msg}")

    frames = []
    for disease in ready_diseases:
        df     = pd.read_csv(test_paths[disease])
        q_cols = [c for c in df.columns if c.startswith("Q_feature")]
        df_m   = df[q_cols].copy()
        df_m["disease_label"] = DISEASE_LABEL.get(disease, 0)
        frames.append(df_m)
        log(f"Test loaded: {disease} ({len(df_m)} samples)")

    df_all  = pd.concat(frames, ignore_index=True)
    df_all.to_csv(MERGED_TEST_PATH, index=False)
    q_cols  = [c for c in df_all.columns if c.startswith("Q_feature")]
    return df_all, q_cols


def compute_shap_unified(gm, X_test_s, feature_cols, classes, emit=None):
    """Compute SHAP values for the unified model."""
    def log(msg, level="info"):
        if emit: emit.log(msg, module="M3D", level=level)
        else:    print(f"  {msg}")

    try:
        import shap
        lr  = gm.to_sklearn(classes=classes)
        bg  = X_test_s[:min(100, len(X_test_s))]
        exp = shap.LinearExplainer(lr, bg, feature_perturbation="interventional")
        sv  = exp.shap_values(bg)

        if isinstance(sv, list):
            mean_ab = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
            raw     = np.mean([s.mean(axis=0) for s in sv], axis=0)
        else:
            mean_ab = np.abs(sv).mean(axis=0)
            raw     = sv.mean(axis=0)

        result = sorted([
            {
                "feature":     feature_cols[i],
                "mean_abs":    round(float(mean_ab[i]), 4),
                "mean_signed": round(float(raw[i]), 4),
                "direction":   "increases risk" if raw[i] > 0 else "decreases risk",
                "rank":        0,
            }
            for i in range(len(feature_cols))
        ], key=lambda x: x["mean_abs"], reverse=True)

        for i, r in enumerate(result):
            r["rank"] = i + 1
        log(f"SHAP computed. Top: {result[0]['feature']}", level="success")
        return result

    except Exception as e:
        log(f"SHAP fallback: {e}", level="warn")
        lr   = gm.to_sklearn(classes=classes)
        coef = np.abs(lr.coef_).mean(axis=0)
        result = sorted([
            {
                "feature":     feature_cols[i],
                "mean_abs":    round(float(coef[i]), 4),
                "mean_signed": round(float(coef[i]), 4),
                "direction":   "increases risk",
                "rank":        0,
            }
            for i in range(len(feature_cols))
        ], key=lambda x: x["mean_abs"], reverse=True)
        for i, r in enumerate(result):
            r["rank"] = i + 1
        return result


def predict_patient(gm, scaler, q_features, classes, ready_diseases):
    """
    Predict which disease(s) a patient has.
    Returns probabilities for each disease class.
    """
    arr    = scaler.transform(np.array(q_features).reshape(1, -1))
    lr     = gm.to_sklearn(classes=classes)
    pred   = int(lr.predict(arr)[0])
    probas = lr.predict_proba(arr)[0]

    # Build per-disease prediction
    disease_predictions = []
    for cls_idx, prob in zip(classes, probas):
        disease = DISEASE_NAMES.get(cls_idx, f"disease_{cls_idx}")
        if disease not in ready_diseases:
            continue
        disp = DISEASE_DISPLAY.get(disease, {})
        disease_predictions.append({
            "disease":     disease,
            "icon":        disp.get("icon", "🔬"),
            "label":       disp.get("label", "Detected"),
            "neg_label":   disp.get("neg", "Not Detected"),
            "probability": round(float(prob) * 100, 1),
            "detected":    cls_idx == pred,
            "risk_level":  "High" if prob >= 0.6 else
                           "Medium" if prob >= 0.35 else "Low",
        })

    # Sort by probability descending
    disease_predictions.sort(key=lambda x: x["probability"], reverse=True)

    # Overall metabolic risk
    max_prob      = max(p["probability"] for p in disease_predictions)
    high_count    = sum(1 for p in disease_predictions if p["risk_level"] == "High")
    overall_risk  = "High"   if high_count >= 2 or max_prob >= 70 else \
                    "Medium" if high_count >= 1 or max_prob >= 40 else "Low"

    primary_disease = DISEASE_NAMES.get(pred, "unknown")

    return {
        "primary_disease":      primary_disease,
        "primary_icon":         DISEASE_DISPLAY.get(primary_disease, {}).get("icon", "🔬"),
        "primary_label":        DISEASE_DISPLAY.get(primary_disease, {}).get("label", "Detected"),
        "overall_risk":         overall_risk,
        "disease_predictions":  disease_predictions,
        "q_features":           [round(float(v), 4) for v in q_features],
    }


def run_predictor_unified(test_paths, ready_diseases, emit=None):
    """
    Run unified predictor — evaluates ONE model on merged test data.
    Shows which disease(s) are detected per patient.
    """
    def section(t):
        if emit: emit.section(t, module="M3D")
        else:    print(f"\n{'='*55}\n  {t}\n{'='*55}")

    def log(msg, level="info"):
        if emit: emit.log(msg, module="M3D", level=level)
        else:    print(f"  {msg}")

    section("3D — Unified Prediction + SHAP")
    log(f"Diseases: {ready_diseases}")

    # ── Load unified model ────────────────────────────────────────────────
    gm, scaler = load_unified_model()
    classes    = list(range(gm.n_classes_))
    log(f"Model loaded. Classes: {classes}", level="success")

    # ── Merge test data ───────────────────────────────────────────────────
    df_all, feature_cols = merge_test_data(test_paths, ready_diseases, emit)
    X_test = scaler.transform(df_all[feature_cols].values.astype(np.float64))
    y_test = df_all["disease_label"].values.astype(np.int32)

    # ── Evaluate ──────────────────────────────────────────────────────────
    lr_eval = gm.to_sklearn(classes=classes)
    y_pred  = lr_eval.predict(X_test)

    metrics = {
        "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
        "f1":               round(float(f1_score(y_test, y_pred,
                                average="weighted", zero_division=0)), 4),
        "precision":        round(float(precision_score(y_test, y_pred,
                                average="weighted", zero_division=0)), 4),
        "recall":           round(float(recall_score(y_test, y_pred,
                                average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples":     int(len(y_test)),
        "feature_cols":     feature_cols,
        "class_names":      {str(k): DISEASE_NAMES.get(k, f"disease_{k}")
                             for k in classes},
    }
    log(f"Test: acc={metrics['accuracy']}  f1={metrics['f1']}", level="success")

    # ── SHAP ──────────────────────────────────────────────────────────────
    shap_values = compute_shap_unified(gm, X_test, feature_cols, classes, emit)

    # ── Sample prediction (first patient from test set) ───────────────────
    sample_q    = X_test[0].tolist()
    sample_pred = predict_patient(gm, scaler, sample_q, classes, ready_diseases)
    log(f"Sample: primary={sample_pred['primary_disease']}  "
        f"risk={sample_pred['overall_risk']}", level="success")

    results = {
        "mode":              "unified",
        "diseases_used":     ready_diseases,
        "test_metrics":      metrics,
        "shap_values":       shap_values,
        "sample_prediction": sample_pred,
        "model_info": {
            "algorithm":    "FedProx Unified",
            "local_model":  "Logistic Regression (multinomial)",
            "n_features":   len(feature_cols),
            "n_classes":    gm.n_classes_,
            "feature_cols": feature_cols,
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────
    pred_path = os.path.join(DATA_FOLDER, "prediction_unified_results.json")
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved: {pred_path}", level="success")

    section("MODULE 3 UNIFIED COMPLETE")
    return results


if __name__ == "__main__":
    ready = [d for d in ["diabetes", "kidney", "heart", "liver"]
             if os.path.exists(os.path.join(DATA_FOLDER, f"quantum_{d}_test.csv"))]
    test_paths = {d: os.path.join(DATA_FOLDER, f"quantum_{d}_test.csv") for d in ready}
    r = run_predictor_unified(test_paths, ready)
    print(f"\n  Test accuracy: {r['test_metrics']['accuracy']}")
    print(f"  Primary disease: {r['sample_prediction']['primary_disease']}")
    print(f"  Overall risk: {r['sample_prediction']['overall_risk']}")