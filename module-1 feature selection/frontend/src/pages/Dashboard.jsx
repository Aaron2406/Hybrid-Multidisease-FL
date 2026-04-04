import React, { useState } from "react"
import axios from "axios"
import UploadCSV          from "../Component/UploadCSV"
import MetricsPanel       from "../Component/MetricsPanel"
import FeatureChart       from "../Component/FeatureChart"
import ConfusionMatrix    from "../Component/ConfusionMatrix"
import PredictionTable    from "../Component/PredictionTable"
import QuantumCircuitInfo from "../Component/QuantumCircuitInfo"
import QuantumFeatureTable from "../Component/QuantumFeatureTable"
import QuantumSimulator   from "../Component/QuantumSimulator"
import "./dashboard.css"

function SectionHeading({ num, text, badge }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
      <span style={{ fontSize: 11, fontWeight: 700, color: "var(--primary)",
                     fontVariantNumeric: "tabular-nums" }}>
        {num}
      </span>
      <span className="section-heading" style={{ margin: 0 }}>{text}</span>
      {badge && (
        <span style={{
          background: "#7c3aed", color: "#fff",
          fontSize: 10, fontWeight: 700, borderRadius: 6,
          padding: "2px 8px", textTransform: "uppercase", letterSpacing: "0.05em",
        }}>
          {badge}
        </span>
      )}
    </div>
  )
}

export default function Dashboard() {
  const [results,        setResults]        = useState(null)
  const [quantumResults, setQuantumResults] = useState(null)
  const [qLoading,       setQLoading]       = useState(false)
  const [qError,         setQError]         = useState(null)

  const runQuantum = async () => {
    setQError(null)
    setQLoading(true)
    try {
      // No file needed — backend uses the CSV saved by /run-model automatically
      const res = await axios.post("http://localhost:5000/run-quantum")
      if (res.data.error) setQError(res.data.error)
      else setQuantumResults(res.data)
    } catch (e) {
      setQError(e.response?.data?.error || e.message || "Quantum encoding failed.")
    } finally {
      setQLoading(false)
    }
  }

  return (
    <div style={{ width: "100%", minHeight: "100vh", background: "var(--bg)" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
      `}</style>

      <header className="header">
        <div>
          <div className="header-title">Disease Prediction Dashboard</div>
          <div className="header-sub">
            XGBoost · WEKA ClassifierAttributeEval · Quantum Feature Encoding
          </div>
        </div>
      </header>

      <main className="main">

        {/* ── 01 Upload ──────────────────────────────────────────────────────── */}
        <section className="section">
          <SectionHeading num="01" text="Upload Dataset" />
          <UploadCSV setResults={setResults} />
        </section>

        {results && (
          <>
            <div className="divider" />

            {/* ── 02 Preprocessing ─────────────────────────────────────────── */}
            <section className="section">
              <SectionHeading num="02" text="Preprocessing Results" />
              <PredictionTable
                predictions={results.predictions}
                classNames={results.class_names}
                preprocessingSummary={results.preprocessing_summary}
              />
            </section>

            <div className="divider" />

            {/* ── 03 Model Performance ─────────────────────────────────────── */}
            <section className="section">
              <SectionHeading num="03" text="Model Performance" />
              <MetricsPanel
                metrics={results.metrics}
                metrics_before={results.metrics_before}
                metrics_after={results.metrics_after}
              />
            </section>

            <div className="divider" />

            {/* ── 04 Feature Importance ────────────────────────────────────── */}
            <section className="section">
              <SectionHeading num="04" text="Feature Importance" />
              <FeatureChart features={results.top_features} />
            </section>

            <div className="divider" />

            {/* ── 05 Confusion Matrix ──────────────────────────────────────── */}
            <section className="section">
              <SectionHeading num="05" text="Confusion Matrix" />
              <ConfusionMatrix matrix={results.confusion_matrix} />
            </section>

            <div className="divider" />

            {/* ── 06 Quantum Encoding trigger ──────────────────────────────── */}
            <section className="section">
              <SectionHeading num="06" text="Quantum Feature Encoding" badge="Module 2" />

              {!quantumResults && (
                <div style={{
                  background: "#faf5ff", border: "1px solid #e9d5ff",
                  borderRadius: 12, padding: 20,
                }}>
                  <div style={{ fontSize: 14, color: "#6d28d9", fontWeight: 600, marginBottom: 8 }}>
                    Ready to encode features into quantum states
                  </div>
                  <div style={{ fontSize: 13, color: "#7c3aed", marginBottom: 16, lineHeight: 1.6 }}>
                    Module 1 selected <strong>{results.top_features?.length ?? 8} features</strong>.
                    Module 2 will encode each feature into a qubit using angle encoding,
                    apply entanglement and variational layers, then measure PauliZ
                    expectation values as the quantum feature representation.
                  </div>
                  <button
                    className="btn"
                    disabled={qLoading}
                    onClick={runQuantum}
                    style={{ background: "#7c3aed", minWidth: 180 }}
                  >
                    {qLoading ? "Running quantum circuit..." : "Run Quantum Encoding"}
                  </button>
                  {qLoading && (
                    <div style={{ fontSize: 12, color: "#8b5cf6", marginTop: 10 }}>
                      Processing {results.top_features?.length ?? 8} qubits through
                      3 variational layers... this may take 30-60 seconds.
                    </div>
                  )}
                </div>
              )}

              {qError && (
                <div className="error-box" style={{ marginTop: 12 }}>
                  {qError}
                </div>
              )}

              {quantumResults && (
                <div style={{
                  background: "#f0fdf4", border: "1px solid #bbf7d0",
                  borderRadius: 10, padding: "10px 16px",
                  fontSize: 13, color: "#15803d", fontWeight: 600,
                  display: "flex", alignItems: "center", gap: 8,
                }}>
                  <span style={{ fontSize: 16 }}>✓</span>
                  Quantum encoding complete — {quantumResults.q_train_shape?.[0]} train
                  samples encoded into {quantumResults.q_train_shape?.[1]}-dimensional
                  quantum feature vectors.
                  <button
                    onClick={() => { setQuantumResults(null); setQError(null) }}
                    style={{ marginLeft: "auto", background: "none", border: "none",
                             color: "#15803d", cursor: "pointer", fontSize: 12 }}
                  >
                    Re-run
                  </button>
                </div>
              )}
            </section>

            {/* ── 07 Circuit Info ──────────────────────────────────────────── */}
            {quantumResults && (
              <>
                <div className="divider" />
                <section className="section">
                  <SectionHeading num="07" text="Quantum Circuit Architecture" badge="Module 2" />
                  <QuantumCircuitInfo circuitInfo={quantumResults.circuit_info} />
                </section>

                <div className="divider" />

                {/* ── 08 Quantum Features ──────────────────────────────────── */}
                <section className="section">
                  <SectionHeading num="08" text="Quantum Feature Vectors" badge="Module 2" />
                  <QuantumFeatureTable
                    quantumSamples={quantumResults.quantum_features_sample}
                    spread={quantumResults.spread}
                    qTrainShape={quantumResults.q_train_shape}
                    qTestShape={quantumResults.q_test_shape}
                  />
                </section>

                <div className="divider" />

                {/* ── 09 Simulator ─────────────────────────────────────────── */}
                <section className="section">
                  <SectionHeading num="09" text="Circuit Simulator" badge="Module 2" />
                  <QuantumSimulator simulation={quantumResults.simulation} />
                </section>
              </>
            )}
          </>
        )}
      </main>
    </div>
  )
}