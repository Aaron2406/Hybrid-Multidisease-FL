import React, { useState } from "react"

const GATE_COLOR = {
  "none":    "#94a3b8",
  "H":       "#7c3aed",
  "RY":      "#0891b2",
  "CNOT":    "#059669",
  "RZ + RY": "#d97706",
  "PauliZ":  "#dc2626",
}

function BlochDot({ z }) {
  // z in [-1,+1]: map to vertical position in a circle
  const cy   = 20 - z * 14   // 6 (top) to 34 (bottom)
  const color = z > 0.3 ? "#dc2626" : z < -0.3 ? "#2563eb" : "#8b5cf6"
  return (
    <svg width="40" height="40" viewBox="0 0 40 40">
      <circle cx="20" cy="20" r="16" fill="none" stroke="#e2e8f0" strokeWidth="1"/>
      <line x1="20" y1="4" x2="20" y2="36" stroke="#e2e8f0" strokeWidth="0.5"/>
      <line x1="4"  y1="20" x2="36" y2="20" stroke="#e2e8f0" strokeWidth="0.5"/>
      <circle cx="20" cy={cy} r="4" fill={color}/>
      <text x="21" y="3"  fontSize="7" fill="#94a3b8">|0&gt;</text>
      <text x="21" y="38" fontSize="7" fill="#94a3b8">|1&gt;</text>
    </svg>
  )
}

function StepCard({ step, isActive, onClick }) {
  const color = GATE_COLOR[step.gate] || "#6366f1"
  return (
    <div
      onClick={onClick}
      style={{
        border: `2px solid ${isActive ? color : "#e2e8f0"}`,
        borderRadius: 10, padding: "12px 14px", cursor: "pointer",
        background: isActive ? `${color}10` : "#fff",
        transition: "all 0.2s",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{
          background: color, color: "#fff", borderRadius: 6,
          padding: "2px 8px", fontSize: 11, fontWeight: 700, minWidth: 28,
          textAlign: "center",
        }}>
          {step.step}
        </span>
        <span style={{ fontSize: 13, fontWeight: 600, color: "#1e293b" }}>
          {step.layer_name}
        </span>
        <span style={{
          marginLeft: "auto", background: `${color}20`, color,
          borderRadius: 4, padding: "1px 6px", fontSize: 11, fontWeight: 600,
        }}>
          {step.gate === "none" ? "init" : step.gate}
        </span>
      </div>
    </div>
  )
}

export default function QuantumSimulator({ simulation }) {
  const [activeStep, setActiveStep] = useState(0)

  if (!simulation || !simulation.layer_trace) return null

  const { layer_trace, classical_vs_quantum, features_raw,
          features_normalized, quantum_vector } = simulation

  const step = layer_trace[activeStep]

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* Header info */}
      <div style={{
        background: "#f0fdf4", border: "1px solid #bbf7d0",
        borderRadius: 10, padding: 14, fontSize: 13, color: "#15803d",
      }}>
        Tracing <strong>Patient sample #1</strong> through the quantum circuit step by step.
        Click any step to inspect the qubit states at that point.
      </div>

      {/* Step selector */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b",
                      textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
          Circuit steps — click to inspect
        </div>
        {layer_trace.map((s, i) => (
          <StepCard
            key={i} step={s}
            isActive={i === activeStep}
            onClick={() => setActiveStep(i)}
          />
        ))}
      </div>

      {/* Active step detail */}
      {step && (
        <div style={{
          border: `2px solid ${GATE_COLOR[step.gate] || "#6366f1"}`,
          borderRadius: 12, padding: 16,
          background: `${GATE_COLOR[step.gate] || "#6366f1"}08`,
        }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#1e293b", marginBottom: 6 }}>
            Step {step.step}: {step.layer_name}
          </div>
          <div style={{ fontSize: 13, color: "#475569", marginBottom: 14, lineHeight: 1.6 }}>
            {step.description}
          </div>

          {/* Bloch sphere view per qubit */}
          <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b",
                        marginBottom: 8, textTransform: "uppercase" }}>
            Qubit states after this step
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {step.qubit_z_values && step.qubit_z_values.map((z, i) => (
              <div key={i} style={{
                display: "flex", flexDirection: "column", alignItems: "center",
                background: "#fff", border: "1px solid #e2e8f0",
                borderRadius: 8, padding: "8px 10px", minWidth: 70,
              }}>
                <div style={{ fontSize: 10, color: "#64748b", fontWeight: 600 }}>
                  q{i}
                </div>
                <BlochDot z={z} />
                <div style={{ fontSize: 10, fontWeight: 700,
                              color: z > 0.3 ? "#dc2626" : z < -0.3 ? "#2563eb" : "#8b5cf6" }}>
                  {z.toFixed(3)}
                </div>
                <div style={{ fontSize: 9, color: "#94a3b8", textAlign: "center",
                              maxWidth: 65, marginTop: 2 }}>
                  {step.qubit_labels?.[i]}
                </div>
              </div>
            ))}
          </div>

          {/* Feature-to-qubit mapping (only on encoding step) */}
          {step.feature_to_qubit_mapping && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b",
                            marginBottom: 8, textTransform: "uppercase" }}>
                Feature to qubit mapping
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {Object.entries(step.feature_to_qubit_mapping).map(([feat, info]) => (
                  <div key={feat} style={{
                    display: "flex", alignItems: "center", gap: 8,
                    fontSize: 12, padding: "4px 0",
                    borderBottom: "1px solid #f1f5f9",
                  }}>
                    <span style={{ minWidth: 160, color: "#374151", fontWeight: 600 }}>
                      {feat}
                    </span>
                    <span style={{ color: "#0891b2" }}>
                      {info.angle_radians.toFixed(4)} rad
                    </span>
                    <span style={{ color: "#64748b" }}>→</span>
                    <span style={{ color: "#7c3aed", fontWeight: 600 }}>
                      z = {info.z_value.toFixed(4)}
                    </span>
                    <span style={{ color: "#94a3b8", fontSize: 11 }}>
                      {info.interpretation}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Classical vs Quantum comparison */}
      {classical_vs_quantum && (
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b",
                        marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Classical vs quantum — feature transformation
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ background: "#f8fafc", borderBottom: "2px solid #e2e8f0" }}>
                  {["Feature", "Classical (raw)", "Normalized (rad)", "Quantum encoded"].map(h => (
                    <th key={h} style={{ padding: "8px 12px", textAlign: "left",
                                         color: "#64748b", fontWeight: 600, fontSize: 11 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {classical_vs_quantum.map((row, i) => (
                  <tr key={i} style={{
                    borderBottom: "1px solid #f1f5f9",
                    background: i % 2 === 0 ? "#fff" : "#f8fafc",
                  }}>
                    <td style={{ padding: "8px 12px", fontWeight: 600, color: "#374151" }}>
                      {row.feature}
                    </td>
                    <td style={{ padding: "8px 12px", color: "#64748b" }}>
                      {row.classical_raw}
                    </td>
                    <td style={{ padding: "8px 12px", color: "#0891b2", fontWeight: 600 }}>
                      {row.classical_normalized.toFixed(4)}
                    </td>
                    <td style={{ padding: "8px 12px" }}>
                      <span style={{
                        color: row.quantum_encoded > 0.3 ? "#dc2626"
                             : row.quantum_encoded < -0.3 ? "#2563eb" : "#8b5cf6",
                        fontWeight: 700, fontSize: 13,
                      }}>
                        {row.quantum_encoded.toFixed(4)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 6 }}>
            Raw feature value → normalized to [0, pi] → encoded into quantum expectation value in [-1, +1]
          </div>
        </div>
      )}

    </div>
  )
}