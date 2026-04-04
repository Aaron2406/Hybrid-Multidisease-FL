import React from "react"

const GATE_COLORS = {
  Hadamard: "#7c3aed",
  RY:       "#0891b2",
  CNOT:     "#059669",
  RZ:       "#d97706",
  PauliZ:   "#dc2626",
}

export default function QuantumCircuitInfo({ circuitInfo }) {
  if (!circuitInfo) return null

  const { n_qubits, n_layers, entanglement, encoding,
          measurement, total_params, gates_used, feature_map } = circuitInfo

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Top stats row */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {[
          { label: "Qubits",      value: n_qubits },
          { label: "Layers",      value: n_layers },
          { label: "Parameters",  value: total_params },
          { label: "Entanglement",value: entanglement },
        ].map(({ label, value }) => (
          <div key={label} style={{
            flex: "1 1 120px",
            background: "var(--card-bg, #f8fafc)",
            border: "1px solid var(--border, #e2e8f0)",
            borderRadius: 10, padding: "12px 16px",
          }}>
            <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600,
                          textTransform: "uppercase", letterSpacing: "0.05em" }}>
              {label}
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: "#1e293b", marginTop: 4 }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* Gates used */}
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b",
                      marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Gates used
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {gates_used.map(gate => (
            <span key={gate} style={{
              background: GATE_COLORS[gate] || "#6366f1",
              color: "#fff", borderRadius: 6,
              padding: "4px 10px", fontSize: 12, fontWeight: 600,
            }}>
              {gate}
            </span>
          ))}
        </div>
      </div>

      {/* Circuit layout diagram */}
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b",
                      marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Circuit layout
        </div>
        <div style={{
          background: "#0f172a", borderRadius: 10, padding: 16,
          fontFamily: "monospace", fontSize: 12, color: "#94a3b8", overflowX: "auto",
        }}>
          {Object.entries(feature_map).map(([qubit, feature], idx) => (
            <div key={qubit} style={{ display: "flex", alignItems: "center",
                                      gap: 6, marginBottom: 6 }}>
              <span style={{ color: "#7c3aed", minWidth: 60 }}>q{idx}:</span>
              <span style={{ color: "#f1f5f9", minWidth: 100 }}>{feature}</span>
              <span style={{ color: "#7c3aed" }}>─H─</span>
              <span style={{ color: "#0891b2" }}>RY(θ)─</span>
              {idx < 7 && <span style={{ color: "#059669" }}>●─</span>}
              <span style={{ color: "#d97706" }}>RZ─RY─</span>
              <span style={{ color: "#d97706" }}>RZ─RY─</span>
              <span style={{ color: "#d97706" }}>RZ─RY─</span>
              <span style={{ color: "#dc2626" }}>⟨Z⟩</span>
            </div>
          ))}
        </div>
      </div>

      {/* Encoding + measurement info */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <div style={{
          flex: "1 1 200px", background: "#eff6ff",
          border: "1px solid #bfdbfe", borderRadius: 10, padding: 14,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#1d4ed8",
                        textTransform: "uppercase", marginBottom: 4 }}>
            Encoding
          </div>
          <div style={{ fontSize: 13, color: "#1e40af" }}>{encoding}</div>
          <div style={{ fontSize: 12, color: "#3b82f6", marginTop: 4 }}>
            Features normalized to [0, pi] then mapped to RY rotation angles
          </div>
        </div>
        <div style={{
          flex: "1 1 200px", background: "#fdf4ff",
          border: "1px solid #e9d5ff", borderRadius: 10, padding: 14,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#7c3aed",
                        textTransform: "uppercase", marginBottom: 4 }}>
            Measurement
          </div>
          <div style={{ fontSize: 13, color: "#6d28d9" }}>{measurement}</div>
          <div style={{ fontSize: 12, color: "#8b5cf6", marginTop: 4 }}>
            Output values in [-1, +1] — one per qubit per patient
          </div>
        </div>
      </div>

    </div>
  )
}