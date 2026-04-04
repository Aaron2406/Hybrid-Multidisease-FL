import React from "react"

function ValueBar({ value }) {
  // value is in [-1, +1], map to 0-100% for bar width
  const pct    = ((value + 1) / 2) * 100
  const color  = value > 0.3 ? "#dc2626" : value < -0.3 ? "#2563eb" : "#6b7280"
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 80, height: 8, background: "#e2e8f0", borderRadius: 4, position: "relative" }}>
        {/* center line */}
        <div style={{
          position: "absolute", left: "50%", top: 0,
          width: 1, height: "100%", background: "#94a3b8",
        }} />
        <div style={{
          position: "absolute",
          left:  value >= 0 ? "50%" : `${pct}%`,
          width: `${Math.abs(pct - 50)}%`,
          height: "100%", background: color, borderRadius: 4,
        }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 600,
                     color, minWidth: 52, textAlign: "right" }}>
        {value.toFixed(4)}
      </span>
    </div>
  )
}

const FEATURES = [
  "HighBP", "GenHlth", "HighChol", "BMI",
  "DifficultyWalk", "Age", "PhysHlth", "HeartDiseaseorAttack",
]

export default function QuantumFeatureTable({ quantumSamples, spread, qTrainShape, qTestShape }) {
  if (!quantumSamples || quantumSamples.length === 0) return null

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Shape + spread stats */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {[
          { label: "Train samples",  value: qTrainShape?.[0] ?? "—" },
          { label: "Test samples",   value: qTestShape?.[0]  ?? "—" },
          { label: "Q-features",     value: qTrainShape?.[1] ?? 8   },
          { label: "Train spread",   value: spread?.train ?? "—"    },
          { label: "Test spread",    value: spread?.test  ?? "—"    },
        ].map(({ label, value }) => (
          <div key={label} style={{
            flex: "1 1 100px",
            background: "var(--card-bg, #f8fafc)",
            border: "1px solid var(--border, #e2e8f0)",
            borderRadius: 10, padding: "10px 14px",
          }}>
            <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600,
                          textTransform: "uppercase", letterSpacing: "0.04em" }}>
              {label}
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#1e293b", marginTop: 3 }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* Sample vectors table */}
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b",
                      marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Sample quantum feature vectors (first 5 patients)
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #e2e8f0" }}>
                <th style={{ padding: "8px 12px", textAlign: "left",
                             color: "#64748b", fontWeight: 600, fontSize: 11 }}>
                  Patient
                </th>
                <th style={{ padding: "8px 12px", textAlign: "left",
                             color: "#64748b", fontWeight: 600, fontSize: 11 }}>
                  Label
                </th>
                {FEATURES.map((f, i) => (
                  <th key={f} style={{ padding: "8px 8px", textAlign: "left",
                                       color: "#64748b", fontWeight: 600, fontSize: 11 }}>
                    Q{i}<br />
                    <span style={{ fontSize: 10, fontWeight: 400 }}>{f}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {quantumSamples.map((row, idx) => (
                <tr key={idx} style={{
                  borderBottom: "1px solid #f1f5f9",
                  background: idx % 2 === 0 ? "#fff" : "#f8fafc",
                }}>
                  <td style={{ padding: "8px 12px", fontWeight: 600, color: "#374151" }}>
                    #{idx + 1}
                  </td>
                  <td style={{ padding: "8px 12px" }}>
                    <span style={{
                      background: row.label === 1 ? "#fee2e2" : "#dcfce7",
                      color:      row.label === 1 ? "#dc2626" : "#16a34a",
                      borderRadius: 6, padding: "2px 8px",
                      fontSize: 11, fontWeight: 700,
                    }}>
                      {row.label === 1 ? "Diabetic" : "No Diabetes"}
                    </span>
                  </td>
                  {row.vector.map((val, qi) => (
                    <td key={qi} style={{ padding: "6px 8px" }}>
                      <ValueBar value={val} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 8 }}>
          Each value is a PauliZ expectation measurement in [-1, +1].
          Red bar = positive (closer to |0>).  Blue bar = negative (closer to |1>).
        </div>
      </div>

    </div>
  )
}