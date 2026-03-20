import React from "react"
import "../pages/dashboard.css"

export default function ConfusionMatrix({ matrix }) {
  if (!matrix || matrix.length === 0) return null

  const size   = matrix.length
  const labels = size === 2
    ? ["No", "Yes"]
    : size === 3
    ? ["High Risk", "Low Risk", "Prediabetes"]
    : matrix.map((_, i) => `Class ${i}`)

  const total = matrix.flat().reduce((a, b) => a + b, 0)

  return (
    <div className="card">
      {/* Legend */}
      <div style={{ display: "flex", gap: "16px", marginBottom: "18px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
          <div style={{ width: 12, height: 12, borderRadius: 3, background: "rgba(34,197,94,0.2)", border: "1px solid rgba(34,197,94,0.4)" }} />
          <span style={{ fontSize: 12, color: "var(--muted)" }}>Correct</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
          <div style={{ width: 12, height: 12, borderRadius: 3, background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.35)" }} />
          <span style={{ fontSize: 12, color: "var(--muted)" }}>Misclassified</span>
        </div>
        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--muted)" }}>
          {total.toLocaleString()} samples
        </span>
      </div>

      <table className="cm-table">
        <thead>
          <tr>
            <th />
            {labels.map(l => (
              <th key={l}>PRED {l.toUpperCase()}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="cm-row-label">ACT {labels[i].toUpperCase()}</td>
              {row.map((cell, j) => {
                const isDiag = i === j
                const cellClass = isDiag
                  ? "cm-cell correct"
                  : cell > 0
                  ? "cm-cell misclassified"
                  : "cm-cell empty"
                return (
                  <td key={j} className={cellClass}>
                    <div className="cm-cell-count">{cell.toLocaleString()}</div>
                    <div className="cm-cell-pct">
                      {total > 0 ? ((cell / total) * 100).toFixed(1) + "%" : "—"}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}