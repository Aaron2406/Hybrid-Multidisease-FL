import React, { useEffect, useState } from "react"
import "../pages/dashboard.css"

const METRICS = [
  { key: "accuracy",  label: "Accuracy",  desc: "Overall correct predictions" },
  { key: "precision", label: "Precision", desc: "Positive predictive value"    },
  { key: "recall",    label: "Recall",    desc: "True positive rate"           },
  { key: "f1",        label: "F1 Score",  desc: "Harmonic mean of P & R"      },
]

function AnimBar({ value }) {
  const [w, setW] = useState(0)
  useEffect(() => {
    const t = setTimeout(() => setW(value * 100), 80)
    return () => clearTimeout(t)
  }, [value])
  return (
    <div className="metric-bar">
      <div className="metric-bar-fill" style={{ width: `${w}%` }} />
    </div>
  )
}

function Delta({ before, after }) {
  const diff = ((after - before) * 100).toFixed(2)
  const up = after >= before
  return (
    <span className={`delta ${up ? "up" : "down"}`}>
      {up ? "▲" : "▼"} {Math.abs(diff)}%
    </span>
  )
}

export default function MetricsPanel({ metrics, metrics_before, metrics_after }) {
  const before    = metrics_before || metrics
  const after     = metrics_after  || metrics
  const hasBoth   = !!(metrics_before && metrics_after)
  const fmt       = (v) => (v * 100).toFixed(2) + "%"
  const avgBefore = METRICS.reduce((s, m) => s + (before[m.key] || 0), 0) / METRICS.length
  const avgAfter  = METRICS.reduce((s, m) => s + (after[m.key]  || 0), 0) / METRICS.length
  const avgDelta  = ((avgAfter - avgBefore) * 100).toFixed(2)
  const improved  = avgAfter >= avgBefore

  return (
    <div className="card">
      {hasBoth && (
        <div className="metrics-compare-header">
          <span className="metrics-compare-title">
            Feature Selection {improved ? "Improved" : "Changed"} Performance
          </span>
          <span
            className="metrics-compare-delta"
            style={{ color: improved ? "var(--success)" : "var(--error)" }}
          >
            {improved ? "+" : ""}{avgDelta}%
          </span>
        </div>
      )}

      <div className="metrics-grid">
        {METRICS.map(({ key, label, desc }) => {
          const bVal = before[key] || 0
          const aVal = after[key]  || 0
          return (
            <div key={key} className="metric-card">
              <div className="metric-label">
                {label}
                {hasBoth && <Delta before={bVal} after={aVal} />}
              </div>
              <div className="metric-value">{fmt(aVal)}</div>
              {hasBoth && (
                <div className="compare-row">
                  <div className="compare-cell">
                    <div className="compare-cell-label">Before</div>
                    <div className="compare-cell-val">{fmt(bVal)}</div>
                  </div>
                  <div className="compare-cell after">
                    <div className="compare-cell-label">After</div>
                    <div className="compare-cell-val">{fmt(aVal)}</div>
                  </div>
                </div>
              )}
              <AnimBar value={aVal} />
              <div className="metric-desc">{desc}</div>
            </div>
          )
        })}
      </div>

      <div className="footer-note">10-Fold CV · Stratified · Seed 42</div>
    </div>
  )
}