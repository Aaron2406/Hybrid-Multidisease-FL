import React, { useState } from "react"
import "../pages/dashboard.css"

const CLASS_CONFIG = {
  0: { label: "High Risk",   cls: "cls-0" },
  1: { label: "Low Risk",    cls: "cls-1" },
  2: { label: "Prediabetes", cls: "cls-2" },
}

function getConfig(classIndex, classNames) {
  const defaults = CLASS_CONFIG[classIndex] || { label: String(classIndex), cls: "" }
  if (classNames && classNames[classIndex]) {
    return { label: classNames[classIndex], cls: defaults.cls }
  }
  return defaults
}

function fmt(n) {
  if (n === undefined || n === null) return "—"
  return Number(n).toLocaleString()
}

function PreprocessingReport({ summary }) {
  if (!summary) return null
  const {
    original_rows, rows_removed_null, rows_removed_age,
    rows_removed_bp, rows_removed_glucose, rows_removed_lifestyle,
    rows_removed_outliers, total_removed, remaining_rows,
  } = summary

  const rules = [
    { label: "Null values",             value: rows_removed_null },
    { label: "Age out of range",        value: rows_removed_age },
    { label: "Blood pressure",          value: rows_removed_bp },
    { label: "Blood glucose",           value: rows_removed_glucose },
    { label: "Lifestyle inconsistency", value: rows_removed_lifestyle },
    { label: "Medical outliers",        value: rows_removed_outliers },
  ].filter(r => r.value !== undefined)

  const retainedPct = original_rows > 0
    ? ((remaining_rows / original_rows) * 100).toFixed(1) : "—"

  return (
    <div className="preprocess-box">
      <div className="preprocess-heading">Healthcare Validation Report</div>

      <div className="preprocess-chips">
        {[
          { label: "Original",  value: fmt(original_rows),  cls: "" },
          { label: "Removed",   value: fmt(total_removed),  cls: "removed" },
          { label: "Remaining", value: fmt(remaining_rows), cls: "retained" },
          { label: "Retained",  value: retainedPct + "%",   cls: "" },
        ].map(c => (
          <div key={c.label} className={`chip ${c.cls}`}>
            {c.value} <span>{c.label}</span>
          </div>
        ))}
      </div>

      <div className="preprocess-rules">
        {rules.map(rule => (
          <div key={rule.label} className={`rule-badge ${rule.value > 0 ? "has-removed" : ""}`}>
            {rule.label}: {rule.value > 0 ? `−${rule.value}` : "—"}
          </div>
        ))}
      </div>

      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${(remaining_rows / original_rows) * 100}%` }}
        />
      </div>
      <div className="progress-labels">
        <span>0</span>
        <span>{retainedPct}% retained after validation</span>
        <span>{fmt(original_rows)}</span>
      </div>
    </div>
  )
}

const PAGE_SIZE = 100

export default function PredictionTable({ predictions, classNames, preprocessingSummary }) {
  const [page, setPage] = useState(0)
  if (!predictions || predictions.length === 0) return null

  const counts     = predictions.reduce((acc, p) => { acc[p] = (acc[p] || 0) + 1; return acc }, {})
  const totalPages = Math.ceil(predictions.length / PAGE_SIZE)
  const pageData   = predictions.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const uniqueClasses = Object.keys(counts).map(Number).sort()

  return (
    <div className="card">
      <PreprocessingReport summary={preprocessingSummary} />

      <div className="predictions-header">
        <div>
          <div className="predictions-title">Prediction Results</div>
          <div className="predictions-sub">
            {predictions.length.toLocaleString()} predictions
            {classNames ? " · " + classNames.map((n, i) => `${i} = ${n}`).join(" · ") : ""}
          </div>
        </div>
        <div className="prediction-pills">
          {uniqueClasses.map(cls => {
            const c   = getConfig(cls, classNames)
            const pct = ((counts[cls] / predictions.length) * 100).toFixed(1)
            return (
              <div key={cls} className={`pill ${c.cls}`}>
                {c.label} <span style={{ fontWeight: 400, color: "inherit", opacity: 0.7 }}>
                  {counts[cls].toLocaleString()} ({pct}%)
                </span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="predictions-grid">
        {pageData.map((p, i) => {
          const c        = getConfig(p, classNames)
          const isLong   = c.label.length > 4
          const shortLbl = c.label.length > 6 ? c.label.slice(0, 5) + "…" : c.label
          return (
            <div
              key={page * PAGE_SIZE + i}
              title={`Row ${page * PAGE_SIZE + i + 1}: ${c.label}`}
              className={`pred-badge ${c.cls} ${isLong ? "long" : ""}`}
            >
              {shortLbl}
            </div>
          )
        })}
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <span className="page-info">
            Page {page + 1} / {totalPages} · {pageData.length} of {predictions.length.toLocaleString()} shown
          </span>
          <div className="page-btns">
            <button
              className="page-btn"
              disabled={page === 0}
              onClick={() => setPage(p => p - 1)}
            >
              ← Prev
            </button>
            <button
              className="page-btn"
              disabled={page === totalPages - 1}
              onClick={() => setPage(p => p + 1)}
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}