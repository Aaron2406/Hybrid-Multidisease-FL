import React, { useState } from "react"
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend,
} from "chart.js"
import { Bar } from "react-chartjs-2"

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

// VIBGYOR neon palette — fully vivid, dark-theme optimised
const PALETTE = [
  "rgba(139,92,246,0.9)",    // Violet
  "rgba(106,0,255,0.9)",     // Indigo
  "rgba(0,212,255,0.9)",     // Blue
  "rgba(0,255,136,0.9)",     // Green
  "rgba(184,255,0,0.9)",     // Yellow-Green
  "rgba(255,214,0,0.9)",     // Yellow
  "rgba(255,109,0,0.9)",     // Orange
  "rgba(255,0,110,0.9)",     // Red
]

const MODES = [
  { id: "horizon", label: "Horizon" },
  { id: "arc",     label: "Arc" },
]

// ─── Horizon Chart ────────────────────────────────────────────────────────────
function HorizonChart({ features }) {
  const sorted = [...features].sort((a, b) => b.score - a.score)
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {sorted.map((f, i) => {
        const pct   = ((f.score - 0.3) / 0.7) * 100
        const color = PALETTE[i % PALETTE.length]
        return (
          <div key={f.feature}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
              <span style={{
                fontFamily: "'Share Tech Mono', monospace", fontSize: "10px",
                color: color,
                background: color.replace(/[\d.]+\)$/, "0.12)"),
                border: `1px solid ${color.replace(/[\d.]+\)$/, "0.35)")}`,
                borderRadius: "4px", padding: "2px 8px", flexShrink: 0,
                boxShadow: `0 0 6px ${color.replace(/[\d.]+\)$/, "0.3)")}`,
              }}>#{i + 1}</span>
              <span style={{
                fontFamily: "'Exo 2', sans-serif", fontSize: "13px", color: "#ffffff",
                flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                fontWeight: "600",
              }}>{f.feature}</span>
              <span style={{
                fontFamily: "'Share Tech Mono', monospace", fontSize: "12px",
                color: color, flexShrink: 0,
                textShadow: `0 0 8px ${color.replace(/[\d.]+\)$/, "0.6)")}`,
              }}>{f.score.toFixed(4)}</span>
            </div>
            <div style={{
              position: "relative", height: "8px",
              background: "rgba(255,255,255,0.06)", borderRadius: "4px", overflow: "hidden",
              border: `1px solid rgba(255,255,255,0.08)`,
            }}>
              <div style={{
                position: "absolute", top: 0, left: 0, height: "100%",
                width: `${pct}%`, background: color,
                borderRadius: "4px",
                boxShadow: `0 0 10px ${color.replace(/[\d.]+\)$/, "0.5)")}`,
              }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: "4px" }}>
              {["0.30", "0.47", "0.65", "0.82", "1.00"].map(t => (
                <span key={t} style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: "9px", color: "#6a5acd" }}>{t}</span>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─── Arc Gauge Grid ───────────────────────────────────────────────────────────
function ArcGaugeGrid({ features, onSelect }) {
  const sorted = [...features].sort((a, b) => b.score - a.score)
  const r = 54, cx = 90, cy = 76
  const minScore = 0.3

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
      gap: "14px",
    }}>
      {sorted.map((f, i) => {
        const color = PALETTE[i % PALETTE.length]
        const pct   = Math.max(0, Math.min(1, (f.score - minScore) / (1 - minScore)))
        const angle = pct * Math.PI
        const x1 = cx - r, y1 = cy
        const x2 = cx + r * Math.cos(Math.PI - angle)
        const y2 = cy - r * Math.sin(angle)
        const large = angle > Math.PI ? 1 : 0

        return (
          <div
            key={f.feature}
            onClick={() => onSelect(f)}
            style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: "10px",
              padding: "24px 18px 18px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              cursor: "pointer",
              transition: "all 0.2s",
              position: "relative",
              boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = color.replace(/[\d.]+\)$/, "0.1)")
              e.currentTarget.style.borderColor = color.replace(/[\d.]+\)$/, "0.5)")
              e.currentTarget.style.boxShadow = `0 0 20px ${color.replace(/[\d.]+\)$/, "0.25)")}`
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = "rgba(255,255,255,0.03)"
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"
              e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.3)"
            }}
          >
            <div style={{
              position: "absolute", top: "12px", right: "14px",
              fontSize: "10px", fontFamily: "'Share Tech Mono', monospace",
              color: color, letterSpacing: "0.06em",
              textShadow: `0 0 6px ${color.replace(/[\d.]+\)$/, "0.5)")}`,
            }}>
              #{i + 1}
            </div>

            <svg viewBox="0 0 180 92" style={{ width: "100%", maxWidth: "180px", overflow: "visible" }}>
              <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
                fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="7" strokeLinecap="round" />
              {angle > 0 && (
                <path d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`}
                  fill="none" stroke={color} strokeWidth="7" strokeLinecap="round"
                  style={{ filter: `drop-shadow(0 0 5px ${color})` }} />
              )}
              <circle cx={x2} cy={y2} r="6" fill={color} opacity={angle > 0 ? 1 : 0}
                style={{ filter: `drop-shadow(0 0 5px ${color})` }} />
              <text x={cx} y={cy - 14} textAnchor="middle" fill={color}
                style={{ fontFamily: "'Exo 2', sans-serif", fontSize: "20px", fontWeight: 700,
                  filter: `drop-shadow(0 0 4px ${color})` }}>
                {f.score.toFixed(3)}
              </text>
              <text x={cx} y={cy + 4} textAnchor="middle" fill="#6a5acd"
                style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: "10px", letterSpacing: "0.1em" }}>
                MERIT
              </text>
            </svg>

            <div style={{
              fontFamily: "'Exo 2', sans-serif",
              fontSize: "13px",
              color: "#e2e8f0",
              textAlign: "center",
              marginTop: "8px",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              width: "100%",
              padding: "0 6px",
              fontWeight: 600,
            }}>
              {f.feature}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─── Comparison Modal ─────────────────────────────────────────────────────────
function ComparisonModal({ selected, allFeatures, onClose }) {
  const sorted   = [...allFeatures].sort((a, b) => b.score - a.score)
  const selIdx   = allFeatures.findIndex(f => f.feature === selected.feature)
  const selColor = PALETTE[selIdx % PALETTE.length]
  const others   = sorted.filter(f => f.feature !== selected.feature)

  const barData = {
    labels: others.map(f => f.feature),
    datasets: [
      {
        label: selected.feature,
        data: others.map(() => selected.score),
        backgroundColor: selColor.replace(/[\d.]+\)$/, "0.35)"),
        borderColor: selColor,
        borderWidth: 1,
        borderRadius: 4,
        borderSkipped: false,
      },
      {
        label: "Compared Feature",
        data: others.map(f => f.score),
        backgroundColor: others.map((f) => {
          const idx = allFeatures.findIndex(a => a.feature === f.feature)
          return PALETTE[idx % PALETTE.length].replace(/[\d.]+\)$/, "0.25)")
        }),
        borderColor: others.map((f) => {
          const idx = allFeatures.findIndex(a => a.feature === f.feature)
          return PALETTE[idx % PALETTE.length]
        }),
        borderWidth: 1,
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  }

  const barOptions = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: "top",
        labels: {
          color: "#e2e8f0",
          font: { family: "'Share Tech Mono', monospace", size: 11 },
          boxWidth: 10, padding: 14,
        },
      },
      tooltip: {
        backgroundColor: "#0f0f2a",
        borderColor: "#2a1a5e",
        borderWidth: 1,
        titleColor: "#ffffff",
        bodyColor: "#a78bfa",
        padding: 10,
        titleFont: { family: "'Share Tech Mono', monospace", size: 11 },
        bodyFont:  { family: "'Share Tech Mono', monospace", size: 12 },
        cornerRadius: 6,
        callbacks: { label: ctx => `  ${ctx.dataset.label}: ${ctx.raw.toFixed(4)}` },
      },
    },
    scales: {
      x: {
        min: 0.25, max: 1.05,
        grid: { color: "rgba(255,255,255,0.05)" },
        ticks: { color: "#6a5acd", font: { family: "'Share Tech Mono', monospace", size: 10 } },
        border: { color: "rgba(255,255,255,0.08)" },
      },
      y: {
        grid: { display: false },
        ticks: { color: "#e2e8f0", font: { family: "'Exo 2', sans-serif", size: 11 } },
        border: { color: "rgba(255,255,255,0.08)" },
      },
    },
  }

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 9999,
        background: "rgba(5,5,20,0.75)",
        backdropFilter: "blur(8px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "24px",
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "#0f0f2a",
          border: "1px solid #2a1a5e",
          borderRadius: "12px",
          width: "100%",
          maxWidth: "800px",
          maxHeight: "88vh",
          overflowY: "auto",
          boxShadow: `0 20px 60px rgba(106,0,255,0.3), 0 0 0 1px rgba(139,92,246,0.15)`,
          padding: "26px",
          fontFamily: "'Share Tech Mono', monospace",
          position: "relative",
        }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "22px" }}>
          <div>
            <div style={{ fontSize: "10px", color: "#6a5acd", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "5px" }}>
              Feature Comparison
            </div>
            <div style={{ fontSize: "15px", fontWeight: 700, color: selColor, fontFamily: "'Exo 2', sans-serif",
              textShadow: `0 0 12px ${selColor.replace(/[\d.]+\)$/, "0.6)")}` }}>
              {selected.feature}
            </div>
            <div style={{ fontSize: "11px", color: "#a78bfa", marginTop: "3px" }}>
              Merit score: {selected.score.toFixed(4)}
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              padding: "5px 12px",
              borderRadius: "5px",
              border: "1px solid rgba(106,0,255,0.4)",
              background: "rgba(106,0,255,0.12)",
              color: "#a78bfa",
              cursor: "pointer",
              fontSize: "11px",
              fontFamily: "'Share Tech Mono', monospace",
              letterSpacing: "0.06em",
              transition: "all 0.2s",
            }}
          >
            ESC
          </button>
        </div>

        <div style={{ height: `${Math.max(220, others.length * 44)}px` }}>
          <Bar data={barData} options={barOptions} />
        </div>
      </div>
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function FeatureChart({ features }) {
  const [mode, setMode]         = useState("arc")
  const [selected, setSelected] = useState(null)

  if (!features || features.length === 0) return null

  const total  = features.reduce((s, f) => s + f.score, 0)
  const sorted = [...features].sort((a, b) => b.score - a.score)
  const best   = sorted[0]

  return (
    <div style={{
      background: "#0f0f2a",
      border: "1px solid #2a1a5e",
      borderRadius: "10px",
      overflow: "hidden",
      fontFamily: "'Share Tech Mono', monospace",
      boxShadow: "0 0 40px rgba(106,0,255,0.15)",
    }}>
      {/* Top bar */}
      <div style={{
        padding: "16px 22px",
        borderBottom: "1px solid #2a1a5e",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        flexWrap: "wrap",
        background: "linear-gradient(135deg, rgba(106,0,255,0.15), rgba(0,212,255,0.06))",
      }}>
        <div style={{ display: "flex", gap: "8px", flex: 1, flexWrap: "wrap" }}>
          {[
            { label: "Top Feature",  value: best.feature,                         color: "#00d4ff" },
            { label: "Best Merit",   value: best.score.toFixed(4),                color: "#8b5cf6" },
            { label: "Avg Merit",    value: (total / features.length).toFixed(4), color: "#ffd600" },
          ].map(chip => (
            <div key={chip.label} style={{
              padding: "5px 10px",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "5px",
            }}>
              <div style={{ fontSize: "9px", color: "#6a5acd", letterSpacing: "0.12em", textTransform: "uppercase" }}>{chip.label}</div>
              <div style={{ fontSize: "12px", color: chip.color, marginTop: "2px", letterSpacing: "0.02em",
                textShadow: `0 0 8px ${chip.color}` }}>{chip.value}</div>
            </div>
          ))}
        </div>

        {/* Mode switcher */}
        <div style={{
          display: "flex",
          background: "rgba(255,255,255,0.05)",
          border: "1px solid rgba(106,0,255,0.35)",
          borderRadius: "5px",
          padding: "3px",
        }}>
          {MODES.map(m => (
            <button key={m.id} onClick={() => setMode(m.id)} style={{
              padding: "4px 12px",
              background: mode === m.id
                ? "linear-gradient(135deg, rgba(106,0,255,0.6), rgba(0,212,255,0.4))"
                : "transparent",
              border: "none",
              borderRadius: "3px",
              color: mode === m.id ? "#ffffff" : "#6a5acd",
              fontSize: "10px",
              fontFamily: "'Share Tech Mono', monospace",
              cursor: "pointer",
              transition: "all 0.15s",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              fontWeight: mode === m.id ? 700 : 400,
              boxShadow: mode === m.id ? "0 0 10px rgba(106,0,255,0.4)" : "none",
            }}>
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {mode === "arc" && (
        <div style={{
          padding: "6px 22px",
          background: "rgba(0,212,255,0.06)",
          borderBottom: "1px solid rgba(0,212,255,0.2)",
          fontSize: "10px",
          color: "#00d4ff",
          letterSpacing: "0.06em",
          textShadow: "0 0 8px rgba(0,212,255,0.4)",
        }}>
          Click any card to compare against all other features
        </div>
      )}

      <div style={{ padding: "22px", background: "transparent" }}>
        {mode === "horizon" && <HorizonChart features={features} />}
        {mode === "arc"     && <ArcGaugeGrid features={features} onSelect={setSelected} />}
      </div>

      {/* Legend */}
      <div style={{
        padding: "10px 22px",
        borderTop: "1px solid #2a1a5e",
        display: "flex",
        flexWrap: "wrap",
        gap: "8px 16px",
        background: "rgba(106,0,255,0.06)",
      }}>
        {sorted.map((f, i) => (
          <div key={f.feature} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <div style={{
              width: "6px", height: "6px", borderRadius: "2px",
              background: PALETTE[i % PALETTE.length],
              flexShrink: 0,
              boxShadow: `0 0 6px ${PALETTE[i % PALETTE.length]}`,
            }} />
            <span style={{ fontSize: "10px", color: "#a78bfa", letterSpacing: "0.02em" }}>{f.feature}</span>
            <span style={{ fontSize: "10px", color: PALETTE[i % PALETTE.length], letterSpacing: "0.02em",
              textShadow: `0 0 6px ${PALETTE[i % PALETTE.length].replace(/[\d.]+\)$/, "0.5)")}` }}>
              {((f.score / total) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      {selected && (
        <ComparisonModal selected={selected} allFeatures={features} onClose={() => setSelected(null)} />
      )}
    </div>
  )
}