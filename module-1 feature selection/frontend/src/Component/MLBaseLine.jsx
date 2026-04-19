import React from "react"

const COLORS = {
  "Random Forest":       "#2563eb",
  "SVM":                 "#7c3aed",
  "MLP":                 "#0891b2",
  "Logistic Regression": "#059669",
}

function MetricBar({ value, color }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8 }}>
      <div style={{ flex:1, height:8, background:"#e2e8f0", borderRadius:4, overflow:"hidden" }}>
        <div style={{ width:`${value*100}%`, height:"100%", background:color, borderRadius:4 }}/>
      </div>
      <span style={{ fontSize:12, fontWeight:600, color, minWidth:42 }}>
        {(value*100).toFixed(1)}%
      </span>
    </div>
  )
}

export default function MLBaseline({ baseline }) {
  if (!baseline) return null
  const { models, best_model, train_samples, test_samples } = baseline

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
      <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
        {[
          { label:"Train samples", value: train_samples?.toLocaleString() ?? "—" },
          { label:"Test samples",  value: test_samples?.toLocaleString()  ?? "—" },
          { label:"Best model",    value: best_model ?? "—" },
        ].map(({ label, value }) => (
          <div key={label} style={{
            flex:"1 1 140px", background:"#f8fafc",
            border:"1px solid #e2e8f0", borderRadius:10, padding:"10px 14px",
          }}>
            <div style={{ fontSize:11, color:"#64748b", fontWeight:600,
                          textTransform:"uppercase", letterSpacing:"0.04em" }}>{label}</div>
            <div style={{ fontSize:17, fontWeight:700, color:"#1e293b", marginTop:3 }}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
        {Object.entries(models).map(([name, m]) => {
          const color  = COLORS[name] || "#6366f1"
          const isBest = name === best_model
          return (
            <div key={name} style={{
              border:`2px solid ${isBest ? color : "#e2e8f0"}`,
              borderRadius:10, padding:14,
              background: isBest ? `${color}08` : "#fff",
            }}>
              <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:10 }}>
                <span style={{ fontSize:13, fontWeight:700, color:"#1e293b" }}>{name}</span>
                {isBest && (
                  <span style={{
                    background:color, color:"#fff", fontSize:10, fontWeight:700,
                    borderRadius:6, padding:"2px 8px",
                  }}>best</span>
                )}
                <span style={{ marginLeft:"auto", fontSize:12, color:"#64748b" }}>
                  AUC: {m.auc ?? "n/a"}
                </span>
              </div>
              {["accuracy","f1","precision","recall"].map(k => (
                <div key={k} style={{ marginBottom:6 }}>
                  <div style={{ fontSize:11, color:"#64748b", marginBottom:2,
                                textTransform:"capitalize" }}>{k}</div>
                  <MetricBar value={m[k] ?? 0} color={color}/>
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}