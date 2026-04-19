import React, { useState } from "react"

const CLIENT_COLORS = ["#2563eb", "#7c3aed", "#059669"]

export default function FLRoundsPanel({ fl }) {
  const [selected, setSelected] = useState(null)
  if (!fl || !fl.rounds) return null

  const { rounds, config, client_profiles } = fl
  const maxAcc = Math.max(...rounds.map(r => r.global_acc))
  const minAcc = Math.min(...rounds.map(r => r.global_acc))

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

      {/* Config badges */}
      <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
        {[
          { label:"Algorithm",   value: config.algorithm },
          { label:"Clients",     value: config.n_clients },
          { label:"Rounds",      value: config.n_rounds },
          { label:"Mu (prox)",   value: config.mu },
          { label:"Local model", value: config.local_model },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background:"#f8fafc", border:"1px solid #e2e8f0",
            borderRadius:8, padding:"8px 12px",
          }}>
            <div style={{ fontSize:10, color:"#64748b", fontWeight:600,
                          textTransform:"uppercase" }}>{label}</div>
            <div style={{ fontSize:13, fontWeight:700, color:"#1e293b" }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Client profiles */}
      <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
        {client_profiles.map((p, i) => (
          <div key={i} style={{
            flex:"1 1 160px",
            border:`2px solid ${CLIENT_COLORS[i]}`,
            borderRadius:10, padding:12,
          }}>
            <div style={{ fontSize:13, fontWeight:700,
                          color: CLIENT_COLORS[i] }}>{p.name}</div>
            <div style={{ fontSize:11, color:"#64748b", marginTop:4 }}>
              {p.n_samples} samples
            </div>
            <div style={{ fontSize:11, color:"#64748b" }}>
              {Object.entries(p.class_dist).map(([k,v]) =>
                `class ${k}: ${v}`
              ).join("  |  ")}
            </div>
          </div>
        ))}
      </div>

      {/* Convergence chart (CSS bars) */}
      <div>
        <div style={{ fontSize:12, fontWeight:600, color:"#64748b",
                      marginBottom:8, textTransform:"uppercase" }}>
          Global model accuracy per round
        </div>
        <div style={{ display:"flex", gap:8, alignItems:"flex-end", height:100 }}>
          {rounds.map(r => {
            const h = minAcc === maxAcc ? 80
              : 20 + ((r.global_acc - minAcc) / (maxAcc - minAcc)) * 70
            const isSelected = selected === r.round
            return (
              <div key={r.round}
                onClick={() => setSelected(isSelected ? null : r.round)}
                style={{ flex:1, cursor:"pointer" }}>
                <div style={{ textAlign:"center", fontSize:10,
                              color:"#64748b", marginBottom:2 }}>
                  {(r.global_acc*100).toFixed(1)}%
                </div>
                <div style={{
                  height: h, borderRadius:"4px 4px 0 0",
                  background: isSelected ? "#7c3aed" : "#818cf8",
                  border: isSelected ? "2px solid #7c3aed" : "none",
                  transition:"height 0.3s",
                }}/>
                <div style={{ textAlign:"center", fontSize:10,
                              color:"#64748b", marginTop:4 }}>R{r.round}</div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Round detail */}
      {selected && (() => {
        const r = rounds.find(x => x.round === selected)
        if (!r) return null
        return (
          <div style={{
            background:"#faf5ff", border:"1px solid #e9d5ff",
            borderRadius:10, padding:14,
          }}>
            <div style={{ fontSize:13, fontWeight:700, color:"#6d28d9",
                          marginBottom:8 }}>
              Round {r.round} detail
            </div>
            <div style={{ fontSize:12, color:"#7c3aed", marginBottom:8 }}>
              Global: acc={r.global_acc}  f1={r.global_f1}  weight_norm={r.weight_norm}
            </div>
            <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
              {r.client_metrics.map((cm, i) => (
                <div key={i} style={{
                  background:"#fff", border:`1px solid ${CLIENT_COLORS[i]}`,
                  borderRadius:8, padding:"8px 12px", flex:"1 1 120px",
                }}>
                  <div style={{ fontSize:11, fontWeight:700,
                                color: CLIENT_COLORS[i] }}>
                    Hospital {i+1}
                  </div>
                  <div style={{ fontSize:11, color:"#64748b" }}>
                    acc={cm.accuracy}  f1={cm.f1}
                  </div>
                  <div style={{ fontSize:10, color:"#94a3b8" }}>
                    n={cm.n_samples}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })()}
    </div>
  )
}