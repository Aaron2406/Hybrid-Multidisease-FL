import React, { useState } from "react"

const CLIENT_COLORS = ["#2563eb", "#7c3aed", "#059669"]

export default function FLSimulator({ simulation }) {
  const [step, setStep] = useState(0)
  if (!simulation || !simulation.round_steps) return null

  const { round_steps, convergence, fedprox_explanation, client_progression } = simulation
  const current = round_steps[step]

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

      {/* FedProx explanation */}
      <div style={{
        background:"#eff6ff", border:"1px solid #bfdbfe",
        borderRadius:10, padding:14,
      }}>
        <div style={{ fontSize:13, fontWeight:700, color:"#1d4ed8", marginBottom:6 }}>
          Why FedProx?
        </div>
        <div style={{ fontSize:12, color:"#1e40af", lineHeight:1.6 }}>
          {fedprox_explanation.why_fedprox}
        </div>
        <div style={{ fontSize:12, color:"#3b82f6", marginTop:8 }}>
          mu={fedprox_explanation.mu} — {fedprox_explanation.proximal_effect}
        </div>
      </div>

      {/* Step navigator */}
      <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
        {round_steps.map((s, i) => (
          <button key={i}
            onClick={() => setStep(i)}
            style={{
              padding:"6px 14px", borderRadius:8, border:"none",
              background: i === step ? "#7c3aed" : "#f1f5f9",
              color:       i === step ? "#fff"     : "#374151",
              fontWeight:  i === step ? 700 : 400,
              cursor:"pointer", fontSize:12,
            }}>
            Round {s.round}
          </button>
        ))}
      </div>

      {/* Current round detail */}
      {current && (
        <div style={{
          border:"2px solid #7c3aed", borderRadius:12, padding:16,
          background:"#faf5ff",
        }}>
          <div style={{ fontSize:14, fontWeight:700, color:"#1e293b", marginBottom:8 }}>
            Round {current.round} — what happened
          </div>
          <div style={{ fontSize:13, color:"#475569", lineHeight:1.7,
                        marginBottom:12 }}>
            {current.description}
          </div>

          <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:12 }}>
            <span style={{ background:"#dcfce7", color:"#15803d",
                           borderRadius:6, padding:"3px 10px", fontSize:12, fontWeight:600 }}>
              Global acc = {current.global_acc}
            </span>
            <span style={{ background:"#eff6ff", color:"#1d4ed8",
                           borderRadius:6, padding:"3px 10px", fontSize:12, fontWeight:600 }}>
              Global F1 = {current.global_f1}
            </span>
            <span style={{ background:"#faf5ff", color:"#7c3aed",
                           borderRadius:6, padding:"3px 10px", fontSize:12, fontWeight:600 }}>
              Weight norm = {current.weight_norm}
            </span>
          </div>

          <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
            {current.client_metrics.map((cm, i) => (
              <div key={i} style={{
                flex:"1 1 130px", background:"#fff",
                border:`2px solid ${CLIENT_COLORS[i]}`,
                borderRadius:8, padding:"10px 12px",
              }}>
                <div style={{ fontSize:12, fontWeight:700,
                              color: CLIENT_COLORS[i] }}>Hospital {i+1}</div>
                <div style={{ fontSize:12, color:"#374151", marginTop:4 }}>
                  acc = {cm.accuracy}
                </div>
                <div style={{ fontSize:12, color:"#64748b" }}>
                  f1  = {cm.f1}
                </div>
                <div style={{ fontSize:11, color:"#94a3b8" }}>
                  n   = {cm.n_samples}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Convergence status */}
      <div style={{
        background: convergence.status.includes("converged") ? "#f0fdf4" : "#fefce8",
        border: `1px solid ${convergence.status.includes("converged") ? "#bbf7d0" : "#fde68a"}`,
        borderRadius:10, padding:12,
        fontSize:13, fontWeight:600,
        color: convergence.status.includes("converged") ? "#15803d" : "#854d0e",
      }}>
        Convergence: {convergence.status}
      </div>

      {/* Per-client accuracy history */}
      {client_progression && (
        <div>
          <div style={{ fontSize:12, fontWeight:600, color:"#64748b",
                        marginBottom:8, textTransform:"uppercase" }}>
            Per-client accuracy history
          </div>
          {client_progression.map((cp, i) => (
            <div key={i} style={{ marginBottom:8 }}>
              <div style={{ fontSize:12, color: CLIENT_COLORS[i],
                            fontWeight:600, marginBottom:4 }}>
                {cp.name}  (final acc={cp.final_acc})
              </div>
              <div style={{ display:"flex", gap:6 }}>
                {cp.acc_history.map((acc, ri) => (
                  <div key={ri} style={{
                    flex:1, textAlign:"center",
                    background: CLIENT_COLORS[i] + "20",
                    border:`1px solid ${CLIENT_COLORS[i]}40`,
                    borderRadius:6, padding:"4px 0",
                    fontSize:11, color: CLIENT_COLORS[i], fontWeight:600,
                  }}>
                    {(acc*100).toFixed(1)}%
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}