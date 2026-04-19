import React from "react"
import "../pages/dashboard.css"

export default function PredictionPanel({ prediction }) {
  if (!prediction) return null
  const { test_metrics, sample_prediction, model_info } = prediction
  const m = test_metrics
  const s = sample_prediction

  const riskColor = {
    High:   { bg:"#fee2e2", border:"#fecaca", text:"#dc2626" },
    Medium: { bg:"#fef9c3", border:"#fde68a", text:"#854d0e" },
    Low:    { bg:"#dcfce7", border:"#bbf7d0", text:"#15803d" },
  }[s.risk_level] || { bg:"#f8fafc", border:"#e2e8f0", text:"#374151" }

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

      {/* Test set metrics */}
      <div>
        <div style={{ fontSize:12, fontWeight:600, color:"#64748b",
                      marginBottom:8, textTransform:"uppercase" }}>
          Global model — test set performance
        </div>
        <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
          {["accuracy","f1","precision","recall","auc"].map(k => (
            <div key={k} style={{
              flex:"1 1 90px", background:"#f8fafc",
              border:"1px solid #e2e8f0", borderRadius:10, padding:"10px 12px",
            }}>
              <div style={{ fontSize:10, color:"#64748b", fontWeight:600,
                            textTransform:"uppercase" }}>{k}</div>
              <div style={{ fontSize:20, fontWeight:700, color:"#1e293b", marginTop:3 }}>
                {m[k] != null ? (m[k]*100).toFixed(1)+"%" : "—"}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model info */}
      <div style={{ fontSize:12, color:"#64748b" }}>
        Algorithm: <strong>{model_info.algorithm}</strong> &nbsp;|&nbsp;
        Local model: <strong>{model_info.local_model}</strong> &nbsp;|&nbsp;
        Features: <strong>{model_info.n_features}</strong>
      </div>

      {/* Sample prediction */}
      <div>
        <div style={{ fontSize:12, fontWeight:600, color:"#64748b",
                      marginBottom:8, textTransform:"uppercase" }}>
          Sample prediction (patient #1 from test set)
        </div>
        <div style={{
          border:`2px solid ${riskColor.border}`,
          background: riskColor.bg, borderRadius:12, padding:16,
        }}>
          <div style={{ display:"flex", alignItems:"center", gap:12 }}>
            <div>
              <div style={{ fontSize:24, fontWeight:700, color: riskColor.text }}>
                {s.label}
              </div>
              <div style={{ fontSize:13, color: riskColor.text, marginTop:2 }}>
                Risk level: {s.risk_level}
              </div>
            </div>
            <div style={{ marginLeft:"auto", textAlign:"right" }}>
              <div style={{ fontSize:13, color:"#64748b" }}>Probability</div>
              <div style={{ fontSize:28, fontWeight:700, color: riskColor.text }}>
                {(s.probability*100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Probability bar */}
          <div style={{ marginTop:12 }}>
            <div style={{ height:10, background:"#e2e8f0",
                          borderRadius:5, overflow:"hidden" }}>
              <div style={{
                width:`${s.probability*100}%`, height:"100%",
                background: riskColor.text, borderRadius:5,
                transition:"width 0.5s",
              }}/>
            </div>
            <div style={{ display:"flex", justifyContent:"space-between",
                          fontSize:10, color:"#94a3b8", marginTop:4 }}>
              <span>0% (No Diabetes)</span>
              <span>100% (Diabetic)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Confusion matrix */}
      {m.confusion_matrix && (
        <div>
          <div style={{ fontSize:12, fontWeight:600, color:"#64748b",
                        marginBottom:8, textTransform:"uppercase" }}>
            Confusion matrix
          </div>
          <table style={{ borderCollapse:"collapse", fontSize:13 }}>
            <thead>
              <tr>
                <th style={{ padding:"6px 12px", color:"#64748b", fontSize:11 }}></th>
                <th style={{ padding:"6px 12px", color:"#64748b", fontSize:11 }}>Pred: No Diabetes</th>
                <th style={{ padding:"6px 12px", color:"#64748b", fontSize:11 }}>Pred: Diabetic</th>
              </tr>
            </thead>
            <tbody>
              {m.confusion_matrix.map((row, i) => (
                <tr key={i}>
                  <td style={{ padding:"6px 12px", fontWeight:600,
                               color:"#374151", fontSize:11 }}>
                    True: {i === 0 ? "No Diabetes" : "Diabetic"}
                  </td>
                  {row.map((v, j) => (
                    <td key={j} style={{
                      padding:"8px 20px", textAlign:"center",
                      fontWeight:700, fontSize:15,
                      background: i === j ? "#dcfce7" : "#fee2e2",
                      color:      i === j ? "#15803d" : "#dc2626",
                      border:"1px solid #e2e8f0",
                    }}>
                      {v.toLocaleString()}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}