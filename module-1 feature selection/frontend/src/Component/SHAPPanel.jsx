import React from "react"

export default function SHAPPanel({ prediction }) {
  if (!prediction?.shap_values) return null
  const shap = prediction.shap_values
  const maxAbs = Math.max(...shap.map(s => s.mean_abs))

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
      <div style={{ fontSize:12, color:"#64748b", lineHeight:1.6 }}>
        SHAP (SHapley Additive exPlanations) shows how much each quantum
        feature contributed to the diabetes prediction. Positive values
        increase the risk prediction; negative values decrease it.
      </div>

      {shap.map((item) => {
        const isPos   = item.mean_signed >= 0
        const barW    = maxAbs > 0 ? (item.mean_abs / maxAbs) * 100 : 0
        const barColor = isPos ? "#dc2626" : "#2563eb"

        return (
          <div key={item.feature} style={{
            background:"#f8fafc", border:"1px solid #e2e8f0",
            borderRadius:8, padding:"10px 14px",
          }}>
            <div style={{ display:"flex", alignItems:"center",
                          gap:8, marginBottom:6 }}>
              <span style={{
                fontSize:11, fontWeight:700, color:"#64748b",
                minWidth:22, textAlign:"right",
              }}>
                #{item.rank}
              </span>
              <span style={{ fontSize:13, fontWeight:600,
                             color:"#1e293b", flex:1 }}>
                {item.feature}
              </span>
              <span style={{
                background: isPos ? "#fee2e2" : "#eff6ff",
                color:       isPos ? "#dc2626" : "#2563eb",
                borderRadius:6, padding:"2px 8px",
                fontSize:11, fontWeight:700,
              }}>
                {item.mean_signed > 0 ? "+" : ""}{item.mean_signed}
              </span>
              <span style={{ fontSize:11, color:"#64748b", minWidth:80 }}>
                {item.direction}
              </span>
            </div>

            <div style={{ display:"flex", alignItems:"center", gap:8 }}>
              <div style={{ flex:1, height:8, background:"#e2e8f0",
                            borderRadius:4, overflow:"hidden" }}>
                <div style={{
                  width:`${barW}%`, height:"100%",
                  background: barColor, borderRadius:4,
                }}/>
              </div>
              <span style={{ fontSize:11, fontWeight:600, color: barColor,
                             minWidth:44 }}>
                {item.mean_abs}
              </span>
            </div>
          </div>
        )
      })}

      <div style={{ fontSize:11, color:"#94a3b8", marginTop:4 }}>
        Bar width = mean absolute SHAP value (importance magnitude).
        Red = increases diabetes risk. Blue = decreases diabetes risk.
      </div>
    </div>
  )
}