import React, { useState, useRef, useEffect } from "react"
import axios from "axios"
import "./dashboard.css"

import MetricsPanel    from "../Component/metricspanel"
import FeatureChart    from "../Component/Featurechart"
import ConfusionMatrix from "../Component/Confusionmatrix"
import PredictionTable from "../Component/PredictionTable"

import QuantumCircuitInfo  from "../Component/QuantumcircuitInfo"
import QuantumFeatureTable from "../Component/QuantumfeatureTable"
import QuantumSimulator    from "../Component/Quantumsimulator"

import MLBaseline      from "../Component/MLBaseLine"
import FLRoundsPanel   from "../Component/FLRoundsPanel"
import FLSimulator     from "../Component/FLsimulator"
import SHAPPanel       from "../Component/SHAPPanel"

const API = "http://localhost:5000"

const DISEASES = [
  { key: "diabetes", label: "Diabetes",       icon: "D", color: "#2563eb" },
  { key: "kidney",   label: "Kidney Disease",  icon: "K", color: "#7c3aed" },
  { key: "heart",    label: "Heart Disease",   icon: "H", color: "#dc2626" },
  { key: "liver",    label: "Liver Disease",   icon: "L", color: "#059669" },
]

const initDiseaseState = () =>
  Object.fromEntries(DISEASES.map(d => [d.key, {
    file: null, m1Results: null, m2Results: null,
    m1Loading: false, m2Loading: false,
    m1Error: null, m2Error: null,
  }]))

function SectionHeading({ num, text, badge, color }) {
  const badgeColors = {
    "Module 1": "#2563eb", "Module 2": "#7c3aed",
    "Module 3": "#059669", "Report":   "#d97706",
  }
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
      <span style={{ fontSize:11, fontWeight:700, color:"var(--primary)",
                     fontVariantNumeric:"tabular-nums" }}>{num}</span>
      <span className="section-heading" style={{ margin:0 }}>{text}</span>
      {badge && (
        <span style={{
          background: color || badgeColors[badge] || "#6366f1",
          color:"#fff", fontSize:10, fontWeight:700,
          borderRadius:6, padding:"2px 8px",
          textTransform:"uppercase", letterSpacing:"0.05em",
        }}>{badge}</span>
      )}
    </div>
  )
}

function LiveLog({ logs, running, done }) {
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ behavior:"smooth" }) }, [logs])
  if (!running && !done && logs.length === 0) return null

  const MC = { M3:"#059669", M3A:"#2563eb", M3B:"#7c3aed", M3C:"#d97706", M3D:"#dc2626" }
  const LI = { success:"v", warn:"!", error:"X", info:">" }

  return (
    <div style={{ background:"#0f172a", borderRadius:12, padding:16,
                  maxHeight:400, overflowY:"auto", border:"1px solid #1e293b", marginTop:12 }}>
      <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:12,
                    paddingBottom:10, borderBottom:"1px solid #1e293b" }}>
        {["#ef4444","#f59e0b","#22c55e"].map(c => (
          <div key={c} style={{ width:10, height:10, borderRadius:"50%", background:c }}/>
        ))}
        <span style={{ marginLeft:8, fontSize:11, color:"#475569", fontFamily:"monospace" }}>
          module3.log {running ? "running..." : "complete"}
        </span>
        {running && <span style={{ marginLeft:"auto", fontSize:11, color:"#22c55e",
                                   fontFamily:"monospace" }}>live</span>}
      </div>
      <div style={{ fontFamily:"monospace" }}>
        {logs.map((ev, i) => {
          if (ev.type === "section") {
            const c = MC[ev.module] || "#94a3b8"
            return (
              <div key={i} style={{ margin:"8px 0 4px", padding:"3px 8px",
                                    background:`${c}20`, borderLeft:`3px solid ${c}`,
                                    borderRadius:"0 4px 4px 0" }}>
                <span style={{ color:c, fontSize:11, fontWeight:700 }}>[{ev.module}]</span>
                <span style={{ color:c, fontSize:11, marginLeft:6 }}>{ev.msg}</span>
              </div>
            )
          }
          if (ev.type === "progress") {
            const c = MC[ev.module] || "#7c3aed"
            return (
              <div key={i} style={{ margin:"3px 0" }}>
                <div style={{ fontSize:11, color:c, marginBottom:2 }}>{ev.label}</div>
                <div style={{ height:5, background:"#1e293b", borderRadius:3 }}>
                  <div style={{ width:`${ev.pct}%`, height:"100%", background:c,
                                borderRadius:3, transition:"width 0.3s" }}/>
                </div>
              </div>
            )
          }
          if (ev.type === "log") {
            const c  = MC[ev.module] || "#94a3b8"
            const lc = ev.level === "success" ? "#22c55e"
                     : ev.level === "warn"    ? "#f59e0b"
                     : ev.level === "error"   ? "#ef4444" : "#94a3b8"
            return (
              <div key={i} style={{ display:"flex", gap:6, padding:"1px 0",
                                    fontSize:11, fontFamily:"monospace" }}>
                <span style={{ color:c, minWidth:28, fontWeight:700 }}>{ev.module}</span>
                <span style={{ color:lc, minWidth:10 }}>{LI[ev.level] || ">"}</span>
                <span style={{ color:lc, flex:1, wordBreak:"break-word" }}>{ev.msg}</span>
              </div>
            )
          }
          return null
        })}
        {running && <div style={{ color:"#22c55e", fontSize:11, marginTop:4 }}>|</div>}
        <div ref={endRef}/>
      </div>
    </div>
  )
}

function UnifiedPredictionPanel({ prediction }) {
  if (!prediction) return null
  const { sample_prediction: sp, test_metrics, model_info, shap_values } = prediction
  if (!sp) return null

  const RC = {
    High:   { bg:"#fee2e2", border:"#fecaca", text:"#dc2626" },
    Medium: { bg:"#fef9c3", border:"#fde68a", text:"#854d0e" },
    Low:    { bg:"#dcfce7", border:"#bbf7d0", text:"#15803d" },
  }
  const rc = RC[sp.overall_risk] || { bg:"#f8fafc", border:"#e2e8f0", text:"#374151" }

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
      <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
        {["accuracy","f1","precision","recall"].map(k => (
          <div key={k} style={{ flex:"1 1 90px", background:"#f8fafc",
                                border:"1px solid #e2e8f0", borderRadius:10, padding:"10px 12px" }}>
            <div style={{ fontSize:10, color:"#64748b", fontWeight:600,
                          textTransform:"uppercase" }}>{k}</div>
            <div style={{ fontSize:20, fontWeight:700, color:"#1e293b", marginTop:3 }}>
              {test_metrics?.[k] != null ? (test_metrics[k]*100).toFixed(1)+"%" : "N/A"}
            </div>
          </div>
        ))}
      </div>

      <div style={{ border:`2px solid ${rc.border}`, background:rc.bg,
                    borderRadius:12, padding:16 }}>
        <div style={{ fontSize:18, fontWeight:700, color:rc.text }}>
          Overall Metabolic Risk: {sp.overall_risk}
        </div>
        <div style={{ fontSize:13, color:rc.text, marginTop:4 }}>
          Primary condition: {sp.primary_icon} {sp.primary_label}
        </div>
      </div>

      <div>
        <div style={{ fontSize:12, fontWeight:600, color:"#64748b",
                      marginBottom:8, textTransform:"uppercase" }}>
          Disease breakdown
        </div>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(2, 1fr)", gap:10 }}>
          {(sp.disease_predictions || []).map(d => (
            <div key={d.disease} style={{
              border:`2px solid ${d.detected ? "#dc2626" : "#e2e8f0"}`,
              borderRadius:10, padding:12,
              background: d.detected ? "#fff1f2" : "#f8fafc",
            }}>
              <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:6 }}>
                <span style={{ fontSize:16 }}>{d.icon}</span>
                <span style={{ fontSize:13, fontWeight:600, color:"#1e293b" }}>
                  {d.disease.charAt(0).toUpperCase() + d.disease.slice(1)}
                </span>
                <span style={{
                  marginLeft:"auto",
                  background: d.risk_level === "High"   ? "#dc2626"
                            : d.risk_level === "Medium" ? "#d97706" : "#059669",
                  color:"#fff", borderRadius:5, padding:"1px 7px",
                  fontSize:10, fontWeight:700,
                }}>{d.risk_level}</span>
              </div>
              <div style={{ fontSize:22, fontWeight:700,
                            color: d.detected ? "#dc2626" : "#64748b" }}>
                {d.probability}%
              </div>
              <div style={{ height:6, background:"#e2e8f0", borderRadius:3,
                            overflow:"hidden", marginTop:6 }}>
                <div style={{ width:`${d.probability}%`, height:"100%",
                              background: d.detected ? "#dc2626" : "#94a3b8",
                              borderRadius:3 }}/>
              </div>
              <div style={{ fontSize:11, color:"#64748b", marginTop:4 }}>
                {d.detected ? d.label : d.neg_label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {shap_values && shap_values.length > 0 && (
        <div>
          <div style={{ fontSize:12, fontWeight:600, color:"#64748b",
                        marginBottom:8, textTransform:"uppercase" }}>
            Top features driving prediction
          </div>
          {shap_values.slice(0,5).map(item => (
            <div key={item.feature} style={{
              background:"#f8fafc", border:"1px solid #e2e8f0",
              borderRadius:8, padding:"8px 12px", marginBottom:6,
              display:"flex", alignItems:"center", gap:10,
            }}>
              <span style={{ fontSize:11, fontWeight:700, color:"#64748b",
                             minWidth:20 }}>#{item.rank}</span>
              <span style={{ fontSize:12, fontWeight:600, color:"#1e293b",
                             flex:1 }}>{item.feature}</span>
              <span style={{
                background: item.mean_signed > 0 ? "#fee2e2" : "#eff6ff",
                color:       item.mean_signed > 0 ? "#dc2626" : "#2563eb",
                borderRadius:5, padding:"1px 7px", fontSize:11, fontWeight:700,
              }}>{item.mean_signed > 0 ? "+" : ""}{item.mean_signed}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ fontSize:12, color:"#64748b" }}>
        Algorithm: {model_info?.algorithm} | Features: {model_info?.n_features} | Classes: {model_info?.n_classes}
      </div>
    </div>
  )
}

function DiseasePipelineCard({ disease, state, onFileChange, onRunM1, onRunM2 }) {
  const { key, label, icon, color } = disease
  const s = state[key]

  return (
    <div style={{ border:`2px solid ${s.m2Results ? color : "#e2e8f0"}`,
                  borderRadius:14, padding:20, background:"#fff" }}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:16 }}>
        <span style={{ width:32, height:32, borderRadius:"50%", background:color,
                       color:"#fff", display:"flex", alignItems:"center",
                       justifyContent:"center", fontSize:13, fontWeight:700 }}>{icon}</span>
        <div>
          <div style={{ fontSize:15, fontWeight:700, color:"#1e293b" }}>{label}</div>
          <div style={{ fontSize:11, color:"#64748b" }}>
            {s.m2Results  ? "Ready for unified FL"
             : s.m1Results ? "Module 1 done - run quantum"
             : s.file      ? "Ready to run Module 1"
             : "Upload dataset to start"}
          </div>
        </div>
        {s.m2Results && (
          <span style={{ marginLeft:"auto", background:color, color:"#fff",
                         borderRadius:6, padding:"3px 10px",
                         fontSize:11, fontWeight:700 }}>Ready</span>
        )}
      </div>

      <div style={{ border:"1px dashed #e2e8f0", borderRadius:8, padding:12,
                    marginBottom:12, cursor:"pointer",
                    background: s.file ? "#f0fdf4" : "#f8fafc",
                    borderColor: s.file ? "#86efac" : "#e2e8f0" }}
        onClick={() => document.getElementById(`file-${key}`).click()}>
        <input id={`file-${key}`} type="file" accept=".csv"
               style={{ display:"none" }}
               onChange={e => onFileChange(key, e.target.files[0])}/>
        <div style={{ fontSize:12, color: s.file ? "#15803d" : "#94a3b8",
                      fontWeight: s.file ? 600 : 400 }}>
          {s.file ? s.file.name : "Click to upload CSV"}
        </div>
      </div>

      <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
        <button disabled={!s.file || s.m1Loading} onClick={() => onRunM1(key)}
          style={{ background: s.m1Results ? "#dcfce7" : color,
                   color: s.m1Results ? "#15803d" : "#fff",
                   border: s.m1Results ? "1px solid #86efac" : "none",
                   borderRadius:8, padding:"8px 16px", fontWeight:600,
                   fontSize:12, cursor:"pointer",
                   opacity: (!s.file || s.m1Loading) ? 0.4 : 1 }}>
          {s.m1Loading ? "Running XGBoost..."
           : s.m1Results ? "Module 1 Complete" : "Run Module 1 (XGBoost)"}
        </button>
        {s.m1Error && <div className="error-box" style={{ fontSize:11 }}>{s.m1Error}</div>}

        <button disabled={!s.m1Results || s.m2Loading} onClick={() => onRunM2(key)}
          style={{ background: s.m2Results ? "#ede9fe" : "#7c3aed",
                   color: s.m2Results ? "#6d28d9" : "#fff",
                   border: s.m2Results ? "1px solid #c4b5fd" : "none",
                   borderRadius:8, padding:"8px 16px", fontWeight:600,
                   fontSize:12, cursor:"pointer",
                   opacity: (!s.m1Results || s.m2Loading) ? 0.4 : 1 }}>
          {s.m2Loading ? "Running quantum..."
           : s.m2Results ? "Module 2 Complete" : "Run Module 2 (Quantum)"}
        </button>
        {s.m2Error && <div className="error-box" style={{ fontSize:11 }}>{s.m2Error}</div>}
      </div>
    </div>
  )
}

function DiseaseResults({ disease, state, sectionOffset }) {
  const { key, label, color } = disease
  const s   = state[key]
  const num = (n) => String(sectionOffset + n).padStart(2, "0")
  if (!s.m1Results) return null

  return (
    <>
      <div className="divider"/>
      <div style={{ padding:"6px 14px", background:`${color}15`,
                    borderLeft:`4px solid ${color}`, borderRadius:"0 8px 8px 0",
                    marginBottom:16, display:"flex", alignItems:"center", gap:8 }}>
        <span style={{ fontSize:14, fontWeight:700, color }}>{label} Results</span>
      </div>

      <section className="section">
        <SectionHeading num={num(0)} text="Preprocessing Results"
                        badge="Module 1" color={color}/>
        <PredictionTable predictions={s.m1Results.predictions}
                         classNames={s.m1Results.class_names}
                         preprocessingSummary={s.m1Results.preprocessing_summary}/>
      </section>

      <div className="divider"/>
      <section className="section">
        <SectionHeading num={num(1)} text="Model Performance"
                        badge="Module 1" color={color}/>
        <MetricsPanel metrics={s.m1Results.metrics}
                      metrics_before={s.m1Results.metrics_before}
                      metrics_after={s.m1Results.metrics_after}/>
      </section>

      <div className="divider"/>
      <section className="section">
        <SectionHeading num={num(2)} text="Feature Importance"
                        badge="Module 1" color={color}/>
        <FeatureChart features={s.m1Results.top_features}/>
      </section>

      <div className="divider"/>
      <section className="section">
        <SectionHeading num={num(3)} text="Confusion Matrix"
                        badge="Module 1" color={color}/>
        <ConfusionMatrix matrix={s.m1Results.confusion_matrix}/>
      </section>

      {s.m2Results && (<>
        <div className="divider"/>
        <section className="section">
          <SectionHeading num={num(4)} text="Quantum Circuit Architecture"
                          badge="Module 2" color="#7c3aed"/>
          <QuantumCircuitInfo circuitInfo={s.m2Results.circuit_info}/>
        </section>

        <div className="divider"/>
        <section className="section">
          <SectionHeading num={num(5)} text="Quantum Feature Vectors"
                          badge="Module 2" color="#7c3aed"/>
          <QuantumFeatureTable quantumSamples={s.m2Results.quantum_features_sample}
                               spread={s.m2Results.spread}
                               qTrainShape={s.m2Results.q_train_shape}
                               qTestShape={s.m2Results.q_test_shape}/>
        </section>

        <div className="divider"/>
        <section className="section">
          <SectionHeading num={num(6)} text="Circuit Simulator"
                          badge="Module 2" color="#7c3aed"/>
          <QuantumSimulator simulation={s.m2Results.simulation}/>
        </section>
      </>)}
    </>
  )
}

export default function Dashboard() {
  const [state,     setState]     = useState(initDiseaseState())
  const [m3Logs,    setM3Logs]    = useState([])
  const [m3Running, setM3Running] = useState(false)
  const [m3Done,    setM3Done]    = useState(false)
  const [m3Error,   setM3Error]   = useState(null)
  const [m3Results, setM3Results] = useState(null)

  const update = (disease, patch) =>
    setState(prev => ({ ...prev, [disease]: { ...prev[disease], ...patch } }))

  const handleFileChange = (disease, file) =>
    update(disease, { file, m1Error: null })

  const runM1 = async (disease) => {
    update(disease, { m1Loading:true, m1Error:null, m1Results:null })
    try {
      const fd = new FormData()
      fd.append("file", state[disease].file)
      const res = await axios.post(`${API}/run-model/${disease}`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      if (res.data.error) update(disease, { m1Error: res.data.error })
      else                update(disease, { m1Results: res.data })
    } catch (e) {
      update(disease, { m1Error: e.response?.data?.error || e.message })
    } finally {
      update(disease, { m1Loading: false })
    }
  }

  const runM2 = async (disease) => {
    update(disease, { m2Loading:true, m2Error:null, m2Results:null })
    try {
      const res = await axios.post(`${API}/run-quantum/${disease}`)
      if (res.data.error) update(disease, { m2Error: res.data.error })
      else                update(disease, { m2Results: res.data })
    } catch (e) {
      update(disease, { m2Error: e.response?.data?.error || e.message })
    } finally {
      update(disease, { m2Loading: false })
    }
  }

  const runM3Unified = async () => {
    setM3Logs([]); setM3Running(true); setM3Done(false)
    setM3Error(null); setM3Results(null)
    try {
      const response = await fetch(`${API}/run-m3-unified`, { method:"POST" })
      if (!response.ok) {
        const err = await response.json()
        setM3Error(err.error || "Server error"); setM3Running(false); return
      }
      const reader  = response.body.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ""
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream:true })
        const lines = buffer.split("\n")
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith("data:")) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          let ev
          try { ev = JSON.parse(raw) } catch { continue }
          if (ev.type === "done")   { setM3Done(true); setM3Running(false); break }
          if (ev.type === "error")  { setM3Error(ev.msg); setM3Running(false); break }
          if (ev.type === "result") { setM3Results(ev.data); continue }
          setM3Logs(prev => [...prev, ev])
        }
      }
    } catch (e) {
      setM3Error(e.message); setM3Running(false)
    }
  }

  const m2ReadyCount = DISEASES.filter(d => state[d.key].m2Results).length
  const canRunM3     = m2ReadyCount >= 2

  return (
    <div style={{ width:"100%", minHeight:"100vh", background:"var(--bg)" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');`}</style>

      <header className="header">
        <div>
          <div className="header-title">Hybrid Multi-Disease Prediction</div>
          <div className="header-sub">
            XGBoost · Quantum Encoding · FedProx FL · Metabolic Syndrome Risk
          </div>
        </div>
      </header>

      <main className="main">

        <section className="section">
          <SectionHeading num="01" text="Disease Pipelines" badge="Module 1"/>
          <div style={{ fontSize:13, color:"#64748b", marginBottom:16 }}>
            Upload CSV for each disease. Run Module 1 and Module 2 per disease.
            Once at least 2 diseases complete Module 2, run the unified federated model.
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(2, 1fr)", gap:16 }}>
            {DISEASES.map(disease => (
              <DiseasePipelineCard key={disease.key} disease={disease} state={state}
                onFileChange={handleFileChange} onRunM1={runM1} onRunM2={runM2}/>
            ))}
          </div>
        </section>

        {canRunM3 && (<>
          <div className="divider"/>
          <section className="section">
            <SectionHeading num="02" text="Unified Federated Learning"
                            badge="Module 3" color="#059669"/>
            <div style={{ fontSize:13, color:"#64748b", marginBottom:16 }}>
              {m2ReadyCount} of 4 diseases ready. The unified model merges all
              quantum features and trains ONE FedProx model across 3 hospital clients.
            </div>
            {!m3Done && (
              <button disabled={m3Running} onClick={runM3Unified}
                style={{ background:"#059669", color:"#fff", border:"none",
                         borderRadius:8, padding:"12px 28px", fontWeight:700,
                         fontSize:14, cursor:"pointer", opacity: m3Running ? 0.6 : 1 }}>
                {m3Running ? "Running unified FedProx..." : "Run Unified Module 3 (All Diseases)"}
              </button>
            )}
            {m3Error && <div className="error-box" style={{ marginTop:12 }}>{m3Error}</div>}
            <LiveLog logs={m3Logs} running={m3Running} done={m3Done}/>
          </section>
        </>)}

        {m3Results && (<>
          <div className="divider"/>
          <section className="section">
            <SectionHeading num="03" text="ML Baseline Comparison"
                            badge="Module 3" color="#059669"/>
            <MLBaseline baseline={m3Results.baseline}/>
          </section>

          <div className="divider"/>
          <section className="section">
            <SectionHeading num="04" text="FL Rounds (FedProx Unified)"
                            badge="Module 3" color="#059669"/>
            <FLRoundsPanel fl={m3Results.fl}/>
          </section>

          <div className="divider"/>
          <section className="section">
            <SectionHeading num="05" text="FL Simulation Trace"
                            badge="Module 3" color="#059669"/>
            <FLSimulator simulation={m3Results.simulation}/>
          </section>

          <div className="divider"/>
          <section className="section">
            <SectionHeading num="06" text="Multi-Disease Prediction Report"
                            badge="Module 3" color="#059669"/>
            <UnifiedPredictionPanel prediction={m3Results.prediction}/>
          </section>
        </>)}

        {DISEASES.map((disease, idx) => (
          <DiseaseResults key={disease.key} disease={disease} state={state}
                          sectionOffset={10 + idx * 10}/>
        ))}

      </main>
    </div>
  )
}