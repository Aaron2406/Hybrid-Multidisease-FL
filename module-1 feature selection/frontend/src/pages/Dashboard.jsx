import React, { useState } from "react"
import UploadCSV      from "../Component/UploadCSV"
import MetricsPanel   from "../Component/MetricsPanel"
import FeatureChart   from "../Component/FeatureChart"
import ConfusionMatrix from "../Component/ConfusionMatrix"
import PredictionTable from "../Component/PredictionTable"
import "./dashboard.css"

function SectionHeading({ num, text }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
      <span style={{ fontSize: 11, fontWeight: 700, color: "var(--primary)", fontVariantNumeric: "tabular-nums" }}>
        {num}
      </span>
      <span className="section-heading" style={{ margin: 0 }}>{text}</span>
    </div>
  )
}

export default function Dashboard() {
  const [results, setResults] = useState(null)

  return (
    <div style={{ width: "100%", minHeight: "100vh", background: "var(--bg)" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
      `}</style>

      <header className="header">
        <div>
          <div className="header-title">Disease Prediction Dashboard</div>
          <div className="header-sub">XGBoost · WEKA ClassifierAttributeEval · 10-Fold CV</div>
        </div>
        {/* <span className="header-badge">XGBoost v3</span> */}
      </header>

      <main className="main">
        <section className="section">
          <SectionHeading num="01" text="Upload Dataset" />
          <UploadCSV setResults={setResults} />
        </section>

        {results && (
          <>
            <div className="divider" />
            <section className="section">
              <SectionHeading num="02" text="Preprocessing Results" />
              <PredictionTable
                predictions={results.predictions}
                classNames={results.class_names}
                preprocessingSummary={results.preprocessing_summary}
              />
            </section>

            <div className="divider" />
            <section className="section">
              <SectionHeading num="03" text="Model Performance" />
              <MetricsPanel
                metrics={results.metrics}
                metrics_before={results.metrics_before}
                metrics_after={results.metrics_after}
              />
            </section>

            <div className="divider" />
            <section className="section">
              <SectionHeading num="04" text="Feature Importance" />
              <FeatureChart features={results.top_features} />
            </section>

            <div className="divider" />
            <section className="section">
              <SectionHeading num="05" text="Confusion Matrix" />
              <ConfusionMatrix matrix={results.confusion_matrix} />
            </section>

            <div className="divider" />
            
          </>
        )}
      </main>
    </div>
  )
}