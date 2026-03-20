import React, { useState, useRef } from "react"
import axios from "axios"
import "../pages/dashboard.css"

export default function UploadCSV({ setResults }) {
  const [file, setFile]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [drag, setDrag]       = useState(false)
  const inputRef              = useRef()

  const handleFile = (f) => { setFile(f); setError(null) }

  const runModel = async () => {
    if (!file) { setError("Please select a CSV file first."); return }
    setError(null); setLoading(true)
    try {
      const fd = new FormData()
      fd.append("file", file)
      const res = await axios.post("http://localhost:5000/run-model", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      if (res.data.error) setError(res.data.error)
      else setResults(res.data)
    } catch (e) {
      setError(e.response?.data?.error || e.message || "Unexpected error.")
    } finally {
      setLoading(false)
    }
  }

  const zoneClass = ["upload-zone", drag ? "drag" : "", file ? "has-file" : ""].filter(Boolean).join(" ")

  return (
    <div>
      <div
        className={zoneClass}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault(); setDrag(false)
          const f = e.dataTransfer.files[0]
          if (f) handleFile(f)
        }}
      >
        <div className="upload-icon">{loading ? "⏳" : file ? "📊" : "📂"}</div>

        <div style={{ flex: 1 }}>
          <div className={`upload-name${file ? " ready" : ""}`}>
            {file ? file.name : "Drop CSV here or click to browse"}
          </div>
          <div className="upload-hint">
            {file
              ? `${(file.size / 1024).toFixed(1)} KB · ready to run`
              : "Accepts .csv files · target column = last column"}
          </div>
        </div>

        <button
          className="btn"
          disabled={loading || !file}
          onClick={(e) => { e.stopPropagation(); runModel() }}
        >
          {loading ? "Running…" : "Run Model"}
        </button>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />

      {error && <div className="error-box">⚠ {error}</div>}
    </div>
  )
}