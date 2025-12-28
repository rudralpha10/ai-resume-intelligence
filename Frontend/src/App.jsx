import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [files, setFiles] = useState([]);
  const [jd, setJd] = useState("");
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");

  // ----------------------------
  // Upload ONE resume
  // ----------------------------
  const uploadOne = async () => {
    if (!files.length) {
      alert("Select a resume first");
      return;
    }

    const fd = new FormData();
    fd.append("file", files[0]);

    try {
      setStatus("Uploading resume...");
      const res = await fetch(`${API_BASE}/resume/upload`, {
        method: "POST",
        body: fd,
      });

      const data = await res.json();
      alert(data.message || "Resume uploaded");
    } catch {
      alert("Upload failed");
    }

    setStatus("");
  };

  // ----------------------------
  // Upload MULTIPLE resumes
  // ----------------------------
  const uploadMultiple = async () => {
    if (!files.length) {
      alert("Select resumes first");
      return;
    }

    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));

    try {
      setStatus("Uploading resumes...");
      const res = await fetch(`${API_BASE}/resumes/upload`, {
        method: "POST",
        body: fd,
      });

      const data = await res.json();
      alert(data.message || "Resumes uploaded");
    } catch {
      alert("Upload failed");
    }

    setStatus("");
  };

  // ----------------------------
  // Match Job Description
  // ----------------------------
  const matchJD = async () => {
    if (!jd.trim()) {
      alert("Paste a Job Description");
      return;
    }

    setLoading(true);
    setMatches([]);
    setStatus("Processing AI embeddings...");

    try {
      const res = await fetch(`${API_BASE}/match`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: jd, top_k: 5 }),
      });

      const data = await res.json();

      if (!data.matches || data.matches.length === 0) {
        alert("No resumes found. Upload resumes first.");
      } else {
        setMatches(data.matches);
      }
    } catch {
      alert("Matching failed");
    }

    setLoading(false);
    setStatus("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#020617] via-[#020b2d] to-black flex items-center justify-center text-white">
      <div className="relative w-[560px] rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-[0_0_60px_rgba(59,130,246,0.15)] p-8">

        <div className="absolute inset-0 -z-10 rounded-2xl bg-gradient-to-r from-blue-500/20 via-indigo-500/10 to-cyan-500/20 blur-2xl" />

        <h1 className="text-3xl font-bold text-center mb-2 tracking-wide">
          🚀 AI Resume Intelligence
        </h1>
        <p className="text-center text-gray-400 mb-6">
          Enterprise-grade Resume Screening Engine
        </p>

        <label className="block mb-3 text-sm text-gray-300">
          Upload Resumes
        </label>
        <input
          type="file"
          multiple
          onChange={(e) => setFiles([...e.target.files])}
          className="w-full mb-4 text-sm text-gray-300
          file:mr-4 file:py-2 file:px-4
          file:rounded-lg file:border-0
          file:bg-blue-600 file:text-white
          hover:file:bg-blue-700
          cursor-pointer"
        />

        <div className="flex gap-3 mb-6">
          <button
            onClick={uploadOne}
            className="flex-1 bg-blue-600 hover:bg-blue-700 transition rounded-lg py-2 font-medium"
          >
            Upload One
          </button>
          <button
            onClick={uploadMultiple}
            className="flex-1 bg-indigo-600 hover:bg-indigo-700 transition rounded-lg py-2 font-medium"
          >
            Upload Multiple
          </button>
        </div>

        <label className="block mb-2 text-sm text-gray-300">
          Job Description
        </label>
        <textarea
          rows={4}
          placeholder="Paste job description here..."
          value={jd}
          onChange={(e) => setJd(e.target.value)}
          className="w-full p-3 mb-5 rounded-lg bg-black/40 border border-white/10 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <button
          onClick={matchJD}
          className="w-full bg-green-600 hover:bg-green-700 transition rounded-lg py-3 font-semibold tracking-wide"
        >
          Match Job Description
        </button>

        {(loading || status) && (
          <p className="text-center mt-4 text-blue-400 animate-pulse">
            {status || "Processing..."}
          </p>
        )}

        {matches.length > 0 && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-3">
              Matching Results
            </h2>

            <div className="space-y-3">
              {matches.map((m, i) => {
                // ✅ NORMALIZATION (IMPORTANT FIX)
                const normalizedScore = Math.max(
                  0,
                  Math.min(100, m.score * 100)
                );

                return (
                  <div
                    key={i}
                    className="bg-white/5 border border-white/10 rounded-lg p-3 hover:bg-white/10 transition"
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium">{m.resume_id}</span>
                      <span className="text-blue-400 font-semibold">
                        {normalizedScore.toFixed(1)}%
                      </span>
                    </div>

                    <div className="h-2 bg-gray-700 rounded">
                      <div
                        className="h-2 bg-gradient-to-r from-green-400 to-blue-500 rounded transition-all duration-500"
                        style={{ width: `${normalizedScore}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {!loading && matches.length === 0 && (
          <p className="text-center mt-6 text-gray-500">
            Upload resumes and match with a job description
          </p>
        )}
      </div>
    </div>
  );
}
