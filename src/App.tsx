import { useEffect, useState } from "react";
import { startSidecar, getScanStatus, cancelScan, preloadModels, getHardwareInfo, ScanStatus } from "./api/client";
import PeopleTab from "./components/PeopleTab/PeopleTab";
import GalleryTab from "./components/GalleryTab/GalleryTab";
import SearchTab from "./components/SearchTab/SearchTab";
import SettingsTab from "./components/SettingsTab/SettingsTab";
import "./App.css";

type Tab = "gallery" | "people" | "search" | "settings";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("gallery");
  const [sidecarReady, setSidecarReady] = useState(false);
  const [sidecarError, setSidecarError] = useState<string | null>(null);
  const [scanStatus, setScanStatus] = useState<ScanStatus | null>(null);
  const [hasCuda, setHasCuda] = useState<boolean | null>(null);

  // Start sidecar on mount, then preload models and fetch hardware info
  useEffect(() => {
    startSidecar()
      .then(() => {
        setSidecarReady(true);
        preloadModels().catch(() => {});
        getHardwareInfo()
          .then((hw) => setHasCuda(hw.has_cuda))
          .catch(() => {});
      })
      .catch((err) => setSidecarError(String(err)));
  }, []);

  // Poll scan status when sidecar is ready
  useEffect(() => {
    if (!sidecarReady) return;
    const poll = async () => {
      try {
        const status = await getScanStatus();
        setScanStatus(status);
      } catch {
        // ignore transient errors
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [sidecarReady]);

  if (sidecarError) {
    return (
      <div className="error-screen">
        <h2>Failed to start AI engine</h2>
        <pre>{sidecarError}</pre>
        <p>Please check that the application is installed correctly.</p>
      </div>
    );
  }

  if (!sidecarReady) {
    return (
      <div className="loading-screen">
        <div className="spinner" />
        <p>Starting AI engine… (first run may take a minute)</p>
      </div>
    );
  }

  const isScanning =
    scanStatus?.status === "running" || scanStatus?.status === "queued";

  return (
    <div className="app">
      {/* Top nav */}
      <header className="app-header">
        <span className="app-logo">Memora</span>
        <nav className="tab-nav">
          {(["gallery", "people", "search", "settings"] as Tab[]).map((t) => (
            <button
              key={t}
              className={`tab-btn${activeTab === t ? " active" : ""}`}
              onClick={() => setActiveTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <div className="scan-progress-bar">
          {isScanning ? (
            <>
              <span>
                {hasCuda != null && (
                  <span style={{ fontSize: 10, marginRight: 6, opacity: 0.7 }}>
                    {hasCuda ? "GPU" : "CPU"}
                  </span>
                )}
                Scanning... {scanStatus?.progress_pct ?? 0}%
              </span>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${scanStatus?.progress_pct ?? 0}%` }}
                />
              </div>
            </>
          ) : (
            <span style={{ color: "var(--text-muted)" }}>
              {scanStatus?.status === "done" ? `Last scan: ${scanStatus.total_files} files` : ""}
            </span>
          )}
          <button
            className="btn btn-danger"
            style={{ padding: "3px 10px", fontSize: 11 }}
            disabled={!isScanning}
            onClick={() => cancelScan().catch(() => {})}
          >
            Stop
          </button>
        </div>
        {scanStatus?.status === "error" && (
          <div className="scan-error-bar" title={scanStatus.error_message ?? undefined}>
            Scan failed{scanStatus.error_message ? `: ${scanStatus.error_message}` : ""}
          </div>
        )}
      </header>

      {/* Tab content */}
      <main className="app-main">
        {activeTab === "gallery" && <GalleryTab />}
        {activeTab === "people" && <PeopleTab />}
        {activeTab === "search" && <SearchTab />}
        {activeTab === "settings" && (
          <SettingsTab
            onScanStart={() => setScanStatus({ status: "queued", progress_pct: 0, total_files: 0, processed_files: 0, current_file: null, started_at: null, completed_at: null })}
          />
        )}
      </main>
    </div>
  );
}
