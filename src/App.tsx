import { useEffect, useState } from "react";
import { startSidecar, getScanStatus, cancelScan, preloadModels, getHardwareInfo, getGallery, ScanStatus } from "./api/client";
import PeopleTab from "./components/PeopleTab/PeopleTab";
import GalleryTab from "./components/GalleryTab/GalleryTab";
import SearchTab from "./components/SearchTab/SearchTab";
import SettingsTab from "./components/SettingsTab/SettingsTab";
import "./App.css";

type Tab = "gallery" | "people" | "search" | "settings";

const NAV_ITEMS: { id: Tab; label: string; icon: JSX.Element }[] = [
  {
    id: "gallery",
    label: "Gallery",
    icon: (
      <svg viewBox="0 0 20 20">
        <rect x="2" y="2" width="16" height="16" rx="2" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="7" cy="7" r="1.5" />
        <path d="M2 14l4-4 3 3 4-5 5 6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    id: "search",
    label: "Search",
    icon: (
      <svg viewBox="0 0 20 20">
        <rect x="2" y="3" width="16" height="14" rx="2" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <line x1="2" y1="7" x2="18" y2="7" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    ),
  },
  {
    id: "people",
    label: "People",
    icon: (
      <svg viewBox="0 0 20 20">
        <circle cx="7" cy="7" r="3" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M1 17c0-3.3 2.7-6 6-6s6 2.7 6 6" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="14" cy="6" r="2.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M13 11c1-.6 2-.9 3-.9 2.8 0 5 2.2 5 5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    ),
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("gallery");
  const [sidecarReady, setSidecarReady] = useState(false);
  const [sidecarError, setSidecarError] = useState<string | null>(null);
  const [scanStatus, setScanStatus] = useState<ScanStatus | null>(null);
  const [hasCuda, setHasCuda] = useState<boolean | null>(null);
  const [totalPhotos, setTotalPhotos] = useState(0);

  useEffect(() => {
    startSidecar()
      .then(() => {
        setSidecarReady(true);
        preloadModels().catch(() => {});
        getHardwareInfo()
          .then((hw) => setHasCuda(hw.has_cuda))
          .catch(() => {});
        getGallery(1, 1)
          .then((g) => setTotalPhotos(g.total))
          .catch(() => {});
      })
      .catch((err) => setSidecarError(String(err)));
  }, []);

  useEffect(() => {
    if (!sidecarReady) return;
    const poll = async () => {
      try {
        const status = await getScanStatus();
        setScanStatus(status);
        if (status.status === "done" || status.status === "running") {
          getGallery(1, 1).then((g) => setTotalPhotos(g.total)).catch(() => {});
        }
      } catch {
        // ignore
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
        <p>Starting AI engine...</p>
      </div>
    );
  }

  const isScanning =
    scanStatus?.status === "running" || scanStatus?.status === "queued";

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">
            <svg viewBox="0 0 24 24">
              <rect x="3" y="3" width="7" height="7" rx="1" />
              <rect x="14" y="3" width="7" height="7" rx="1" />
              <rect x="3" y="14" width="7" height="7" rx="1" />
              <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
          </div>
          <div className="sidebar-logo-text">
            <span className="sidebar-logo-title">Memora</span>
            <span className="sidebar-logo-subtitle">Studio</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-item${activeTab === item.id ? " active" : ""}`}
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        {/* Scan progress */}
        {isScanning && (
          <div className="sidebar-scan-progress">
            <span>
              Scanning... {scanStatus?.progress_pct ?? 0}%
              {hasCuda != null && (
                <span style={{ opacity: 0.6, marginLeft: 6, fontSize: 10 }}>
                  {hasCuda ? "GPU" : "CPU"}
                </span>
              )}
            </span>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${scanStatus?.progress_pct ?? 0}%` }}
              />
            </div>
            <button
              className="scan-stop-btn"
              onClick={() => cancelScan().catch(() => {})}
            >
              Stop scan
            </button>
          </div>
        )}

        {scanStatus?.status === "error" && (
          <div className="scan-error-bar" title={scanStatus.error_message ?? undefined}>
            Scan failed{scanStatus.error_message ? `: ${scanStatus.error_message}` : ""}
          </div>
        )}

        <div className="sidebar-footer">
          <div className="sidebar-photo-count">
            {totalPhotos.toLocaleString()} photo{totalPhotos !== 1 ? "s" : ""}
          </div>
          <button
            className="sidebar-settings-btn"
            onClick={() => setActiveTab("settings")}
          >
            <svg viewBox="0 0 20 20">
              <circle cx="10" cy="10" r="3" />
              <path d="M10 1v2m0 14v2M1 10h2m14 0h2M3.5 3.5l1.4 1.4m10.2 10.2l1.4 1.4M3.5 16.5l1.4-1.4m10.2-10.2l1.4-1.4" strokeLinecap="round" />
            </svg>
            Settings
          </button>
        </div>
      </aside>

      {/* Main content */}
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
