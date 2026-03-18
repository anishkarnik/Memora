import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  getSettings,
  updateSettings,
  startScan,
  Settings,
  CaptionModel,
  EmbeddingModel,
} from "../../api/client";

const CAPTION_MODELS: { value: CaptionModel; label: string; meta: string }[] = [
  {
    value: "moondream2",
    label: "Moondream2",
    meta: "~3.8 GB · Best quality · CPU/GPU",
  },
  {
    value: "florence2",
    label: "Florence-2 Base",
    meta: "~270 MB · Fast · CPU-friendly",
  },
  {
    value: "blip",
    label: "BLIP Base (legacy)",
    meta: "~990 MB · Older model",
  },
];

const EMBEDDING_MODELS: { value: EmbeddingModel; label: string; meta: string }[] = [
  {
    value: "clip",
    label: "CLIP ViT-B/32",
    meta: "~350 MB · 512-d · Fast CPU · Default",
  },
  {
    value: "siglip2",
    label: "SigLIP2 So400m",
    meta: "~4.5 GB · 1152-d · GPU recommended · Better quality",
  },
];

const DEFAULT_SETTINGS: Settings = {
  scan_paths: [],
  auto_scan_enabled: false,
  caption_model: "moondream2",
  embedding_model: "clip",
};

export default function SettingsTab({ onScanStart }: { onScanStart: () => void }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [savedEmbeddingModel, setSavedEmbeddingModel] = useState<EmbeddingModel>("clip");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getSettings().then((s) => {
      const merged = { ...DEFAULT_SETTINGS, ...s };
      setSettings(merged);
      setSavedEmbeddingModel(merged.embedding_model);
    }).finally(() => setLoading(false));
  }, []);

  const addPath = async () => {
    try {
      const selected = await open({ directory: true, multiple: true });
      if (!selected) return;
      const paths = Array.isArray(selected) ? selected : [selected];
      setSettings((s) => ({
        ...s,
        scan_paths: [...new Set([...s.scan_paths, ...paths])],
      }));
    } catch (e) {
      alert("Could not open folder picker: " + String(e));
    }
  };

  const removePath = (path: string) => {
    setSettings((s) => ({
      ...s,
      scan_paths: s.scan_paths.filter((p) => p !== path),
    }));
  };

  const save = async () => {
    setSaving(true);
    try {
      await updateSettings(settings);
      setSavedEmbeddingModel(settings.embedding_model);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } finally {
      setSaving(false);
    }
  };

  const triggerScan = async () => {
    if (settings.scan_paths.length === 0) {
      alert("Add at least one scan path first.");
      return;
    }
    setScanning(true);
    try {
      await updateSettings(settings);
      await startScan(settings.scan_paths);
      onScanStart();
    } catch (e) {
      alert("Failed to start scan: " + String(e));
    } finally {
      setScanning(false);
    }
  };

  const embeddingModelChanged = settings.embedding_model !== savedEmbeddingModel;

  if (loading) {
    return <div className="search-empty" style={{ marginTop: 80 }}><p>Loading settings…</p></div>;
  }

  return (
    <div className="settings-wrap">

      {/* Scan Paths */}
      <div className="settings-section">
        <h2>Scan Paths</h2>
        {settings.scan_paths.length === 0 ? (
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 10 }}>
            No paths added yet.
          </p>
        ) : (
          <ul className="path-list">
            {settings.scan_paths.map((p) => (
              <li key={p}>
                <span>{p}</span>
                <button onClick={() => removePath(p)} title="Remove">×</button>
              </li>
            ))}
          </ul>
        )}
        <button className="btn btn-secondary" onClick={addPath}>
          + Add Folder
        </button>
      </div>

      {/* AI Models */}
      <div className="settings-section">
        <h2>AI Models</h2>
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 14 }}>
          Model changes take effect after restarting the sidecar.
        </p>

        {/* Caption model */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
            Caption Model
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {CAPTION_MODELS.map((m) => (
              <label key={m.value} className="model-option">
                <input
                  type="radio"
                  name="caption_model"
                  value={m.value}
                  checked={settings.caption_model === m.value}
                  onChange={() => setSettings((s) => ({ ...s, caption_model: m.value }))}
                />
                <div>
                  <span style={{ fontWeight: 500, fontSize: 13 }}>{m.label}</span>
                  <span style={{ color: "var(--text-muted)", fontSize: 11, marginLeft: 8 }}>
                    {m.meta}
                  </span>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Embedding model */}
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
            Search Embedding Model
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {EMBEDDING_MODELS.map((m) => (
              <label key={m.value} className="model-option">
                <input
                  type="radio"
                  name="embedding_model"
                  value={m.value}
                  checked={settings.embedding_model === m.value}
                  onChange={() => setSettings((s) => ({ ...s, embedding_model: m.value }))}
                />
                <div>
                  <span style={{ fontWeight: 500, fontSize: 13 }}>{m.label}</span>
                  <span style={{ color: "var(--text-muted)", fontSize: 11, marginLeft: 8 }}>
                    {m.meta}
                  </span>
                </div>
              </label>
            ))}
          </div>
          {embeddingModelChanged && (
            <div style={{
              marginTop: 10, padding: "8px 12px",
              background: "rgba(251, 191, 36, 0.1)",
              border: "1px solid rgba(251, 191, 36, 0.4)",
              borderRadius: 6, fontSize: 12, color: "#fbbf24",
            }}>
              Changing the embedding model requires a full re-scan — existing search
              index is incompatible with the new model's dimensions.
            </div>
          )}
        </div>
      </div>

      {/* Auto-scan toggle */}
      <div className="settings-section">
        <h2>Auto Scan</h2>
        <div className="toggle-row">
          <span style={{ fontSize: 13 }}>Watch paths for new photos and scan automatically</span>
          <label className="toggle">
            <input
              type="checkbox"
              checked={settings.auto_scan_enabled}
              onChange={(e) =>
                setSettings((s) => ({ ...s, auto_scan_enabled: e.target.checked }))
              }
            />
            <span className="toggle-slider" />
          </label>
        </div>
      </div>

      {/* Actions */}
      <div className="settings-section">
        <h2>Actions</h2>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="btn btn-secondary" onClick={save} disabled={saving}>
            {saved ? "Saved!" : saving ? "Saving…" : "Save Settings"}
          </button>
          <button
            className="btn btn-primary"
            onClick={triggerScan}
            disabled={scanning || settings.scan_paths.length === 0}
          >
            {scanning ? "Starting…" : "Start Scan Now"}
          </button>
        </div>
      </div>

      {/* Storage info */}
      <div className="settings-section">
        <h2>Storage</h2>
        <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
          Database and models are stored in <code>~/.memora/</code>
        </p>
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>
          All data stays on your machine — nothing is ever sent to the cloud.
        </p>
      </div>

    </div>
  );
}
