import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  getGallery,
  getMedia,
  getSettings,
  updateSettings,
  startScan,
  MediaSummary,
  MediaDetail,
  ensureSidecarPort,
} from "../../api/client";

function ImgProxy({ relUrl, alt, className }: { relUrl: string; alt: string; className?: string }) {
  const [src, setSrc] = useState<string>("");
  useEffect(() => {
    ensureSidecarPort().then((port) => {
      const sep = relUrl.includes("?") ? "&" : "?";
      setSrc(`http://127.0.0.1:${port}${relUrl}${sep}_t=${Date.now()}`);
    });
  }, [relUrl]);
  return <img src={src} alt={alt} className={className} loading="lazy" />;
}

function Lightbox({ mediaId, onClose }: { mediaId: number; onClose: () => void }) {
  const [detail, setDetail] = useState<MediaDetail | null>(null);
  const [imgSrc, setImgSrc] = useState("");

  useEffect(() => {
    getMedia(mediaId).then(setDetail);
    ensureSidecarPort().then((port) =>
      setImgSrc(`http://127.0.0.1:${port}/media/${mediaId}/file`)
    );
  }, [mediaId]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <button className="lightbox-close" onClick={onClose}>×</button>
      <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
        {imgSrc && <img src={imgSrc} alt="Full" className="lightbox-img" />}
        {detail && (
          <div className="lightbox-meta">
            <h3>Details</h3>
            {detail.caption && (
              <div className="meta-row">
                <span className="meta-label">Caption</span>
                <span>{detail.caption}</span>
              </div>
            )}
            {detail.date_taken && (
              <div className="meta-row">
                <span className="meta-label">Date Taken</span>
                <span>{new Date(detail.date_taken).toLocaleString()}</span>
              </div>
            )}
            {detail.camera_model && (
              <div className="meta-row">
                <span className="meta-label">Camera</span>
                <span>{detail.camera_model}</span>
              </div>
            )}
            {detail.width && detail.height && (
              <div className="meta-row">
                <span className="meta-label">Dimensions</span>
                <span>{detail.width} × {detail.height}</span>
              </div>
            )}
            {detail.format && (
              <div className="meta-row">
                <span className="meta-label">Format</span>
                <span>{detail.format}</span>
              </div>
            )}
            {detail.gps_lat != null && detail.gps_lon != null && (
              <div className="meta-row">
                <span className="meta-label">Location</span>
                <span>{detail.gps_lat.toFixed(5)}, {detail.gps_lon.toFixed(5)}</span>
              </div>
            )}
            {detail.faces.length > 0 && (
              <div className="meta-row">
                <span className="meta-label">Faces Detected</span>
                <span>{detail.faces.length}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function groupByMonth(items: MediaSummary[]): { label: string; sortKey: string; items: MediaSummary[] }[] {
  const groups = new Map<string, MediaSummary[]>();
  const noDateItems: MediaSummary[] = [];

  for (const item of items) {
    if (item.date_taken) {
      const d = new Date(item.date_taken);
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(item);
    } else {
      noDateItems.push(item);
    }
  }

  const result: { label: string; sortKey: string; items: MediaSummary[] }[] = [];

  // Sort groups by date descending
  const sortedKeys = [...groups.keys()].sort((a, b) => b.localeCompare(a));
  for (const key of sortedKeys) {
    const groupItems = groups.get(key)!;
    const d = new Date(key + "-01");
    const label = d.toLocaleDateString("en-US", { month: "long", year: "numeric" });
    result.push({ label, sortKey: key, items: groupItems });
  }

  if (noDateItems.length > 0) {
    result.push({ label: "Unknown Date", sortKey: "0000-00", items: noDateItems });
  }

  return result;
}

export default function GalleryTab() {
  const [items, setItems] = useState<MediaSummary[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const PAGE_SIZE = 60;

  const loadMore = useCallback(async () => {
    if (loading || items.length >= total) return;
    setLoading(true);
    try {
      const result = await getGallery(page, PAGE_SIZE);
      setItems((prev) => [...prev, ...result.items]);
      setTotal(result.total);
      setPage((p) => p + 1);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [loading, items.length, total, page]);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const result = await getGallery(1, PAGE_SIZE);
        setItems(result.items);
        setTotal(result.total);
        setPage(2);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    const sentinel = loaderRef.current;
    if (!sentinel) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) loadMore(); },
      { threshold: 0.1, root: scrollRef.current }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loadMore]);

  const handleAddFolder = async () => {
    try {
      const selected = await open({ directory: true, multiple: true });
      if (!selected) return;
      const paths = Array.isArray(selected) ? selected : [selected];
      // Get current settings, add paths, save, start scan
      const settings = await getSettings();
      const newPaths = [...new Set([...settings.scan_paths, ...paths])];
      await updateSettings({ ...settings, scan_paths: newPaths });
      await startScan(newPaths);
    } catch (e) {
      console.error("Add folder failed:", e);
    }
  };

  const monthGroups = useMemo(() => groupByMonth(items), [items]);

  if (!loading && items.length === 0) {
    return (
      <>
        <div className="content-header">
          <h1 className="content-title">All Photos</h1>
          <div className="content-header-actions">
            <button className="btn btn-primary" onClick={handleAddFolder}>
              + ADD FOLDER
            </button>
          </div>
        </div>
        <div className="search-empty" style={{ marginTop: 80 }}>
          <p>No images found.</p>
          <p style={{ marginTop: 8, fontSize: 13, color: "var(--text-muted)" }}>
            Click "+ ADD FOLDER" to add a photo directory and start scanning.
          </p>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="content-header">
        <h1 className="content-title">All Photos</h1>
        <div className="content-header-actions">
          <button className="btn btn-secondary" style={{ gap: 6 }}>
            <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8">
              <line x1="3" y1="5" x2="17" y2="5" />
              <line x1="5" y1="10" x2="15" y2="10" />
              <line x1="7" y1="15" x2="13" y2="15" />
            </svg>
            Filter
          </button>
          <button className="btn btn-primary" onClick={handleAddFolder}>
            + ADD FOLDER
          </button>
        </div>
      </div>

      <div className="gallery-scroll" ref={scrollRef}>
        {monthGroups.map((group) => (
          <div key={group.sortKey}>
            <h2 className="gallery-month-header">{group.label}</h2>
            <div className="gallery-month-divider" />
            <div className="gallery-month-grid">
              {group.items.map((item) => (
                <div
                  key={item.id}
                  className="media-card"
                  onClick={() => setSelectedId(item.id)}
                >
                  <ImgProxy relUrl={item.thumbnail_url} alt={item.caption ?? ""} />
                  {item.caption && (
                    <div className="caption-tooltip">{item.caption}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
        <div ref={loaderRef} style={{ height: 40 }} />
        {loading && (
          <div style={{ textAlign: "center", padding: 20, color: "var(--text-muted)" }}>
            Loading more...
          </div>
        )}
      </div>

      {selectedId != null && (
        <Lightbox mediaId={selectedId} onClose={() => setSelectedId(null)} />
      )}
    </>
  );
}
