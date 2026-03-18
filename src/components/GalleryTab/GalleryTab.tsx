import { useEffect, useRef, useState, useCallback } from "react";
import {
  getGallery,
  getMedia,
  MediaSummary,
  MediaDetail,
  ensureSidecarPort,
} from "../../api/client";

function ImgProxy({ relUrl, alt, className }: { relUrl: string; alt: string; className?: string }) {
  const [src, setSrc] = useState<string>("");
  useEffect(() => {
    ensureSidecarPort().then((port) =>
      setSrc(`http://127.0.0.1:${port}${relUrl}`)
    );
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

export default function GalleryTab() {
  const [items, setItems] = useState<MediaSummary[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  const PAGE_SIZE = 30;

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

  // Initial load
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

  // Infinite scroll sentinel
  useEffect(() => {
    const sentinel = loaderRef.current;
    if (!sentinel) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) loadMore(); },
      { threshold: 0.1 }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loadMore]);

  if (!loading && items.length === 0) {
    return (
      <div className="search-empty" style={{ marginTop: 80 }}>
        <p>No images found.</p>
        <p style={{ marginTop: 8, fontSize: 12 }}>
          Go to Settings to add scan paths.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="media-grid" style={{ height: "100%" }}>
        {items.map((item) => (
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
        <div ref={loaderRef} style={{ height: 40 }} />
      </div>

      {selectedId != null && (
        <Lightbox mediaId={selectedId} onClose={() => setSelectedId(null)} />
      )}
    </>
  );
}
