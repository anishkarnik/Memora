import { useState, useEffect } from "react";
import {
  search,
  listPeople,
  MediaSummary,
  PersonSummary,
  ensureSidecarPort,
} from "../../api/client";

function ImgProxy({ relUrl, alt }: { relUrl: string; alt: string }) {
  const [src, setSrc] = useState("");
  useEffect(() => {
    ensureSidecarPort().then((port) =>
      setSrc(`http://127.0.0.1:${port}${relUrl}`)
    );
  }, [relUrl]);
  return <img src={src} alt={alt} loading="lazy" />;
}

function FaceChip({
  person,
  selected,
  onClick,
}: {
  person: PersonSummary;
  selected: boolean;
  onClick: () => void;
}) {
  const [src, setSrc] = useState("");
  useEffect(() => {
    ensureSidecarPort().then((port) =>
      setSrc(`http://127.0.0.1:${port}${person.face_thumbnail_url}`)
    );
  }, [person.face_thumbnail_url]);

  const label = person.name ?? `#${person.id}`;

  return (
    <button
      className={`face-chip${selected ? " selected" : ""}`}
      onClick={onClick}
      title={label}
    >
      <div className="face-chip-avatar">
        {src && <img src={src} alt={label} />}
      </div>
      <span className="face-chip-label">{label}</span>
    </button>
  );
}

export default function SearchTab() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<(MediaSummary & { score: number })[]>([]);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [people, setPeople] = useState<PersonSummary[]>([]);
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);

  useEffect(() => {
    listPeople().then(setPeople).catch(() => {});
  }, []);

  const togglePerson = (id: number) => {
    const newId = id === selectedPersonId ? null : id;
    setSelectedPersonId(newId);
    // Always re-run search when toggling a person (even with no text query)
    runSearch(newId);
  };

  const runSearch = async (personIdOverride?: number | null) => {
    const personId = personIdOverride !== undefined ? personIdOverride : selectedPersonId;
    if (!query.trim() && personId === null) return;
    setSearching(true);
    setError(null);
    try {
      const res = await search(query.trim(), 40, personId ?? undefined);
      setResults(res.results);
      setSearched(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setSearching(false);
    }
  };

  const selectedPerson = people.find((p) => p.id === selectedPersonId);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>

      {/* ── People filter bar ─────────────────────────────────── */}
      {people.length > 0 && (
        <div className="face-filter-bar">
          <span className="face-filter-label">Filter by person</span>
          <div className="face-filter-scroll">
            {people.map((p) => (
              <FaceChip
                key={p.id}
                person={p}
                selected={p.id === selectedPersonId}
                onClick={() => togglePerson(p.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Search bar ───────────────────────────────────────────── */}
      <div className="search-bar-wrap">
        <div className="search-bar">
          <div style={{ position: "relative", flex: 1 }}>
            <input
              className="input"
              placeholder={
                selectedPerson
                  ? `Search photos of ${selectedPerson.name ?? `Unknown #${selectedPerson.id}`}…`
                  : 'Search your photos… e.g. "mountains at sunset"'
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") runSearch(); }}
              style={selectedPerson ? { paddingLeft: 36 } : {}}
            />
            {/* Active person indicator inside input */}
            {selectedPerson && (
              <ActivePersonBadge
                person={selectedPerson}
                onClear={() => {
                  setSelectedPersonId(null);
                  if (searched && query.trim()) runSearch(null);
                }}
              />
            )}
          </div>
          <button
            className="btn btn-primary"
            onClick={() => runSearch()}
            disabled={searching || (!query.trim() && selectedPersonId === null)}
            style={{ flexShrink: 0 }}
          >
            {searching ? "Searching…" : "Search"}
          </button>
        </div>
        {error && <p style={{ color: "#f87171", marginTop: 8, fontSize: 12 }}>{error}</p>}
      </div>

      {/* ── Results ──────────────────────────────────────────────── */}
      <div className="search-results" style={{ flex: 1, overflow: "hidden" }}>
        {!searched && !searching && (
          <div className="search-empty">
            {selectedPersonId !== null ? (
              <>
                <p>Showing all photos of {selectedPerson?.name ?? `Unknown #${selectedPersonId}`}.</p>
                <p style={{ marginTop: 8, fontSize: 12 }}>Add a description to narrow results further.</p>
              </>
            ) : (
              <>
                <p>Enter a description or select a person above to find photos.</p>
                <p style={{ marginTop: 8, fontSize: 12 }}>Powered by CLIP — understands meaning, not just keywords.</p>
              </>
            )}
          </div>
        )}
        {searched && results.length === 0 && (
          <div className="search-empty">
            <p>No results found.</p>
          </div>
        )}
        {results.length > 0 && (
          <div className="media-grid" style={{ height: "100%", overflow: "auto" }}>
            {results.map((item) => (
              <div key={item.id} className="media-card">
                <ImgProxy relUrl={item.thumbnail_url} alt={item.caption ?? ""} />
                {item.caption && (
                  <div className="caption-tooltip">{item.caption}</div>
                )}
                {process.env.NODE_ENV === "development" && (
                  <div style={{
                    position: "absolute", top: 4, right: 4,
                    background: "rgba(0,0,0,0.7)", borderRadius: 4,
                    padding: "2px 5px", fontSize: 10, color: "#a5f3fc",
                  }}>
                    {(item.score * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Small face avatar shown inside the search input when a person is active
function ActivePersonBadge({
  person,
  onClear,
}: {
  person: PersonSummary;
  onClear: () => void;
}) {
  const [src, setSrc] = useState("");
  useEffect(() => {
    ensureSidecarPort().then((port) =>
      setSrc(`http://127.0.0.1:${port}${person.face_thumbnail_url}`)
    );
  }, [person.face_thumbnail_url]);

  return (
    <div
      style={{
        position: "absolute",
        left: 8,
        top: "50%",
        transform: "translateY(-50%)",
        display: "flex",
        alignItems: "center",
        gap: 4,
        pointerEvents: "none",
      }}
    >
      <div style={{
        width: 20, height: 20, borderRadius: "50%", overflow: "hidden",
        background: "var(--surface2)", flexShrink: 0, border: "1px solid var(--accent)",
      }}>
        {src && <img src={src} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />}
      </div>
      <button
        onClick={onClear}
        style={{
          pointerEvents: "auto",
          background: "var(--accent)",
          border: "none",
          color: "#fff",
          borderRadius: "50%",
          width: 14,
          height: 14,
          fontSize: 9,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
        title="Clear person filter"
      >
        ×
      </button>
    </div>
  );
}
