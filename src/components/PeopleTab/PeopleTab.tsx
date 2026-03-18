import { useEffect, useState } from "react";
import {
  listPeople,
  getPersonImages,
  setPersonName,
  mergePeople,
  recluster,
  PersonSummary,
  MediaSummary,
  ensureSidecarPort,
} from "../../api/client";

function FaceImg({ relUrl, alt }: { relUrl: string; alt: string }) {
  const [src, setSrc] = useState("");
  useEffect(() => {
    ensureSidecarPort().then((port) => setSrc(`http://127.0.0.1:${port}${relUrl}`));
  }, [relUrl]);
  return (
    <img
      src={src}
      alt={alt}
      style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
      loading="lazy"
    />
  );
}

function ImgProxy({ relUrl, alt }: { relUrl: string; alt: string }) {
  const [src, setSrc] = useState("");
  useEffect(() => {
    ensureSidecarPort().then((port) => setSrc(`http://127.0.0.1:${port}${relUrl}`));
  }, [relUrl]);
  return <img src={src} alt={alt} loading="lazy" />;
}

// ── Person detail (photos of one person) ────────────────────────────────────

function PersonDetail({
  person,
  onBack,
  onRefresh,
}: {
  person: PersonSummary;
  onBack: () => void;
  onRefresh: () => void;
}) {
  const [images, setImages] = useState<MediaSummary[]>([]);
  const [editing, setEditing] = useState(false);
  const [nameInput, setNameInput] = useState(person.name ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => { getPersonImages(person.id).then(setImages); }, [person.id]);

  const saveName = async () => {
    setSaving(true);
    try {
      await setPersonName(person.id, nameInput);
      onRefresh();
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ padding: "12px 16px", display: "flex", alignItems: "center", gap: 12, borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
        {/* Face avatar */}
        <div style={{ width: 40, height: 40, borderRadius: "50%", overflow: "hidden", flexShrink: 0, background: "var(--surface2)" }}>
          <FaceImg relUrl={person.face_thumbnail_url} alt={person.name ?? "Unknown"} />
        </div>

        <button className="btn btn-secondary" onClick={onBack}>← Back</button>

        {editing ? (
          <>
            <input
              className="input" style={{ width: 200 }} value={nameInput}
              onChange={(e) => setNameInput(e.target.value)} autoFocus
              onKeyDown={(e) => { if (e.key === "Enter") saveName(); if (e.key === "Escape") setEditing(false); }}
            />
            <button className="btn btn-primary" onClick={saveName} disabled={saving}>Save</button>
            <button className="btn btn-secondary" onClick={() => setEditing(false)}>Cancel</button>
          </>
        ) : (
          <>
            <span style={{ fontWeight: 600, fontSize: 16 }}>
              {person.name ?? `Unknown #${person.id}`}
            </span>
            <button className="btn btn-secondary" onClick={() => { setEditing(true); setNameInput(person.name ?? ""); }}>
              Rename
            </button>
          </>
        )}
        <span style={{ color: "var(--text-muted)", marginLeft: "auto", fontSize: 12 }}>
          {images.length} photo{images.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="media-grid" style={{ flex: 1, overflow: "auto" }}>
        {images.map((img) => (
          <div key={img.id} className="media-card">
            <ImgProxy relUrl={img.thumbnail_url} alt={img.caption ?? ""} />
            {img.caption && <div className="caption-tooltip">{img.caption}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Person card ──────────────────────────────────────────────────────────────

type MergeState = "none" | "source" | "target" | "pickable";

function PersonCard({
  person,
  mergeState,
  onClick,
}: {
  person: PersonSummary;
  mergeState: MergeState;
  onClick: () => void;
}) {
  const isNamed = person.name !== null;

  const borderColor =
    mergeState === "source" ? "#f97316"       // orange = selected as source
    : mergeState === "target" ? "var(--accent)" // accent = selectable target
    : isNamed ? "var(--accent)"
    : "transparent";

  const overlay =
    mergeState === "source" ? { label: "Merging…", bg: "#f97316" }
    : mergeState === "target" ? { label: "Merge into", bg: "var(--accent)" }
    : isNamed ? { label: "Named", bg: "var(--accent)" }
    : null;

  return (
    <div
      className="person-card"
      onClick={onClick}
      style={{
        borderColor,
        background: (mergeState === "source" || isNamed) ? "var(--surface2)" : undefined,
        opacity: mergeState === "source" ? 0.75 : 1,
      }}
    >
      <div style={{ position: "relative", width: "100%", aspectRatio: "1", overflow: "hidden", background: "var(--surface2)" }}>
        <FaceImg relUrl={person.face_thumbnail_url} alt={person.name ?? "Unknown"} />
        {overlay && (
          <div style={{
            position: "absolute", top: 6, right: 6,
            background: overlay.bg, borderRadius: 4,
            padding: "2px 6px", fontSize: 10, color: "#fff", fontWeight: 600,
          }}>
            {overlay.label}
          </div>
        )}
      </div>
      <div className="person-info">
        <div className="person-name" style={isNamed ? { color: "var(--accent)" } : {}}>
          {person.name ?? `Unknown #${person.id}`}
        </div>
        <div className="face-count">{person.face_count} photo{person.face_count !== 1 ? "s" : ""}</div>
      </div>
    </div>
  );
}

// ── Main PeopleTab ───────────────────────────────────────────────────────────

export default function PeopleTab() {
  const [people, setPeople] = useState<PersonSummary[]>([]);
  const [selectedPerson, setSelectedPerson] = useState<PersonSummary | null>(null);
  const [mergeSource, setMergeSource] = useState<PersonSummary | null>(null);
  const [mergeMode, setMergeMode] = useState(false);
  const [reclustering, setReclustering] = useState(false);

  const loadPeople = () => listPeople().then(setPeople);

  useEffect(() => { loadPeople(); }, []);

  const handleRecluster = async () => {
    setReclustering(true);
    try {
      const result = await recluster();
      await loadPeople();
      alert(`Done — found ${result.clusters} people across ${result.total_faces} faces.`);
    } catch (e) {
      alert("Re-cluster failed: " + String(e));
    } finally {
      setReclustering(false);
    }
  };

  const cancelMerge = () => { setMergeMode(false); setMergeSource(null); };

  // In merge mode: first click picks source, second click on a different card triggers merge.
  const handleCardClickInMerge = async (person: PersonSummary) => {
    if (!mergeSource) {
      setMergeSource(person);
      return;
    }
    if (mergeSource.id === person.id) {
      // Clicking source again → deselect
      setMergeSource(null);
      return;
    }
    // Second different person → execute merge
    try {
      await mergePeople(mergeSource.id, person.id);
      await loadPeople();
      cancelMerge();
    } catch (e) {
      alert("Merge failed: " + String(e));
    }
  };

  if (selectedPerson) {
    return (
      <PersonDetail
        person={selectedPerson}
        onBack={() => setSelectedPerson(null)}
        onRefresh={loadPeople}
      />
    );
  }

  const namedPeople = people.filter((p) => p.name !== null);
  const unnamedPeople = people.filter((p) => p.name === null);

  const toolbar = (
    <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)", flexShrink: 0, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
      <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
        {namedPeople.length} named · {unnamedPeople.length} unknown
      </span>
      <button className="btn btn-secondary" onClick={handleRecluster} disabled={reclustering || mergeMode}>
        {reclustering ? "Clustering…" : "Re-cluster Faces"}
      </button>
      {mergeMode ? (
        <>
          <span style={{ color: "var(--accent)", fontSize: 13 }}>
            {mergeSource
              ? `Now click who to merge "${mergeSource.name ?? `Unknown #${mergeSource.id}`}" into`
              : "Click the person to merge away"}
          </span>
          <button className="btn btn-secondary" onClick={cancelMerge}>Cancel</button>
        </>
      ) : (
        <button
          className="btn btn-secondary"
          style={{ marginLeft: "auto" }}
          onClick={() => setMergeMode(true)}
          disabled={people.length < 2}
        >
          Merge two people
        </button>
      )}
    </div>
  );

  if (people.length === 0) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
        {toolbar}
        <div className="search-empty" style={{ marginTop: 60 }}>
          <p>No people found.</p>
          <p style={{ marginTop: 8, fontSize: 12 }}>Scan a folder with photos, then click Re-cluster Faces.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {toolbar}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {/* Named people section */}
        {namedPeople.length > 0 && (
          <div style={{ padding: "12px 16px 4px" }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 10 }}>
              Named People
            </div>
            <div className="person-grid" style={{ height: "auto", overflow: "visible", padding: 0 }}>
              {namedPeople.map((p) => (
                <PersonCard
                  key={p.id}
                  person={p}
                  mergeState={
                    !mergeMode ? "none"
                    : mergeSource?.id === p.id ? "source"
                    : mergeSource ? "target"
                    : "pickable"
                  }
                  onClick={() => mergeMode ? handleCardClickInMerge(p) : setSelectedPerson(p)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Unknown people section */}
        {unnamedPeople.length > 0 && (
          <div style={{ padding: "12px 16px 16px" }}>
            {namedPeople.length > 0 && (
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 10 }}>
                Unknown People
              </div>
            )}
            <div className="person-grid" style={{ height: "auto", overflow: "visible", padding: 0 }}>
              {unnamedPeople.map((p) => (
                <PersonCard
                  key={p.id}
                  person={p}
                  mergeState={
                    !mergeMode ? "none"
                    : mergeSource?.id === p.id ? "source"
                    : mergeSource ? "target"
                    : "pickable"
                  }
                  onClick={() => mergeMode ? handleCardClickInMerge(p) : setSelectedPerson(p)}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
