"""
Face clustering using Agglomerative Clustering (average linkage).

Why not DBSCAN:
  DBSCAN with min_samples=1 degenerates to single-linkage — it chains through
  the embedding space, pulling different people into the same cluster whenever
  there is an intermediate "bridge" face. This causes messy sets.

Why Agglomerative + average linkage:
  Merges two clusters only when the AVERAGE distance between all their members
  is below the threshold. No chaining. Tight, clean clusters.

Incremental assignment:
  New faces from a scan are first matched against existing person centroids.
  Only truly novel faces (no match) are batch-clustered. This prevents the
  instability of full recluster on every scan.
"""
from datetime import datetime
from typing import Optional

import numpy as np

# Two faces are considered the same person if their cosine distance is below this.
# InsightFace buffalo_sc: same-person ~0.2-0.4, different-person ~0.5+
DISTANCE_THRESHOLD = 0.45

# Minimum face bbox area (px²) — tiny faces yield unreliable embeddings
MIN_FACE_AREA = 1600  # 40×40 px minimum

# Similarity threshold to match a new cluster to an existing NAMED person
NAMED_MERGE_THRESHOLD = 0.60
# Similarity threshold to match to an existing UNNAMED person
UNNAMED_MERGE_THRESHOLD = 0.55

# Similarity threshold for incremental assignment (slightly stricter than
# cluster merge to avoid false assignments during scan)
INCREMENTAL_ASSIGN_THRESHOLD = 0.55


def assign_face_to_person(face_embedding: np.ndarray, session) -> Optional[int]:
    """Try to assign a single face to an existing person by centroid similarity.

    Returns person_id if matched, None if no match (needs batch clustering later).
    This is called during scan to incrementally assign faces without full recluster.
    """
    from models import Person
    from face_engine import bytes_to_embedding, cosine_similarity

    emb = face_embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    people = session.query(Person).filter(Person.representative_embedding.isnot(None)).all()
    if not people:
        return None

    best_person_id = None
    best_sim = 0.0

    for person in people:
        centroid = bytes_to_embedding(person.representative_embedding)
        sim = cosine_similarity(emb, centroid)
        threshold = NAMED_MERGE_THRESHOLD if person.name else INCREMENTAL_ASSIGN_THRESHOLD
        if sim > threshold and sim > best_sim:
            best_sim = sim
            best_person_id = person.id

    return best_person_id


def update_person_centroid(person_id: int, session) -> None:
    """Recalculate a person's representative embedding from all their faces."""
    from models import Face, Person
    from face_engine import bytes_to_embedding, embedding_to_bytes

    person = session.query(Person).filter(Person.id == person_id).first()
    if not person:
        return

    faces = session.query(Face).filter(Face.person_id == person_id).all()
    if not faces:
        return

    embeddings = np.stack([bytes_to_embedding(f.embedding) for f in faces])

    # Quality-weighted mean: higher det_score faces contribute more
    weights = np.array([
        f.det_score if f.det_score is not None else 0.5
        for f in faces
    ], dtype=np.float32)
    # Normalize weights
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(faces), dtype=np.float32) / len(faces)

    centroid = (embeddings * weights[:, np.newaxis]).sum(axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    person.representative_embedding = embedding_to_bytes(centroid)


def run_clustering(session) -> dict:
    """Recluster — preserves named people and their face assignments.

    Faces assigned to named people are PINNED — they are never unassigned
    or reclustered. Only faces belonging to unnamed people (or unassigned)
    are reclustered. This preserves manual merges into named people.

    Uses quality-weighted centroids: faces with higher det_score contribute
    more to the person's representative embedding.
    """
    from models import Face, Person
    from face_engine import bytes_to_embedding, cosine_similarity, embedding_to_bytes
    import json

    all_faces = session.query(Face).all()
    if not all_faces:
        return {"clusters": 0, "noise": 0, "total_faces": 0}

    # ── Identify named people and their pinned faces ─────────────────────────
    named_people = session.query(Person).filter(Person.name.isnot(None)).all()
    named_person_ids = {p.id for p in named_people}

    # ── Quality filter + separate pinned vs free faces ───────────────────────
    pinned_faces = []   # belong to named people — do NOT recluster
    free_faces = []     # will be reclustered
    skipped = 0

    for f in all_faces:
        bbox = json.loads(f.bbox_json)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w * h < MIN_FACE_AREA:
            skipped += 1
            f.person_id = None
            continue

        if f.person_id in named_person_ids:
            pinned_faces.append(f)
        else:
            free_faces.append(f)

    # Clear unnamed persons — they'll be rebuilt from clustering
    for f in free_faces:
        f.person_id = None
    session.query(Person).filter(Person.name.is_(None)).delete(synchronize_session=False)
    session.flush()

    # Recalculate named people centroids from their pinned faces
    for person in named_people:
        update_person_centroid(person.id, session)

    if not free_faces:
        session.commit()
        return {
            "clusters": 0,
            "noise": skipped,
            "total_faces": len(all_faces),
            "pinned": len(pinned_faces),
        }

    # ── Cluster only free (non-pinned) faces ─────────────────────────────────
    embeddings = np.stack([bytes_to_embedding(f.embedding) for f in free_faces])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms > 0, norms, 1)

    if len(free_faces) == 1:
        labels = np.array([0])
    else:
        cosine_dist = np.clip(1 - normed @ normed.T, 0, 2)
        try:
            from sklearn.cluster import AgglomerativeClustering
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=DISTANCE_THRESHOLD,
                metric="precomputed",
                linkage="average",
            ).fit_predict(cosine_dist)
        except Exception as e:
            return {"clusters": 0, "noise": 0, "total_faces": len(all_faces), "error": str(e)}

    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    # ── Map each cluster to a Person ─────────────────────────────────────────
    # Named people are candidates for matching (they survived the delete above)
    existing_people = list(named_people)
    label_to_person: dict[int, Person] = {}

    det_scores = np.array([
        f.det_score if f.det_score is not None else 0.5
        for f in free_faces
    ], dtype=np.float32)

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_embeddings = normed[mask]
        cluster_scores = det_scores[mask]

        # Quality-weighted centroid
        weights = cluster_scores / (cluster_scores.sum() or 1.0)
        centroid = (cluster_embeddings * weights[:, np.newaxis]).sum(axis=0).astype(np.float32)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Try to match to existing people (named or already-created unnamed)
        best_person: Optional[Person] = None
        best_sim = 0.0
        for person in existing_people:
            if person.representative_embedding is None:
                continue
            threshold = NAMED_MERGE_THRESHOLD if person.name else UNNAMED_MERGE_THRESHOLD
            sim = cosine_similarity(centroid, bytes_to_embedding(person.representative_embedding))
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_person = person

        if best_person is not None:
            label_to_person[label] = best_person
        else:
            person = Person(
                name=None,
                representative_embedding=embedding_to_bytes(centroid),
                created_at=datetime.utcnow(),
            )
            session.add(person)
            session.flush()
            label_to_person[label] = person
            existing_people.append(person)

    # Assign person_id to each free face
    for i, face in enumerate(free_faces):
        face.person_id = label_to_person[labels[i]].id

    # Recalculate centroids for all people that got new faces
    affected_people = set(label_to_person.values())
    for person in affected_people:
        all_person_faces = session.query(Face).filter(Face.person_id == person.id).all()
        if all_person_faces:
            p_embeds = np.stack([bytes_to_embedding(f.embedding) for f in all_person_faces])
            p_scores = np.array([
                f.det_score if f.det_score is not None else 0.5
                for f in all_person_faces
            ], dtype=np.float32)
            p_weights = p_scores / (p_scores.sum() or 1.0)
            p_centroid = (p_embeds * p_weights[:, np.newaxis]).sum(axis=0).astype(np.float32)
            p_norm = np.linalg.norm(p_centroid)
            if p_norm > 0:
                p_centroid = p_centroid / p_norm
            person.representative_embedding = embedding_to_bytes(p_centroid)

    session.commit()
    return {
        "clusters": n_clusters,
        "noise": skipped,
        "total_faces": len(all_faces),
        "pinned": len(pinned_faces),
        "quality_filtered": skipped,
    }
