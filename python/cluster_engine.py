"""
Face clustering using Agglomerative Clustering (average linkage).

Why not DBSCAN:
  DBSCAN with min_samples=1 degenerates to single-linkage — it chains through
  the embedding space, pulling different people into the same cluster whenever
  there is an intermediate "bridge" face. This causes messy sets.

Why Agglomerative + average linkage:
  Merges two clusters only when the AVERAGE distance between all their members
  is below the threshold. No chaining. Tight, clean clusters.
"""
from datetime import datetime
from typing import Optional

import numpy as np

# Two faces are considered the same person if their cosine distance is below this.
# InsightFace buffalo_sc: same-person ~0.2-0.4, different-person ~0.5+
DISTANCE_THRESHOLD = 0.45

# Minimum detection confidence score to use a face embedding.
# Low-confidence detections (small/blurry/profile) produce noisy embeddings.
MIN_DET_SCORE = 0.0   # set > 0 once det_score is stored; filtered by bbox size for now

# Minimum face bbox area (px²) — tiny faces yield unreliable embeddings
MIN_FACE_AREA = 1600  # 40×40 px minimum

# Similarity threshold to match a new cluster to an existing NAMED person
NAMED_MERGE_THRESHOLD = 0.60
# Similarity threshold to match to an existing UNNAMED person
UNNAMED_MERGE_THRESHOLD = 0.55


def run_clustering(session) -> dict:
    from models import Face, Person
    from face_engine import bytes_to_embedding, cosine_similarity, embedding_to_bytes
    import json

    all_faces = session.query(Face).all()
    if not all_faces:
        return {"clusters": 0, "noise": 0, "total_faces": 0}

    # ── Quality filter: skip tiny/unreliable faces ───────────────────────────
    faces = []
    skipped = 0
    for f in all_faces:
        bbox = json.loads(f.bbox_json)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w * h >= MIN_FACE_AREA:
            faces.append(f)
        else:
            skipped += 1
            f.person_id = None  # unassign tiny faces

    if not faces:
        session.commit()
        return {"clusters": 0, "noise": skipped, "total_faces": len(all_faces)}

    embeddings = np.stack([bytes_to_embedding(f.embedding) for f in faces])

    # ── Normalise embeddings ─────────────────────────────────────────────────
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms > 0, norms, 1)

    # ── Agglomerative clustering (average linkage, cosine distance) ──────────
    if len(faces) == 1:
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
            return {"clusters": 0, "noise": 0, "total_faces": len(faces), "error": str(e)}

    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    # ── Map each cluster to a Person ─────────────────────────────────────────
    existing_people = session.query(Person).all()
    label_to_person: dict[int, Person] = {}

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_embeddings = normed[mask]
        cluster_faces = [f for f, m in zip(faces, mask) if m]

        centroid = cluster_embeddings.mean(axis=0).astype(np.float32)
        centroid = centroid / (np.linalg.norm(centroid) or 1)

        # Named people first, then unnamed — pick best match above threshold
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
            # Refine representative embedding toward new data
            old = bytes_to_embedding(best_person.representative_embedding)
            updated = (old + centroid) / 2
            updated = updated / (np.linalg.norm(updated) or 1)
            best_person.representative_embedding = embedding_to_bytes(updated)
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

    # Assign person_id to each face
    for i, face in enumerate(faces):
        face.person_id = label_to_person[labels[i]].id

    session.commit()
    return {
        "clusters": n_clusters,
        "noise": skipped,
        "total_faces": len(all_faces),
        "quality_filtered": skipped,
    }
