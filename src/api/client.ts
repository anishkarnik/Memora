import { invoke } from "@tauri-apps/api/core";

export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

async function request<T>(
  method: HttpMethod,
  path: string,
  body?: unknown
): Promise<T> {
  const result = await invoke<T>("api_request", {
    method,
    path,
    body: body ?? null,
  });
  return result;
}

// ── Scan ────────────────────────────────────────────────────────────────────

export interface ScanStatus {
  job_id?: number;
  status: "idle" | "queued" | "running" | "done" | "error" | "cancelled";
  progress_pct: number;
  total_files: number;
  processed_files: number;
  current_file: string | null;
  error_message?: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export const startScan = (paths: string[]) =>
  request<{ job_id: number; status: string }>("POST", "/scan/start", {
    paths,
    mode: "manual",
  });

export const getScanStatus = () => request<ScanStatus>("GET", "/scan/status");

export const cancelScan = () =>
  request<{ status: string }>("POST", "/scan/cancel");

export const recluster = () =>
  request<{ clusters: number; noise: number; total_faces: number }>("POST", "/cluster");

// ── People ──────────────────────────────────────────────────────────────────

export interface PersonSummary {
  id: number;
  name: string | null;
  face_count: number;
  face_thumbnail_url: string;
  created_at: string | null;
}

export interface MediaSummary {
  id: number;
  path: string;
  thumbnail_url: string;
  file_url: string;
  width: number | null;
  height: number | null;
  format: string | null;
  date_taken: string | null;
  caption: string | null;
  media_type: string;
  score?: number;
}

export interface MediaDetail extends MediaSummary {
  gps_lat: number | null;
  gps_lon: number | null;
  camera_model: string | null;
  processed_at: string | null;
  faces: { id: number; bbox: number[]; person_id: number | null }[];
}

export const listPeople = () => request<PersonSummary[]>("GET", "/people");

export const getPersonImages = (personId: number) =>
  request<MediaSummary[]>("GET", `/people/${personId}/images`);

export const setPersonName = (personId: number, name: string) =>
  request<{ id: number; name: string }>("POST", `/people/${personId}/name`, {
    name,
  });

export const mergePeople = (sourceId: number, targetId: number) =>
  request<{ merged_into: number }>("POST", "/people/merge", {
    source_id: sourceId,
    target_id: targetId,
  });

// ── Gallery ─────────────────────────────────────────────────────────────────

export interface GalleryPage {
  total: number;
  page: number;
  page_size: number;
  items: MediaSummary[];
}

export const getGallery = (page = 1, pageSize = 50) =>
  request<GalleryPage>("GET", `/gallery?page=${page}&page_size=${pageSize}`);

export const getMedia = (mediaId: number) =>
  request<MediaDetail>("GET", `/media/${mediaId}`);

// ── Search ──────────────────────────────────────────────────────────────────

export interface SearchResult {
  query: string;
  results: (MediaSummary & { score: number })[];
}

export const search = (q: string, limit = 20, personId?: number) => {
  let path = `/search?q=${encodeURIComponent(q)}&limit=${limit}`;
  if (personId != null) path += `&person_id=${personId}`;
  return request<SearchResult>("GET", path);
};

// ── Settings ─────────────────────────────────────────────────────────────────

export type CaptionModel = "moondream2" | "florence2" | "blip";
export type EmbeddingModel = "clip" | "siglip2";

export interface Settings {
  scan_paths: string[];
  auto_scan_enabled: boolean;
  caption_model: CaptionModel;
  embedding_model: EmbeddingModel;
}

export const getSettings = () => request<Settings>("GET", "/settings");

export const updateSettings = (settings: Settings) =>
  request<Settings>("POST", "/settings", settings);

// ── Sidecar ──────────────────────────────────────────────────────────────────

export const startSidecar = () => invoke<number>("start_sidecar");
export const getSidecarPort = () => invoke<number | null>("get_sidecar_port");

// ── Image URL helper ─────────────────────────────────────────────────────────
// Construct a proxied URL via the Tauri sidecar port stored at runtime.
let _sidecarPort: number | null = null;

export async function ensureSidecarPort(): Promise<number> {
  if (_sidecarPort) return _sidecarPort;
  _sidecarPort = await getSidecarPort();
  if (!_sidecarPort) {
    _sidecarPort = await startSidecar();
  }
  return _sidecarPort!;
}

export async function mediaUrl(relPath: string): Promise<string> {
  const port = await ensureSidecarPort();
  return `http://127.0.0.1:${port}${relPath}`;
}
