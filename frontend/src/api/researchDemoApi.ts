/**
 * frontend/src/api/researchDemoApi.ts
 *
 * Dedicated client for the SeismoFNO Professor-Facing Research Demonstration API endpoints.
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export interface DemoModel {
  model_id: string;
  display_name: string;
  phase: string;
  architecture: string;
  representation: string;
  conditioning_type: string;
  cond_dim: number;
  parameter_count: number;
  checkpoint_path: string;
  sha256: string;
  full_sha256: string;
  status: string;
  primary_role: string;
  verified_key_metric: string;
  supports_live_inference: boolean;
  notes: string;
}

export interface DemoStructure {
  archetype_id: string;
  display_name: string;
  n_stories: number;
  category: string;
  total_mass_kg: number;
  story_mass_kg: number;
  story_stiffness_kN_m: number;
  total_height_m: number;
  story_height_m: number;
  yield_drift_mm: number;
  T1_s: number;
  T2_s: number;
  T3_s: number;
  omega1_rad_s: number;
  omega2_rad_s: number;
  omega3_rad_s: number;
  mode_shapes: number[][];
  modal_origin: string;
}

export interface DemoEarthquake {
  record_id: string;
  event_name: string;
  station: string;
  year: number;
  pga_g: number;
  duration_s: number;
  dt: number;
  role: string;
}

export interface ProgressionStep {
  step: number;
  phase: string;
  title: string;
  representation: string;
  result_highlight: string;
  status: string;
  finding: string;
  color: string;
}

export interface OODMatrix {
  partitions: string[];
  peak_disp_error_pct: Record<string, number[]>;
  rel_l2_u_pct: Record<string, number[]>;
  key_finding: string;
}

export interface AblationData {
  hypothesis: string;
  protocol: string;
  metrics: Array<{
    partition: string;
    true_multimodal_err: number;
    shuffled_err: number;
    delta_percentage_points: number;
    degradation_pct: number;
  }>;
  interpretation: string;
}

export interface FailureMode {
  id: string;
  title: string;
  phase: string;
  symptom: string;
  mechanism: string;
  resolution: string;
}

export interface BenchmarkItem {
  name: string;
  batch_size: number;
  latency_ms: number;
  throughput_sim_s: number;
  speedup: number;
  badge: string;
}

export interface BenchmarkData {
  hardware: string;
  benchmark_protocol: string;
  models: BenchmarkItem[];
  conclusion: string;
}

export interface DemoSimulationResponse {
  archetype: DemoStructure;
  earthquake: DemoEarthquake;
  model: DemoModel;
  selected_floor: number;
  total_floors: number;
  is_live_inference: boolean;
  time: number[];
  ag: number[];
  u_true: number[];
  u_pred: number[];
  metrics: {
    rel_l2_pct: number;
    peak_disp_err_pct: number;
    pearson_r: number;
    delta_t_peak_s: number;
    peak_pred_m: number;
    peak_true_m: number;
    latency_ms: number;
    provenance: string;
  };
  data_provenance: {
    ground_truth: string;
    prediction: string;
  };
}

export async function fetchDemoModels(): Promise<DemoModel[]> {
  const res = await fetch(`${API_BASE}/api/v1/demo/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
  return await res.json();
}

export async function fetchDemoStructures(): Promise<DemoStructure[]> {
  const res = await fetch(`${API_BASE}/api/v1/demo/structures`);
  if (!res.ok) throw new Error(`Failed to fetch structures: ${res.status}`);
  return await res.json();
}

export async function fetchDemoEarthquakes(): Promise<DemoEarthquake[]> {
  const res = await fetch(`${API_BASE}/api/v1/demo/earthquakes`);
  if (!res.ok) throw new Error(`Failed to fetch earthquakes: ${res.status}`);
  return await res.json();
}

export async function fetchDemoProgression(): Promise<ProgressionStep[]> {
  const res = await fetch(`${API_BASE}/api/v1/demo/progression`);
  if (!res.ok) throw new Error(`Failed to fetch progression: ${res.status}`);
  return await res.json();
}

export async function fetchDemoOODMatrix(): Promise<OODMatrix> {
  const res = await fetch(`${API_BASE}/api/v1/demo/ood-matrix`);
  if (!res.ok) throw new Error(`Failed to fetch OOD matrix: ${res.status}`);
  return await res.json();
}

export async function fetchDemoAblation(): Promise<AblationData> {
  const res = await fetch(`${API_BASE}/api/v1/demo/ablation`);
  if (!res.ok) throw new Error(`Failed to fetch ablation: ${res.status}`);
  return await res.json();
}

export async function fetchDemoFailures(): Promise<FailureMode[]> {
  const res = await fetch(`${API_BASE}/api/v1/demo/failures`);
  if (!res.ok) throw new Error(`Failed to fetch failure modes: ${res.status}`);
  return await res.json();
}

export async function fetchDemoBenchmark(): Promise<BenchmarkData> {
  const res = await fetch(`${API_BASE}/api/v1/demo/benchmark`);
  if (!res.ok) throw new Error(`Failed to fetch benchmark: ${res.status}`);
  return await res.json();
}

export async function runDemoSimulation(payload: {
  archetype_id: string;
  record_id: string;
  model_id: string;
  selected_floor?: number;
}): Promise<DemoSimulationResponse> {
  const res = await fetch(`${API_BASE}/api/v1/demo/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Simulation failed (${res.status})`);
  }
  return await res.json();
}

export interface LiveEarthquakeEvent {
  event_id: string;
  magnitude: number;
  location: string;
  depth_km: number;
  latitude: number;
  longitude: number;
  origin_time: string;
  time_epoch_ms: number;
  status: string;
  source: string;
  data_type: string;
  has_compatible_waveform: boolean;
  distance_km?: number | null;
  url: string;
}

export interface LiveEarthquakeResponse {
  source: string;
  status: string;
  data_type: string;
  is_fallback: boolean;
  status_message: string;
  last_updated: string;
  count: number;
  events: LiveEarthquakeEvent[];
}

export interface LiveEarthquakeQueryParams {
  minmagnitude?: number;
  limit?: number;
  hours?: number;
  latitude?: number;
  longitude?: number;
  radius_km?: number;
}

export async function fetchLiveEarthquakes(
  params?: LiveEarthquakeQueryParams
): Promise<LiveEarthquakeResponse> {
  const query = new URLSearchParams();
  if (params?.minmagnitude !== undefined) query.append("minmagnitude", params.minmagnitude.toString());
  if (params?.limit !== undefined) query.append("limit", params.limit.toString());
  if (params?.hours !== undefined) query.append("hours", params.hours.toString());
  if (params?.latitude !== undefined) query.append("latitude", params.latitude.toString());
  if (params?.longitude !== undefined) query.append("longitude", params.longitude.toString());
  if (params?.radius_km !== undefined) query.append("radius_km", params.radius_km.toString());

  const url = `${API_BASE}/api/v1/demo/earthquakes/live${query.toString() ? `?${query.toString()}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch live earthquakes: ${res.status}`);
  return await res.json();
}

export async function fetchLiveEarthquakeDetails(
  eventId: string
): Promise<LiveEarthquakeEvent> {
  const res = await fetch(`${API_BASE}/api/v1/demo/earthquakes/${eventId}`);
  if (!res.ok) throw new Error(`Failed to fetch earthquake detail: ${res.status}`);
  return await res.json();
}

