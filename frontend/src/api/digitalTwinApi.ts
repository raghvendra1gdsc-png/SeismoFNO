/**
 * frontend/src/api/digitalTwinApi.ts
 *
 * Dedicated client for the SeismoFNO Digital-Twin Application API endpoints.
 */

export interface SystemInfo {
  status: string;
  version: string;
  model_loaded: boolean;
  device: string;
  model_parameters: number;
  checkpoint_path: string;
  config_path: string;
  modes: number;
  width: number;
  is_mock: boolean;
  scientific_integrity: string;
}

export interface IndianEarthquake {
  id: string;
  name: string;
  year: number;
  magnitude: number;
  depth_km: number;
  latitude: number;
  longitude: number;
  is1893_zone: string;
  region_state: string;
  tectonic_regime: string;
  historical_notes: string;
  associated_peer_proxy: string;
}

export interface PeerRecord {
  record_id: string;
  filename: string;
  earthquake_name: string;
  year: number;
  station: string;
  magnitude: number;
  r_rup_km: number;
  vs30_ms: number;
  raw_pga_g: number;
  raw_dt: number;
  source: string;
}

export interface BuildingArchetype {
  id: string;
  name: string;
  structural_system: string;
  stories: number;
  total_height_m: number;
  story_height_m: number;
  fundamental_period_s: number;
  damping_ratio: number;
  yield_displacement_m: number;
  post_yield_ratio: number;
  material_type: string;
  seismic_design_category: string;
  description: string;
}

export interface ScenarioInputPayload {
  earthquake_id: string;
  pga_g: number;
  dt?: number;
  duration_sec?: number;
  T0: number;
  damping_ratio: number;
  stiffness?: number;
  yield_displacement_m: number;
  post_yield_ratio: number;
  material_type: "bilinear" | "elastic";
  mass_kg?: number;
  building_height_m?: number;
  include_ground_truth?: boolean;
  stride?: number;
}

export interface TrajectoryData {
  time: number[];
  ag: number[];
  u: number[];
  v: number[];
  fr: number[];
  up: number[];
  eh: number[];
  u_gt?: number[];
  fr_gt?: number[];
  eh_gt?: number[];
}

export interface ValidationComparison {
  relative_l2_u_percent: number;
  relative_l2_fr_percent: number;
  relative_l2_eh_percent: number;
  peak_u_error_percent: number;
  rmse_u_mm: number;
  speedup_factor: number;
  fno_latency_ms: number;
  opensees_latency_ms: number;
  benchmark_status: string;
}

export interface ScenarioPredictionResponse {
  scenario: Record<string, any>;
  trajectories: TrajectoryData;
  metrics: Record<string, any>;
  validation?: ValidationComparison;
  inference_time_ms: number;
  is_mock: boolean;
  uncertainty_note: string;
  model_provenance: Record<string, any>;
}

const API_BASE = "";

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${API_BASE}/api/v1/system/info`);
  if (!res.ok) throw new Error(`Failed to fetch system info (${res.status})`);
  return await res.json();
}

export async function fetchEarthquakeCatalog(): Promise<{
  indian_catalog: IndianEarthquake[];
  peer_records: PeerRecord[];
  total_indian: number;
  total_peer: number;
}> {
  const res = await fetch(`${API_BASE}/api/v1/earthquakes`);
  if (!res.ok) throw new Error(`Failed to fetch earthquake catalog (${res.status})`);
  return await res.json();
}

export async function fetchBuildingArchetypes(): Promise<{
  buildings: BuildingArchetype[];
  total: number;
}> {
  const res = await fetch(`${API_BASE}/api/v1/buildings`);
  if (!res.ok) throw new Error(`Failed to fetch building archetypes (${res.status})`);
  return await res.json();
}

export async function validateScenario(payload: ScenarioInputPayload): Promise<{
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  domain_status: string;
  computed_properties: Record<string, any>;
}> {
  const res = await fetch(`${API_BASE}/api/v1/scenario/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return await res.json();
}

export async function predictDigitalTwin(payload: ScenarioInputPayload): Promise<ScenarioPredictionResponse> {
  const res = await fetch(`${API_BASE}/api/v1/scenario/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail?.errors?.join("; ") || err.detail || `Prediction failed (${res.status})`);
  }
  return await res.json();
}
