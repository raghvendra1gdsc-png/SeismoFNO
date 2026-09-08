export type MaterialType = "bilinear" | "elastic";

export interface PresetRecord {
  id: string;
  name: string;
  path: string;
  pga_g?: number;
  duration_s?: number;
  dt_s?: number;
}

export interface SimulationPayload {
  preset_id: string;
  pga_g: number;
  T: number;
  zeta: number;
  material_type: MaterialType;
  u_y: number;
  alpha: number;
  custom_content?: string | null;
  custom_filename?: string | null;
}

export interface SimulationMetrics {
  err_u_rel_l2: number;
  err_fr_rel_l2: number;
  err_eh_rel_l2: number;
  err_umax_rel: number;
  u_max_gt_m: number;
  u_max_pred_m: number;
  ductility_mu: number;
  pga_g: number;
  t_fno_ms: number;
  t_gt_ms: number;
  speedup: number;
  device: string;
}

import type { OODSummary } from "./agent";

export interface SimulationResponse {
  time: number[];
  ag: number[];
  u_gt: number[];
  u_pred: number[];
  fr_gt: number[];
  fr_pred: number[];
  eh_gt: number[];
  eh_pred: number[];
  metrics: SimulationMetrics;
  ood?: OODSummary;
  record_name?: string;
  status?: string;
  error?: string;
}

export interface SystemHealth {
  status: string;
  model_loaded: boolean;
  device: string;
  num_params: number;
}
