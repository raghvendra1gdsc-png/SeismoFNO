/**
 * frontend/src/types/agent.ts
 *
 * Strongly typed interfaces for SeismoAgent Gateway API contracts.
 * Matches backend Pydantic schemas in seismo_agent/server/schemas.py.
 */

export interface SystemHealth {
  status: string;
  service: string;
  version: string;
  orchestrator: string;
  nebius_configured: boolean;
  nebius_live_verified: boolean;
  model: string;
  research_core: string;
  tools_count: number;
  device: string;
  concurrency_policy: string;
  opensees_execution: string;
}

export interface ToolInfo {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface ToolsResponse {
  count: number;
  tools: ToolInfo[];
}

export interface ToolTraceItem {
  step: number;
  tool_name: string;
  tool_input: Record<string, unknown>;
  success: boolean;
  runtime_ms: number;
  output_summary: Record<string, unknown>;
  error?: string;
}

export interface TimingBreakdown {
  llm_latency_ms?: number;
  tool_latency_ms?: number;
  orchestrator_latency_ms?: number;
  total_server_latency_ms: number;
}

export interface AgentRunResponse {
  request_id: string;
  status: "completed" | "error" | "aborted";
  final_response: string;
  tool_trace: ToolTraceItem[];
  tool_results: Record<string, unknown>;
  steps_used: number;
  errors: string[];
  timing: TimingBreakdown;
}

export interface StreamEvent {
  event:
    | "request_started"
    | "planning"
    | "tool_started"
    | "tool_completed"
    | "tool_failed"
    | "synthesis_started"
    | "completed"
    | "error";
  request_id: string;
  step?: number;
  tool?: string;
  status?: string;
  elapsed_ms?: number;
  summary?: Record<string, unknown>;
  message?: string;
  data?: Record<string, unknown>;
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
  phase_error_rad?: number | null;
  yield_time_error_ms?: number | null;
  residual_drift_error_m?: number | null;
}

export interface OODSummary {
  is_ood: boolean;
  domain_violations: string[];
  uncertainty_status: "NOT_QUANTIFIED" | string;
  recommendation: string;
}

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
  ood: OODSummary;
  record_name: string;
  status: string;
  error?: string;
}

export interface PresetRecord {
  id: string;
  name: string;
  path: string;
  pga_g?: number;
  duration_s?: number;
  dt_s?: number;
}

export interface SweepPointResult {
  parameter_value: number;
  u_max_fno?: number | null;
  u_max_physics?: number | null;
  rel_l2_error_pct?: number | null;
  rel_l2_u_pct?: number | null;
  fno_runtime_ms?: number | null;
  physics_runtime_ms?: number | null;
  speedup?: number | null;
  converged?: boolean;
}

export interface SweepResponse {
  param_name: string;
  results: SweepPointResult[];
  total_evaluations: number;
}

export interface ReportResponse {
  report_title: string;
  content: string;
  format: string;
  metrics: Record<string, unknown>;
}

export interface ExperimentRecord {
  id: string;
  timestamp: string;
  recordName: string;
  pgaG: number;
  T: number;
  zeta: number;
  materialType: string;
  uy: number;
  alpha: number;
  errRelL2: number;
  speedup: number;
  status: "success" | "error";
  simulationData?: SimulationResponse;
}
