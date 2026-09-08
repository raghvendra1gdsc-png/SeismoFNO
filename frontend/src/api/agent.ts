/**
 * frontend/src/api/agent.ts
 *
 * Typed API client for SeismoAgent Gateway endpoints and WebSockets.
 * Interacts with FastAPI server on /api/... without exposing secrets.
 */

import type {
  SystemHealth,
  ToolsResponse,
  PresetRecord,
  SimulationResponse,
  AgentRunResponse,
  StreamEvent,
  SweepResponse,
  ReportResponse,
} from "../types/agent";

const BASE_URL = "";

/**
 * Fetch system health and runtime indicators from backend.
 */
export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${BASE_URL}/api/agent/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Fetch list of registered deterministic tools with their schemas.
 */
export async function fetchTools(): Promise<ToolsResponse> {
  const res = await fetch(`${BASE_URL}/api/agent/tools`);
  if (!res.ok) {
    throw new Error(`Failed to fetch tool registry: ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Fetch ground motion presets.
 */
export async function fetchPresets(): Promise<PresetRecord[]> {
  const res = await fetch(`${BASE_URL}/api/presets`);
  if (!res.ok) {
    throw new Error(`Failed to fetch presets: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return data.presets || [];
}

/**
 * Run deterministic direct simulation (FNO + OpenSeesPy reference).
 */
export async function runSimulation(payload: {
  preset_id?: string;
  pga_g: number;
  T: number;
  zeta: number;
  material_type: "bilinear" | "elastic";
  u_y: number;
  alpha: number;
  mass?: number;
  custom_content?: string | null;
  custom_filename?: string | null;
  solver?: "opensees" | "newmark";
}): Promise<SimulationResponse> {
  const res = await fetch(`${BASE_URL}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || errData.message || `Simulation failed (${res.status})`);
  }
  return await res.json();
}

/**
 * Run synchronous/bounded agent query via POST /api/agent/run.
 */
export async function runAgent(
  message: string,
  context?: Record<string, unknown>
): Promise<AgentRunResponse> {
  const res = await fetch(`${BASE_URL}/api/agent/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, context }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    const msg = errData?.error?.message || `Agent run failed (${res.status})`;
    throw new Error(msg);
  }
  return await res.json();
}

/**
 * Establish WebSocket connection for live telemetry streaming.
 * Returns an unsubscribe/disconnect function.
 */
export function createAgentStream(
  message: string,
  onEvent: (event: StreamEvent) => void,
  onError: (error: string) => void,
  onComplete: () => void
): () => void {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/agent/stream`;

  let ws: WebSocket | null = null;
  let closedManually = false;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ message }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const parsed: StreamEvent = JSON.parse(event.data);
        onEvent(parsed);

        if (parsed.event === "completed") {
          onComplete();
          if (ws) ws.close();
        } else if (parsed.event === "error") {
          onError(parsed.message || "An unexpected error occurred during execution.");
          if (ws) ws.close();
        }
      } catch (err) {
        onError(`Failed to parse stream event: ${err}`);
      }
    };

    ws.onerror = () => {
      if (!closedManually) {
        onError("WebSocket connection encountered an error.");
      }
    };

    ws.onclose = () => {
      if (!closedManually) {
        onComplete();
      }
    };
  } catch (err) {
    onError(`Failed to initialize WebSocket: ${err}`);
  }

  return () => {
    closedManually = true;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close();
    }
  };
}

/**
 * Execute parametric sweep across structural parameters.
 */
export async function runSweep(payload: {
  param_name: "pga_g" | "T" | "zeta" | "u_y" | "alpha" | "mass";
  param_values: number[];
  preset_id?: string;
  pga_g?: number;
  T?: number;
  zeta?: number;
  material_type?: "bilinear" | "elastic";
  u_y?: number;
  alpha?: number;
}): Promise<SweepResponse> {
  const res = await fetch(`${BASE_URL}/api/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Sweep failed with status ${res.status}`);
  }
  return await res.json();
}

/**
 * Generate formal engineering audit report.
 */
export async function generateReport(payload: {
  preset_id?: string;
  pga_g?: number;
  T?: number;
  zeta?: number;
  material_type?: "bilinear" | "elastic";
  u_y?: number;
  alpha?: number;
  format?: "markdown" | "json";
}): Promise<ReportResponse> {
  const res = await fetch(`${BASE_URL}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Report generation failed (${res.status})`);
  }
  return await res.json();
}
