import type { PresetRecord, SimulationPayload, SimulationResponse, SystemHealth } from "../types/simulation";

const BASE_URL = "";

export async function fetchPresets(): Promise<PresetRecord[]> {
  const res = await fetch(`${BASE_URL}/api/presets`);
  if (!res.ok) {
    throw new Error(`Failed to fetch presets: ${res.statusText}`);
  }
  const data = await res.json();
  return data.presets || [];
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) {
    throw new Error(`Failed to fetch health status: ${res.statusText}`);
  }
  return await res.json();
}

export async function runSimulationAPI(payload: SimulationPayload): Promise<SimulationResponse> {
  const res = await fetch(`${BASE_URL}/api/simulate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.error || `Simulation failed with status ${res.status}`);
  }

  return await res.json();
}
