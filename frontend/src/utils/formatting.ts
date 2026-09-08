import type { SimulationResponse } from "../types/simulation";

export function formatNumber(val: number, decimals: number = 2): string {
  if (val === undefined || val === null || isNaN(val)) return "—";
  return val.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatScientific(val: number, decimals: number = 2): string {
  if (val === undefined || val === null || isNaN(val)) return "—";
  return val.toExponential(decimals);
}

export function exportSimulationCSV(data: SimulationResponse, filename: string = "seismofno_simulation.csv") {
  const headers = ["Time_s", "GroundAccel_mps2", "Displacement_GT_m", "Displacement_FNO_m", "Force_GT_N", "Force_FNO_N", "Energy_GT_J", "Energy_FNO_J"];
  const rows: string[] = [headers.join(",")];

  for (let i = 0; i < data.time.length; i++) {
    rows.push([
      data.time[i].toFixed(4),
      data.ag[i].toFixed(6),
      data.u_gt[i].toFixed(8),
      data.u_pred[i].toFixed(8),
      data.fr_gt[i].toFixed(6),
      data.fr_pred[i].toFixed(6),
      data.eh_gt[i].toFixed(6),
      data.eh_pred[i].toFixed(6),
    ].join(","));
  }

  const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function exportSimulationJSON(data: SimulationResponse, filename: string = "seismofno_simulation.json") {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
