import React, { useState } from "react";
import { Sliders, Play, Loader2 } from "lucide-react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { runSweep } from "../../api/agent";
import type { SweepResponse } from "../../types/agent";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface ParametricSweepWorkspaceProps {
  currentPresetId: string;
  currentPga: number;
  currentT: number;
  currentZeta: number;
  currentMaterialType: "bilinear" | "elastic";
  currentUy: number;
  currentAlpha: number;
}

export const ParametricSweepWorkspace: React.FC<ParametricSweepWorkspaceProps> = ({
  currentPresetId,
  currentPga,
  currentT,
  currentZeta,
  currentMaterialType,
  currentUy,
  currentAlpha,
}) => {
  const [paramName, setParamName] = useState<"T" | "zeta" | "u_y" | "alpha" | "pga_g">("T");
  const [startVal, setStartVal] = useState<number>(0.2);
  const [endVal, setEndVal] = useState<number>(1.2);
  const [steps, setSteps] = useState<number>(5);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [sweepData, setSweepData] = useState<SweepResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunSweep = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const stepSize = steps > 1 ? (endVal - startVal) / (steps - 1) : 0;
      const values: number[] = [];
      for (let i = 0; i < steps; i++) {
        values.push(Number((startVal + i * stepSize).toFixed(4)));
      }

      const res = await runSweep({
        param_name: paramName,
        param_values: values,
        preset_id: currentPresetId,
        pga_g: currentPga,
        T: currentT,
        zeta: currentZeta,
        material_type: currentMaterialType,
        u_y: currentUy,
        alpha: currentAlpha,
      });

      setSweepData(res);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setIsLoading(false);
    }
  };

  // Prepare chart data if sweep exists
  const chartData = sweepData
    ? {
        labels: sweepData.results.map((r) => r.parameter_value.toString()),
        datasets: [
          {
            label: "SeismoFNO Prediction (u_max mm)",
            data: sweepData.results.map((r) => (r.u_max_fno != null ? r.u_max_fno * 1000 : null)),
            borderColor: "#00D2FF",
            backgroundColor: "rgba(0, 210, 255, 0.1)",
            borderWidth: 2,
            pointRadius: 4,
            pointBackgroundColor: "#00D2FF",
          },
          {
            label: "OpenSeesPy Reference (u_max mm)",
            data: sweepData.results.map((r) => (r.u_max_physics != null ? r.u_max_physics * 1000 : null)),
            borderColor: "#F59E0B",
            backgroundColor: "rgba(245, 158, 11, 0.1)",
            borderWidth: 2,
            pointRadius: 4,
            pointBackgroundColor: "#F59E0B",
          },
        ],
      }
    : null;

  return (
    <div className="p-4 space-y-4 font-mono text-xs select-none">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center space-x-2">
          <Sliders size={16} className="text-fno" />
          <span className="font-bold text-text-primary uppercase tracking-wider text-sm">
            PARAMETRIC EXPERIMENT WORKSPACE
          </span>
        </div>
        <div className="text-[11px] text-text-muted">
          SweepTool · Incremental Dynamic Analysis
        </div>
      </div>

      {/* Control Configuration Bar */}
      <div className="bg-panel-base border border-border-subtle rounded-md p-3 grid grid-cols-2 md:grid-cols-5 gap-3 items-end">
        <div>
          <label className="block text-[10px] text-text-muted uppercase mb-1">
            Sweep Parameter
          </label>
          <select
            value={paramName}
            onChange={(e) => {
              const p = e.target.value as any;
              setParamName(p);
              if (p === "T") {
                setStartVal(0.2);
                setEndVal(1.2);
              } else if (p === "zeta") {
                setStartVal(0.02);
                setEndVal(0.1);
              } else if (p === "u_y") {
                setStartVal(0.005);
                setEndVal(0.03);
              } else if (p === "alpha") {
                setStartVal(0.02);
                setEndVal(0.15);
              } else if (p === "pga_g") {
                setStartVal(0.1);
                setEndVal(0.8);
              }
            }}
            className="w-full bg-background-base border border-border-subtle rounded px-2 py-1 text-xs text-text-primary focus:border-fno focus:outline-none"
          >
            <option value="T">Period T (s)</option>
            <option value="zeta">Damping ratio ζ</option>
            <option value="u_y">Yield Disp u_y (m)</option>
            <option value="alpha">Post-yield ratio α</option>
            <option value="pga_g">Ground Accel PGA (g)</option>
          </select>
        </div>

        <div>
          <label className="block text-[10px] text-text-muted uppercase mb-1">Start Value</label>
          <input
            type="number"
            step="any"
            value={startVal}
            onChange={(e) => setStartVal(parseFloat(e.target.value) || 0)}
            className="w-full bg-background-base border border-border-subtle rounded px-2 py-1 text-xs text-text-primary focus:border-fno focus:outline-none"
          />
        </div>

        <div>
          <label className="block text-[10px] text-text-muted uppercase mb-1">End Value</label>
          <input
            type="number"
            step="any"
            value={endVal}
            onChange={(e) => setEndVal(parseFloat(e.target.value) || 0)}
            className="w-full bg-background-base border border-border-subtle rounded px-2 py-1 text-xs text-text-primary focus:border-fno focus:outline-none"
          />
        </div>

        <div>
          <label className="block text-[10px] text-text-muted uppercase mb-1">Steps Count</label>
          <input
            type="number"
            min="2"
            max="10"
            value={steps}
            onChange={(e) => setSteps(parseInt(e.target.value, 10) || 3)}
            className="w-full bg-background-base border border-border-subtle rounded px-2 py-1 text-xs text-text-primary focus:border-fno focus:outline-none"
          />
        </div>

        <div>
          <button
            onClick={handleRunSweep}
            disabled={isLoading}
            className="w-full flex items-center justify-center space-x-1.5 px-3 py-1.5 rounded bg-fno/10 text-fno border border-fno/30 hover:bg-fno hover:text-background-deep font-semibold transition disabled:opacity-50 cursor-pointer"
          >
            {isLoading ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
            <span>{isLoading ? "Running..." : "Run Sweep"}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-danger/10 border border-danger/30 rounded text-danger text-xs">
          Sweep execution failed: {error}
        </div>
      )}

      {/* Sweep Results Plot */}
      {chartData && (
        <div className="bg-panel-base border border-border-subtle rounded-md p-4 space-y-3">
          <div className="flex items-center justify-between text-xs text-text-muted border-b border-border-subtle/60 pb-2">
            <span className="text-text-primary font-bold">
              PEAK DISPLACEMENT vs {paramName.toUpperCase()}
            </span>
            <span>Evaluations: {sweepData?.total_evaluations} cases</span>
          </div>

          <div className="h-64">
            <Line
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#8B96A5", font: { family: "JetBrains Mono", size: 10 } },
                    title: { display: true, text: `Parameter: ${paramName}`, color: "#8B96A5" },
                  },
                  y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#8B96A5", font: { family: "JetBrains Mono", size: 10 } },
                    title: { display: true, text: "Peak Displacement (mm)", color: "#8B96A5" },
                  },
                },
                plugins: {
                  legend: {
                    labels: { color: "#E7ECF2", font: { family: "JetBrains Mono", size: 11 } },
                  },
                },
              }}
            />
          </div>
        </div>
      )}

      {/* Sweep Results Table */}
      {sweepData && sweepData.results.length > 0 && (
        <div className="bg-panel-base border border-border-subtle rounded-md overflow-hidden">
          <table className="w-full text-left text-xs">
            <thead className="bg-background-base text-[10px] uppercase text-text-muted border-b border-border-subtle">
              <tr>
                <th className="px-3 py-2">Value ({paramName})</th>
                <th className="px-3 py-2">FNO u_max (mm)</th>
                <th className="px-3 py-2">Physics u_max (mm)</th>
                <th className="px-3 py-2">Rel L2 Error (%)</th>
                <th className="px-3 py-2">Speedup Factor</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle/50 text-text-secondary">
              {sweepData.results.map((pt, idx) => (
                <tr key={idx}>
                  <td className="px-3 py-2 font-bold text-text-primary">{pt.parameter_value}</td>
                  <td className="px-3 py-2 text-fno font-semibold">
                    {pt.u_max_fno != null ? (pt.u_max_fno * 1000).toFixed(2) : "N/A"}
                  </td>
                  <td className="px-3 py-2 text-opensees font-semibold">
                    {pt.u_max_physics != null ? (pt.u_max_physics * 1000).toFixed(2) : "N/A"}
                  </td>
                  <td className="px-3 py-2 text-text-primary">
                    {(pt.rel_l2_u_pct ?? pt.rel_l2_error_pct) != null
                      ? `${(pt.rel_l2_u_pct ?? pt.rel_l2_error_pct)!.toFixed(2)}%`
                      : "N/A"}
                  </td>
                  <td className="px-3 py-2 text-energy font-bold">
                    {pt.physics_runtime_ms && pt.fno_runtime_ms
                      ? `${(pt.physics_runtime_ms / Math.max(1e-4, pt.fno_runtime_ms)).toFixed(1)}x`
                      : pt.speedup != null
                      ? `${pt.speedup.toFixed(1)}x`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
