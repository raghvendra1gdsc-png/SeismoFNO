import React, { useState, useEffect, useMemo } from "react";
import {
  GitCompare,
  HelpCircle,
  Play,
} from "lucide-react";
import { predictDigitalTwin, type ScenarioInputPayload, type ScenarioPredictionResponse } from "../../api/digitalTwinApi";

interface ScenarioLabViewProps {
  baseScenario: ScenarioInputPayload;
}

type InterventionType =
  | "STIFFNESS_PLUS_20"
  | "YIELD_PLUS_20"
  | "DAMPING_PLUS_2PC"
  | "ALTERNATIVE_EARTHQUAKE"
  | "CUSTOM";

export const ScenarioLabView: React.FC<ScenarioLabViewProps> = ({ baseScenario }) => {
  const [intervention, setIntervention] = useState<InterventionType>("YIELD_PLUS_20");

  // Comparison Scenarios
  const [scenarioA, setScenarioA] = useState<ScenarioInputPayload>(baseScenario);
  const [scenarioB, setScenarioB] = useState<ScenarioInputPayload>({
    ...baseScenario,
    yield_displacement_m: baseScenario.yield_displacement_m * 1.2,
  });

  const [predA, setPredA] = useState<ScenarioPredictionResponse | null>(null);
  const [predB, setPredB] = useState<ScenarioPredictionResponse | null>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [activeOverlay, setActiveOverlay] = useState<"disp" | "force" | "energy">("disp");

  // Apply Intervention Preset to Scenario B
  const applyIntervention = (type: InterventionType) => {
    setIntervention(type);
    const newB = { ...scenarioA };

    if (type === "STIFFNESS_PLUS_20") {
      newB.T0 = parseFloat((scenarioA.T0 / Math.sqrt(1.2)).toFixed(3));
    } else if (type === "YIELD_PLUS_20") {
      newB.yield_displacement_m = parseFloat((scenarioA.yield_displacement_m * 1.2).toFixed(4));
    } else if (type === "DAMPING_PLUS_2PC") {
      newB.damping_ratio = parseFloat((scenarioA.damping_ratio + 0.02).toFixed(3));
    } else if (type === "ALTERNATIVE_EARTHQUAKE") {
      newB.earthquake_id = "IND-2001-BHUJ";
      newB.pga_g = 0.40;
    }
    setScenarioB(newB);
  };

  // Run Comparative Digital-Twin Execution
  const runComparison = async () => {
    setIsComparing(true);
    try {
      const [resA, resB] = await Promise.all([
        predictDigitalTwin(scenarioA),
        predictDigitalTwin(scenarioB),
      ]);
      setPredA(resA);
      setPredB(resB);
    } catch (err) {
      console.error("Comparison execution failed:", err);
    } finally {
      setIsComparing(false);
    }
  };

  // Initial Run on Mount or baseScenario change
  useEffect(() => {
    setScenarioA(baseScenario);
    applyIntervention(intervention);
  }, [baseScenario]);

  useEffect(() => {
    runComparison();
  }, [scenarioA, scenarioB.T0, scenarioB.yield_displacement_m, scenarioB.damping_ratio, scenarioB.earthquake_id]);

  // Derived Delta Computations
  const mA = predA?.metrics;
  const mB = predB?.metrics;

  const calcDelta = (valB?: number, valA?: number) => {
    if (valA === undefined || valB === undefined || Math.abs(valA) < 1e-6) return 0.0;
    return ((valB - valA) / Math.abs(valA)) * 100.0;
  };

  const deltaU = calcDelta(mB?.peak_displacement_mm, mA?.peak_displacement_mm);
  const deltaDrift = calcDelta(mB?.drift_ratio_percent, mA?.drift_ratio_percent);
  const deltaMu = calcDelta(mB?.ductility_demand_mu, mA?.ductility_demand_mu);
  const deltaFR = calcDelta(mB?.peak_restoring_force_N, mA?.peak_restoring_force_N);
  const deltaEh = calcDelta(mB?.total_hysteretic_energy_J, mA?.total_hysteretic_energy_J);

  // Generate Grounded "What Changed?" Explanation
  const whatChangedExplanation = useMemo(() => {
    if (!mA || !mB) return "Awaiting comparative simulation...";

    let inputDesc = "";
    if (intervention === "STIFFNESS_PLUS_20") {
      inputDesc = `Lateral stiffness increased by +20% (fundamental period reduced from ${scenarioA.T0.toFixed(2)}s to ${scenarioB.T0.toFixed(2)}s).`;
    } else if (intervention === "YIELD_PLUS_20") {
      inputDesc = `Yield strength capacity increased by +20% (yield displacement uy expanded from ${(scenarioA.yield_displacement_m * 1000).toFixed(1)} mm to ${(scenarioB.yield_displacement_m * 1000).toFixed(1)} mm).`;
    } else if (intervention === "DAMPING_PLUS_2PC") {
      inputDesc = `Supplemental damping increased by +2.0% (viscous damping ratio zeta modified from ${(scenarioA.damping_ratio * 100).toFixed(1)}% to ${(scenarioB.damping_ratio * 100).toFixed(1)}%).`;
    } else if (intervention === "ALTERNATIVE_EARTHQUAKE") {
      inputDesc = `Excitation altered from ${scenarioA.earthquake_id} to ${scenarioB.earthquake_id}.`;
    } else {
      inputDesc = "Custom multi-parameter intervention applied.";
    }

    const dispDir = deltaU <= 0 ? "reduced" : "increased";
    const driftDir = deltaDrift <= 0 ? "decreased" : "increased";

    return (
      `${inputDesc}\n\n` +
      `ENGINEERING IMPACT ON DYNAMIC DEMAND:\n` +
      `• Peak structural displacement ${dispDir} from ${mA.peak_displacement_mm?.toFixed(1)} mm to ${mB.peak_displacement_mm?.toFixed(1)} mm (${deltaU > 0 ? "+" : ""}${deltaU.toFixed(1)}%).\n` +
      `• Maximum inter-story drift ${driftDir} from ${mA.drift_ratio_percent?.toFixed(2)}% to ${mB.drift_ratio_percent?.toFixed(2)}% (${deltaDrift > 0 ? "+" : ""}${deltaDrift.toFixed(1)}%).\n` +
      `• Hysteretic dissipated energy changed from ${mA.total_hysteretic_energy_J?.toFixed(1)} J to ${mB.total_hysteretic_energy_J?.toFixed(1)} J (${deltaEh > 0 ? "+" : ""}${deltaEh.toFixed(1)}%).\n` +
      `• Initial plastic yield onset: Scenario A yielded at t = ${mA.yield_time_sec !== null && mA.yield_time_sec !== undefined ? `${mA.yield_time_sec?.toFixed(2)}s` : "None (Elastic)"} vs Scenario B at t = ${mB.yield_time_sec !== null && mB.yield_time_sec !== undefined ? `${mB.yield_time_sec?.toFixed(2)}s` : "None (Elastic)"}.`
    );
  }, [mA, mB, intervention, scenarioA, scenarioB, deltaU, deltaDrift, deltaEh]);

  return (
    <div className="p-6 space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-4 bg-white p-5 rounded border">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#161616]">
              SCENARIO LAB — DUAL DIGITAL-TWIN COMPARISON
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-semibold">
              FEATURED COMPARATIVE TOOL
            </span>
          </div>
          <p className="text-xs text-[#525252] mt-1 font-sans">
            Execute side-by-side scenario evaluations in sub-milliseconds to assess retrofits, stiffness variations, and seismic sensitivity.
          </p>
        </div>

        <button
          onClick={runComparison}
          disabled={isComparing}
          className="px-4 py-2 bg-[#0F62FE] hover:bg-[#0353E9] disabled:opacity-50 text-white text-xs font-mono font-bold rounded flex items-center space-x-2 cursor-pointer transition shadow-sm"
        >
          <Play size={14} />
          <span>{isComparing ? "EVALUATING BOTH..." : "EVALUATE SCENARIOS (< 5 ms)"}</span>
        </button>
      </div>

      {/* Preset Intervention Selectors */}
      <div className="bg-white border border-[#E0E0E0] rounded p-4 space-y-2">
        <div className="text-xs font-mono font-bold text-[#525252] uppercase">
          Select Engineering Intervention for Scenario B:
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => applyIntervention("STIFFNESS_PLUS_20")}
            className={`px-3 py-1.5 rounded text-xs font-mono cursor-pointer transition border ${
              intervention === "STIFFNESS_PLUS_20"
                ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                : "bg-[#F4F4F4] border-[#E0E0E0] text-[#525252] hover:text-[#161616] hover:bg-[#EBEBEB]"
            }`}
          >
            +20% Lateral Stiffness (Retrofit Bracing)
          </button>
          <button
            onClick={() => applyIntervention("YIELD_PLUS_20")}
            className={`px-3 py-1.5 rounded text-xs font-mono cursor-pointer transition border ${
              intervention === "YIELD_PLUS_20"
                ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                : "bg-[#F4F4F4] border-[#E0E0E0] text-[#525252] hover:text-[#161616] hover:bg-[#EBEBEB]"
            }`}
          >
            +20% Yield Strength (Steel Jacketing)
          </button>
          <button
            onClick={() => applyIntervention("DAMPING_PLUS_2PC")}
            className={`px-3 py-1.5 rounded text-xs font-mono cursor-pointer transition border ${
              intervention === "DAMPING_PLUS_2PC"
                ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                : "bg-[#F4F4F4] border-[#E0E0E0] text-[#525252] hover:text-[#161616] hover:bg-[#EBEBEB]"
            }`}
          >
            +2.0% Supplemental Damping (Fluid Dampers)
          </button>
          <button
            onClick={() => applyIntervention("ALTERNATIVE_EARTHQUAKE")}
            className={`px-3 py-1.5 rounded text-xs font-mono cursor-pointer transition border ${
              intervention === "ALTERNATIVE_EARTHQUAKE"
                ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                : "bg-[#F4F4F4] border-[#E0E0E0] text-[#525252] hover:text-[#161616] hover:bg-[#EBEBEB]"
            }`}
          >
            Alternative Earthquake (Bhuj 2001)
          </button>
          <button
            onClick={() => setIntervention("CUSTOM")}
            className={`px-3 py-1.5 rounded text-xs font-mono cursor-pointer transition border ${
              intervention === "CUSTOM"
                ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                : "bg-[#F4F4F4] border-[#E0E0E0] text-[#525252] hover:text-[#161616] hover:bg-[#EBEBEB]"
            }`}
          >
            Custom Scenario
          </button>
        </div>
      </div>

      {/* Side-by-Side Metric Comparison Table (White table, #F4F4F4 header, alternating rows) */}
      <div className="bg-white border border-[#E0E0E0] rounded overflow-hidden">
        <div className="p-4 border-b border-[#E0E0E0] flex items-center justify-between text-xs font-mono bg-white">
          <div className="flex items-center space-x-2">
            <GitCompare size={15} className="text-[#0F62FE]" />
            <span className="font-bold text-[#161616]">QUANTITATIVE COMPARISON: SCENARIO A vs SCENARIO B</span>
          </div>
          <div className="flex items-center space-x-4 text-[11px]">
            <span className="flex items-center space-x-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-[#0F62FE] inline-block" />
              <span className="text-[#161616] font-semibold">Scenario A (Baseline)</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-[#B28600] inline-block" />
              <span className="text-[#161616] font-semibold">Scenario B (Intervention)</span>
            </span>
          </div>
        </div>

        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-[#F4F4F4] text-[#525252] uppercase text-[10px] border-b border-[#E0E0E0]">
            <tr>
              <th className="p-3 font-sans font-semibold">Engineering Parameter / Metric</th>
              <th className="p-3">
                <span className="inline-flex items-center">
                  <span className="w-2 h-2 rounded-full bg-[#0F62FE] mr-1.5" />
                  Scenario A (Baseline)
                </span>
              </th>
              <th className="p-3">
                <span className="inline-flex items-center">
                  <span className="w-2 h-2 rounded-full bg-[#B28600] mr-1.5" />
                  Scenario B (Intervention)
                </span>
              </th>
              <th className="p-3 text-right">Delta (%)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#E0E0E0]">
            {/* Row 1: Peak Relative Displacement */}
            <tr className="bg-white hover:bg-[#F4F4F4] transition-colors">
              <td className="p-3 font-semibold text-[#161616] font-sans">Peak Relative Displacement (mm)</td>
              <td className="p-3 text-[#161616] font-bold font-mono">{mA?.peak_displacement_mm?.toFixed(1) ?? "—"} mm</td>
              <td className="p-3 text-[#161616] font-bold font-mono">{mB?.peak_displacement_mm?.toFixed(1) ?? "—"} mm</td>
              <td
                className={`p-3 text-right font-bold font-mono ${
                  deltaU <= 0 ? "text-[#198038]" : "text-[#DA1E28]"
                }`}
              >
                {deltaU > 0 ? `+${deltaU.toFixed(1)}%` : `${deltaU.toFixed(1)}%`}
              </td>
            </tr>

            {/* Row 2: Inter-Story Drift Demand */}
            <tr className="bg-[#FAFAFA] hover:bg-[#F4F4F4] transition-colors">
              <td className="p-3 font-semibold text-[#161616] font-sans">Inter-Story Drift Demand (%)</td>
              <td className="p-3 text-[#161616] font-mono">{mA?.drift_ratio_percent?.toFixed(2) ?? "—"} %</td>
              <td className="p-3 text-[#161616] font-mono">{mB?.drift_ratio_percent?.toFixed(2) ?? "—"} %</td>
              <td
                className={`p-3 text-right font-bold font-mono ${
                  deltaDrift <= 0 ? "text-[#198038]" : "text-[#DA1E28]"
                }`}
              >
                {deltaDrift > 0 ? `+${deltaDrift.toFixed(1)}%` : `${deltaDrift.toFixed(1)}%`}
              </td>
            </tr>

            {/* Row 3: Ductility Demand */}
            <tr className="bg-white hover:bg-[#F4F4F4] transition-colors">
              <td className="p-3 font-semibold text-[#161616] font-sans">Ductility Demand (μ = u_max / uy)</td>
              <td className="p-3 text-[#161616] font-mono">{mA?.ductility_demand_mu?.toFixed(2) ?? "—"}</td>
              <td className="p-3 text-[#161616] font-mono">{mB?.ductility_demand_mu?.toFixed(2) ?? "—"}</td>
              <td
                className={`p-3 text-right font-bold font-mono ${
                  deltaMu <= 0 ? "text-[#198038]" : "text-[#DA1E28]"
                }`}
              >
                {deltaMu > 0 ? `+${deltaMu.toFixed(1)}%` : `${deltaMu.toFixed(1)}%`}
              </td>
            </tr>

            {/* Row 4: Peak Restoring Force */}
            <tr className="bg-[#FAFAFA] hover:bg-[#F4F4F4] transition-colors">
              <td className="p-3 font-semibold text-[#161616] font-sans">Peak Restoring Force (N)</td>
              <td className="p-3 text-[#161616] font-mono">{mA?.peak_restoring_force_N?.toFixed(1) ?? "—"} N</td>
              <td className="p-3 text-[#161616] font-mono">{mB?.peak_restoring_force_N?.toFixed(1) ?? "—"} N</td>
              <td
                className={`p-3 text-right font-bold font-mono ${
                  deltaFR <= 0 ? "text-[#198038]" : "text-[#DA1E28]"
                }`}
              >
                {deltaFR > 0 ? `+${deltaFR.toFixed(1)}%` : `${deltaFR.toFixed(1)}%`}
              </td>
            </tr>

            {/* Row 5: Total Dissipated Hysteretic Energy */}
            <tr className="bg-white hover:bg-[#F4F4F4] transition-colors">
              <td className="p-3 font-semibold text-[#161616] font-sans">Total Dissipated Hysteretic Energy (J)</td>
              <td className="p-3 text-[#161616] font-bold font-mono">{mA?.total_hysteretic_energy_J?.toFixed(1) ?? "—"} J</td>
              <td className="p-3 text-[#161616] font-bold font-mono">{mB?.total_hysteretic_energy_J?.toFixed(1) ?? "—"} J</td>
              <td
                className={`p-3 text-right font-bold font-mono ${
                  deltaEh <= 0 ? "text-[#198038]" : "text-[#DA1E28]"
                }`}
              >
                {deltaEh > 0 ? `+${deltaEh.toFixed(1)}%` : `${deltaEh.toFixed(1)}%`}
              </td>
            </tr>

            {/* Row 6: Initial Yield Onset Time */}
            <tr className="bg-[#FAFAFA] hover:bg-[#F4F4F4] transition-colors">
              <td className="p-3 font-semibold text-[#161616] font-sans">Initial Yield Onset Time</td>
              <td className="p-3 text-[#525252] font-mono">
                {mA?.yield_time_sec !== null && mA?.yield_time_sec !== undefined
                  ? `t = ${mA.yield_time_sec.toFixed(2)}s`
                  : "Elastic (No Yield)"}
              </td>
              <td className="p-3 text-[#525252] font-mono">
                {mB?.yield_time_sec !== null && mB?.yield_time_sec !== undefined
                  ? `t = ${mB.yield_time_sec.toFixed(2)}s`
                  : "Elastic (No Yield)"}
              </td>
              <td className="p-3 text-right text-[#525252] text-[11px] font-mono">
                {(mB?.yield_time_sec ?? 0) > (mA?.yield_time_sec ?? 0) ? (
                  <span className="text-[#198038] font-bold">Yield Delayed</span>
                ) : (
                  "Similar"
                )}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Comparative Overlaid Trajectory Plot */}
      <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-3">
        <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3 text-xs font-mono">
          <div className="flex items-center space-x-2">
            <span className="font-bold text-[#161616]">OVERLAID DYNAMIC RESPONSE TRAJECTORIES</span>
          </div>
          <div className="flex space-x-1">
            <button
              onClick={() => setActiveOverlay("disp")}
              className={`px-3 py-1 rounded cursor-pointer transition font-medium ${
                activeOverlay === "disp" ? "bg-[#0F62FE] text-white font-bold" : "text-[#525252] hover:text-[#161616] hover:bg-[#F4F4F4]"
              }`}
            >
              Displacement u(t)
            </button>
            <button
              onClick={() => setActiveOverlay("force")}
              className={`px-3 py-1 rounded cursor-pointer transition font-medium ${
                activeOverlay === "force" ? "bg-[#0F62FE] text-white font-bold" : "text-[#525252] hover:text-[#161616] hover:bg-[#F4F4F4]"
              }`}
            >
              Restoring Force FR(t)
            </button>
            <button
              onClick={() => setActiveOverlay("energy")}
              className={`px-3 py-1 rounded cursor-pointer transition font-medium ${
                activeOverlay === "energy" ? "bg-[#0F62FE] text-white font-bold" : "text-[#525252] hover:text-[#161616] hover:bg-[#F4F4F4]"
              }`}
            >
              Energy Dissipation Eh(t)
            </button>
          </div>
        </div>

        {/* Dual Waveform Overlay Canvas - #F4F4F4 chart background */}
        <div className="h-64 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-center p-3 relative">
          {predA && predB && predA.trajectories.u.length > 0 ? (
            <svg className="w-full h-full" viewBox="0 0 600 240">
              <line x1="20" y1="120" x2="580" y2="120" stroke="#8D8D8D" strokeDasharray="3 3" />

              {/* Scenario A Line (Carbon blue-60 solid) */}
              <path
                d={(activeOverlay === "disp"
                  ? predA.trajectories.u
                  : activeOverlay === "force"
                  ? predA.trajectories.fr
                  : predA.trajectories.eh
                )
                  .map((val, i, arr) => {
                    const x = 20 + (i / (arr.length - 1)) * 560;
                    const maxVal =
                      Math.max(
                        ...arr.map(Math.abs),
                        ...(activeOverlay === "disp"
                          ? predB.trajectories.u
                          : activeOverlay === "force"
                          ? predB.trajectories.fr
                          : predB.trajectories.eh
                        ).map(Math.abs)
                      ) || 1e-4;
                    const y = activeOverlay === "energy" ? 220 - (val / maxVal) * 200 : 120 - (val / maxVal) * 100;
                    return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                  })
                  .join(" ")}
                fill="none"
                stroke="#0F62FE"
                strokeWidth="2.0"
              />

              {/* Scenario B Line (Carbon amber-60 dashed) */}
              <path
                d={(activeOverlay === "disp"
                  ? predB.trajectories.u
                  : activeOverlay === "force"
                  ? predB.trajectories.fr
                  : predB.trajectories.eh
                )
                  .map((val, i, arr) => {
                    const x = 20 + (i / (arr.length - 1)) * 560;
                    const maxVal =
                      Math.max(
                        ...arr.map(Math.abs),
                        ...(activeOverlay === "disp"
                          ? predA.trajectories.u
                          : activeOverlay === "force"
                          ? predA.trajectories.fr
                          : predA.trajectories.eh
                        ).map(Math.abs)
                      ) || 1e-4;
                    const y = activeOverlay === "energy" ? 220 - (val / maxVal) * 200 : 120 - (val / maxVal) * 100;
                    return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                  })
                  .join(" ")}
                fill="none"
                stroke="#B28600"
                strokeWidth="2.0"
                strokeDasharray="4 3"
              />
            </svg>
          ) : (
            <div className="text-[#525252] text-xs font-mono">Evaluating scenarios...</div>
          )}
        </div>
      </div>

      {/* "WHAT CHANGED?" Root-Cause Engineering Analysis Panel */}
      <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-3">
        <div className="flex items-center space-x-2 text-[#0F62FE] font-mono font-bold text-xs uppercase tracking-wider">
          <HelpCircle size={16} />
          <span>"WHAT CHANGED?" — ENGINEERING ROOT-CAUSE INTERPRETATION</span>
        </div>
        <div className="bg-[#F4F4F4] p-4 rounded border border-[#E0E0E0] font-mono text-xs text-[#161616] whitespace-pre-line leading-relaxed">
          {whatChangedExplanation}
        </div>
        <p className="text-[10px] font-mono text-[#525252]">
          All conclusions are deterministically derived from computed spectral surrogate integration and physical mechanics formulas.
        </p>
      </div>
    </div>
  );
};
