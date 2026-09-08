import React, { useState } from "react";
import {
  Building2,
  Sliders,
  Play,
} from "lucide-react";
import type { BuildingArchetype, ScenarioPredictionResponse } from "../../api/digitalTwinApi";

interface StructuralTwinViewProps {
  prediction: ScenarioPredictionResponse | null;
  buildingArchetypes: BuildingArchetype[];
  selectedBuildingId: string;
  onSelectBuilding: (bld: BuildingArchetype) => void;
  onRunSimulation: () => void;
  isLoading: boolean;
  // Parameter setters
  T0: number;
  setT0: (val: number) => void;
  damping: number;
  setDamping: (val: number) => void;
  uy: number;
  setUy: (val: number) => void;
  alpha: number;
  setAlpha: (val: number) => void;
  materialType: "bilinear" | "elastic";
  setMaterialType: (val: "bilinear" | "elastic") => void;
  pgaG: number;
  setPgaG: (val: number) => void;
}

export const StructuralTwinView: React.FC<StructuralTwinViewProps> = ({
  prediction,
  buildingArchetypes,
  selectedBuildingId,
  onSelectBuilding,
  onRunSimulation,
  isLoading,
  T0,
  setT0,
  damping,
  setDamping,
  uy,
  setUy,
  alpha,
  setAlpha,
  materialType,
  setMaterialType,
  pgaG,
  setPgaG,
}) => {
  const [scrubIndex, setScrubIndex] = useState<number>(0);
  const [activeTabPlot, setActiveTabPlot] = useState<"disp" | "hysteresis" | "plastic" | "energy">("disp");

  const activeBuilding = buildingArchetypes.find((b) => b.id === selectedBuildingId) || buildingArchetypes[0];
  const traj = prediction?.trajectories;
  const metrics = prediction?.metrics;

  const totalPoints = traj?.time.length || 0;
  const currentIdx = Math.min(scrubIndex, totalPoints > 0 ? totalPoints - 1 : 0);

  const currTime = traj ? traj.time[currentIdx] : 0.0;
  const currU = traj ? traj.u[currentIdx] * 1000.0 : 0.0; // mm

  const peakU = metrics?.peak_displacement_mm || 50.0;
  const stories = activeBuilding?.stories || 3;

  // Compute story deflections based on standard first-mode shape
  const storyDeflections = Array.from({ length: stories }, (_, i) => {
    const fraction = (i + 1) / stories;
    const modeWeight = Math.pow(fraction, 1.2);
    return currU * modeWeight;
  });

  return (
    <div className="p-6 space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-4 bg-white p-5 rounded border">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#161616]">
              STRUCTURAL DIGITAL TWIN & DYNAMICS VISUALIZER
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-semibold">
              NONLINEAR KINEMATIC SIMULATION
            </span>
          </div>
          <p className="text-xs text-[#525252] mt-1 font-sans">
            Interactive structural oscillator physics: Continuous displacement, restoring force hysteresis, and plastic deformation state.
          </p>
        </div>

        <button
          onClick={onRunSimulation}
          disabled={isLoading}
          className="px-4 py-2 bg-[#0F62FE] hover:bg-[#0353E9] disabled:opacity-50 text-white text-xs font-mono font-bold rounded flex items-center space-x-2 cursor-pointer transition shadow-sm"
        >
          <Play size={14} />
          <span>{isLoading ? "SOLVING SURROGATE..." : "RECOMPUTE RESPONSE (< 2 ms)"}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Parameter Controls Panel (4 Cols) - Clean white card */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
            <div className="flex items-center space-x-2 border-b border-[#E0E0E0] pb-3 text-xs font-mono font-bold text-[#161616]">
              <Sliders size={15} className="text-[#0F62FE]" />
              <span>STRUCTURAL PROPERTIES</span>
            </div>

            {/* Building Archetype Selector */}
            <div className="space-y-1 text-xs font-mono">
              <label className="text-[#525252] uppercase text-[10px] font-bold font-sans">Building Archetype</label>
              <select
                value={selectedBuildingId}
                onChange={(e) => {
                  const bld = buildingArchetypes.find((b) => b.id === e.target.value);
                  if (bld) onSelectBuilding(bld);
                }}
                className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded p-2 text-[#161616] text-xs cursor-pointer font-sans"
              >
                {buildingArchetypes.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name} ({b.stories} Stories)
                  </option>
                ))}
              </select>
            </div>

            {/* Fundamental Period Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#525252]">Period (T0):</span>
                <span className="text-[#161616] font-bold font-mono">{T0.toFixed(2)} s</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="2.50"
                step="0.05"
                value={T0}
                onChange={(e) => setT0(parseFloat(e.target.value))}
                className="w-full cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#8D8D8D]">
                <span>0.10s (Stiff)</span>
                <span>2.50s (Flexible)</span>
              </div>
            </div>

            {/* Viscous Damping Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#525252]">Damping Ratio (ζ):</span>
                <span className="text-[#161616] font-bold font-mono">{(damping * 100).toFixed(1)} %</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.15"
                step="0.005"
                value={damping}
                onChange={(e) => setDamping(parseFloat(e.target.value))}
                className="w-full cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#8D8D8D]">
                <span>1% (Undamped)</span>
                <span>15% (High)</span>
              </div>
            </div>

            {/* Yield Displacement Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#525252]">Yield Displacement (uy):</span>
                <span className="text-[#161616] font-bold font-mono">{(uy * 1000).toFixed(1)} mm</span>
              </div>
              <input
                type="range"
                min="0.002"
                max="0.040"
                step="0.001"
                value={uy}
                onChange={(e) => setUy(parseFloat(e.target.value))}
                className="w-full cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#8D8D8D]">
                <span>2 mm (Brittle)</span>
                <span>40 mm (Ductile)</span>
              </div>
            </div>

            {/* Post-Yield Ratio Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#525252]">Post-Yield Stiffness Ratio (α):</span>
                <span className="text-[#161616] font-bold font-mono">{(alpha * 100).toFixed(0)} %</span>
              </div>
              <input
                type="range"
                min="0.00"
                max="0.25"
                step="0.01"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#8D8D8D]">
                <span>0% (Elastoplastic)</span>
                <span>25% (Hardening)</span>
              </div>
            </div>

            {/* Excitation PGA Scaling Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#525252]">Target Excitation PGA:</span>
                <span className="text-[#198038] font-bold font-mono">{pgaG.toFixed(2)} g</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.20"
                step="0.05"
                value={pgaG}
                onChange={(e) => setPgaG(parseFloat(e.target.value))}
                className="w-full cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#8D8D8D]">
                <span>0.05g (Low)</span>
                <span>1.20g (Extreme)</span>
              </div>
            </div>

            {/* Constitutive Law Selector */}
            <div className="space-y-1 text-xs font-sans">
              <label className="text-[#525252] uppercase text-[10px] font-bold font-sans">Constitutive Material Law</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setMaterialType("bilinear")}
                  className={`py-1.5 rounded border text-xs font-mono cursor-pointer transition ${
                    materialType === "bilinear"
                      ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                      : "bg-[#F4F4F4] border-[#E0E0E0] text-[#525252] hover:bg-[#EBEBEB]"
                  }`}
                >
                  Bilinear (Elastoplastic)
                </button>
                <button
                  onClick={() => setMaterialType("elastic")}
                  className={`py-1.5 rounded border text-xs font-mono cursor-pointer transition ${
                    materialType === "elastic"
                      ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                      : "bg-[#F4F4F4] border-[#E0E0E0] text-[#525252] hover:bg-[#EBEBEB]"
                  }`}
                >
                  Linear Elastic
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Center & Right: Physical Visualization & Plots (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          {/* Top Half: 2D Building Kinematic Deflection - ETABS/AutoCAD Engineering Drawing Aesthetic */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3 mb-3 text-xs font-mono">
              <div className="flex items-center space-x-2">
                <Building2 size={15} className="text-[#0F62FE]" />
                <span className="font-bold text-[#161616]">
                  {activeBuilding.name} — LATERAL DEFLECTION AT t = {currTime.toFixed(2)}s
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-[#525252]">Story Drift:</span>
                <span className="text-[#161616] font-bold font-mono">{currU.toFixed(1)} mm</span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono ${
                    Math.abs(currU) > uy * 1000
                      ? "bg-[#FFF8E1] text-[#B28600] border border-[#F1C21B]"
                      : "bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C]"
                  }`}
                >
                  {Math.abs(currU) > uy * 1000 ? "YIELDED (PLASTIC)" : "ELASTIC"}
                </span>
              </div>
            </div>

            {/* Schematic Structural Drawing Canvas (#F4F4F4 background with #161616 CAD structural members) */}
            <div className="h-60 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-center p-4 relative overflow-hidden">
              <svg className="w-full h-full max-w-md" viewBox="0 0 400 210">
                {/* Engineering Grid / Reference Lines */}
                <line x1="50" y1="180" x2="350" y2="180" stroke="#161616" strokeWidth="2.5" />
                {/* Foundation Earth Hatching */}
                {[50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350].map((hx) => (
                  <line key={hx} x1={hx} y1="180" x2={hx - 10} y2="195" stroke="#8D8D8D" strokeWidth="1.2" />
                ))}

                {/* Base Column Foundation Pins */}
                <circle cx="150" cy="180" r="4.5" fill="#161616" />
                <circle cx="250" cy="180" r="4.5" fill="#161616" />

                {/* Multi-story columns and floor slabs */}
                {storyDeflections.map((dispMm, storyIdx) => {
                  const numStories = stories;
                  const floorY = 180 - ((storyIdx + 1) / numStories) * 140;
                  const prevFloorY = storyIdx === 0 ? 180 : 180 - (storyIdx / numStories) * 140;
                  const prevDisp = storyIdx === 0 ? 0 : storyDeflections[storyIdx - 1];

                  // Scale displacement visually (clamp to 40px max screen shift)
                  const screenShift = (dispMm / Math.max(10.0, peakU)) * 40;
                  const prevScreenShift = (prevDisp / Math.max(10.0, peakU)) * 40;

                  const colLeftX1 = 150 + prevScreenShift;
                  const colLeftX2 = 150 + screenShift;
                  const colRightX1 = 250 + prevScreenShift;
                  const colRightX2 = 250 + screenShift;

                  const isFloorYielded = Math.abs(dispMm) > uy * 1000;

                  return (
                    <g key={`story-${storyIdx}`}>
                      {/* Left Column (#161616 CAD black or #DA1E28 red if yielded) */}
                      <line
                        x1={colLeftX1}
                        y1={prevFloorY}
                        x2={colLeftX2}
                        y2={floorY}
                        stroke={isFloorYielded ? "#DA1E28" : "#161616"}
                        strokeWidth="3.5"
                      />
                      {/* Right Column */}
                      <line
                        x1={colRightX1}
                        y1={prevFloorY}
                        x2={colRightX2}
                        y2={floorY}
                        stroke={isFloorYielded ? "#DA1E28" : "#161616"}
                        strokeWidth="3.5"
                      />
                      {/* Horizontal Floor Beam / Slab (#161616 solid structural line) */}
                      <line
                        x1={colLeftX2 - 15}
                        y1={floorY}
                        x2={colRightX2 + 15}
                        y2={floorY}
                        stroke="#161616"
                        strokeWidth="4"
                      />
                      {/* Floor Joint Nodes */}
                      <circle cx={colLeftX2} cy={floorY} r="3.5" fill={isFloorYielded ? "#DA1E28" : "#161616"} />
                      <circle cx={colRightX2} cy={floorY} r="3.5" fill={isFloorYielded ? "#DA1E28" : "#161616"} />

                      {/* Story Level Marker */}
                      <text
                        x={colRightX2 + 25}
                        y={floorY + 3}
                        fill="#525252"
                        fontSize="9"
                        fontFamily="IBM Plex Mono"
                        fontWeight="600"
                      >
                        L{storyIdx + 1}: {dispMm.toFixed(1)}mm
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Time Scrubber Slider */}
            <div className="mt-3 space-y-1.5 font-mono text-xs">
              <div className="flex justify-between items-center text-[11px] text-[#525252]">
                <span>SCRUB TIME HISTORY:</span>
                <span className="font-bold text-[#161616]">
                  t = {currTime.toFixed(2)}s / {traj ? traj.time[traj.time.length - 1].toFixed(2) : "20.48"}s
                </span>
              </div>
              <input
                type="range"
                min="0"
                max={Math.max(0, totalPoints - 1)}
                value={currentIdx}
                onChange={(e) => setScrubIndex(parseInt(e.target.value))}
                className="w-full cursor-pointer"
              />
            </div>
          </div>

          {/* Bottom Half: Multi-tab Engineering Plots */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-2 text-xs font-mono">
              <span className="font-bold text-[#161616]">DYNAMIC ENGINEERING PLOTS</span>
              <div className="flex space-x-1">
                <button
                  onClick={() => setActiveTabPlot("disp")}
                  className={`px-2.5 py-1 rounded cursor-pointer transition font-medium ${
                    activeTabPlot === "disp"
                      ? "bg-[#0F62FE] text-white font-bold"
                      : "text-[#525252] hover:text-[#161616] hover:bg-[#F4F4F4]"
                  }`}
                >
                  Displacement u(t)
                </button>
                <button
                  onClick={() => setActiveTabPlot("hysteresis")}
                  className={`px-2.5 py-1 rounded cursor-pointer transition font-medium ${
                    activeTabPlot === "hysteresis"
                      ? "bg-[#0F62FE] text-white font-bold"
                      : "text-[#525252] hover:text-[#161616] hover:bg-[#F4F4F4]"
                  }`}
                >
                  Hysteresis F(u)
                </button>
                <button
                  onClick={() => setActiveTabPlot("energy")}
                  className={`px-2.5 py-1 rounded cursor-pointer transition font-medium ${
                    activeTabPlot === "energy"
                      ? "bg-[#0F62FE] text-white font-bold"
                      : "text-[#525252] hover:text-[#161616] hover:bg-[#F4F4F4]"
                  }`}
                >
                  Dissipated Energy Eh(t)
                </button>
              </div>
            </div>

            {/* Plot Surface Canvas with contrasting #F4F4F4 background */}
            <div className="h-56 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-center p-3 relative">
              {traj && traj.u.length > 0 ? (
                <svg className="w-full h-full" viewBox="0 0 600 200">
                  {/* Axis Zero Line */}
                  <line x1="20" y1="100" x2="580" y2="100" stroke="#8D8D8D" strokeDasharray="3 3" />
                  <line x1="20" y1="20" x2="20" y2="180" stroke="#8D8D8D" />

                  {/* Active Plot Content */}
                  {activeTabPlot === "disp" && (
                    <path
                      d={traj.u
                        .map((val, idx, arr) => {
                          const x = 20 + (idx / (arr.length - 1)) * 560;
                          const maxVal = Math.max(...arr.map(Math.abs)) || 1e-4;
                          const y = 100 - (val / maxVal) * 80;
                          return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                        })
                        .join(" ")}
                      fill="none"
                      stroke="#0F62FE"
                      strokeWidth="2.0"
                    />
                  )}

                  {activeTabPlot === "hysteresis" && (
                    <path
                      d={traj.u
                        .map((uVal, idx) => {
                          const frVal = traj.fr[idx];
                          const maxU = Math.max(...traj.u.map(Math.abs)) || 1e-4;
                          const maxFR = Math.max(...traj.fr.map(Math.abs)) || 1e-4;
                          const x = 300 + (uVal / maxU) * 260;
                          const y = 100 - (frVal / maxFR) * 80;
                          return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                        })
                        .join(" ")}
                      fill="none"
                      stroke="#161616"
                      strokeWidth="1.8"
                    />
                  )}

                  {activeTabPlot === "energy" && (
                    <path
                      d={traj.eh
                        .map((val, idx, arr) => {
                          const x = 20 + (idx / (arr.length - 1)) * 560;
                          const maxVal = Math.max(...arr) || 1e-4;
                          const y = 180 - (val / maxVal) * 160;
                          return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                        })
                        .join(" ")}
                      fill="none"
                      stroke="#198038"
                      strokeWidth="2.0"
                    />
                  )}

                  {/* Scrub Marker vertical line */}
                  <line
                    x1={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y1="10"
                    x2={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y2="190"
                    stroke="#0F62FE"
                    strokeWidth="1.5"
                  />
                </svg>
              ) : (
                <div className="text-xs font-mono text-[#525252]">No simulation data computed.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
