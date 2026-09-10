import React, { useState, useEffect, useRef } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Sliders,
  Activity,
  Sparkles,
} from "lucide-react";
import { Structural3DViewer } from "./Structural3DViewer";
import type { BuildingArchetype, ScenarioPredictionResponse } from "../../api/digitalTwinApi";

interface StructuralTwinViewProps {
  prediction: ScenarioPredictionResponse | null;
  buildingArchetypes: BuildingArchetype[];
  selectedBuildingId: string;
  onSelectBuilding: (bld: BuildingArchetype) => void;
  onRunSimulation: () => void;
  isLoading: boolean;
  T0: number;
  setT0: (v: number) => void;
  damping: number;
  setDamping: (v: number) => void;
  uy: number;
  setUy: (v: number) => void;
  alpha: number;
  setAlpha: (v: number) => void;
  materialType: "bilinear" | "elastic";
  setMaterialType: (v: "bilinear" | "elastic") => void;
  pgaG: number;
  setPgaG: (v: number) => void;
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
  // Animation / Time scrubber state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [scrubIndex, setScrubIndex] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [activeTabPlot, setActiveTabPlot] = useState<"disp" | "hysteresis" | "energy">("disp");

  const animTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const traj = prediction?.trajectories;
  const totalPoints = traj?.time.length || 0;

  // Active Building Meta
  const activeBuilding = buildingArchetypes.find((b) => b.id === selectedBuildingId) || {
    id: "BLD-RC-03",
    name: "3-Story RC Moment Frame",
    stories: 3,
    fundamental_period_s: 0.5,
    damping_ratio: 0.05,
    yield_displacement_m: 0.01,
    post_yield_ratio: 0.05,
  };
  const stories = activeBuilding.stories;

  // Current time and frame deflections
  const currentIdx = Math.min(scrubIndex, Math.max(0, totalPoints - 1));
  const currTime = traj && traj.time.length > 0 ? traj.time[currentIdx] : 0.0;
  const roofDispM = traj && traj.u.length > 0 ? traj.u[currentIdx] : 0.0;
  const roofDispMm = roofDispM * 1000.0;

  // Calculate story deflections assuming triangular/first-mode distribution
  const storyDeflections: number[] = [];
  for (let s = 1; s <= stories; s++) {
    const ratio = s / stories;
    storyDeflections.push(roofDispMm * ratio);
  }

  const yieldMm = uy * 1000.0;
  const peakU = (prediction?.metrics?.peak_displacement_mm || 0.0);
  const isCurrentlyYielded = Math.abs(roofDispMm) >= yieldMm;

  // Playback timer effect
  useEffect(() => {
    if (!isPlaying || totalPoints === 0) {
      if (animTimerRef.current) clearInterval(animTimerRef.current);
      return;
    }

    const intervalMs = Math.max(16, Math.floor(25 / playbackSpeed));
    animTimerRef.current = setInterval(() => {
      setScrubIndex((prev) => {
        if (prev >= totalPoints - 1) {
          return 0; // loop back
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => {
      if (animTimerRef.current) clearInterval(animTimerRef.current);
    };
  }, [isPlaying, totalPoints, playbackSpeed]);

  return (
    <div className="p-4 md:p-6 space-y-6 font-sans text-[#E8E8DE] max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-lg border border-white/[0.08] bg-[#0E1B17] shadow-lg">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#E8E8DE] flex items-center gap-2">
              <Sparkles size={18} className="text-[#73E6B5]" />
              3D STRUCTURAL DIGITAL TWIN & SEISMIC WORKSTATION
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#73E6B5]/10 text-[#73E6B5] border border-[#73E6B5]/30 font-bold">
              SUB-2ms NEURAL OPERATOR
            </span>
          </div>
          <p className="text-xs text-[#82928B] mt-1">
            Real-time nonlinear structural dynamics under earthquake excitation: 3D volumetric sway, plastic hinge formation, and hysteretic energy dissipation.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={onRunSimulation}
            disabled={isLoading}
            className="px-5 py-2.5 bg-[#73E6B5] hover:bg-[#5cd4a2] text-[#07110F] text-xs font-mono font-bold rounded-lg flex items-center space-x-2 cursor-pointer transition shadow-md disabled:opacity-50"
          >
            <Play size={14} className="fill-[#07110F]" />
            <span>{isLoading ? "SOLVING SURROGATE..." : "RECOMPUTE SURROGATE (< 2 ms)"}</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Physical & Structural Controls (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 space-y-4 shadow-lg">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3 text-xs font-mono font-bold">
              <div className="flex items-center space-x-2 text-[#73E6B5]">
                <Sliders size={15} />
                <span>PHYSICAL PARAMETERS</span>
              </div>
              <span className="text-[10px] text-[#82928B]">INELASTIC HARDENING</span>
            </div>

            {/* Building Archetype Selector */}
            <div className="space-y-1.5 text-xs font-mono">
              <label className="text-[#82928B] uppercase text-[10px] font-bold">Building Archetype</label>
              <select
                value={selectedBuildingId}
                onChange={(e) => {
                  const bld = buildingArchetypes.find((b) => b.id === e.target.value);
                  if (bld) onSelectBuilding(bld);
                }}
                className="w-full bg-[#07110F] border border-white/[0.08] rounded-lg p-2.5 text-[#E8E8DE] text-xs cursor-pointer font-sans focus:border-[#73E6B5] outline-none transition"
              >
                {buildingArchetypes.map((b) => (
                  <option key={b.id} value={b.id} className="bg-[#07110F]">
                    {b.name} ({b.stories} Stories)
                  </option>
                ))}
              </select>
            </div>

            {/* Fundamental Period Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#82928B]">Fundamental Period (T₁):</span>
                <span className="text-[#73E6B5] font-bold font-mono">{T0.toFixed(2)} s</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="2.50"
                step="0.05"
                value={T0}
                onChange={(e) => setT0(parseFloat(e.target.value))}
                className="w-full cursor-pointer accent-[#73E6B5]"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#82928B]">
                <span>0.10s (Stiff)</span>
                <span>2.50s (Flexible)</span>
              </div>
            </div>

            {/* Viscous Damping Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#82928B]">Rayleigh Damping (ζ):</span>
                <span className="text-[#73E6B5] font-bold font-mono">{(damping * 100).toFixed(1)} %</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.15"
                step="0.005"
                value={damping}
                onChange={(e) => setDamping(parseFloat(e.target.value))}
                className="w-full cursor-pointer accent-[#73E6B5]"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#82928B]">
                <span>1% (Light)</span>
                <span>15% (Heavy)</span>
              </div>
            </div>

            {/* Yield Displacement Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#82928B]">Yield Drift Limit (uᵧ):</span>
                <span className="text-[#D6B56D] font-bold font-mono">{yieldMm.toFixed(1)} mm</span>
              </div>
              <input
                type="range"
                min="0.002"
                max="0.040"
                step="0.001"
                value={uy}
                onChange={(e) => setUy(parseFloat(e.target.value))}
                className="w-full cursor-pointer accent-[#D6B56D]"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#82928B]">
                <span>2 mm (Low Ductility)</span>
                <span>40 mm (High Ductility)</span>
              </div>
            </div>

            {/* Post-Yield Ratio Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#82928B]">Post-Yield Ratio (α):</span>
                <span className="text-[#D6B56D] font-bold font-mono">{(alpha * 100).toFixed(0)} %</span>
              </div>
              <input
                type="range"
                min="0.00"
                max="0.25"
                step="0.01"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full cursor-pointer accent-[#D6B56D]"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#82928B]">
                <span>0% (Elastoplastic)</span>
                <span>25% (Strain Hardening)</span>
              </div>
            </div>

            {/* Excitation PGA Scaling Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#82928B]">Target Excitation PGA:</span>
                <span className="text-[#73E6B5] font-bold font-mono">{pgaG.toFixed(2)} g</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.20"
                step="0.05"
                value={pgaG}
                onChange={(e) => setPgaG(parseFloat(e.target.value))}
                className="w-full cursor-pointer accent-[#73E6B5]"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#82928B]">
                <span>0.05g (Minor)</span>
                <span>1.20g (Severe MCE)</span>
              </div>
            </div>

            {/* Constitutive Law Selector */}
            <div className="space-y-1.5 text-xs font-sans">
              <label className="text-[#82928B] uppercase text-[10px] font-bold">Constitutive Law</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setMaterialType("bilinear")}
                  className={`py-2 rounded-lg border text-xs font-mono cursor-pointer transition ${
                    materialType === "bilinear"
                      ? "bg-[#73E6B5]/20 border-[#73E6B5] text-[#73E6B5] font-bold shadow-sm"
                      : "bg-[#07110F] border-white/[0.08] text-[#82928B] hover:text-[#E8E8DE]"
                  }`}
                >
                  Bilinear Inelastic
                </button>
                <button
                  onClick={() => setMaterialType("elastic")}
                  className={`py-2 rounded-lg border text-xs font-mono cursor-pointer transition ${
                    materialType === "elastic"
                      ? "bg-[#73E6B5]/20 border-[#73E6B5] text-[#73E6B5] font-bold shadow-sm"
                      : "bg-[#07110F] border-white/[0.08] text-[#82928B] hover:text-[#E8E8DE]"
                  }`}
                >
                  Linear Elastic
                </button>
              </div>
            </div>

            {/* Engineering Metrics Summary Badge */}
            <div className="pt-3 border-t border-white/[0.08] grid grid-cols-2 gap-2 text-[11px] font-mono">
              <div className="bg-[#07110F] p-2 rounded border border-white/[0.06]">
                <span className="text-[#82928B] block text-[9px] uppercase">Peak Roof Drift</span>
                <span className="text-[#E8E8DE] font-bold text-xs">{peakU.toFixed(1)} mm</span>
              </div>
              <div className="bg-[#07110F] p-2 rounded border border-white/[0.06]">
                <span className="text-[#82928B] block text-[9px] uppercase">Ductility μ</span>
                <span className="text-[#73E6B5] font-bold text-xs">{(peakU / (yieldMm || 1)).toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Center & Right Column: 3D Building Visualizer & Engineering Plots (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          {/* Top: 3D Interactive Three.js Structural Twin */}
          <div className="space-y-3">
            <Structural3DViewer
              stories={stories}
              storyDeflectionsMm={storyDeflections}
              yieldDisplacementMm={yieldMm}
              currentTimeS={currTime}
              peakDisplacementMm={peakU}
              isYielded={isCurrentlyYielded}
              buildingName={activeBuilding.name}
            />

            {/* Playback Controls & Time Scrubber Deck */}
            <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4 space-y-2.5 shadow-lg">
              <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 font-bold cursor-pointer transition ${
                      isPlaying
                        ? "bg-[#E35D5D]/20 border border-[#E35D5D] text-[#E35D5D]"
                        : "bg-[#73E6B5]/20 border border-[#73E6B5] text-[#73E6B5]"
                    }`}
                  >
                    {isPlaying ? <Pause size={13} /> : <Play size={13} className="fill-current" />}
                    <span>{isPlaying ? "PAUSE" : "PLAY"}</span>
                  </button>

                  <button
                    onClick={() => {
                      setIsPlaying(false);
                      setScrubIndex(0);
                    }}
                    title="Reset to t = 0s"
                    className="p-1.5 rounded-lg bg-[#07110F] hover:bg-[#101D19] border border-white/[0.08] text-[#82928B] hover:text-[#E8E8DE] cursor-pointer transition"
                  >
                    <RotateCcw size={14} />
                  </button>

                  <div className="flex items-center space-x-1 pl-2 border-l border-white/[0.08] text-[10px]">
                    <span className="text-[#82928B]">SPEED:</span>
                    {[0.5, 1, 2].map((s) => (
                      <button
                        key={s}
                        onClick={() => setPlaybackSpeed(s)}
                        className={`px-1.5 py-0.5 rounded cursor-pointer ${
                          playbackSpeed === s
                            ? "bg-[#73E6B5]/20 text-[#73E6B5] font-bold border border-[#73E6B5]/40"
                            : "text-[#82928B] hover:text-[#E8E8DE]"
                        }`}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                </div>

                <div className="text-[#82928B]">
                  TIME: <span className="text-[#E8E8DE] font-bold">{currTime.toFixed(2)}s</span> /{" "}
                  <span className="text-[#82928B]">{traj ? traj.time[traj.time.length - 1].toFixed(2) : "20.48"}s</span>
                </div>
              </div>

              {/* Time Scrubber Slider */}
              <input
                type="range"
                min="0"
                max={Math.max(0, totalPoints - 1)}
                value={currentIdx}
                onChange={(e) => {
                  setIsPlaying(false);
                  setScrubIndex(parseInt(e.target.value));
                }}
                className="w-full cursor-pointer accent-[#73E6B5]"
              />
            </div>
          </div>

          {/* Bottom: Dynamic Engineering Waveform & Hysteresis Plots */}
          <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 space-y-3 shadow-lg">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] pb-3 text-xs font-mono">
              <div className="flex items-center space-x-2 text-[#E8E8DE] font-bold">
                <Activity size={15} className="text-[#73E6B5]" />
                <span>DYNAMIC RESPONSE TRAJECTORY & HYSTERESIS</span>
              </div>
              <div className="flex space-x-1 bg-[#07110F] p-1 rounded border border-white/[0.06]">
                <button
                  onClick={() => setActiveTabPlot("disp")}
                  className={`px-3 py-1 rounded text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "disp"
                      ? "bg-[#73E6B5] text-[#07110F] font-bold"
                      : "text-[#82928B] hover:text-[#E8E8DE]"
                  }`}
                >
                  Displacement u(t)
                </button>
                <button
                  onClick={() => setActiveTabPlot("hysteresis")}
                  className={`px-3 py-1 rounded text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "hysteresis"
                      ? "bg-[#73E6B5] text-[#07110F] font-bold"
                      : "text-[#82928B] hover:text-[#E8E8DE]"
                  }`}
                >
                  Hysteresis Loop f(u)
                </button>
                <button
                  onClick={() => setActiveTabPlot("energy")}
                  className={`px-3 py-1 rounded text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "energy"
                      ? "bg-[#73E6B5] text-[#07110F] font-bold"
                      : "text-[#82928B] hover:text-[#E8E8DE]"
                  }`}
                >
                  Energy Dissipation Eₕ(t)
                </button>
              </div>
            </div>

            {/* Plot Surface Canvas */}
            <div className="h-56 bg-[#07110F] rounded-lg border border-white/[0.08] flex items-center justify-center p-3 relative overflow-hidden">
              {traj && traj.u.length > 0 ? (
                <svg className="w-full h-full" viewBox="0 0 600 200">
                  {/* Axis Zero Line */}
                  <line x1="20" y1="100" x2="580" y2="100" stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3" />
                  <line x1="20" y1="20" x2="20" y2="180" stroke="rgba(255,255,255,0.12)" />

                  {/* Active Plot Content */}
                  {activeTabPlot === "disp" && (
                    <>
                      {/* Yield lines */}
                      <line
                        x1="20"
                        y1={100 - (yieldMm / (peakU || 1)) * 80}
                        x2="580"
                        y2={100 - (yieldMm / (peakU || 1)) * 80}
                        stroke="#E35D5D"
                        strokeDasharray="2 2"
                        strokeOpacity="0.6"
                      />
                      <line
                        x1="20"
                        y1={100 + (yieldMm / (peakU || 1)) * 80}
                        x2="580"
                        y2={100 + (yieldMm / (peakU || 1)) * 80}
                        stroke="#E35D5D"
                        strokeDasharray="2 2"
                        strokeOpacity="0.6"
                      />

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
                        stroke="#73E6B5"
                        strokeWidth="2.0"
                      />
                    </>
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
                      stroke="#D6B56D"
                      strokeWidth="2.0"
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
                      stroke="#73E6B5"
                      strokeWidth="2.0"
                    />
                  )}

                  {/* Real-time Scrub Marker Indicator */}
                  <line
                    x1={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y1="10"
                    x2={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y2="190"
                    stroke="#E35D5D"
                    strokeWidth="1.8"
                  />
                  <circle
                    cx={20 + (currentIdx / (totalPoints - 1)) * 560}
                    cy={
                      activeTabPlot === "disp"
                        ? 100 - (traj.u[currentIdx] / (Math.max(...traj.u.map(Math.abs)) || 1e-4)) * 80
                        : activeTabPlot === "energy"
                        ? 180 - (traj.eh[currentIdx] / (Math.max(...traj.eh) || 1e-4)) * 160
                        : 100
                    }
                    r="4"
                    fill="#E35D5D"
                  />
                </svg>
              ) : (
                <div className="text-xs font-mono text-[#82928B]">Computing structural dynamics...</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StructuralTwinView;
