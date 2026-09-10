import React, { useState, useEffect, useRef } from "react";
import {
  Sliders,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  Activity,
} from "lucide-react";
import type { BuildingArchetype, ScenarioPredictionResponse } from "../../api/digitalTwinApi";
import { Structural3DViewer } from "./Structural3DViewer";

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
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [activeTabPlot, setActiveTabPlot] = useState<"disp" | "hysteresis" | "energy">("disp");

  const animTimerRef = useRef<number | null>(null);

  const activeBuilding = buildingArchetypes.find((b) => b.id === selectedBuildingId) || buildingArchetypes[0] || {
    id: "BLD-RC-03",
    name: "3-Story RC Moment Frame",
    stories: 3,
  };

  const traj = prediction?.trajectories;
  const metrics = prediction?.metrics;
  const totalPoints = traj?.time.length || 0;
  const currentIdx = Math.min(scrubIndex, totalPoints > 0 ? totalPoints - 1 : 0);

  const currTime = traj ? traj.time[currentIdx] : 0.0;
  const currU = traj ? traj.u[currentIdx] * 1000.0 : 0.0; // mm

  const peakU = metrics?.peak_displacement_mm || 50.0;
  const stories = activeBuilding?.stories || 3;

  // Compute multi-story deflection profile (mode shape scaled)
  const storyDeflections = Array.from({ length: stories }, (_, i) => {
    const fraction = (i + 1) / stories;
    const modeWeight = Math.pow(fraction, 1.25);
    return currU * modeWeight;
  });

  const yieldMm = uy * 1000;
  const isCurrentlyYielded = Math.abs(currU) > yieldMm;

  // Playback Loop
  useEffect(() => {
    if (!isPlaying || totalPoints === 0) {
      if (animTimerRef.current) clearInterval(animTimerRef.current);
      return;
    }

    const intervalMs = Math.max(16, Math.floor(25 / playbackSpeed));
    animTimerRef.current = window.setInterval(() => {
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
    <div className="p-4 md:p-6 space-y-6 font-sans text-slate-100 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-xl border border-white/10 bg-[#0B0F1A]/80 backdrop-blur-xl shadow-2xl">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-white flex items-center gap-2">
              <Sparkles size={18} className="text-[#00F0FF]" />
              3D STRUCTURAL DIGITAL TWIN & SEISMIC WORKSTATION
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#00F0FF]/15 text-[#00F0FF] border border-[#00F0FF]/40 font-bold">
              SUB-2ms NEURAL OPERATOR
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time nonlinear structural dynamics under earthquake excitation: 3D volumetric sway, plastic hinge formation, and hysteretic energy dissipation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onRunSimulation}
            disabled={isLoading}
            className="px-5 py-2.5 bg-gradient-to-r from-[#00F0FF] to-[#0099FF] hover:opacity-95 text-[#070A11] text-xs font-mono font-bold rounded-lg flex items-center space-x-2 cursor-pointer transition shadow-[0_0_20px_rgba(0,240,255,0.4)] disabled:opacity-50"
          >
            <Play size={14} className="fill-[#070A11]" />
            <span>{isLoading ? "SOLVING SURROGATE..." : "RECOMPUTE SURROGATE (< 2 ms)"}</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Physical & Structural Controls (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-[#0B0F1A]/90 border border-white/10 rounded-xl p-5 space-y-4 shadow-xl backdrop-blur-md">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 text-xs font-mono font-bold">
              <div className="flex items-center space-x-2 text-cyan-400">
                <Sliders size={15} />
                <span>PHYSICAL PARAMETERS</span>
              </div>
              <span className="text-[10px] text-slate-400">INELASTIC HARDENING</span>
            </div>

            {/* Building Archetype Selector */}
            <div className="space-y-1.5 text-xs font-mono">
              <label className="text-slate-400 uppercase text-[10px] font-bold">Building Archetype</label>
              <select
                value={selectedBuildingId}
                onChange={(e) => {
                  const bld = buildingArchetypes.find((b) => b.id === e.target.value);
                  if (bld) onSelectBuilding(bld);
                }}
                className="w-full bg-[#111827] border border-white/10 rounded-lg p-2.5 text-white text-xs cursor-pointer font-sans focus:border-[#00F0FF] outline-none transition"
              >
                {buildingArchetypes.map((b) => (
                  <option key={b.id} value={b.id} className="bg-[#0B0F1A]">
                    {b.name} ({b.stories} Stories)
                  </option>
                ))}
              </select>
            </div>

            {/* Fundamental Period Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-slate-400">Fundamental Period (T₁):</span>
                <span className="text-cyan-300 font-bold font-mono">{T0.toFixed(2)} s</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="2.50"
                step="0.05"
                value={T0}
                onChange={(e) => setT0(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>0.10s (Stiff)</span>
                <span>2.50s (Flexible)</span>
              </div>
            </div>

            {/* Viscous Damping Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-slate-400">Rayleigh Damping (ζ):</span>
                <span className="text-emerald-400 font-bold font-mono">{(damping * 100).toFixed(1)} %</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.15"
                step="0.005"
                value={damping}
                onChange={(e) => setDamping(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>1% (Light)</span>
                <span>15% (Heavy)</span>
              </div>
            </div>

            {/* Yield Displacement Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-slate-400">Yield Drift Limit (uᵧ):</span>
                <span className="text-[#FF9900] font-bold font-mono">{yieldMm.toFixed(1)} mm</span>
              </div>
              <input
                type="range"
                min="0.002"
                max="0.040"
                step="0.001"
                value={uy}
                onChange={(e) => setUy(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>2 mm (Low Ductility)</span>
                <span>40 mm (High Ductility)</span>
              </div>
            </div>

            {/* Post-Yield Ratio Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-slate-400">Post-Yield Ratio (α):</span>
                <span className="text-purple-400 font-bold font-mono">{(alpha * 100).toFixed(0)} %</span>
              </div>
              <input
                type="range"
                min="0.00"
                max="0.25"
                step="0.01"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>0% (Elastoplastic)</span>
                <span>25% (Strain Hardening)</span>
              </div>
            </div>

            {/* Excitation PGA Scaling Slider */}
            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-slate-400">Target Excitation PGA:</span>
                <span className="text-[#00F0FF] font-bold font-mono">{pgaG.toFixed(2)} g</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.20"
                step="0.05"
                value={pgaG}
                onChange={(e) => setPgaG(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>0.05g (Minor)</span>
                <span>1.20g (Severe MCE)</span>
              </div>
            </div>

            {/* Constitutive Law Selector */}
            <div className="space-y-1.5 text-xs font-sans">
              <label className="text-slate-400 uppercase text-[10px] font-bold">Constitutive Law</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setMaterialType("bilinear")}
                  className={`py-2 rounded-lg border text-xs font-mono cursor-pointer transition ${
                    materialType === "bilinear"
                      ? "bg-[#00F0FF]/15 border-[#00F0FF] text-[#00F0FF] font-bold shadow-[0_0_12px_rgba(0,240,255,0.2)]"
                      : "bg-[#111827] border-white/10 text-slate-400 hover:text-white"
                  }`}
                >
                  Bilinear Inelastic
                </button>
                <button
                  onClick={() => setMaterialType("elastic")}
                  className={`py-2 rounded-lg border text-xs font-mono cursor-pointer transition ${
                    materialType === "elastic"
                      ? "bg-[#00F0FF]/15 border-[#00F0FF] text-[#00F0FF] font-bold shadow-[0_0_12px_rgba(0,240,255,0.2)]"
                      : "bg-[#111827] border-white/10 text-slate-400 hover:text-white"
                  }`}
                >
                  Linear Elastic
                </button>
              </div>
            </div>

            {/* Engineering Metrics Summary Badge */}
            <div className="pt-3 border-t border-white/10 grid grid-cols-2 gap-2 text-[11px] font-mono">
              <div className="bg-[#111827]/80 p-2 rounded border border-white/5">
                <span className="text-slate-400 block text-[9px] uppercase">Peak Roof Drift</span>
                <span className="text-white font-bold text-xs">{peakU.toFixed(1)} mm</span>
              </div>
              <div className="bg-[#111827]/80 p-2 rounded border border-white/5">
                <span className="text-slate-400 block text-[9px] uppercase">Ductility μ</span>
                <span className="text-cyan-400 font-bold text-xs">{(peakU / yieldMm).toFixed(2)}</span>
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
            <div className="bg-[#0B0F1A]/90 border border-white/10 rounded-xl p-4 space-y-2.5 shadow-xl backdrop-blur-md">
              <div className="flex items-center justify-between text-xs font-mono">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 font-bold cursor-pointer transition ${
                      isPlaying
                        ? "bg-[#FF2A6D]/20 border border-[#FF2A6D] text-[#FF2A6D]"
                        : "bg-[#00F0FF]/20 border border-[#00F0FF] text-[#00F0FF]"
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
                    className="p-1.5 rounded-lg bg-[#111827] hover:bg-[#1E293B] border border-white/10 text-slate-300 cursor-pointer transition"
                  >
                    <RotateCcw size={14} />
                  </button>

                  <div className="flex items-center space-x-1 pl-2 border-l border-white/10 text-[10px]">
                    <span className="text-slate-400">SPEED:</span>
                    {[0.5, 1, 2].map((s) => (
                      <button
                        key={s}
                        onClick={() => setPlaybackSpeed(s)}
                        className={`px-1.5 py-0.5 rounded cursor-pointer ${
                          playbackSpeed === s
                            ? "bg-[#00F0FF]/20 text-cyan-300 font-bold border border-[#00F0FF]/40"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                </div>

                <div className="text-slate-400">
                  TIME: <span className="text-white font-bold">{currTime.toFixed(2)}s</span> /{" "}
                  <span className="text-slate-400">{traj ? traj.time[traj.time.length - 1].toFixed(2) : "20.48"}s</span>
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
                className="w-full cursor-pointer"
              />
            </div>
          </div>

          {/* Bottom: Dynamic Engineering Waveform & Hysteresis Plots */}
          <div className="bg-[#0B0F1A]/90 border border-white/10 rounded-xl p-5 space-y-3 shadow-xl backdrop-blur-md">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 text-xs font-mono">
              <div className="flex items-center space-x-2 text-white font-bold">
                <Activity size={15} className="text-[#00F0FF]" />
                <span>DYNAMIC RESPONSE TRAJECTORY & HYSTERESIS</span>
              </div>
              <div className="flex space-x-1">
                <button
                  onClick={() => setActiveTabPlot("disp")}
                  className={`px-3 py-1 rounded-lg text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "disp"
                      ? "bg-[#00F0FF]/20 text-[#00F0FF] border border-[#00F0FF]/50 font-bold"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Displacement u(t)
                </button>
                <button
                  onClick={() => setActiveTabPlot("hysteresis")}
                  className={`px-3 py-1 rounded-lg text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "hysteresis"
                      ? "bg-[#00F0FF]/20 text-[#00F0FF] border border-[#00F0FF]/50 font-bold"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Hysteresis Loop f(u)
                </button>
                <button
                  onClick={() => setActiveTabPlot("energy")}
                  className={`px-3 py-1 rounded-lg text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "energy"
                      ? "bg-[#00F0FF]/20 text-[#00F0FF] border border-[#00F0FF]/50 font-bold"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Energy Dissipation Eₕ(t)
                </button>
              </div>
            </div>

            {/* Plot Surface Canvas */}
            <div className="h-56 bg-[#080C16] rounded-lg border border-white/10 flex items-center justify-center p-3 relative overflow-hidden">
              {traj && traj.u.length > 0 ? (
                <svg className="w-full h-full" viewBox="0 0 600 200">
                  <defs>
                    <linearGradient id="cyanGlow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00F0FF" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#00F0FF" stopOpacity="0.0" />
                    </linearGradient>
                    <linearGradient id="emeraldGlow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00E676" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#00E676" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>

                  {/* Axis Zero Line */}
                  <line x1="20" y1="100" x2="580" y2="100" stroke="#334155" strokeDasharray="3 3" />
                  <line x1="20" y1="20" x2="20" y2="180" stroke="#334155" />

                  {/* Active Plot Content */}
                  {activeTabPlot === "disp" && (
                    <>
                      {/* Yield lines */}
                      <line
                        x1="20"
                        y1={100 - (yieldMm / (peakU || 1)) * 80}
                        x2="580"
                        y2={100 - (yieldMm / (peakU || 1)) * 80}
                        stroke="#FF2A6D"
                        strokeDasharray="2 2"
                        strokeOpacity="0.5"
                      />
                      <line
                        x1="20"
                        y1={100 + (yieldMm / (peakU || 1)) * 80}
                        x2="580"
                        y2={100 + (yieldMm / (peakU || 1)) * 80}
                        stroke="#FF2A6D"
                        strokeDasharray="2 2"
                        strokeOpacity="0.5"
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
                        stroke="#00F0FF"
                        strokeWidth="2.2"
                        filter="drop-shadow(0 0 4px #00F0FF)"
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
                      stroke="#FF9900"
                      strokeWidth="2.0"
                      filter="drop-shadow(0 0 5px #FF9900)"
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
                      stroke="#00E676"
                      strokeWidth="2.2"
                      filter="drop-shadow(0 0 4px #00E676)"
                    />
                  )}

                  {/* Real-time Scrub Marker Indicator */}
                  <line
                    x1={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y1="10"
                    x2={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y2="190"
                    stroke="#FF2A6D"
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
                    fill="#FF2A6D"
                    filter="drop-shadow(0 0 6px #FF2A6D)"
                  />
                </svg>
              ) : (
                <div className="text-xs font-mono text-slate-400">Computing structural dynamics...</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
