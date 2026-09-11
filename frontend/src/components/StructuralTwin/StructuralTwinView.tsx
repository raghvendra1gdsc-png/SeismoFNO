import React, { useState, useEffect, useRef } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Sliders,
  Activity,
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
  const peakU = prediction?.metrics?.peak_displacement_mm || 0.0;
  const isCurrentlyYielded = Math.abs(roofDispMm) >= yieldMm;

  // Playback timer effect
  useEffect(() => {
    if (!isPlaying || totalPoints === 0) {
      if (animTimerRef.current) clearInterval(animTimerRef.current);
      return;
    }

    const intervalMs = Math.max(10, Math.round(20 / playbackSpeed));
    animTimerRef.current = setInterval(() => {
      setScrubIndex((prev) => {
        if (prev >= totalPoints - 1) {
          setIsPlaying(false);
          return 0;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => {
      if (animTimerRef.current) clearInterval(animTimerRef.current);
    };
  }, [isPlaying, totalPoints, playbackSpeed]);

  // Live Auto-Recompute on Slider Change (debounced at 180ms)
  const isFirstMount = useRef(true);
  useEffect(() => {
    if (isFirstMount.current) {
      isFirstMount.current = false;
      return;
    }
    const timer = setTimeout(() => {
      onRunSimulation();
    }, 180);
    return () => clearTimeout(timer);
  }, [T0, damping, uy, alpha, pgaG, materialType, selectedBuildingId, onRunSimulation]);

  // When a new simulation finishes, auto-play dynamic vibration from t=0 so the building visibly vibrates
  const prevPredTimeRef = useRef<number | null>(null);
  useEffect(() => {
    if (prediction && prediction.inference_time_ms !== prevPredTimeRef.current && totalPoints > 0) {
      prevPredTimeRef.current = prediction.inference_time_ms;
      setScrubIndex(0);
      setIsPlaying(true);
    }
  }, [prediction, totalPoints]);

  return (
    <div className="p-4 md:p-6 space-y-4 font-sans text-[#0F172A] max-w-7xl mx-auto">
      {/* 1. Academic Header & Input -> Model -> Output Technical Strip */}
      <div className="panel-workstation p-4 space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="w-2 h-2 bg-[#047857]" />
              <h1 className="text-base font-bold font-mono tracking-tight text-[#0F172A]">
                STRUCTURAL RESPONSE: Nonlinear SDOF/MDOF Dynamic Analysis
              </h1>
              <span className="badge-tech-green">
                SURROGATE: {prediction ? `${prediction.inference_time_ms.toFixed(1)} ms` : "1.84 ms"}
              </span>
            </div>
            <p className="text-xs text-[#475569] mt-1 leading-relaxed">
              Earthquake ground motion excitation + structural physical parameters → continuous Fourier Neural Operator → nonlinear displacement and hysteretic dissipation.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={onRunSimulation}
              disabled={isLoading}
              className="btn-engineering px-4 py-2 bg-[#047857] hover:bg-[#065F46] text-white font-mono text-xs font-bold rounded transition cursor-pointer flex items-center justify-center gap-2 disabled:opacity-50 shadow-xs"
            >
              <Play size={12} className={isLoading ? "animate-spin" : "fill-white"} />
              <span>{isLoading ? "COMPUTING FORWARD PASS..." : "Execute Surrogate Simulation"}</span>
            </button>
          </div>
        </div>

        {/* Real-Time Forward Pass Execution Banner */}
        {prediction && (
          <div className="p-2.5 bg-[#ECFDF5] border border-[#A7F3D0] rounded text-xs font-mono text-[#047857] flex items-center justify-between shadow-xs">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#047857] animate-ping" />
              <span className="font-semibold">
                Surrogate forward pass active on {activeBuilding.name} · Peak Roof Drift = {peakU.toFixed(2)} mm · Dynamic 3D response playing
              </span>
            </div>
            <span className="text-[10px] text-[#065F46] font-bold shrink-0 ml-2">
              {prediction.inference_time_ms.toFixed(1)} ms · Apple MPS
            </span>
          </div>
        )}

        {/* Scientific Input -> Model -> Output Pipeline Strip */}
        <div className="pt-2.5 border-t border-[#E2E8F0] grid grid-cols-1 md:grid-cols-3 gap-2.5 text-xs font-mono">
          <div className="bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
            <span className="text-[10px] text-[#64748B] uppercase font-semibold block">1. INPUT EXCITATION & STRUCTURAL PARAMETERS</span>
            <span className="text-[#0F172A] font-medium text-[11px] truncate block mt-0.5">
              ü_g(t) · T₁={T0.toFixed(2)}s · ζ={(damping * 100).toFixed(1)}% · uᵧ={yieldMm.toFixed(1)}mm · α={(alpha * 100).toFixed(0)}% · PGA={pgaG.toFixed(2)}g
            </span>
          </div>

          <div className="bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
            <span className="text-[10px] text-[#64748B] uppercase font-semibold block">2. NEURAL OPERATOR SURROGATE</span>
            <span className="text-[#047857] font-semibold text-[11px] truncate block mt-0.5">
              Continuous FNO-1D / Modal GNO · 16 Modes · Apple MPS
            </span>
          </div>

          <div className="bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
            <span className="text-[10px] text-[#64748B] uppercase font-semibold block">3. OUTPUT PREDICTION</span>
            <span className="text-[#0F172A] font-medium text-[11px] truncate block mt-0.5">
              Floor u(t) · Interstory Drift IDR · Hysteresis F_s(u)
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Physical & Structural Controls (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="panel-workstation p-4 space-y-3.5">
            <div className="flex items-center justify-between border-b border-[#E2E8F0] pb-2 text-xs font-mono font-bold">
              <div className="flex items-center space-x-1.5 text-[#0F172A]">
                <Sliders size={13} className="text-[#047857]" />
                <span>PHYSICAL PARAMETERS</span>
              </div>
              <span className="text-[10px] text-[#64748B]">INELASTIC LAW</span>
            </div>

            {/* Building Archetype Selector */}
            <div className="space-y-1 text-xs font-mono">
              <label className="text-[#64748B] uppercase text-[10px] font-bold">Building Archetype</label>
              <select
                value={selectedBuildingId}
                onChange={(e) => {
                  const bld = buildingArchetypes.find((b) => b.id === e.target.value);
                  if (bld) onSelectBuilding(bld);
                }}
                className="w-full bg-[#FFFFFF] border border-[#CBD5E1] rounded p-2 text-[#0F172A] text-xs cursor-pointer font-sans focus:border-[#047857] outline-none transition"
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
                <span className="text-[#475569] font-medium">Fundamental Period (T₁):</span>
                <span className="text-[#0F172A] font-bold font-mono">{T0.toFixed(2)} s</span>
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
              <div className="flex justify-between text-[10px] font-mono text-[#64748B]">
                <span>0.10 s (Stiff)</span>
                <span>2.50 s (Flexible)</span>
              </div>
            </div>

            {/* Viscous Damping Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#475569] font-medium">Rayleigh Damping (ζ):</span>
                <span className="text-[#0F172A] font-bold font-mono">{(damping * 100).toFixed(1)} %</span>
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
              <div className="flex justify-between text-[10px] font-mono text-[#64748B]">
                <span>1% (Light)</span>
                <span>15% (Heavy)</span>
              </div>
            </div>

            {/* Yield Displacement Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#475569] font-medium">Yield Drift Limit (uᵧ):</span>
                <span className="text-[#B45309] font-bold font-mono">{yieldMm.toFixed(1)} mm</span>
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
              <div className="flex justify-between text-[10px] font-mono text-[#64748B]">
                <span>2.0 mm (Low Ductility)</span>
                <span>40.0 mm (High Ductility)</span>
              </div>
            </div>

            {/* Post-Yield Ratio Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#475569] font-medium">Post-Yield Ratio (α):</span>
                <span className="text-[#B45309] font-bold font-mono">{(alpha * 100).toFixed(0)} %</span>
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
              <div className="flex justify-between text-[10px] font-mono text-[#64748B]">
                <span>0% (Elastoplastic)</span>
                <span>25% (Strain Hardening)</span>
              </div>
            </div>

            {/* Excitation PGA Scaling Slider */}
            <div className="space-y-1 text-xs font-sans">
              <div className="flex justify-between items-baseline">
                <span className="text-[#475569] font-medium">Target Excitation PGA:</span>
                <span className="text-[#047857] font-bold font-mono">{pgaG.toFixed(2)} g</span>
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
              <div className="flex justify-between text-[10px] font-mono text-[#64748B]">
                <span>0.05 g (Moderate)</span>
                <span>1.20 g (Severe MCE)</span>
              </div>
            </div>

            {/* Constitutive Law Selector */}
            <div className="space-y-1 text-xs font-sans">
              <label className="text-[#64748B] uppercase text-[10px] font-bold">Constitutive Law</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setMaterialType("bilinear")}
                  className={`py-1.5 rounded border text-xs font-mono cursor-pointer transition ${
                    materialType === "bilinear"
                      ? "bg-[#0F172A] text-white font-semibold border-[#0F172A]"
                      : "bg-[#FFFFFF] border-[#CBD5E1] text-[#475569] hover:text-[#0F172A]"
                  }`}
                >
                  Bilinear Inelastic
                </button>
                <button
                  onClick={() => setMaterialType("elastic")}
                  className={`py-1.5 rounded border text-xs font-mono cursor-pointer transition ${
                    materialType === "elastic"
                      ? "bg-[#0F172A] text-white font-semibold border-[#0F172A]"
                      : "bg-[#FFFFFF] border-[#CBD5E1] text-[#475569] hover:text-[#0F172A]"
                  }`}
                >
                  Linear Elastic
                </button>
              </div>
            </div>

            {/* Technical Response Status Readout */}
            <div className="pt-3 border-t border-[#E2E8F0] space-y-1.5 font-mono text-[11px]">
              <span className="text-[10px] text-[#64748B] uppercase font-semibold block">STRUCTURAL STATUS</span>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
                  <span className="text-[#64748B] block text-[9px] uppercase">State</span>
                  <span className={`font-bold text-xs ${isCurrentlyYielded ? "text-[#DC2626]" : "text-[#047857]"}`}>
                    {isCurrentlyYielded ? "PLASTIC YIELD" : "ELASTIC"}
                  </span>
                </div>
                <div className="bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
                  <span className="text-[#64748B] block text-[9px] uppercase">Time</span>
                  <span className="font-bold text-xs text-[#0F172A]">{currTime.toFixed(2)} s</span>
                </div>
                <div className="bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
                  <span className="text-[#64748B] block text-[9px] uppercase">Peak Roof Drift</span>
                  <span className="text-[#0F172A] font-bold text-xs">{peakU.toFixed(1)} mm</span>
                </div>
                <div className="bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
                  <span className="text-[#64748B] block text-[9px] uppercase">Ductility Demand</span>
                  <span className="text-[#047857] font-bold text-xs">μ = {(peakU / (yieldMm || 1)).toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Center & Right Column: 3D Building Visualizer & Scientific Plots (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          {/* Top: 3D Interactive Three.js Structural Twin */}
          <div className="space-y-2">
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
            <div className="panel-workstation p-3 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="btn-engineering-secondary"
                  >
                    {isPlaying ? <Pause size={12} /> : <Play size={12} className="fill-current" />}
                    <span>{isPlaying ? "Pause" : "Play"}</span>
                  </button>

                  <button
                    onClick={() => {
                      setIsPlaying(false);
                      setScrubIndex(0);
                    }}
                    title="Reset to t = 0s"
                    className="p-1.5 rounded bg-[#FFFFFF] hover:bg-[#F8FAFC] border border-[#CBD5E1] text-[#64748B] hover:text-[#0F172A] cursor-pointer transition"
                  >
                    <RotateCcw size={12} />
                  </button>

                  <div className="flex items-center space-x-1 pl-2 border-l border-[#CBD5E1] text-[10px]">
                    <span className="text-[#64748B]">SPEED:</span>
                    {[0.5, 1, 2].map((s) => (
                      <button
                        key={s}
                        onClick={() => setPlaybackSpeed(s)}
                        className={`px-1.5 py-0.5 rounded cursor-pointer ${
                          playbackSpeed === s
                            ? "bg-[#0F172A] text-white font-bold"
                            : "text-[#64748B] hover:text-[#0F172A]"
                        }`}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                </div>

                <div className="text-[#64748B]">
                  TIME: <span className="text-[#0F172A] font-bold">{currTime.toFixed(2)} s</span> /{" "}
                  <span className="text-[#64748B]">{traj ? traj.time[traj.time.length - 1].toFixed(2) : "20.48"} s</span>
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

          {/* Bottom: Scientific Engineering Waveform & Hysteresis Plots */}
          <div className="panel-workstation p-4 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E2E8F0] pb-2.5 text-xs font-mono">
              <div className="flex items-center space-x-1.5 text-[#0F172A] font-bold">
                <Activity size={14} className="text-[#047857]" />
                <span>DYNAMIC RESPONSE TRAJECTORY & HYSTERESIS</span>
              </div>
              <div className="flex space-x-1 bg-[#F1F5F9] p-0.5 rounded border border-[#CBD5E1]">
                <button
                  onClick={() => setActiveTabPlot("disp")}
                  className={`px-2.5 py-0.5 rounded text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "disp"
                      ? "bg-[#FFFFFF] text-[#0F172A] font-bold shadow-xs border border-[#CBD5E1]"
                      : "text-[#64748B] hover:text-[#0F172A]"
                  }`}
                >
                  Displacement u(t)
                </button>
                <button
                  onClick={() => setActiveTabPlot("hysteresis")}
                  className={`px-2.5 py-0.5 rounded text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "hysteresis"
                      ? "bg-[#FFFFFF] text-[#0F172A] font-bold shadow-xs border border-[#CBD5E1]"
                      : "text-[#64748B] hover:text-[#0F172A]"
                  }`}
                >
                  Hysteresis Loop f(u)
                </button>
                <button
                  onClick={() => setActiveTabPlot("energy")}
                  className={`px-2.5 py-0.5 rounded text-xs font-mono cursor-pointer transition ${
                    activeTabPlot === "energy"
                      ? "bg-[#FFFFFF] text-[#0F172A] font-bold shadow-xs border border-[#CBD5E1]"
                      : "text-[#64748B] hover:text-[#0F172A]"
                  }`}
                >
                  Energy Dissipation E_h(t)
                </button>
              </div>
            </div>

            {/* Scientific Plot Surface (Clean White Canvas) */}
            <div className="h-56 bg-[#FFFFFF] rounded border border-[#E2E8F0] flex items-center justify-center p-3 relative overflow-hidden">
              {traj && traj.u.length > 0 ? (
                <svg className="w-full h-full" viewBox="0 0 600 200">
                  {/* Axis Zero Line & Neutral Grid */}
                  <line x1="20" y1="100" x2="580" y2="100" stroke="#CBD5E1" strokeDasharray="3 3" />
                  <line x1="20" y1="20" x2="20" y2="180" stroke="#CBD5E1" />

                  {/* Active Plot Content */}
                  {activeTabPlot === "disp" && (
                    <>
                      {/* Yield lines (Red Dashed) */}
                      <line
                        x1="20"
                        y1={100 - (yieldMm / (peakU || 1)) * 80}
                        x2="580"
                        y2={100 - (yieldMm / (peakU || 1)) * 80}
                        stroke="#DC2626"
                        strokeDasharray="2 2"
                        strokeOpacity="0.7"
                      />
                      <line
                        x1="20"
                        y1={100 + (yieldMm / (peakU || 1)) * 80}
                        x2="580"
                        y2={100 + (yieldMm / (peakU || 1)) * 80}
                        stroke="#DC2626"
                        strokeDasharray="2 2"
                        strokeOpacity="0.7"
                      />

                      {/* Displacement curve: Clean Restrained Green */}
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
                        stroke="#047857"
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
                      stroke="#B45309"
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
                      stroke="#047857"
                      strokeWidth="2.0"
                    />
                  )}

                  {/* Real-time Scrub Marker Indicator */}
                  <line
                    x1={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y1="10"
                    x2={20 + (currentIdx / (totalPoints - 1)) * 560}
                    y2="190"
                    stroke="#DC2626"
                    strokeWidth="1.5"
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
                    r="3.5"
                    fill="#DC2626"
                  />
                </svg>
              ) : (
                <div className="text-xs font-mono text-[#64748B]">Computing structural dynamics...</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StructuralTwinView;
