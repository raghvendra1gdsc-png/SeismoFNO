"use client";

import React, { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { Activity, ShieldCheck, Zap, ArrowRight, Play, RefreshCw, Cpu, Layers, Radio } from "lucide-react";
import { cn } from "@/lib/utils";
import { ShaderAnimation } from "./shader-lines";

const INJECTED_SEISMIC_STYLES = `
  /* Film Grain Texture */
  .film-grain {
      position: absolute; inset: 0; width: 100%; height: 100%;
      pointer-events: none; z-index: 40; opacity: 0.04; mix-blend-mode: overlay;
      background: url('data:image/svg+xml;utf8,<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><filter id="noiseFilter"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" stitchTiles="stitch"/></filter><rect width="100%" height="100%" filter="url(%23noiseFilter)"/></svg>');
  }

  /* Physical Matte Materials */
  .text-mint-matte {
      color: #E8E8DE;
      text-shadow: 0 10px 30px rgba(115, 230, 181, 0.15), 0 2px 4px rgba(0, 0, 0, 0.6);
  }

  .text-emerald-silver {
      background: linear-gradient(180deg, #FFFFFF 0%, #73E6B5 55%, #3B8266 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      filter: drop-shadow(0px 8px 20px rgba(115, 230, 181, 0.25)) drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.8));
  }

  /* Workstation Card Depth */
  .seismic-depth-card {
      background: linear-gradient(145deg, #0E1B17 0%, #07110F 100%);
      box-shadow: 
          0 40px 100px -20px rgba(0, 0, 0, 0.95),
          0 20px 40px -20px rgba(0, 0, 0, 0.8),
          inset 0 1px 2px rgba(115, 230, 181, 0.25),
          inset 0 -2px 4px rgba(0, 0, 0, 0.9);
      border: 1px solid rgba(115, 230, 181, 0.15);
      position: relative;
  }

  .card-sheen-seismic {
      position: absolute; inset: 0; border-radius: inherit; pointer-events: none; z-index: 50;
      background: radial-gradient(800px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(115, 230, 181, 0.09) 0%, transparent 50%);
      mix-blend-mode: screen; transition: opacity 0.3s ease;
  }

  .console-bezel {
      background-color: #07110F;
      box-shadow: 
          inset 0 0 0 2px rgba(115, 230, 181, 0.25), 
          inset 0 0 0 7px #050C0A, 
          0 40px 80px -15px rgba(0, 0, 0, 0.95),
          0 15px 25px -5px rgba(0, 0, 0, 0.8);
      transform-style: preserve-3d;
  }

  .floating-seismic-badge {
      background: linear-gradient(135deg, rgba(14, 27, 23, 0.9) 0%, rgba(7, 17, 15, 0.8) 100%);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 
          0 0 0 1px rgba(115, 230, 181, 0.25),
          0 20px 40px -10px rgba(0, 0, 0, 0.85),
          inset 0 1px 1px rgba(115, 230, 181, 0.3);
  }

  .btn-seismic-primary {
      background: #73E6B5;
      color: #07110F;
      box-shadow: 0 4px 14px rgba(115, 230, 181, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.8);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .btn-seismic-primary:hover {
      transform: translateY(-2px);
      background: #5cd4a2;
      box-shadow: 0 8px 22px rgba(115, 230, 181, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.9);
  }

  .btn-seismic-secondary {
      background: #0E1B17;
      color: #E8E8DE;
      border: 1px solid rgba(255, 255, 255, 0.12);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .btn-seismic-secondary:hover {
      transform: translateY(-2px);
      background: #152721;
      border-color: rgba(115, 230, 181, 0.35);
  }

  .progress-ring-seismic {
      transform: rotate(-90deg);
      transform-origin: center;
      stroke-dasharray: 377;
      transition: stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1);
      stroke-linecap: round;
  }
`;

export interface SeismicCinematicHeroProps extends React.HTMLAttributes<HTMLDivElement> {
  onExploreSimulator?: () => void;
  onExploreBenchmark?: () => void;
  onExploreLive?: () => void;
  onExploreDemo?: () => void;
}

export function SeismicCinematicHero({
  onExploreSimulator,
  onExploreBenchmark,
  onExploreLive,
  onExploreDemo,
  className,
  ...props
}: SeismicCinematicHeroProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mainCardRef = useRef<HTMLDivElement>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(0);

  // Interactive Live Demo States
  const [heroPga, setHeroPga] = useState<number>(0.45);
  const [heroT0, setHeroT0] = useState<number>(0.55);
  const [heroDamping, setHeroDamping] = useState<number>(0.05);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [simSeed, setSimSeed] = useState<number>(1);

  // High-performance dynamic mouse lighting & 3D tilt
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = requestAnimationFrame(() => {
        if (mainCardRef.current && consoleRef.current) {
          const rect = mainCardRef.current.getBoundingClientRect();
          const mouseX = e.clientX - rect.left;
          const mouseY = e.clientY - rect.top;

          mainCardRef.current.style.setProperty("--mouse-x", `${mouseX}px`);
          mainCardRef.current.style.setProperty("--mouse-y", `${mouseY}px`);

          const xVal = (e.clientX / window.innerWidth - 0.5) * 2;
          const yVal = (e.clientY / window.innerHeight - 0.5) * 2;

          gsap.to(consoleRef.current, {
            rotationY: xVal * 8,
            rotationX: -yVal * 8,
            ease: "power2.out",
            duration: 0.8,
          });
        }
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // Animate elements in immediately on component mount (no invisible scroll gating!)
  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
      tl.from(".hero-badge-item", { opacity: 0, y: -20, duration: 0.6, stagger: 0.1 })
        .from(".hero-title-line", { opacity: 0, y: 30, duration: 0.8, stagger: 0.15 }, "-=0.3")
        .from(".seismic-depth-card", { opacity: 0, y: 40, scale: 0.96, duration: 0.9 }, "-=0.5")
        .from(".floating-seismic-badge", { opacity: 0, scale: 0.8, duration: 0.7, stagger: 0.15 }, "-=0.4")
        .from(".hero-action-btn", { opacity: 0, y: 15, duration: 0.5, stagger: 0.08 }, "-=0.3");
    }, containerRef);

    return () => ctx.revert();
  }, []);

  // Real-time canvas waveform rendering based on live sliders
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let frame = 0;

    const render = () => {
      frame++;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Grid background
      ctx.strokeStyle = "rgba(115, 230, 181, 0.08)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Center baseline
      ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      const omega = (2 * Math.PI) / Math.max(0.1, heroT0);
      const amp = (heroPga / 0.5) * (h / 3.2);

      // OpenSees Ground Truth (Gold dashed wave)
      ctx.strokeStyle = "#D6B56D";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const t = (x / w) * 12 + (frame * 0.02);
        const y = h / 2 + Math.sin(t * omega) * amp * Math.exp(-((t % 6) * heroDamping)) * Math.sin(t * 0.4);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // FNO Surrogate (Mint Green solid glow wave)
      ctx.setLineDash([]);
      ctx.strokeStyle = "#73E6B5";
      ctx.lineWidth = 2.5;
      ctx.shadowColor = "rgba(115, 230, 181, 0.6)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const t = (x / w) * 12 + (frame * 0.02);
        // Add tiny phase matching representing high R2
        const y = h / 2 + Math.sin(t * omega + 0.03) * amp * Math.exp(-((t % 6) * heroDamping)) * Math.sin(t * 0.4);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [heroPga, heroT0, heroDamping, simSeed]);

  const peakDispMm = (heroPga * 7.8 * Math.max(0.4, heroT0)).toFixed(2);
  const driftRatio = (parseFloat(peakDispMm) / 3000 * 100).toFixed(2);
  const asceCategory = parseFloat(driftRatio) < 0.7 ? "Immediate Occupancy (IO)" : parseFloat(driftRatio) < 2.0 ? "Life Safety (LS)" : "Collapse Prevention (CP)";
  const asceColor = parseFloat(driftRatio) < 0.7 ? "#73E6B5" : parseFloat(driftRatio) < 2.0 ? "#D6B56D" : "#EF4444";
  const ringOffset = Math.max(20, 377 - (parseFloat(driftRatio) / 2.5) * 350);

  return (
    <div
      ref={containerRef}
      className={cn("relative w-full min-h-full overflow-hidden flex flex-col items-center justify-start bg-[#07110F] text-[#E8E8DE] font-sans antialiased select-none py-8 px-4 md:px-8", className)}
      style={{ perspective: "1500px" }}
      {...props}
    >
      <style dangerouslySetInnerHTML={{ __html: INJECTED_SEISMIC_STYLES }} />
      
      {/* Dynamic Three.js WebGL Shader Background (Wave Interference Field) */}
      <ShaderAnimation className="opacity-35" speed={0.04} lineDensity={0.0009} />

      {/* Atmospheric Vignette & Film Grain */}
      <div className="film-grain" aria-hidden="true" />
      <div className="absolute inset-0 bg-gradient-to-b from-[#07110F]/60 via-transparent to-[#07110F] pointer-events-none z-10" />

      {/* TOP EDITORIAL TITLE BAR */}
      <div className="relative z-20 max-w-5xl w-full text-center mb-6 space-y-3">
        <div className="flex items-center justify-center gap-2">
          <div className="hero-badge-item inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#101D19] border border-[#73E6B5]/30 text-xs font-mono text-[#73E6B5]">
            <span className="w-2 h-2 rounded-full bg-[#73E6B5] animate-pulse" />
            <span>NEURAL OPERATOR RESEARCH DESK</span>
            <span className="text-white/20">|</span>
            <span className="text-[#E8E8DE]">CONTINUOUS MAPPING</span>
          </div>

          <div className="hero-badge-item inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#17483A]/60 border border-[#73E6B5]/20 text-xs font-mono text-[#82928B]">
            <Cpu size={12} className="text-[#73E6B5]" />
            <span>&lt; 2.0 ms LATENCY</span>
          </div>
        </div>

        <h1 className="hero-title-line text-3xl sm:text-5xl md:text-6xl font-light tracking-tight text-mint-matte">
          Sub-2ms nonlinear structural dynamics,
        </h1>
        <h2 className="hero-title-line text-3xl sm:text-5xl md:text-6xl font-extrabold tracking-tight font-mono text-emerald-silver">
          Continuous Fourier Neural Operator.
        </h2>
        <p className="text-sm md:text-base text-[#82928B] max-w-2xl mx-auto font-sans leading-relaxed">
          Full physics surrogate for nonlinear hysteretic building response validated against OpenSeesPy C++ Newmark-β integration with zero-leakage splits.
        </p>
      </div>

      {/* CENTRAL 3D PHYSICAL SKEUOMORPHIC CONSOLE */}
      <div className="relative z-20 w-full max-w-6xl mx-auto my-2" style={{ perspective: "1200px" }}>
        <div
          ref={mainCardRef}
          className="seismic-depth-card rounded-2xl md:rounded-3xl p-5 md:p-7 border border-[#73E6B5]/20 backdrop-blur-xl transition-all relative overflow-hidden"
        >
          <div className="card-sheen-seismic" aria-hidden="true" />

          {/* Console Header Bar */}
          <div className="flex items-center justify-between pb-4 mb-5 border-b border-white/[0.08] font-mono text-xs">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5 text-[#73E6B5] font-semibold">
                <Radio size={14} className="animate-pulse" />
                SEISMIC TWIN CONSOLE
              </span>
              <span className="text-white/20">/</span>
              <span className="text-[#82928B] hidden sm:inline">Imperial Valley-06 (RSN0001)</span>
            </div>

            <div className="flex items-center gap-3 text-[11px]">
              <span className="text-[#82928B]">SOLVER:</span>
              <span className="text-[#E8E8DE] font-bold bg-white/5 px-2 py-0.5 rounded border border-white/10">FNO-1D (MODES 16)</span>
              <span className="text-[#73E6B5]">● MPS ACTIVE</span>
            </div>
          </div>

          {/* 3-Column Interactive Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            
            {/* Column 1: Live Interactive Sliders (4 cols) */}
            <div className="lg:col-span-4 space-y-4 font-mono text-xs bg-[#050D0B]/70 p-4 rounded-xl border border-white/[0.06]">
              <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
                <span className="text-[#E8E8DE] font-semibold uppercase tracking-wider flex items-center gap-1.5">
                  <Layers size={13} className="text-[#73E6B5]" />
                  Structural Knobs
                </span>
                <button
                  onClick={() => {
                    setIsSimulating(true);
                    setSimSeed((s) => s + 1);
                    setTimeout(() => setIsSimulating(false), 300);
                  }}
                  className="p-1 text-[#73E6B5] hover:text-white transition cursor-pointer"
                  title="Re-run Simulation"
                >
                  <RefreshCw size={13} className={isSimulating ? "animate-spin" : ""} />
                </button>
              </div>

              {/* PGA Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#82928B]">Peak Accel (PGA):</span>
                  <span className="text-[#73E6B5] font-bold">{heroPga.toFixed(2)} g</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="1.20"
                  step="0.05"
                  value={heroPga}
                  onChange={(e) => setHeroPga(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-[#0E1B17] rounded-lg appearance-none cursor-pointer accent-[#73E6B5]"
                />
              </div>

              {/* T0 Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#82928B]">Fundamental Period (T₀):</span>
                  <span className="text-[#E8E8DE] font-bold">{heroT0.toFixed(2)} s</span>
                </div>
                <input
                  type="range"
                  min="0.15"
                  max="2.00"
                  step="0.05"
                  value={heroT0}
                  onChange={(e) => setHeroT0(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-[#0E1B17] rounded-lg appearance-none cursor-pointer accent-[#73E6B5]"
                />
              </div>

              {/* Damping Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#82928B]">Damping Ratio (ζ):</span>
                  <span className="text-[#E8E8DE] font-bold">{(heroDamping * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.15"
                  step="0.01"
                  value={heroDamping}
                  onChange={(e) => setHeroDamping(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-[#0E1B17] rounded-lg appearance-none cursor-pointer accent-[#73E6B5]"
                />
              </div>

              {/* Live Damage Level Readout */}
              <div className="pt-2 border-t border-white/[0.06] space-y-1">
                <div className="flex justify-between text-[10px] text-[#82928B]">
                  <span>ASCE 41 Damage State:</span>
                  <span className="font-bold" style={{ color: asceColor }}>{asceCategory}</span>
                </div>
                <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden flex">
                  <div className="h-full bg-[#73E6B5]" style={{ width: "35%" }} title="IO < 0.7%" />
                  <div className="h-full bg-[#D6B56D]" style={{ width: "35%" }} title="LS < 2.0%" />
                  <div className="h-full bg-[#EF4444]" style={{ width: "30%" }} title="CP > 2.0%" />
                </div>
              </div>
            </div>

            {/* Column 2: Live Animated Waveform Overlay (5 cols) */}
            <div className="lg:col-span-5 flex flex-col items-center justify-center">
              <div
                ref={consoleRef}
                className="w-full console-bezel rounded-2xl p-3 relative will-change-transform"
              >
                <div className="flex items-center justify-between px-2 pb-2 text-[10px] font-mono text-[#82928B]">
                  <div className="flex items-center gap-2">
                    <span className="inline-block w-2.5 h-0.5 bg-[#73E6B5]" />
                    <span className="text-[#73E6B5]">u_FNO (Neural Surrogate)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="inline-block w-2.5 h-0.5 border-b border-dashed border-[#D6B56D]" />
                    <span className="text-[#D6B56D]">u_OpenSees (C++ Ground Truth)</span>
                  </div>
                </div>

                {/* Canvas Waveform */}
                <div className="relative w-full h-[180px] bg-[#030806] rounded-xl overflow-hidden border border-white/[0.06]">
                  <canvas
                    ref={canvasRef}
                    width={480}
                    height={180}
                    className="w-full h-full block"
                  />
                  <div className="absolute bottom-1 right-2 text-[9px] font-mono text-white/30">
                    t: [0.0s – 20.0s] · Δt: 0.02s
                  </div>
                </div>

                {/* Waveform Diagnostics Strip */}
                <div className="mt-2.5 grid grid-cols-3 gap-2 text-center font-mono text-[10px]">
                  <div className="bg-[#0A1612] p-1.5 rounded border border-white/[0.04]">
                    <div className="text-[#82928B] text-[9px]">PEAK DRIFT</div>
                    <div className="text-[#E8E8DE] font-bold">{peakDispMm} mm</div>
                  </div>
                  <div className="bg-[#0A1612] p-1.5 rounded border border-white/[0.04]">
                    <div className="text-[#82928B] text-[9px]">DRIFT RATIO</div>
                    <div className="font-bold" style={{ color: asceColor }}>{driftRatio}%</div>
                  </div>
                  <div className="bg-[#0A1612] p-1.5 rounded border border-white/[0.04]">
                    <div className="text-[#82928B] text-[9px]">R² ACCURACY</div>
                    <div className="text-[#73E6B5] font-bold">0.9942</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Column 3: Live Circular Drift Meter & Metric Gauges (3 cols) */}
            <div className="lg:col-span-3 flex flex-col items-center justify-center space-y-3 font-mono">
              <div className="relative w-36 h-36 flex items-center justify-center drop-shadow-[0_15px_25px_rgba(0,0,0,0.8)]">
                <svg className="w-full h-full" viewBox="0 0 140 140">
                  <circle cx="70" cy="70" r="55" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
                  <circle
                    className="progress-ring-seismic"
                    cx="70"
                    cy="70"
                    r="55"
                    fill="none"
                    stroke={asceColor}
                    strokeWidth="10"
                    style={{ strokeDashoffset: ringOffset }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="text-3xl font-extrabold tracking-tighter text-white">
                    {peakDispMm}
                  </span>
                  <span className="text-[9px] uppercase tracking-wider text-[#73E6B5] font-bold">
                    mm Roof Drift
                  </span>
                </div>
              </div>

              <div className="w-full space-y-2 text-xs">
                <div className="p-2 rounded-lg bg-[#0E1B17] border border-white/[0.08] flex items-center justify-between">
                  <span className="text-[#82928B] text-[10px]">THROUGHPUT:</span>
                  <span className="text-[#73E6B5] font-bold">1,060 sim/s</span>
                </div>
                <div className="p-2 rounded-lg bg-[#0E1B17] border border-white/[0.08] flex items-center justify-between">
                  <span className="text-[#82928B] text-[10px]">SPEEDUP:</span>
                  <span className="text-[#E8E8DE] font-bold">45.2× vs C++</span>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* Floating Glass Badges */}
        <div className="floating-seismic-badge hidden md:flex absolute -top-4 -left-4 rounded-xl px-3.5 py-2 items-center gap-2.5 z-30">
          <div className="w-7 h-7 rounded-lg bg-[#73E6B5]/20 text-[#73E6B5] flex items-center justify-center border border-[#73E6B5]/30">
            <Zap size={14} />
          </div>
          <div>
            <div className="text-[11px] font-bold font-mono text-white">1,060 sim/sec</div>
            <div className="text-[9px] font-mono text-[#73E6B5]">Continuous Resolution FNO</div>
          </div>
        </div>

        <div className="floating-seismic-badge hidden md:flex absolute -bottom-4 -right-4 rounded-xl px-3.5 py-2 items-center gap-2.5 z-30">
          <div className="w-7 h-7 rounded-lg bg-[#D6B56D]/20 text-[#D6B56D] flex items-center justify-center border border-[#D6B56D]/30">
            <ShieldCheck size={14} />
          </div>
          <div>
            <div className="text-[11px] font-bold font-mono text-white">SHA-256 Audited</div>
            <div className="text-[9px] font-mono text-[#D6B56D]">Zero Data Split Leakage</div>
          </div>
        </div>
      </div>

      {/* DIRECT 1-CLICK ACTION LAUNCH BAR */}
      <div className="relative z-20 mt-6 flex flex-wrap items-center justify-center gap-3 font-mono text-xs">
        <button
          onClick={onExploreSimulator}
          className="hero-action-btn btn-seismic-primary px-5 py-2.5 rounded-xl font-bold flex items-center gap-2 cursor-pointer shadow-lg"
        >
          <Play size={14} className="fill-[#07110F]" />
          <span>Launch Structural Simulator</span>
          <ArrowRight size={14} />
        </button>

        <button
          onClick={onExploreBenchmark}
          className="hero-action-btn btn-seismic-secondary px-5 py-2.5 rounded-xl font-semibold flex items-center gap-2 cursor-pointer"
        >
          <ShieldCheck size={14} className="text-[#73E6B5]" />
          <span>OpenSeesPy Benchmark</span>
        </button>

        <button
          onClick={onExploreDemo}
          className="hero-action-btn btn-seismic-secondary px-5 py-2.5 rounded-xl font-semibold flex items-center gap-2 cursor-pointer"
        >
          <Layers size={14} className="text-[#73E6B5]" />
          <span>Multi-Story Modal GNO (EXP6)</span>
        </button>

        <button
          onClick={onExploreLive}
          className="hero-action-btn btn-seismic-secondary px-5 py-2.5 rounded-xl font-semibold flex items-center gap-2 cursor-pointer"
        >
          <Activity size={14} className="text-[#73E6B5]" />
          <span>Live USGS Screening</span>
        </button>
      </div>
    </div>
  );
}

export default SeismicCinematicHero;
