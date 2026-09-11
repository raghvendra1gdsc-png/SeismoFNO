import React, { useState } from "react";
import { ArrowUpRight, Activity, ShieldCheck, MapPin, Sliders, Radio, Cpu, BookOpen, Layers } from "lucide-react";

export type WorkspaceTab =
  | "structural_twin"
  | "model_validation"
  | "earthquake_intel"
  | "scenario_lab"
  | "live_earthquake"
  | "research_demo"
  | "explainability"
  | "command_center";

interface WorkspaceNavProps {
  activeTab: WorkspaceTab;
  onSelectTab: (tab: WorkspaceTab) => void;
  latencyMs?: number;
  onOpenGreeting?: () => void;
}

export const WorkspaceNav: React.FC<WorkspaceNavProps> = ({
  activeTab,
  onSelectTab,
  latencyMs,
  onOpenGreeting,
}) => {
  const [hoveredId, setHoveredId] = useState<WorkspaceTab | null>(null);

  const navItems: {
    id: WorkspaceTab;
    code: string;
    label: string;
    category: string;
    previewSummary: string;
    icon: React.ComponentType<{ size?: number; className?: string }>;
  }[] = [
    {
      id: "structural_twin",
      code: "01",
      label: "Structural Simulator",
      category: "NLTHA SDOF/MDOF",
      previewSummary: "Nonlinear hysteretic response under continuous Fourier operator",
      icon: Activity,
    },
    {
      id: "model_validation",
      code: "02",
      label: "OpenSees Benchmark",
      category: "C++ VERIFICATION",
      previewSummary: "Physical Newmark-β integration vs FNO continuous inference",
      icon: ShieldCheck,
    },
    {
      id: "earthquake_intel",
      code: "03",
      label: "Seismic Hazard & Map",
      category: "IS 1893 & PEER",
      previewSummary: "Indian seismic microzonation & PEER ground motion catalog",
      icon: MapPin,
    },
    {
      id: "scenario_lab",
      code: "04",
      label: "Retrofit Lab",
      category: "DUAL COMPARISON",
      previewSummary: "Structural yield displacement & period retrofit testing",
      icon: Sliders,
    },
    {
      id: "live_earthquake",
      code: "05",
      label: "Live USGS Screening",
      category: "REAL-TIME FEED",
      previewSummary: "Continuous worldwide event ingestion & ASCE 41 damage state",
      icon: Radio,
    },
    {
      id: "research_demo",
      code: "06",
      label: "Multi-Story Research",
      category: "MODAL GNO (EXP6)",
      previewSummary: "Physics/Modal-conditioned spatiotemporal graph operator",
      icon: Layers,
    },
    {
      id: "explainability",
      code: "07",
      label: "Model Architecture",
      category: "CONTINUOUS FNO",
      previewSummary: "Spectral convolution, zero-leakage splits & physics losses",
      icon: BookOpen,
    },
    {
      id: "command_center",
      code: "08",
      label: "Overview & KPIs",
      category: "EXECUTIVE AUDIT",
      previewSummary: "305 passing unit tests & forensic benchmark metrics",
      icon: Cpu,
    },
  ];

  return (
    <aside className="w-[280px] shrink-0 bg-[#07110F]/90 backdrop-blur-xl border-r border-white/[0.08] flex flex-col justify-between py-5 px-3.5 select-none text-[#E8E8DE] font-sans relative z-20">
      <div className="space-y-6">
        {/* Kinetic Header / Brand Studio Index */}
        <div className="px-1.5 space-y-1.5">
          <button
            onClick={onOpenGreeting}
            className="group flex items-center justify-between w-full text-left transition cursor-pointer"
            title="Click to view Welcome Overview"
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#73E6B5] group-hover:scale-125 transition-transform" />
              <span className="text-xs font-bold tracking-wider text-[#E8E8DE] uppercase font-mono group-hover:text-[#73E6B5] transition-colors">
                SEISMOFNO
              </span>
            </div>
            <span className="text-[10px] font-mono text-[#73E6B5] opacity-80 group-hover:opacity-100 flex items-center gap-0.5">
              INTRO ↗
            </span>
          </button>
          <div className="text-[10px] font-mono text-[#82928B] pl-4 uppercase tracking-wider">
            Neural Operator Research Desk
          </div>
        </div>

        {/* Section Header */}
        <div className="space-y-1">
          <div className="px-1.5 flex items-center justify-between text-[9px] font-mono tracking-widest uppercase text-[#82928B] pb-1 border-b border-white/[0.06]">
            <span>Workspaces</span>
            <span>Index '26</span>
          </div>

          {/* Kinetic Interactive Navigation Menu (inspired by 21st.dev Kinetic Team Hybrid) */}
          <nav
            className="space-y-1 pt-1"
            onMouseLeave={() => setHoveredId(null)}
          >
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              const isHovered = hoveredId === item.id;
              const isAnyHovered = hoveredId !== null;

              return (
                <div key={item.id} className="relative">
                  <button
                    onClick={() => onSelectTab(item.id)}
                    onMouseEnter={() => setHoveredId(item.id)}
                    className={`group w-full flex flex-col px-3 py-2.5 rounded-xl transition-all duration-200 cursor-pointer text-left border ${
                      isActive
                        ? "bg-[#0E1B17] border-[#73E6B5]/40 text-[#E8E8DE] shadow-lg shadow-black/40"
                        : isHovered
                        ? "bg-[#0E1B17]/70 border-white/10 text-white"
                        : isAnyHovered
                        ? "opacity-40 border-transparent text-[#82928B] hover:opacity-100"
                        : "border-transparent text-[#A3B2AC] hover:text-[#E8E8DE] hover:bg-[#0E1B17]/40"
                    }`}
                  >
                    {/* Top Row: Index number, Label, and Arrow */}
                    <div className="flex items-baseline justify-between w-full">
                      <div className="flex items-center gap-2.5 truncate">
                        <span
                          className={`font-mono text-[10px] transition-colors ${
                            isActive
                              ? "text-[#73E6B5] font-bold"
                              : isHovered
                              ? "text-[#73E6B5]"
                              : "text-[#556660]"
                          }`}
                        >
                          {item.code}
                        </span>
                        <span
                          className={`text-xs tracking-tight transition-colors ${
                            isActive
                              ? "font-bold text-[#E8E8DE]"
                              : isHovered
                              ? "font-semibold text-white"
                              : "font-medium"
                          }`}
                        >
                          {item.label}
                        </span>
                      </div>

                      <div className="flex items-center gap-1 shrink-0 ml-2">
                        {isActive && (
                          <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5] mr-1 shadow-[0_0_8px_#73E6B5]" />
                        )}
                        <ArrowUpRight
                          size={12}
                          className={`transition-all duration-200 ${
                            isActive
                              ? "text-[#73E6B5] translate-x-0.5 -translate-y-0.5"
                              : isHovered
                              ? "text-[#73E6B5] translate-x-0.5 -translate-y-0.5 opacity-100"
                              : "text-[#556660] opacity-50"
                          }`}
                        />
                      </div>
                    </div>

                    {/* Bottom Row: Category & Role Monospace Tag */}
                    <div className="flex items-center justify-between mt-1 pl-5">
                      <span className="text-[9px] font-mono tracking-wider uppercase text-[#82928B]">
                        {item.category}
                      </span>
                    </div>
                  </button>

                  {/* Kinetic Hover Card Preview (Inspired by 21st.dev Kinetic Team Hybrid) */}
                  {isHovered && !isActive && (
                    <div className="absolute left-[290px] top-0 z-50 w-64 p-3.5 bg-[#0E1B17]/95 backdrop-blur-xl rounded-xl border border-[#73E6B5]/30 shadow-2xl pointer-events-none transform -translate-y-1 animate-in fade-in zoom-in-95 duration-150">
                      <div className="flex items-center gap-2 text-[#73E6B5] text-[10px] font-mono uppercase tracking-wider pb-1.5 border-b border-white/[0.08]">
                        <item.icon size={12} />
                        <span>{item.category}</span>
                      </div>
                      <div className="text-xs font-bold text-[#E8E8DE] mt-2 mb-1">
                        {item.label}
                      </div>
                      <div className="text-[11px] text-[#82928B] leading-relaxed">
                        {item.previewSummary}
                      </div>
                      <div className="mt-2.5 pt-1.5 border-t border-white/[0.06] flex items-center justify-between text-[9px] font-mono text-[#73E6B5]">
                        <span>OPEN WORKSPACE</span>
                        <span>[ENTER] ↗</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Telemetry Strip */}
      <div className="space-y-2 px-1.5 pt-4 border-t border-white/[0.06] font-mono text-[10px] text-[#82928B]">
        <div className="text-[9px] uppercase tracking-wider text-[#82928B] flex items-center justify-between">
          <span>System Status</span>
          <span className="text-[#73E6B5]">ONLINE</span>
        </div>
        <div className="space-y-1 bg-[#050D0B]/80 p-2.5 rounded-lg border border-white/[0.04]">
          <div className="flex justify-between">
            <span>Inference:</span>
            <span className="text-[#E8E8DE] font-semibold">
              {latencyMs ? `${latencyMs.toFixed(2)} ms` : "< 2.0 ms"}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Throughput:</span>
            <span className="text-[#73E6B5] font-bold">1,060 sim/s</span>
          </div>
          <div className="flex justify-between">
            <span>Verification:</span>
            <span className="text-[#E8E8DE]">305 Tests Passing</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default WorkspaceNav;
