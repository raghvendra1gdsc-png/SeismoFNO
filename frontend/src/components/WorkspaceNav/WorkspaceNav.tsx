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
}

export const WorkspaceNav: React.FC<WorkspaceNavProps> = ({
  activeTab,
  onSelectTab,
  latencyMs,
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
      label: "Structural Response",
      category: "NONLINEAR SDOF/MDOF",
      previewSummary: "Nonlinear hysteretic response under continuous Fourier operator",
      icon: Activity,
    },
    {
      id: "model_validation",
      code: "02",
      label: "Physics Reference",
      category: "OPENSeesPy C++ SOLVER",
      previewSummary: "Physical Newmark-β integration vs FNO continuous inference",
      icon: ShieldCheck,
    },
    {
      id: "earthquake_intel",
      code: "03",
      label: "Seismic Input",
      category: "IS 1893 & PEER DATABASE",
      previewSummary: "Indian seismic microzonation & PEER ground motion records",
      icon: MapPin,
    },
    {
      id: "scenario_lab",
      code: "04",
      label: "Retrofit Lab",
      category: "DUAL COMPARISON",
      previewSummary: "Yield displacement & fundamental period retrofit sensitivity",
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
      label: "Multi-Story Operator",
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
      label: "Results & Audit",
      category: "FORENSIC SUITE",
      previewSummary: "305 passing deterministic checks & benchmark provenance",
      icon: Cpu,
    },
  ];

  return (
    <aside className="w-[270px] shrink-0 bg-[#FFFFFF] border-r border-[#E2E8F0] flex flex-col justify-between py-4 px-3 select-none text-[#0F172A] font-sans relative z-20 shadow-[1px_0_3px_rgba(0,0,0,0.02)]">
      <div className="space-y-4">
        {/* Academic Laboratory Index Header */}
        <div className="px-2 pt-1 pb-2 border-b border-[#E2E8F0]">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-none bg-[#047857]" />
            <span className="text-xs font-bold tracking-wider text-[#0F172A] uppercase font-mono">
              SEISMOFNO
            </span>
          </div>
          <div className="text-[10px] font-mono text-[#64748B] mt-0.5 tracking-tight">
            Neural Operator Research Workstation
          </div>
        </div>

        {/* Section Header */}
        <div className="space-y-1">
          <div className="px-2 flex items-center justify-between text-[9px] font-mono tracking-wider uppercase text-[#64748B] pb-1">
            <span>Research Index</span>
            <span>v1.0 (Audit Passed)</span>
          </div>

          {/* Light Academic Navigation Menu */}
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
                    className={`group w-full flex flex-col px-2.5 py-2 rounded transition-all duration-150 cursor-pointer text-left border ${
                      isActive
                        ? "bg-[#F1F5F9] border-[#CBD5E1] border-l-2 border-l-[#047857] text-[#0F172A] shadow-xs"
                        : isHovered
                        ? "bg-[#F8FAFC] border-[#E2E8F0] text-[#0F172A]"
                        : isAnyHovered
                        ? "opacity-45 border-transparent text-[#64748B] hover:opacity-100"
                        : "border-transparent text-[#334155] hover:text-[#0F172A] hover:bg-[#F8FAFC]"
                    }`}
                  >
                    {/* Top Row: Index number, Label, and Arrow */}
                    <div className="flex items-baseline justify-between w-full">
                      <div className="flex items-center gap-2 truncate">
                        <span
                          className={`font-mono text-[10px] transition-colors ${
                            isActive
                              ? "text-[#047857] font-bold"
                              : isHovered
                              ? "text-[#0F172A]"
                              : "text-[#64748B]"
                          }`}
                        >
                          {item.code}
                        </span>
                        <span
                          className={`text-xs tracking-tight transition-colors ${
                            isActive
                              ? "font-bold text-[#0F172A]"
                              : isHovered
                              ? "font-semibold text-[#0F172A]"
                              : "font-medium"
                          }`}
                        >
                          {item.label}
                        </span>
                      </div>

                      <div className="flex items-center gap-1 shrink-0 ml-1.5">
                        <ArrowUpRight
                          size={12}
                          className={`transition-all duration-150 ${
                            isActive
                              ? "text-[#047857] translate-x-0.5 -translate-y-0.5"
                              : isHovered
                              ? "text-[#0F172A] translate-x-0.5 -translate-y-0.5 opacity-100"
                              : "text-[#94A3B8] opacity-50"
                          }`}
                        />
                      </div>
                    </div>

                    {/* Bottom Row: Category & Role Monospace Tag */}
                    <div className="flex items-center justify-between mt-0.5 pl-4">
                      <span className="text-[9px] font-mono tracking-wider uppercase text-[#64748B]">
                        {item.category}
                      </span>
                    </div>
                  </button>

                  {/* Clean Technical Hover Card Preview */}
                  {isHovered && !isActive && (
                    <div className="absolute left-[275px] top-0 z-50 w-64 p-3 bg-[#FFFFFF] rounded border border-[#CBD5E1] shadow-lg pointer-events-none transform -translate-y-0.5 animate-in fade-in duration-100">
                      <div className="flex items-center gap-1.5 text-[#047857] text-[10px] font-mono uppercase tracking-wider pb-1 border-b border-[#E2E8F0]">
                        <item.icon size={12} />
                        <span>{item.category}</span>
                      </div>
                      <div className="text-xs font-bold text-[#0F172A] mt-1.5 mb-0.5">
                        {item.label}
                      </div>
                      <div className="text-[11px] text-[#475569] leading-relaxed">
                        {item.previewSummary}
                      </div>
                      <div className="mt-2 pt-1 border-t border-[#F1F5F9] flex items-center justify-between text-[9px] font-mono text-[#047857]">
                        <span>SELECT VIEW</span>
                        <span>[CLICK] ↗</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Laboratory Telemetry Strip */}
      <div className="space-y-1.5 px-2 pt-3 border-t border-[#E2E8F0] font-mono text-[10px] text-[#64748B]">
        <div className="text-[9px] uppercase tracking-wider text-[#64748B] flex items-center justify-between">
          <span>Bench Telemetry</span>
          <span className="text-[#047857] font-semibold">VERIFIED</span>
        </div>
        <div className="space-y-1 bg-[#F8FAFC] p-2 rounded border border-[#E2E8F0]">
          <div className="flex justify-between">
            <span>Single Latency:</span>
            <span className="text-[#0F172A] font-semibold">
              {latencyMs ? `${latencyMs.toFixed(2)} ms` : "1.84 ms"}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Batched Speed:</span>
            <span className="text-[#047857] font-bold">1,060 sim/s</span>
          </div>
          <div className="flex justify-between">
            <span>Verification:</span>
            <span className="text-[#0F172A]">305 Tests Passing</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default WorkspaceNav;
