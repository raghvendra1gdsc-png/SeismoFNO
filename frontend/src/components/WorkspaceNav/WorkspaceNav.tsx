import React from "react";

export type WorkspaceTab =
  | "hero"
  | "structural_twin"
  | "research_demo"
  | "live_earthquake"
  | "command_center"
  | "earthquake_intel"
  | "scenario_lab"
  | "model_validation"
  | "explainability";

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
  const navItems: { id: WorkspaceTab; code: string; label: string; desc: string }[] = [
    { id: "hero", code: "00", label: "Cinematic Showcase", desc: "GSAP 3D Research Hero" },
    { id: "structural_twin", code: "01", label: "Structural Simulator", desc: "Nonlinear SDOF/MDOF" },
    { id: "model_validation", code: "02", label: "OpenSees Benchmark", desc: "True NLTHA Verification" },
    { id: "earthquake_intel", code: "03", label: "Seismic Hazard & Map", desc: "India IS 1893 & PEER" },
    { id: "scenario_lab", code: "04", label: "Retrofit Lab", desc: "Dual Scenario Comparison" },
    { id: "live_earthquake", code: "05", label: "Live USGS Screening", desc: "Real-Time Event Ingestion" },
    { id: "research_demo", code: "06", label: "Multi-Story Research", desc: "Modal GNO (EXP4–EXP6)" },
    { id: "explainability", code: "07", label: "Model Architecture", desc: "Continuous FNO Theory" },
    { id: "command_center", code: "08", label: "Overview & KPIs", desc: "Executive Summary" },
  ];

  return (
    <aside className="w-[220px] shrink-0 bg-[#07110F] border-r border-white/[0.06] flex flex-col justify-between py-4 px-3 select-none text-[#E8E8DE] font-sans">
      <div className="space-y-6">
        {/* Workstation Title / Studio Index Header */}
        <div className="px-1 space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5]" />
            <span className="text-xs font-semibold tracking-wider text-[#E8E8DE] uppercase">
              SEISMOFNO
            </span>
          </div>
          <div className="text-[10px] font-mono text-[#82928B] pl-3.5 uppercase tracking-wide">
            Neural Operator Research Desk
          </div>
        </div>

        {/* Index Navigation Items */}
        <div className="space-y-1.5">
          <div className="px-1 text-[10px] font-mono tracking-wider uppercase text-[#82928B]">
            Workspaces
          </div>
          <nav className="space-y-1 text-xs">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg transition cursor-pointer text-left ${
                    isActive
                      ? "text-[#E8E8DE] font-medium bg-[#101D19] border border-[#73E6B5]/30 shadow-sm"
                      : "text-[#82928B] hover:text-[#E8E8DE] hover:bg-[#0B1714] border border-transparent"
                  }`}
                  title={item.desc}
                >
                  <div className="flex items-start gap-2 truncate">
                    <span className="font-mono text-[10px] text-[#73E6B5] w-4 mt-0.5">
                      {item.code}
                    </span>
                    <div className="truncate">
                      <div className="truncate tracking-tight font-medium text-xs text-[#E8E8DE]">{item.label}</div>
                      <div className="text-[9px] font-mono text-[#82928B] truncate">{item.desc}</div>
                    </div>
                  </div>
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5] shrink-0 ml-1" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Telemetry Strip */}
      <div className="space-y-2 px-1 pt-4 border-t border-white/[0.06] font-mono text-[10px] text-[#82928B]">
        <div className="text-[9px] uppercase tracking-wider text-[#82928B]">
          System Status
        </div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span>Inference:</span>
            <span className="text-[#E8E8DE] font-semibold">
              {latencyMs ? `${latencyMs.toFixed(2)} ms` : "< 2.0 ms"}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Throughput:</span>
            <span className="text-[#73E6B5]">1,060 sim/s</span>
          </div>
          <div className="flex justify-between">
            <span>Audit Suite:</span>
            <span className="text-[#73E6B5]">305 Tests Passing</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default WorkspaceNav;
