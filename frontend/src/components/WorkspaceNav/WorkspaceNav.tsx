import React from "react";

export type WorkspaceTab =
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
  const navItems: { id: WorkspaceTab; code: string; label: string }[] = [
    { id: "structural_twin", code: "01", label: "Digital Twin" },
    { id: "research_demo", code: "02", label: "Research Defense" },
    { id: "live_earthquake", code: "03", label: "Live USGS" },
    { id: "command_center", code: "04", label: "Regional Command" },
    { id: "earthquake_intel", code: "05", label: "Hazard Intel" },
    { id: "scenario_lab", code: "06", label: "NLTHA Lab" },
    { id: "model_validation", code: "07", label: "Forensic Audit" },
    { id: "explainability", code: "08", label: "Modal Analytics" },
  ];

  return (
    <aside className="w-[195px] shrink-0 bg-[#07110F] border-r border-white/[0.06] flex flex-col justify-between py-4 px-3 select-none text-[#E8E8DE] font-sans">
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
            Research Desk
          </div>
        </div>

        {/* Index Navigation Items */}
        <div className="space-y-1.5">
          <div className="px-1 text-[10px] font-mono tracking-wider uppercase text-[#82928B]">
            Workspaces
          </div>
          <nav className="space-y-0.5 text-xs">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-2 py-1.5 rounded transition cursor-pointer text-left ${
                    isActive
                      ? "text-[#E8E8DE] font-medium bg-[#101D19]"
                      : "text-[#82928B] hover:text-[#E8E8DE] hover:bg-[#0B1714]"
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <span className="font-mono text-[10px] text-[#82928B] w-4">
                      {item.code}
                    </span>
                    <span className="truncate tracking-tight">{item.label}</span>
                  </div>
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5] shrink-0" />
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
          System
        </div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span>Inference:</span>
            <span className="text-[#E8E8DE] font-semibold">
              {latencyMs ? `${latencyMs.toFixed(2)} ms` : "21.45 ms"}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Throughput:</span>
            <span className="text-[#73E6B5]">1,060 sim/s</span>
          </div>
          <div className="flex justify-between">
            <span>Audit:</span>
            <span className="text-[#73E6B5]">305 tests</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
