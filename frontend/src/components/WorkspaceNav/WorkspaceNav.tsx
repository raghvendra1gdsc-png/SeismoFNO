import React from "react";
import {
  LayoutDashboard,
  Globe2,
  Building2,
  GitCompare,
  FileCheck,
  BookOpen,
  ShieldCheck,
  Zap,
  GraduationCap,
  Radio,
} from "lucide-react";

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
  const navItems: { id: WorkspaceTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: "structural_twin",
      label: "3D STRUCTURAL DIGITAL TWIN",
      icon: <Building2 size={14} />,
      badge: "3D",
    },
    {
      id: "research_demo",
      label: "RESEARCH DEFENSE CONSOLE",
      icon: <GraduationCap size={14} />,
      badge: "EXP6",
    },
    {
      id: "live_earthquake",
      label: "LIVE USGS EARTHQUAKE",
      icon: <Radio size={14} />,
      badge: "LIVE",
    },
    {
      id: "command_center",
      label: "REGIONAL COMMAND CENTER",
      icon: <LayoutDashboard size={14} />,
    },
    {
      id: "earthquake_intel",
      label: "SEISMIC HAZARD INTEL",
      icon: <Globe2 size={14} />,
    },
    {
      id: "scenario_lab",
      label: "NLTHA COMPARISON LAB",
      icon: <GitCompare size={14} />,
    },
    {
      id: "model_validation",
      label: "FORENSIC AUDIT SUITE",
      icon: <FileCheck size={14} />,
    },
    {
      id: "explainability",
      label: "MODAL FiLM ANALYTICS",
      icon: <BookOpen size={14} />,
    },
  ];

  return (
    <aside className="w-[235px] shrink-0 bg-[#080C12] border-r border-white/[0.07] flex flex-col justify-between py-3 select-none text-[#E8EDF3]">
      <div className="space-y-3">
        {/* Workstation Navigation Header */}
        <div className="px-3 pb-2 border-b border-white/[0.07]">
          <div className="flex items-center space-x-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#28D7FF] shadow-[0_0_6px_#28D7FF]" />
            <span className="text-[11px] font-mono font-bold tracking-wider text-[#E8EDF3] uppercase">
              WORKSPACES
            </span>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="space-y-0.5 px-1.5 text-[11px] font-mono">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center justify-between px-2.5 py-2 rounded-[2px] transition cursor-pointer text-left ${
                  isActive
                    ? "bg-[#151D27] text-[#E8EDF3] font-bold border-l-2 border-l-[#28D7FF]"
                    : "text-[#8D9AAA] hover:text-[#E8EDF3] hover:bg-[#111821] border-l-2 border-l-transparent"
                }`}
              >
                <div className="flex items-center space-x-2 truncate">
                  <span className={isActive ? "text-[#28D7FF]" : "text-[#667487]"}>
                    {item.icon}
                  </span>
                  <span className="truncate text-[10px] tracking-tight">{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[8px] font-mono font-bold px-1 py-0.2 rounded-[2px] ${
                      isActive
                        ? "bg-[#28D7FF]/20 text-[#28D7FF] border border-[#28D7FF]/40"
                        : "bg-white/[0.05] text-[#667487]"
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Instrumentation Telemetry */}
      <div className="space-y-3 px-2">
        <div className="p-2.5 bg-[#0B1018] rounded-[2px] border border-white/[0.07] text-[10px] font-mono space-y-1.5">
          <div className="text-[9px] text-[#28D7FF] uppercase font-bold flex items-center justify-between border-b border-white/[0.05] pb-1">
            <span className="flex items-center gap-1">
              <Zap size={10} className="text-[#28D7FF]" />
              OPERATIONAL TELEMETRY
            </span>
            <span className="text-[#31D17C]">ONLINE</span>
          </div>

          <div className="flex justify-between items-baseline text-[#8D9AAA]">
            <span className="text-[#667487]">INFERENCE:</span>
            <span className="text-[#E8EDF3] font-bold font-mono">
              {latencyMs ? `${latencyMs.toFixed(2)} ms` : "21.45 ms"}
            </span>
          </div>

          <div className="flex justify-between items-baseline text-[#8D9AAA]">
            <span className="text-[#667487]">MPS ACCEL:</span>
            <span className="text-[#31D17C] font-bold font-mono">2.55×</span>
          </div>

          <div className="flex justify-between items-baseline text-[#8D9AAA]">
            <span className="text-[#667487]">THROUGHPUT:</span>
            <span className="text-[#28D7FF] font-bold font-mono">1,060 sim/s</span>
          </div>

          <div className="flex justify-between items-baseline text-[#8D9AAA]">
            <span className="text-[#667487]">TESTS:</span>
            <span className="text-[#31D17C] font-bold font-mono">305 / 305</span>
          </div>
        </div>

        {/* Footer Build Tag */}
        <div className="px-1 text-[9px] font-mono text-[#667487] flex items-center justify-between">
          <span className="flex items-center gap-1 text-[#31D17C]">
            <ShieldCheck size={11} />
            <span>EXP6 FROZEN</span>
          </span>
          <span>SHA-256 AUDITED</span>
        </div>
      </div>
    </aside>
  );
};
