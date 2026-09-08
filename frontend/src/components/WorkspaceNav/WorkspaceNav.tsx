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
  | "research_demo"
  | "live_earthquake"
  | "command_center"
  | "earthquake_intel"
  | "structural_twin"
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
      id: "research_demo",
      label: "Research Defense Demo",
      icon: <GraduationCap size={15} className="text-[#0F62FE]" />,
      badge: "EXP4–6",
    },
    {
      id: "command_center",
      label: "Command Center",
      icon: <LayoutDashboard size={15} />,
    },
    {
      id: "earthquake_intel",
      label: "Earthquake Intelligence",
      icon: <Globe2 size={15} />,
      badge: "INDIA MAP",
    },
    {
      id: "live_earthquake",
      label: "00 Live Earthquake",
      icon: <Radio size={15} className="text-[#0F62FE]" />,
      badge: "USGS",
    },
    {
      id: "structural_twin",
      label: "Structural Digital Twin",
      icon: <Building2 size={15} />,
    },
    {
      id: "scenario_lab",
      label: "Scenario Lab",
      icon: <GitCompare size={15} />,
      badge: "COMPARE",
    },
    {
      id: "model_validation",
      label: "Model Validation",
      icon: <FileCheck size={15} />,
    },
    {
      id: "explainability",
      label: "Research / Explainability",
      icon: <BookOpen size={15} />,
    },
  ];

  return (
    <aside className="w-60 shrink-0 bg-white border-r border-[#E0E0E0] flex flex-col justify-between py-3 select-none">
      <div className="space-y-4">
        {/* Navigation Category */}
        <div>
          <div className="px-3 pb-2 text-[10px] font-mono tracking-widest uppercase text-[#525252] font-semibold">
            Workspaces
          </div>
          <nav className="space-y-0.5 text-xs font-sans">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 transition cursor-pointer text-left ${
                    isActive
                      ? "bg-[#EDF5FF] text-[#161616] font-semibold border-l-[3px] border-[#0F62FE]"
                      : "text-[#525252] hover:text-[#161616] hover:bg-[#F4F4F4] border-l-[3px] border-transparent"
                  }`}
                >
                  <div className="flex items-center space-x-2.5 truncate">
                    <span className={isActive ? "text-[#0F62FE]" : "text-[#525252]"}>
                      {item.icon}
                    </span>
                    <span className="truncate">{item.label}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded ${
                        isActive
                          ? "bg-[#0F62FE] text-white"
                          : "bg-[#F4F4F4] text-[#525252] border border-[#E0E0E0]"
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

        {/* Runtime Performance Card */}
        <div className="mx-3 p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] text-[11px] font-mono space-y-1.5">
          <div className="text-[10px] text-[#525252] uppercase font-bold flex items-center space-x-1">
            <Zap size={12} className="text-[#0F62FE]" />
            <span>Fourier Neural Operator</span>
          </div>
          <div className="flex justify-between items-baseline">
            <span className="text-[#525252]">Inference:</span>
            <span className="text-[#161616] font-bold font-mono">
              {latencyMs ? `${latencyMs.toFixed(2)} ms` : "< 2.0 ms"}
            </span>
          </div>
          <div className="flex justify-between items-baseline text-[10px] text-[#525252]">
            <span>Throughput:</span>
            <span className="font-mono">~1,380 rec/s</span>
          </div>
        </div>
      </div>

      {/* Footer System Guarantee */}
      <div className="px-3 pt-3 border-t border-[#E0E0E0] text-[10px] font-mono text-[#525252]">
        <div className="flex items-center space-x-1.5 text-[#198038]">
          <ShieldCheck size={13} />
          <span className="font-bold">RESEARCH CORE FROZEN</span>
        </div>
        <p className="mt-1 text-[9px] text-[#525252] leading-tight font-sans">
          Scientific models immutable. Validated against OpenSeesPy.
        </p>
      </div>
    </aside>
  );
};
