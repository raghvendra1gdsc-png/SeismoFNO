import React from "react";
import type { SystemHealth, SimulationResponse } from "../../types/agent";
import { exportSimulationCSV, exportSimulationJSON } from "../../utils/formatting";

interface TopBarProps {
  health: SystemHealth | null;
  simulationData: SimulationResponse | null;
  isSimulating?: boolean;
  onOpenMetadata: () => void;
  onTakeSnapshot: () => void;
  onResetExperiment: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  health,
  simulationData,
  isSimulating = false,
  onOpenMetadata,
  onTakeSnapshot,
  onResetExperiment,
}) => {
  // Runtime states derived strictly from backend health
  const nebiusState = health?.nebius_configured
    ? health?.nebius_live_verified
      ? { label: "READY", color: "bg-energy text-energy" }
      : { label: "CONFIGURED", color: "bg-opensees text-opensees" }
    : { label: "OFFLINE", color: "bg-danger text-danger" };

  const nemotronState = health?.nebius_configured
    ? { label: "READY", color: "bg-energy text-energy" }
    : { label: "OFFLINE", color: "bg-danger text-danger" };

  const fnoState = isSimulating
    ? { label: "RUNNING", color: "bg-fno text-fno" }
    : { label: "READY", color: "bg-energy text-energy" };

  const physicsState = isSimulating
    ? { label: "SIMULATING", color: "bg-opensees text-opensees" }
    : { label: "SERIALIZED", color: "bg-energy text-energy" };

  return (
    <header className="bg-background-base border-b border-border-subtle px-4 py-2 flex items-center justify-between sticky top-0 z-30 select-none">
      {/* Brand & Engineering Subtitle */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-sm bg-fno shadow-[0_0_8px_rgba(0,210,255,0.6)]"></span>
          <span className="text-xs font-mono font-bold tracking-widest text-text-primary uppercase">
            SEISMOAGENT
          </span>
        </div>
        <div className="h-3 w-px bg-border-subtle"></div>
        <div className="text-[11px] font-mono text-text-muted tracking-wide hidden sm:inline">
          AI-Driven Seismic Structural Analysis
        </div>
      </div>

      {/* Center: Live Hardware & Orchestration Runtime Indicators */}
      <div className="hidden md:flex items-center space-x-4 text-[10px] font-mono">
        {/* Nebius Cloud Indicator */}
        <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-panel-base border border-border-subtle">
          <span className="text-text-muted">NEBIUS</span>
          <span className={`w-1.5 h-1.5 rounded-full ${nebiusState.color.split(" ")[0]}`}></span>
          <span className={nebiusState.color.split(" ")[1]}>{nebiusState.label}</span>
        </div>

        {/* Nemotron Orchestration Indicator */}
        <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-panel-base border border-border-subtle">
          <span className="text-text-muted">NEMOTRON</span>
          <span className={`w-1.5 h-1.5 rounded-full ${nemotronState.color.split(" ")[0]}`}></span>
          <span className={nemotronState.color.split(" ")[1]}>{nemotronState.label}</span>
        </div>

        {/* SeismoFNO Surrogate Indicator */}
        <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-panel-base border border-border-subtle">
          <span className="text-text-muted">FNO</span>
          <span className={`w-1.5 h-1.5 rounded-full ${fnoState.color.split(" ")[0]}`}></span>
          <span className={fnoState.color.split(" ")[1]}>{fnoState.label}</span>
        </div>

        {/* OpenSeesPy Reference Indicator */}
        <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-panel-base border border-border-subtle">
          <span className="text-text-muted">PHYSICS</span>
          <span className={`w-1.5 h-1.5 rounded-full ${physicsState.color.split(" ")[0]}`}></span>
          <span className={physicsState.color.split(" ")[1]}>{physicsState.label}</span>
        </div>
      </div>

      {/* Right Side: Device & Export Actions */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        <div className="flex items-center space-x-1.5 text-[11px] text-text-secondary bg-panel-base px-2 py-0.5 rounded border border-border-subtle">
          <span className="text-text-muted">DEVICE:</span>
          <span className="text-fno font-bold uppercase">{health?.device || "MPS"}</span>
        </div>

        <div className="h-3 w-px bg-border-subtle"></div>

        {/* Spec Modal Trigger */}
        <button
          onClick={onOpenMetadata}
          className="text-[11px] text-text-muted hover:text-text-primary px-2 py-0.5 rounded hover:bg-panel-hover transition uppercase tracking-wider cursor-pointer"
          title="Inspect Model Architecture & Checkpoint Spec"
        >
          SPEC
        </button>

        {/* Export Data */}
        {simulationData && (
          <div className="flex items-center space-x-1 text-[11px] text-text-muted">
            <button
              onClick={() => exportSimulationCSV(simulationData)}
              className="text-text-secondary hover:text-text-primary px-1.5 py-0.5 rounded hover:bg-panel-hover transition cursor-pointer"
              title="Download CSV trajectory"
            >
              CSV
            </button>
            <span className="text-border-medium">/</span>
            <button
              onClick={() => exportSimulationJSON(simulationData)}
              className="text-text-secondary hover:text-text-primary px-1.5 py-0.5 rounded hover:bg-panel-hover transition cursor-pointer"
              title="Download JSON trajectory"
            >
              JSON
            </button>
          </div>
        )}

        <button
          onClick={onTakeSnapshot}
          className="text-[11px] text-text-muted hover:text-text-primary px-2 py-0.5 rounded hover:bg-panel-hover transition uppercase tracking-wider cursor-pointer"
          title="Print or export current view"
        >
          SNAPSHOT
        </button>

        <button
          onClick={onResetExperiment}
          className="text-[11px] text-danger hover:bg-danger/10 px-2 py-0.5 rounded border border-danger/30 transition uppercase tracking-wider cursor-pointer font-semibold"
          title="Reset active charts, metrics, and state"
        >
          RESET
        </button>
      </div>
    </header>
  );
};

