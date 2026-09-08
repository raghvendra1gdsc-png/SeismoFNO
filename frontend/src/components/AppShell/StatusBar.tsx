import React from "react";
import type { SystemHealth, SimulationResponse } from "../../types/agent";

interface StatusBarProps {
  health: SystemHealth | null;
  simulationData: SimulationResponse | null;
  experimentId: string;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  health,
  simulationData,
  experimentId,
}) => {
  const isOod = simulationData?.ood?.is_ood ?? false;
  const fnoLatency = simulationData?.metrics?.t_fno_ms != null
    ? `${simulationData.metrics.t_fno_ms.toFixed(2)} ms`
    : "< 1.0 ms";
  const speedup = simulationData?.metrics?.speedup != null
    ? `${simulationData.metrics.speedup.toFixed(1)}x`
    : "—";

  return (
    <footer className="h-7 bg-background-deep border-t border-border-subtle px-4 flex items-center justify-between text-[11px] font-mono text-text-muted select-none z-20">
      {/* Left items: Core System & Model Latency */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-energy"></span>
          <span className="text-text-secondary">SYSTEM:</span>
          <span className="text-text-primary uppercase">{health?.status || "ONLINE"}</span>
        </div>

        <div className="h-3 w-px bg-border-subtle"></div>

        <div className="flex items-center space-x-1.5">
          <span className="text-text-secondary">FNO:</span>
          <span className="text-fno font-semibold">{fnoLatency}</span>
        </div>

        <div className="h-3 w-px bg-border-subtle"></div>

        <div className="flex items-center space-x-1.5">
          <span className="text-text-secondary">PHYSICS:</span>
          <span className="text-opensees">OpenSeesPy (Serialized)</span>
        </div>

        <div className="h-3 w-px bg-border-subtle hidden sm:block"></div>

        <div className="hidden sm:flex items-center space-x-1.5">
          <span className="text-text-secondary">SPEEDUP:</span>
          <span className="text-energy font-semibold">{speedup}</span>
        </div>
      </div>

      {/* Right items: Domain Safety & Experiment ID */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-1.5">
          <span className="text-text-secondary">DOMAIN:</span>
          {isOod ? (
            <span className="px-1.5 py-0.2 rounded bg-danger/20 text-danger text-[10px] font-bold">
              OOD WARNING
            </span>
          ) : (
            <span className="px-1.5 py-0.2 rounded bg-energy/10 text-energy text-[10px] font-semibold">
              IN-DOMAIN
            </span>
          )}
        </div>

        <div className="h-3 w-px bg-border-subtle"></div>

        <div className="flex items-center space-x-1.5">
          <span className="text-text-secondary">EXP:</span>
          <span className="text-text-primary">{experimentId}</span>
        </div>
      </div>
    </footer>
  );
};
