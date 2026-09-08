import React from "react";
import { History, ArrowRight, Trash2 } from "lucide-react";
import type { ExperimentRecord } from "../../types/agent";

interface ExperimentHistoryWorkspaceProps {
  history: ExperimentRecord[];
  onLoadExperiment: (exp: ExperimentRecord) => void;
  onClearHistory: () => void;
}

export const ExperimentHistoryWorkspace: React.FC<ExperimentHistoryWorkspaceProps> = ({
  history,
  onLoadExperiment,
  onClearHistory,
}) => {
  return (
    <div className="p-4 space-y-4 font-mono text-xs select-none">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center space-x-2">
          <History size={16} className="text-fno" />
          <span className="font-bold text-text-primary uppercase tracking-wider text-sm">
            SESSION EXPERIMENT REGISTRY ({history.length} RUNS)
          </span>
        </div>
        <div className="flex items-center space-x-3">
          <span className="text-[10px] text-text-muted">In-Memory Local Session History</span>
          {history.length > 0 && (
            <button
              onClick={onClearHistory}
              className="flex items-center space-x-1 px-2 py-1 rounded bg-panel-base border border-border-subtle hover:text-danger hover:border-danger/30 transition text-text-muted cursor-pointer"
            >
              <Trash2 size={12} />
              <span>Clear</span>
            </button>
          )}
        </div>
      </div>

      {history.length === 0 ? (
        <div className="bg-panel-base border border-border-subtle rounded-md p-12 text-center text-text-muted space-y-2">
          <History size={32} className="mx-auto text-text-muted/60" />
          <div className="text-text-secondary font-semibold">No experiments recorded in this session</div>
          <p className="text-xs text-text-muted max-w-sm mx-auto">
            Run an analysis or send a prompt to Nemotron Copilot. Each completed simulation will be automatically
            logged here for instant side-by-side recall.
          </p>
        </div>
      ) : (
        <div className="bg-panel-base border border-border-subtle rounded-md overflow-hidden">
          <table className="w-full text-left text-xs">
            <thead className="bg-background-base text-[10px] uppercase text-text-muted border-b border-border-subtle">
              <tr>
                <th className="px-3 py-2">Experiment ID</th>
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">Record</th>
                <th className="px-3 py-2">PGA</th>
                <th className="px-3 py-2">Structure (T/ζ/uy)</th>
                <th className="px-3 py-2">Rel L2 Error</th>
                <th className="px-3 py-2">Speedup</th>
                <th className="px-3 py-2 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle/50 text-text-secondary">
              {history.map((exp) => (
                <tr key={exp.id} className="hover:bg-panel-hover transition">
                  <td className="px-3 py-2 font-bold text-text-primary">{exp.id}</td>
                  <td className="px-3 py-2 text-text-muted">{exp.timestamp}</td>
                  <td className="px-3 py-2 text-text-primary truncate max-w-[120px]">{exp.recordName}</td>
                  <td className="px-3 py-2 text-text-secondary">{exp.pgaG.toFixed(2)} g</td>
                  <td className="px-3 py-2 text-text-secondary">
                    {exp.T}s / {(exp.zeta * 100).toFixed(0)}% / {(exp.uy * 1000).toFixed(1)}mm
                  </td>
                  <td className="px-3 py-2 text-fno font-bold">{exp.errRelL2.toFixed(2)}%</td>
                  <td className="px-3 py-2 text-energy font-bold">{exp.speedup.toFixed(1)}x</td>
                  <td className="px-3 py-2 text-right">
                    <button
                      onClick={() => onLoadExperiment(exp)}
                      className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-fno/10 text-fno border border-fno/30 hover:bg-fno hover:text-background-deep font-semibold transition cursor-pointer text-[11px]"
                    >
                      <span>Load</span>
                      <ArrowRight size={11} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
