import React from "react";
import { Check, Loader2, Play } from "lucide-react";

interface RunAnalysisButtonProps {
  isLoading: boolean;
  onRun: () => void;
  lastLatencyMs: number | null;
}

export const RunAnalysisButton: React.FC<RunAnalysisButtonProps> = ({
  isLoading,
  onRun,
  lastLatencyMs,
}) => {
  return (
    <div className="space-y-2 pt-2 border-t border-border-subtle">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-mono font-bold text-fno bg-fno-dim px-1.5 py-0.5 rounded">
            05
          </span>
          <span className="text-xs font-semibold text-text-primary uppercase tracking-wider font-mono">
            Execution
          </span>
        </div>
        {lastLatencyMs !== null && !isLoading && (
          <span className="text-[10px] text-energy font-mono">
            {lastLatencyMs.toFixed(2)} ms inference
          </span>
        )}
      </div>

      <button
        type="button"
        disabled={isLoading}
        onClick={onRun}
        className={`w-full py-2.5 px-4 rounded font-mono font-bold text-xs uppercase tracking-wider transition flex items-center justify-center space-x-2 ${
          isLoading
            ? "bg-panel-hover text-text-muted border border-border-subtle cursor-not-allowed"
            : "bg-fno hover:bg-fno-light active:bg-fno-dark text-background-deep shadow-md hover:shadow-fno/20 cursor-pointer"
        }`}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-fno" />
            <span>Running NLTHA & FNO...</span>
          </>
        ) : lastLatencyMs !== null ? (
          <>
            <Check className="w-4 h-4 text-background-deep" />
            <span>Re-Run Analysis</span>
          </>
        ) : (
          <>
            <Play className="w-4 h-4 fill-current" />
            <span>Run Analysis</span>
          </>
        )}
      </button>
    </div>
  );
};
