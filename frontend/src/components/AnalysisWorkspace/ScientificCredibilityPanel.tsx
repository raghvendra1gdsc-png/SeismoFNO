import React, { useState } from "react";
import { Shield, ChevronDown, ChevronUp, Cpu, Activity, Database, CheckCircle, AlertTriangle } from "lucide-react";

export const ScientificCredibilityPanel: React.FC = () => {
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  return (
    <div className="bg-panel-base/90 border-b border-border-subtle font-mono text-xs select-none">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-1.5 flex items-center justify-between text-text-muted hover:text-text-primary transition cursor-pointer bg-background-base/40"
      >
        <div className="flex items-center space-x-2">
          <Shield size={13} className="text-energy" />
          <span className="text-[11px] font-bold uppercase tracking-wider text-text-primary">
            SCIENTIFIC CREDIBILITY & METHOD PROVENANCE ARCHITECTURE
          </span>
          <span className="text-[10px] text-text-muted hidden sm:inline">
            · Dual-Track Validation Protocol
          </span>
        </div>
        <div className="flex items-center space-x-1 text-[10px]">
          <span>{isExpanded ? "COLLAPSE" : "DISCLOSE"}</span>
          {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </div>
      </button>

      {/* Expanded Provenance Grid */}
      {isExpanded && (
        <div className="p-3 grid grid-cols-2 md:grid-cols-5 gap-2.5 bg-panel-base border-t border-border-subtle/50 text-[11px]">
          {/* Track 1: AI Reasoning */}
          <div className="bg-background-deep p-2 rounded border border-border-subtle space-y-1">
            <div className="text-[9px] uppercase tracking-wider text-text-muted flex items-center space-x-1">
              <Cpu size={10} className="text-fno" />
              <span>AI REASONING</span>
            </div>
            <div className="text-text-primary font-bold">NVIDIA Nemotron</div>
            <div className="text-[10px] text-text-muted">via Nebius Token Factory</div>
            <div className="text-[9px] text-fno">Autonomous ReAct Planning</div>
          </div>

          {/* Track 2: Neural Operator Surrogate */}
          <div className="bg-background-deep p-2 rounded border border-fno/30 space-y-1">
            <div className="text-[9px] uppercase tracking-wider text-text-muted flex items-center space-x-1">
              <Activity size={10} className="text-fno" />
              <span>LEARNED SURROGATE</span>
            </div>
            <div className="text-fno font-bold">SeismoFNO</div>
            <div className="text-[10px] text-text-muted">Fourier Neural Operator</div>
            <div className="text-[9px] text-energy">Zero-Shot Resolution Invariant</div>
          </div>

          {/* Track 3: Numerical Physics Reference */}
          <div className="bg-background-deep p-2 rounded border border-opensees/30 space-y-1">
            <div className="text-[9px] uppercase tracking-wider text-text-muted flex items-center space-x-1">
              <Database size={10} className="text-opensees" />
              <span>PHYSICS REFERENCE</span>
            </div>
            <div className="text-opensees font-bold">OpenSeesPy NLTHA</div>
            <div className="text-[10px] text-text-muted">Nonlinear Time History</div>
            <div className="text-[9px] text-text-secondary">Process Mutex Serialized</div>
          </div>

          {/* Track 4: Validation Matrix */}
          <div className="bg-background-deep p-2 rounded border border-border-subtle space-y-1">
            <div className="text-[9px] uppercase tracking-wider text-text-muted flex items-center space-x-1">
              <CheckCircle size={10} className="text-energy" />
              <span>CROSS-VALIDATION</span>
            </div>
            <div className="text-text-primary font-bold">FNO vs Physics</div>
            <div className="text-[10px] text-text-muted">Relative L2 & Peak Error</div>
            <div className="text-[9px] text-text-muted">Phase & Yield Discrepancy</div>
          </div>

          {/* Track 5: Epistemic Trust */}
          <div className="bg-background-deep p-2 rounded border border-border-subtle space-y-1">
            <div className="text-[9px] uppercase tracking-wider text-text-muted flex items-center space-x-1">
              <AlertTriangle size={10} className="text-opensees" />
              <span>EPISTEMIC TRUST</span>
            </div>
            <div className="text-energy font-bold">OOD Detection</div>
            <div className="text-[10px] text-text-muted">Domain Boundary Audit</div>
            <div className="text-[9px] text-text-secondary">Uncertainty: NOT QUANTIFIED</div>
          </div>
        </div>
      )}
    </div>
  );
};
