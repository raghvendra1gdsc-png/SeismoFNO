import React from "react";
import { ShieldCheck, ShieldAlert } from "lucide-react";
import type { OODSummary } from "../../types/agent";

interface TrustBannerProps {
  ood?: OODSummary | null;
}

export const TrustBanner: React.FC<TrustBannerProps> = ({ ood }) => {
  const isOod = ood?.is_ood ?? false;
  const violations = ood?.domain_violations ?? [];
  const recommendation =
    ood?.recommendation ||
    (isOod
      ? "Input exceeds calibrated operating envelope. Use OpenSeesPy numerical simulation for authoritative verification."
      : "Operating parameters within validated Fourier Neural Operator envelope.");

  return (
    <div
      className={`px-4 py-2.5 border-b text-xs font-mono flex flex-col md:flex-row md:items-center justify-between gap-2 select-none ${
        isOod
          ? "bg-danger/10 border-danger/30 text-danger"
          : "bg-background-base border-border-subtle text-text-secondary"
      }`}
    >
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-1.5 font-bold">
          {isOod ? (
            <>
              <ShieldAlert size={14} className="text-danger animate-pulse" />
              <span className="text-danger tracking-wider">OOD WARNING: OUT-OF-DOMAIN</span>
            </>
          ) : (
            <>
              <ShieldCheck size={14} className="text-energy" />
              <span className="text-energy tracking-wider">DOMAIN STATUS: IN-DOMAIN (VALIDATED)</span>
            </>
          )}
        </div>

        <div className="h-3 w-px bg-border-subtle hidden sm:block"></div>

        <div className="text-[11px] text-text-muted">
          {isOod && violations.length > 0 ? (
            <span className="text-danger/90">
              Violations: {violations.join(", ")}
            </span>
          ) : (
            <span>{recommendation}</span>
          )}
        </div>
      </div>

      <div className="flex items-center space-x-3 text-[11px]">
        <div className="flex items-center space-x-1.5">
          <span className="text-text-muted">EPISTEMIC UNCERTAINTY:</span>
          <span className="font-bold text-text-primary bg-panel-base px-2 py-0.5 rounded border border-border-subtle">
            {ood?.uncertainty_status || "NOT_QUANTIFIED"}
          </span>
        </div>
      </div>
    </div>
  );
};
