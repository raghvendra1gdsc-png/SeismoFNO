import React, { useState } from "react";
import { FileText, Download, Copy, Check, Loader2 } from "lucide-react";
import { generateReport } from "../../api/agent";
import type { ReportResponse } from "../../types/agent";

interface ReportWorkspaceProps {
  currentPresetId: string;
  currentPga: number;
  currentT: number;
  currentZeta: number;
  currentMaterialType: "bilinear" | "elastic";
  currentUy: number;
  currentAlpha: number;
}

export const ReportWorkspace: React.FC<ReportWorkspaceProps> = ({
  currentPresetId,
  currentPga,
  currentT,
  currentZeta,
  currentMaterialType,
  currentUy,
  currentAlpha,
}) => {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await generateReport({
        preset_id: currentPresetId,
        pga_g: currentPga,
        T: currentT,
        zeta: currentZeta,
        material_type: currentMaterialType,
        u_y: currentUy,
        alpha: currentAlpha,
        format: "markdown",
      });
      setReport(res);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (report) {
      navigator.clipboard.writeText(report.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    if (!report) return;
    const blob = new Blob([report.content], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `SeismoAgent_Audit_${Date.now()}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="p-4 space-y-4 font-mono text-xs select-none">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center space-x-2">
          <FileText size={16} className="text-fno" />
          <span className="font-bold text-text-primary uppercase tracking-wider text-sm">
            ENGINEERING AUDIT REPORT GENERATOR
          </span>
        </div>
        <div className="flex items-center space-x-2">
          {report && (
            <>
              <button
                onClick={handleCopy}
                className="flex items-center space-x-1 px-2.5 py-1 rounded bg-panel-base border border-border-subtle hover:bg-panel-hover text-text-secondary hover:text-text-primary transition cursor-pointer"
              >
                {copied ? <Check size={12} className="text-energy" /> : <Copy size={12} />}
                <span>{copied ? "Copied" : "Copy"}</span>
              </button>
              <button
                onClick={handleDownload}
                className="flex items-center space-x-1 px-2.5 py-1 rounded bg-panel-base border border-border-subtle hover:bg-panel-hover text-text-secondary hover:text-text-primary transition cursor-pointer"
              >
                <Download size={12} />
                <span>Export .md</span>
              </button>
            </>
          )}

          <button
            onClick={handleGenerate}
            disabled={isLoading}
            className="flex items-center space-x-1.5 px-3 py-1 rounded bg-fno/10 text-fno border border-fno/30 hover:bg-fno hover:text-background-deep font-semibold transition disabled:opacity-50 cursor-pointer"
          >
            {isLoading ? <Loader2 size={13} className="animate-spin" /> : <FileText size={13} />}
            <span>{isLoading ? "Generating..." : "Generate Audit Report"}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-danger/10 border border-danger/30 rounded text-danger text-xs">
          Report generation failed: {error}
        </div>
      )}

      {/* Report Content Display */}
      {report ? (
        <div className="bg-panel-base border border-border-subtle rounded-md p-6 overflow-y-auto max-h-[calc(100vh-230px)] shadow-inner">
          <pre className="font-mono text-xs text-text-primary leading-relaxed whitespace-pre-wrap selection:bg-fno/30">
            {report.content}
          </pre>
        </div>
      ) : (
        <div className="bg-panel-base border border-border-subtle rounded-md p-12 text-center space-y-3">
          <FileText size={32} className="text-text-muted mx-auto" />
          <div className="text-text-secondary font-semibold text-sm">
            No report generated for current experiment
          </div>
          <p className="text-text-muted max-w-md mx-auto text-xs leading-relaxed">
            Click "Generate Audit Report" above to compile a comprehensive, deterministic scientific audit
            incorporating OpenSeesPy ground truth, SeismoFNO predictions, and epistemic boundaries.
          </p>
        </div>
      )}
    </div>
  );
};
