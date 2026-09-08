import React, { useRef, useState } from "react";
import type { MaterialType, PresetRecord } from "../../types/simulation";

interface AnalysisSidebarProps {
  presets: PresetRecord[];
  selectedPresetId: string;
  onSelectPreset: (id: string) => void;
  customFileName: string | null;
  onFileUpload: (content: string, filename: string) => void;
  onClearCustomFile: () => void;
  pgaG: number;
  onChangePga: (val: number) => void;
  T: number;
  onChangeT: (val: number) => void;
  zeta: number;
  onChangeZeta: (val: number) => void;
  materialType: MaterialType;
  onChangeMaterialType: (type: MaterialType) => void;
  uy: number;
  onChangeUy: (val: number) => void;
  alpha: number;
  onChangeAlpha: (val: number) => void;
  isLoading: boolean;
  onRun: () => void;
  lastLatencyMs: number | null;
  onLoadDemoScenario?: (scenario: "in_domain" | "ood") => void;
}

export const AnalysisSidebar: React.FC<AnalysisSidebarProps> = ({
  presets,
  selectedPresetId,
  onSelectPreset,
  customFileName,
  onFileUpload,
  onClearCustomFile,
  pgaG,
  onChangePga,
  T,
  onChangeT,
  zeta,
  onChangeZeta,
  materialType,
  onChangeMaterialType,
  uy,
  onChangeUy,
  alpha,
  onChangeAlpha,
  isLoading,
  onRun,
  lastLatencyMs,
  onLoadDemoScenario,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const omegaN = 2.0 * Math.PI / Math.max(1e-4, T);
  const k0 = 1.0 * (omegaN ** 2);
  const isBilinear = materialType === "bilinear";

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (content) {
        onFileUpload(content, file.name);
      }
    };
    reader.readAsText(file);
  };

  return (
    <aside className="w-full lg:w-[280px] bg-background-base border-r border-border-subtle flex flex-col font-mono text-xs select-none shrink-0 h-full">
      {/* Rail Header */}
      <div className="px-4 py-2.5 border-b border-border-subtle flex items-center justify-between text-text-muted text-[10px] tracking-widest uppercase">
        <span>ANALYSIS SETUP</span>
        <span>INSTRUMENT</span>
      </div>

      <div className="p-4 space-y-5 overflow-y-auto flex-1">
        {/* QUICK DEMO SCENARIO SHORTCUTS */}
        {onLoadDemoScenario && (
          <div className="space-y-1 bg-background-deep p-2 border border-border-subtle">
            <div className="text-[9px] font-mono text-text-muted uppercase tracking-widest">
              FROZEN DEMO SCENARIOS
            </div>
            <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
              <button
                type="button"
                onClick={() => onLoadDemoScenario("in_domain")}
                className="px-1.5 py-1 bg-panel-base border border-energy/40 text-energy hover:bg-energy/10 rounded transition text-left cursor-pointer truncate font-semibold"
                title="Load In-Domain Demo Scenario (Imperial Valley 0.4g, T=0.5s)"
              >
                ● IN-DOMAIN
              </button>
              <button
                type="button"
                onClick={() => onLoadDemoScenario("ood")}
                className="px-1.5 py-1 bg-panel-base border border-danger/40 text-danger hover:bg-danger/10 rounded transition text-left cursor-pointer truncate font-semibold"
                title="Load Out-of-Domain Scenario (PGA=1.6g, T=3.8s, uy=60mm)"
              >
                ▲ OOD STRESS
              </button>
            </div>
          </div>
        )}

        {/* 01 GROUND MOTION */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[10px] text-text-muted uppercase tracking-wider">
            <span>01  GROUND MOTION</span>
            {customFileName && <span className="text-energy">CUSTOM</span>}
          </div>

          <div className="space-y-1.5">
            <select
              value={customFileName ? "custom" : selectedPresetId}
              disabled={!!customFileName}
              onChange={(e) => onSelectPreset(e.target.value)}
              className="w-full bg-background-deep border border-border-subtle rounded-none px-2 py-1.5 text-xs font-mono text-text-primary focus:border-border-strong focus:outline-none disabled:opacity-60 appearance-none cursor-pointer"
            >
              {customFileName && <option value="custom">{customFileName}</option>}
              {presets.map((p) => (
                <option key={p.id} value={p.id} className="bg-background-deep text-text-primary">
                  {p.name}
                </option>
              ))}
            </select>

            {/* Custom file upload strip */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragOver(false);
                if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`border border-dashed border-border-subtle p-2 text-center cursor-pointer transition text-[10px] text-text-muted ${
                isDragOver ? "bg-panel-hover border-text-muted" : "hover:border-border-medium"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".AT2,.at2,.txt,.csv,.dat"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFile(e.target.files[0]);
                }}
              />
              {customFileName ? (
                <div className="flex items-center justify-between">
                  <span className="text-text-primary truncate">{customFileName}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onClearCustomFile();
                    }}
                    className="text-danger hover:underline ml-2"
                  >
                    RESET
                  </button>
                </div>
              ) : (
                <span>UPLOAD .AT2 / .CSV</span>
              )}
            </div>
          </div>

          {/* PGA Control */}
          <div className="space-y-1 pt-1">
            <div className="flex justify-between text-[11px] text-text-secondary">
              <span>PGA</span>
              <span className="text-text-primary font-mono-num">{pgaG.toFixed(2)} g</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="1.50"
              step="0.05"
              value={pgaG}
              onChange={(e) => onChangePga(parseFloat(e.target.value))}
              className="w-full cursor-pointer"
            />
          </div>
        </div>

        {/* 02 STRUCTURAL MODEL */}
        <div className="space-y-3 pt-3 border-t border-border-subtle">
          <div className="flex items-center justify-between text-[10px] text-text-muted uppercase tracking-wider">
            <span>02  STRUCTURAL MODEL</span>
            <span>SDOF</span>
          </div>

          {/* T_n */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px] text-text-secondary">
              <span>T<sub>n</sub> (PERIOD)</span>
              <span className="text-text-primary font-mono-num">{T.toFixed(2)} s</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="2.00"
              step="0.05"
              value={T}
              onChange={(e) => onChangeT(parseFloat(e.target.value))}
              className="w-full cursor-pointer"
            />
          </div>

          {/* Damping zeta */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px] text-text-secondary">
              <span>ζ (DAMPING)</span>
              <span className="text-text-primary font-mono-num">{(zeta * 100).toFixed(1)} %</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.15"
              step="0.01"
              value={zeta}
              onChange={(e) => onChangeZeta(parseFloat(e.target.value))}
              className="w-full cursor-pointer"
            />
          </div>

          {/* Derived dynamic values */}
          <div className="text-[10px] text-text-muted space-y-0.5 pt-0.5">
            <div className="flex justify-between">
              <span>ω<sub>n</sub></span>
              <span className="text-text-secondary font-mono-num">{omegaN.toFixed(2)} rad/s</span>
            </div>
            <div className="flex justify-between">
              <span>k<sub>0</sub></span>
              <span className="text-text-secondary font-mono-num">{k0.toFixed(1)} N/m</span>
            </div>
          </div>
        </div>

        {/* 03 MATERIAL */}
        <div className="space-y-3 pt-3 border-t border-border-subtle">
          <div className="flex items-center justify-between text-[10px] text-text-muted uppercase tracking-wider">
            <span>03  CONSTITUTIVE LAW</span>
          </div>

          {/* Material Type Toggle */}
          <div className="grid grid-cols-2 text-[11px] border border-border-subtle">
            <button
              type="button"
              onClick={() => onChangeMaterialType("bilinear")}
              className={`py-1 text-center transition cursor-pointer ${
                isBilinear
                  ? "bg-panel-hover text-text-primary font-semibold"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              BILINEAR
            </button>
            <button
              type="button"
              onClick={() => onChangeMaterialType("elastic")}
              className={`py-1 text-center border-l border-border-subtle transition cursor-pointer ${
                !isBilinear
                  ? "bg-panel-hover text-text-primary font-semibold"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              ELASTIC
            </button>
          </div>

          {isBilinear && (
            <div className="space-y-2 pt-1">
              {/* Yield displacement u_y */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-text-secondary">
                  <span>u<sub>y</sub> (YIELD)</span>
                  <span className="text-text-primary font-mono-num">{(uy * 1000).toFixed(1)} mm</span>
                </div>
                <input
                  type="range"
                  min="0.002"
                  max="0.050"
                  step="0.002"
                  value={uy}
                  onChange={(e) => onChangeUy(parseFloat(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </div>

              {/* Post-yield ratio alpha */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-text-secondary">
                  <span>α (HARDENING)</span>
                  <span className="text-text-primary font-mono-num">{alpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.00"
                  max="0.20"
                  step="0.01"
                  value={alpha}
                  onChange={(e) => onChangeAlpha(parseFloat(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="flex justify-between text-[10px] text-text-muted pt-0.5">
                <span>F<sub>y</sub></span>
                <span className="text-text-secondary font-mono-num">{(k0 * uy).toFixed(1)} N</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Trigger Button Footer */}
      <div className="p-4 border-t border-border-subtle bg-background-base">
        <button
          type="button"
          disabled={isLoading}
          onClick={onRun}
          className={`w-full py-2 px-3 text-xs font-mono font-bold tracking-wider uppercase transition cursor-pointer border ${
            isLoading
              ? "bg-panel-hover text-text-muted border-border-subtle cursor-not-allowed"
              : "bg-text-primary text-background-deep hover:bg-white border-transparent"
          }`}
        >
          {isLoading ? "RUNNING SIMULATION..." : "RUN ANALYSIS"}
        </button>
        {lastLatencyMs !== null && !isLoading && (
          <div className="text-[10px] text-text-muted text-center mt-1.5">
            LATENCY: {lastLatencyMs.toFixed(2)} ms
          </div>
        )}
      </div>
    </aside>
  );
};
