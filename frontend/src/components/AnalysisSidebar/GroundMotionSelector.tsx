import React, { useRef, useState } from "react";
import { CheckCircle2, FileText, UploadCloud, Waves } from "lucide-react";
import type { PresetRecord } from "../../types/simulation";

interface GroundMotionSelectorProps {
  presets: PresetRecord[];
  selectedPresetId: string;
  onSelectPreset: (id: string) => void;
  customFileName: string | null;
  onFileUpload: (content: string, filename: string) => void;
  onClearCustomFile: () => void;
  pgaG: number;
}

export const GroundMotionSelector: React.FC<GroundMotionSelectorProps> = ({
  presets,
  selectedPresetId,
  onSelectPreset,
  customFileName,
  onFileUpload,
  onClearCustomFile,
  pgaG,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

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

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-mono font-bold text-fno bg-fno-dim px-1.5 py-0.5 rounded">
            01
          </span>
          <span className="text-xs font-semibold text-text-primary uppercase tracking-wider font-mono">
            Input Record
          </span>
        </div>
        {customFileName ? (
          <span className="text-[10px] text-energy font-mono flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" /> Custom File
          </span>
        ) : (
          <span className="text-[10px] text-text-muted font-mono">PEER NGA-West2</span>
        )}
      </div>

      {/* Preset Selector Dropdown */}
      <div className="relative">
        <select
          value={customFileName ? "custom" : selectedPresetId}
          disabled={!!customFileName}
          onChange={(e) => onSelectPreset(e.target.value)}
          className="w-full bg-background-deep border border-border-medium rounded px-3 py-2 text-xs font-mono text-text-primary focus:border-fno focus:outline-none transition disabled:opacity-50 appearance-none cursor-pointer"
        >
          {customFileName && <option value="custom">📁 Custom: {customFileName}</option>}
          {presets.map((p) => (
            <option key={p.id} value={p.id} className="bg-panel-base text-text-primary py-1">
              {p.name}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-text-muted">
          <Waves className="w-3.5 h-3.5" />
        </div>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border border-dashed rounded p-3 text-center cursor-pointer transition flex flex-col items-center justify-center gap-1 ${
          isDragOver
            ? "border-fno bg-fno-dim"
            : customFileName
            ? "border-energy bg-energy-dim/20"
            : "border-border-medium bg-background-deep/50 hover:bg-background-deep hover:border-text-muted"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".AT2,.at2,.txt,.csv,.dat"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        {customFileName ? (
          <div className="flex items-center justify-between w-full px-1">
            <div className="flex items-center space-x-2 text-left truncate">
              <FileText className="w-4 h-4 text-energy shrink-0" />
              <div className="truncate">
                <p className="text-[11px] font-mono text-text-primary truncate">{customFileName}</p>
                <p className="text-[9px] text-text-muted font-mono">Resampled to dt = 0.010 s</p>
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onClearCustomFile();
              }}
              className="text-[10px] text-danger hover:underline font-mono ml-2 shrink-0"
            >
              Reset
            </button>
          </div>
        ) : (
          <>
            <UploadCloud className="w-4 h-4 text-text-muted mb-0.5" />
            <p className="text-[11px] text-text-secondary">
              Drag & drop <span className="font-mono text-fno">.AT2</span>,{" "}
              <span className="font-mono text-fno">.CSV</span>, or{" "}
              <span className="font-mono text-fno">.TXT</span>
            </p>
            <p className="text-[9px] text-text-muted font-mono">Automatic baseline correction & resampling</p>
          </>
        )}
      </div>

      {/* Record Technical Metadata Strip */}
      <div className="grid grid-cols-3 gap-1.5 p-2 rounded bg-background-deep border border-border-subtle font-mono text-[10px]">
        <div>
          <span className="text-text-muted block text-[9px]">SAMPLING dt</span>
          <span className="text-text-secondary font-mono-num">0.010 s</span>
        </div>
        <div>
          <span className="text-text-muted block text-[9px]">DURATION</span>
          <span className="text-text-secondary font-mono-num">20.48 s</span>
        </div>
        <div>
          <span className="text-text-muted block text-[9px]">SCALED PGA</span>
          <span className="text-fno font-semibold font-mono-num">{pgaG.toFixed(2)} g</span>
        </div>
      </div>
    </div>
  );
};
