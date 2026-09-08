import React from "react";

interface ExcitationControlsProps {
  pgaG: number;
  onChangePga: (val: number) => void;
}

export const ExcitationControls: React.FC<ExcitationControlsProps> = ({
  pgaG,
  onChangePga,
}) => {
  return (
    <div className="space-y-3 pt-2 border-t border-border-subtle">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-mono font-bold text-fno bg-fno-dim px-1.5 py-0.5 rounded">
            02
          </span>
          <span className="text-xs font-semibold text-text-primary uppercase tracking-wider font-mono">
            Excitation Scaling
          </span>
        </div>
        <span className="text-[10px] text-text-muted font-mono">Peak Intensity</span>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-[11px] text-text-secondary">
            Target PGA Scale (a<sub>g,max</sub>)
          </label>
          <div className="flex items-center space-x-1">
            <input
              type="number"
              min="0.05"
              max="1.50"
              step="0.05"
              value={pgaG}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v) && v >= 0.05 && v <= 1.5) onChangePga(v);
              }}
              className="w-16 bg-background-deep border border-border-medium rounded px-2 py-0.5 text-right text-xs font-mono text-fno focus:border-fno focus:outline-none"
            />
            <span className="text-xs font-mono text-text-muted">g</span>
          </div>
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

        <div className="flex justify-between text-[9px] font-mono text-text-muted px-0.5">
          <span>0.05 g (Service)</span>
          <span>0.40 g (Design DBE)</span>
          <span>1.20 g (MCE)</span>
        </div>
      </div>
    </div>
  );
};
