import React from "react";

interface StructuralParametersProps {
  T: number;
  onChangeT: (val: number) => void;
  zeta: number;
  onChangeZeta: (val: number) => void;
}

export const StructuralParameters: React.FC<StructuralParametersProps> = ({
  T,
  onChangeT,
  zeta,
  onChangeZeta,
}) => {
  const omegaN = 2.0 * Math.PI / Math.max(1e-4, T);
  const k0 = 1.0 * (omegaN ** 2);
  const cDamping = 2.0 * zeta * 1.0 * omegaN;

  return (
    <div className="space-y-3 pt-2 border-t border-border-subtle">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-mono font-bold text-fno bg-fno-dim px-1.5 py-0.5 rounded">
            03
          </span>
          <span className="text-xs font-semibold text-text-primary uppercase tracking-wider font-mono">
            Structural Model
          </span>
        </div>
        <span className="text-[10px] text-text-muted font-mono">SDOF Oscillator</span>
      </div>

      {/* Natural Period T_n */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-[11px] text-text-secondary">
            Natural Period (T<sub>n</sub>)
          </label>
          <div className="flex items-center space-x-1">
            <input
              type="number"
              min="0.10"
              max="2.00"
              step="0.05"
              value={T}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v) && v >= 0.1 && v <= 2.0) onChangeT(v);
              }}
              className="w-16 bg-background-deep border border-border-medium rounded px-2 py-0.5 text-right text-xs font-mono text-text-primary focus:border-fno focus:outline-none"
            />
            <span className="text-xs font-mono text-text-muted">s</span>
          </div>
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

      {/* Damping Ratio zeta */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-[11px] text-text-secondary">
            Viscous Damping Ratio (ζ)
          </label>
          <div className="flex items-center space-x-1">
            <input
              type="number"
              min="1"
              max="15"
              step="1"
              value={Math.round(zeta * 100)}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v) && v >= 1 && v <= 15) onChangeZeta(v / 100);
              }}
              className="w-16 bg-background-deep border border-border-medium rounded px-2 py-0.5 text-right text-xs font-mono text-text-primary focus:border-fno focus:outline-none"
            />
            <span className="text-xs font-mono text-text-muted">%</span>
          </div>
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

      {/* Computed Dynamics Properties */}
      <div className="grid grid-cols-3 gap-1.5 p-2 rounded bg-background-deep border border-border-subtle font-mono text-[10px]">
        <div>
          <span className="text-text-muted block text-[9px]">ω<sub>n</sub> (FREQ)</span>
          <span className="text-text-secondary font-mono-num">{omegaN.toFixed(2)} rad/s</span>
        </div>
        <div>
          <span className="text-text-muted block text-[9px]">k<sub>0</sub> (STIFFNESS)</span>
          <span className="text-text-secondary font-mono-num">{k0.toFixed(1)} N/m</span>
        </div>
        <div>
          <span className="text-text-muted block text-[9px]">c (DAMPING)</span>
          <span className="text-text-secondary font-mono-num">{cDamping.toFixed(2)} N·s/m</span>
        </div>
      </div>
    </div>
  );
};
