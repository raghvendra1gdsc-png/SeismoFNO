import React from "react";
import type { MaterialType } from "../../types/simulation";

interface ConstitutiveModelProps {
  materialType: MaterialType;
  onChangeMaterialType: (type: MaterialType) => void;
  uy: number;
  onChangeUy: (val: number) => void;
  alpha: number;
  onChangeAlpha: (val: number) => void;
  k0: number;
}

export const ConstitutiveModel: React.FC<ConstitutiveModelProps> = ({
  materialType,
  onChangeMaterialType,
  uy,
  onChangeUy,
  alpha,
  onChangeAlpha,
  k0,
}) => {
  const isBilinear = materialType === "bilinear";
  const fyYield = k0 * uy;

  return (
    <div className="space-y-3 pt-2 border-t border-border-subtle">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-mono font-bold text-fno bg-fno-dim px-1.5 py-0.5 rounded">
            04
          </span>
          <span className="text-xs font-semibold text-text-primary uppercase tracking-wider font-mono">
            Constitutive Law
          </span>
        </div>
        <span className="text-[10px] text-text-muted font-mono">Material Physics</span>
      </div>

      {/* Segmented Control */}
      <div className="grid grid-cols-2 p-0.5 rounded bg-background-deep border border-border-medium font-mono text-xs">
        <button
          type="button"
          onClick={() => onChangeMaterialType("bilinear")}
          className={`py-1.5 px-2 rounded transition font-medium ${
            isBilinear
              ? "bg-panel-hover text-fno shadow border border-border-subtle"
              : "text-text-muted hover:text-text-secondary"
          }`}
        >
          Bilinear Hysteretic
        </button>
        <button
          type="button"
          onClick={() => onChangeMaterialType("elastic")}
          className={`py-1.5 px-2 rounded transition font-medium ${
            !isBilinear
              ? "bg-panel-hover text-fno shadow border border-border-subtle"
              : "text-text-muted hover:text-text-secondary"
          }`}
        >
          Linear Elastic
        </button>
      </div>

      {/* Bilinear Parameters */}
      {isBilinear ? (
        <div className="space-y-2.5 transition-all">
          {/* Yield Displacement u_y */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-[11px] text-text-secondary">
                Yield Displacement (u<sub>y</sub>)
              </label>
              <div className="flex items-center space-x-1">
                <input
                  type="number"
                  min="2"
                  max="50"
                  step="1"
                  value={Math.round(uy * 1000)}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v) && v >= 2 && v <= 50) onChangeUy(v / 1000);
                  }}
                  className="w-16 bg-background-deep border border-border-medium rounded px-2 py-0.5 text-right text-xs font-mono text-text-primary focus:border-fno focus:outline-none"
                />
                <span className="text-xs font-mono text-text-muted">mm</span>
              </div>
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

          {/* Post-Yield Ratio alpha */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-[11px] text-text-secondary">
                Post-Yield Ratio (α = k<sub>p</sub>/k<sub>0</sub>)
              </label>
              <div className="flex items-center space-x-1">
                <input
                  type="number"
                  min="0"
                  max="0.20"
                  step="0.01"
                  value={alpha}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v) && v >= 0 && v <= 0.20) onChangeAlpha(v);
                  }}
                  className="w-16 bg-background-deep border border-border-medium rounded px-2 py-0.5 text-right text-xs font-mono text-text-primary focus:border-fno focus:outline-none"
                />
              </div>
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

          {/* Yield Force Reference Strip */}
          <div className="flex items-center justify-between p-2 rounded bg-background-deep border border-border-subtle font-mono text-[10px]">
            <span className="text-text-muted">YIELD STRENGTH (F<sub>y</sub>)</span>
            <span className="text-opensees font-semibold font-mono-num">{fyYield.toFixed(2)} N</span>
          </div>
        </div>
      ) : (
        <div className="p-2.5 rounded bg-background-deep border border-border-subtle font-mono text-[11px] text-text-muted">
          Linear elastic regime (no yield limit or hysteretic dissipation).
        </div>
      )}
    </div>
  );
};
