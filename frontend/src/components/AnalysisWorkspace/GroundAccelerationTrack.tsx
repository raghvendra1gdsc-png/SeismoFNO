import React, { useEffect, useRef, useState } from "react";
import type { SimulationResponse } from "../../types/simulation";

interface GroundAccelerationTrackProps {
  data: SimulationResponse | null;
  hoverTimeIdx: number | null;
  onHoverTimeIdx: (idx: number | null) => void;
}

export const GroundAccelerationTrack: React.FC<GroundAccelerationTrackProps> = ({
  data,
  hoverTimeIdx,
  onHoverTimeIdx,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [resizeTrigger, setResizeTrigger] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(() => setResizeTrigger((p) => p + 1));
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.time.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    ctx.scale(dpr, dpr);
    const width = rect.width;
    const height = rect.height;

    // Margins matching Displacement plot exactly
    const padLeft = 65;
    const padRight = 20;
    const padTop = 6;
    const padBottom = 16;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    const times = data.time;
    const ag = data.ag;

    const tMin = times[0];
    const tMax = times[times.length - 1];

    let agMax = Math.max(...ag.map(Math.abs), 0.5);
    agMax = agMax * 1.15;
    const agMin = -agMax;

    const getX = (t: number) => padLeft + ((t - tMin) / (tMax - tMin || 1)) * plotW;
    const getY = (v: number) => padTop + plotH / 2 - (v / (agMax - agMin || 1)) * plotH;

    // Background
    ctx.fillStyle = "#090C10";
    ctx.fillRect(0, 0, width, height);

    // Baseline 0
    const yZero = getY(0);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.10)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, yZero);
    ctx.lineTo(width - padRight, yZero);
    ctx.stroke();

    // Axis limits
    ctx.fillStyle = "#586271";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`+${(agMax / 9.80665).toFixed(2)}g`, padLeft - 8, padTop + 4);
    ctx.fillText(`-${(agMax / 9.80665).toFixed(2)}g`, padLeft - 8, height - padBottom - 4);

    // Waveform Outline
    ctx.beginPath();
    ctx.strokeStyle = "#8993A1";
    ctx.lineWidth = 1.0;
    for (let i = 0; i < times.length; i++) {
      const x = getX(times[i]);
      const y = getY(ag[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Hover Cursor Line
    if (hoverTimeIdx !== null && hoverTimeIdx < times.length) {
      const xHover = getX(times[hoverTimeIdx]);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(xHover, padTop);
      ctx.lineTo(xHover, height - padBottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [data, hoverTimeIdx, resizeTrigger]);

  return (
    <div className="select-none border-t border-border-subtle bg-background-base">
      <div className="px-6 py-1.5 flex items-center justify-between text-[11px] font-mono text-text-muted">
        <span>GROUND ACCELERATION ä<sub>g</sub>(t)</span>
        {data && (
          <span className="text-text-secondary font-mono-num">
            PGA: {data.metrics.pga_g.toFixed(2)} g ({(data.metrics.pga_g * 9.80665).toFixed(2)} m/s²)
          </span>
        )}
      </div>

      <div ref={containerRef} className="relative w-full h-[60px] bg-background-deep overflow-hidden">
        <canvas
          ref={canvasRef}
          className="w-full h-full cursor-crosshair"
          onMouseMove={(e) => {
            if (!data || data.time.length === 0) return;
            const canvas = canvasRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const padLeft = 65;
            const padRight = 20;
            const plotW = rect.width - padLeft - padRight;
            if (x >= padLeft && x <= rect.width - padRight) {
              const pct = (x - padLeft) / plotW;
              const idx = Math.round(pct * (data.time.length - 1));
              onHoverTimeIdx(idx);
            }
          }}
          onMouseLeave={() => onHoverTimeIdx(null)}
        />
      </div>
    </div>
  );
};
