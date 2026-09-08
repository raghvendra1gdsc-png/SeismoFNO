import React, { useEffect, useRef, useState } from "react";
import type { SimulationResponse } from "../../types/simulation";

interface DisplacementPlotProps {
  data: SimulationResponse | null;
  hoverTimeIdx: number | null;
  onHoverTimeIdx: (idx: number | null) => void;
}

export const DisplacementPlot: React.FC<DisplacementPlotProps> = ({
  data,
  hoverTimeIdx,
  onHoverTimeIdx,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomRange, setZoomRange] = useState<[number, number] | null>(null);
  const [showFNO, setShowFNO] = useState(true);
  const [showGT, setShowGT] = useState(true);
  const [resizeTrigger, setResizeTrigger] = useState(0);

  // ResizeObserver on container
  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(() => setResizeTrigger((p) => p + 1));
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // Redraw Canvas
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

    // Margins
    const padLeft = 65;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 25;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Time indices slice
    const totalPts = data.time.length;
    const startIdx = zoomRange ? zoomRange[0] : 0;
    const endIdx = zoomRange ? zoomRange[1] : totalPts - 1;

    const times = data.time.slice(startIdx, endIdx + 1);
    const uGT = data.u_gt.slice(startIdx, endIdx + 1);
    const uPred = data.u_pred.slice(startIdx, endIdx + 1);

    const tMin = times[0];
    const tMax = times[times.length - 1];

    let uMin = Math.min(...uGT, ...uPred) * 1000;
    let uMax = Math.max(...uGT, ...uPred) * 1000;
    const absMax = Math.max(Math.abs(uMin), Math.abs(uMax), 1.0);
    uMin = -absMax * 1.15;
    uMax = absMax * 1.15;

    const getX = (t: number) => padLeft + ((t - tMin) / (tMax - tMin || 1)) * plotW;
    const getY = (uMm: number) => padTop + plotH / 2 - (uMm / (uMax - uMin || 1)) * plotH;

    // Plot Background
    ctx.fillStyle = "#090C10";
    ctx.fillRect(0, 0, width, height);

    // Grid & Axis Ticks
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.fillStyle = "#586271";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const val = uMin + (i / yTicks) * (uMax - uMin);
      const y = getY(val);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();

      ctx.fillText(`${val.toFixed(1)} mm`, padLeft - 8, y);
    }

    // Zero baseline
    const yZero = getY(0);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.beginPath();
    ctx.moveTo(padLeft, yZero);
    ctx.lineTo(width - padRight, yZero);
    ctx.stroke();

    // Time Ticks
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const tSpan = tMax - tMin;
    const tStep = tSpan <= 5 ? 1 : tSpan <= 10 ? 2 : 5;
    const firstTick = Math.ceil(tMin / tStep) * tStep;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    for (let t = firstTick; t <= tMax; t += tStep) {
      const x = getX(t);
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, height - padBottom);
      ctx.stroke();

      ctx.fillText(`${t.toFixed(0)}s`, x, height - padBottom + 6);
    }

    // OpenSeesPy Ground Truth (Amber dashed line)
    if (showGT) {
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = "#F59E0B";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);

      for (let i = 0; i < times.length; i++) {
        const x = getX(times[i]);
        const y = getY(uGT[i] * 1000);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // SeismoFNO Prediction (Cyan solid line)
    if (showFNO) {
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = "#00D2FF";
      ctx.lineWidth = 1.75;

      for (let i = 0; i < times.length; i++) {
        const x = getX(times[i]);
        const y = getY(uPred[i] * 1000);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Synchronized Cursor Line & Highlight Dots
    if (hoverTimeIdx !== null && hoverTimeIdx >= startIdx && hoverTimeIdx <= endIdx) {
      const idx = hoverTimeIdx - startIdx;
      const tHover = times[idx];
      const xHover = getX(tHover);
      const yGtHover = getY(uGT[idx] * 1000);
      const yFnoHover = getY(uPred[idx] * 1000);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(xHover, padTop);
      ctx.lineTo(xHover, height - padBottom);
      ctx.stroke();
      ctx.setLineDash([]);

      if (showGT) {
        ctx.fillStyle = "#F59E0B";
        ctx.beginPath();
        ctx.arc(xHover, yGtHover, 3.5, 0, 2 * Math.PI);
        ctx.fill();
      }
      if (showFNO) {
        ctx.fillStyle = "#00D2FF";
        ctx.beginPath();
        ctx.arc(xHover, yFnoHover, 3.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }, [data, zoomRange, showFNO, showGT, hoverTimeIdx, resizeTrigger]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || data.time.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const padLeft = 65;
    const padRight = 20;
    const plotW = rect.width - padLeft - padRight;

    if (x < padLeft || x > rect.width - padRight) {
      onHoverTimeIdx(null);
      return;
    }

    const totalPts = data.time.length;
    const startIdx = zoomRange ? zoomRange[0] : 0;
    const endIdx = zoomRange ? zoomRange[1] : totalPts - 1;
    const pct = Math.max(0, Math.min(1, (x - padLeft) / plotW));
    const targetIdx = Math.round(startIdx + pct * (endIdx - startIdx));
    onHoverTimeIdx(targetIdx);
  };

  return (
    <div className="select-none">
      {/* Section Technical Label & Controls */}
      <div className="px-6 py-2.5 border-b border-border-subtle bg-background-base flex items-center justify-between font-mono text-xs">
        <div className="flex items-center space-x-3">
          <span className="text-text-primary uppercase tracking-wider font-semibold">
            DISPLACEMENT RESPONSE
          </span>
          <span className="text-text-muted">
            u(t) — SEISMOFNO / OPENSEESPY
          </span>
        </div>

        {/* Legend / Toggles */}
        <div className="flex items-center space-x-3 text-[11px]">
          <button
            type="button"
            onClick={() => setShowFNO(!showFNO)}
            className={`flex items-center space-x-1.5 transition cursor-pointer ${
              showFNO ? "text-fno font-semibold" : "text-text-muted line-through"
            }`}
          >
            <span className="w-3 h-0.5 bg-fno inline-block"></span>
            <span>SeismoFNO</span>
          </button>

          <button
            type="button"
            onClick={() => setShowGT(!showGT)}
            className={`flex items-center space-x-1.5 transition cursor-pointer ${
              showGT ? "text-opensees font-semibold" : "text-text-muted line-through"
            }`}
          >
            <span className="w-3 h-0.5 border-b border-dashed border-opensees inline-block"></span>
            <span>OpenSeesPy</span>
          </button>

          {data && zoomRange && (
            <button
              type="button"
              onClick={() => setZoomRange(null)}
              className="text-[10px] text-text-muted hover:text-text-primary uppercase ml-2 cursor-pointer"
            >
              [RESET ZOOM]
            </button>
          )}
        </div>
      </div>

      {/* Plot Canvas */}
      <div ref={containerRef} className="relative w-full h-[290px] bg-background-deep overflow-hidden">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => onHoverTimeIdx(null)}
          className="w-full h-full cursor-crosshair"
        />

        {/* Synchronized Inspection Readout Tooltip */}
        {data && hoverTimeIdx !== null && hoverTimeIdx < data.time.length && (
          <div className="absolute top-3 right-4 pointer-events-none bg-background-base/95 border border-border-medium px-3 py-2 font-mono text-[11px] space-y-1 shadow-none">
            <div className="text-text-muted">t = {data.time[hoverTimeIdx].toFixed(2)} s</div>
            <div className="text-fno">SeismoFNO &nbsp;&nbsp; {(data.u_pred[hoverTimeIdx] * 1000).toFixed(2)} mm</div>
            <div className="text-opensees">OpenSeesPy &nbsp; {(data.u_gt[hoverTimeIdx] * 1000).toFixed(2)} mm</div>
            <div className="text-text-secondary border-t border-border-subtle pt-1">
              Δu &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {Math.abs((data.u_pred[hoverTimeIdx] - data.u_gt[hoverTimeIdx]) * 1000).toFixed(2)} mm
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
