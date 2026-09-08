import React, { useEffect, useRef, useState } from "react";
import type { SimulationResponse } from "../../types/simulation";
import { formatNumber } from "../../utils/formatting";

interface EnergyPlotProps {
  data: SimulationResponse | null;
  hoverTimeIdx: number | null;
}

export const EnergyPlot: React.FC<EnergyPlotProps> = ({
  data,
  hoverTimeIdx,
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

    const padLeft = 60;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 25;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    const times = data.time;
    const ehGT = data.eh_gt;
    const ehPred = data.eh_pred;

    const tMin = times[0];
    const tMax = times[times.length - 1];

    const ehMax = Math.max(...ehGT, ...ehPred, 1.0) * 1.15;
    const ehMin = 0;

    const getX = (t: number) => padLeft + ((t - tMin) / (tMax - tMin || 1)) * plotW;
    const getY = (eh: number) => padTop + plotH - ((eh - ehMin) / (ehMax - ehMin || 1)) * plotH;

    // Background
    ctx.fillStyle = "#090C10";
    ctx.fillRect(0, 0, width, height);

    // Gridlines & Ticks
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#586271";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const yTicks = 3;
    for (let i = 0; i <= yTicks; i++) {
      const val = ehMin + (i / yTicks) * (ehMax - ehMin);
      const y = getY(val);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();

      ctx.fillText(`${val.toFixed(1)} J`, padLeft - 6, y);
    }

    // Baseline 0 J
    ctx.strokeStyle = "rgba(255, 255, 255, 0.10)";
    ctx.beginPath();
    ctx.moveTo(padLeft, getY(0));
    ctx.lineTo(width - padRight, getY(0));
    ctx.stroke();

    // Time Ticks
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const tStep = (tMax - tMin) <= 10 ? 2 : 5;
    for (let t = 0; t <= tMax; t += tStep) {
      const x = getX(t);
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, height - padBottom);
      ctx.stroke();
      ctx.fillText(`${t.toFixed(0)}s`, x, height - padBottom + 4);
    }

    // OpenSeesPy Ground Truth (Amber dashed line)
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = "#F59E0B";
    ctx.lineWidth = 1.25;
    ctx.setLineDash([4, 3]);
    for (let i = 0; i < times.length; i++) {
      const x = getX(times[i]);
      const y = getY(ehGT[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // SeismoFNO Predicted Curve (Cyan solid)
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = "#00D2FF";
    ctx.lineWidth = 1.5;
    for (let i = 0; i < times.length; i++) {
      const x = getX(times[i]);
      const y = getY(ehPred[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // Synchronized Cursor Line & Dots
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

      const yGtHover = getY(ehGT[hoverTimeIdx]);
      const yFnoHover = getY(ehPred[hoverTimeIdx]);

      ctx.fillStyle = "#F59E0B";
      ctx.beginPath();
      ctx.arc(xHover, yGtHover, 3.5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "#00D2FF";
      ctx.beginPath();
      ctx.arc(xHover, yFnoHover, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [data, hoverTimeIdx, resizeTrigger]);

  const finalGT = data && data.eh_gt.length > 0 ? data.eh_gt[data.eh_gt.length - 1] : 0;
  const finalFNO = data && data.eh_pred.length > 0 ? data.eh_pred[data.eh_pred.length - 1] : 0;

  return (
    <div className="select-none flex flex-col font-mono text-xs">
      <div className="px-6 py-2.5 border-b border-border-subtle bg-background-base flex items-center justify-between">
        <span className="text-text-primary uppercase tracking-wider font-semibold">
          HYSTERETIC ENERGY E<sub>h</sub>(t)
        </span>
        {data && (
          <div className="flex items-center space-x-2 text-[11px] text-text-muted font-mono-num">
            <span>GT: <strong className="text-opensees">{formatNumber(finalGT, 1)} J</strong></span>
            <span>·</span>
            <span>FNO: <strong className="text-fno">{formatNumber(finalFNO, 1)} J</strong></span>
          </div>
        )}
      </div>

      <div ref={containerRef} className="relative w-full h-[220px] bg-background-deep overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
};
