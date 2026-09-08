import React, { useEffect, useRef, useState } from "react";
import type { SimulationResponse } from "../../types/simulation";
import { formatNumber } from "../../utils/formatting";

interface HysteresisPlotProps {
  data: SimulationResponse | null;
  uy: number;
  hoverTimeIdx: number | null;
}

export const HysteresisPlot: React.FC<HysteresisPlotProps> = ({
  data,
  uy,
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

    const uGT = data.u_gt.map((u) => u * 1000);
    const uPred = data.u_pred.map((u) => u * 1000);
    const frGT = data.fr_gt;
    const frPred = data.fr_pred;

    const uMaxAbs = Math.max(...uGT.map(Math.abs), ...uPred.map(Math.abs), uy * 1000 * 1.2, 1.0) * 1.15;
    const frMaxAbs = Math.max(...frGT.map(Math.abs), ...frPred.map(Math.abs), 5.0) * 1.15;

    const uMin = -uMaxAbs;
    const uMax = uMaxAbs;
    const frMin = -frMaxAbs;
    const frMax = frMaxAbs;

    const getX = (uMm: number) => padLeft + plotW / 2 + (uMm / (uMax - uMin)) * plotW;
    const getY = (frN: number) => padTop + plotH / 2 - (frN / (frMax - frMin)) * plotH;

    // Background
    ctx.fillStyle = "#090C10";
    ctx.fillRect(0, 0, width, height);

    // Axis lines (u=0, Fr=0)
    const xZero = getX(0);
    const yZero = getY(0);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.10)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, yZero);
    ctx.lineTo(width - padRight, yZero);
    ctx.moveTo(xZero, padTop);
    ctx.lineTo(xZero, height - padBottom);
    ctx.stroke();

    // Yield Limits (±uy)
    const uyMm = uy * 1000;
    const xUyPos = getX(uyMm);
    const xUyNeg = getX(-uyMm);

    ctx.strokeStyle = "rgba(245, 158, 11, 0.25)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(xUyPos, padTop);
    ctx.lineTo(xUyPos, height - padBottom);
    ctx.moveTo(xUyNeg, padTop);
    ctx.lineTo(xUyNeg, height - padBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Axis Labels & Ticks
    ctx.fillStyle = "#586271";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(`+u_y`, xUyPos, padTop - 4);
    ctx.fillText(`-u_y`, xUyNeg, padTop - 4);

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`${frMax.toFixed(0)}N`, padLeft - 6, padTop + 4);
    ctx.fillText(`0`, padLeft - 6, yZero);
    ctx.fillText(`${frMin.toFixed(0)}N`, padLeft - 6, height - padBottom - 4);

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`${uMin.toFixed(1)} mm`, padLeft + 15, height - padBottom + 4);
    ctx.fillText(`${uMax.toFixed(1)} mm`, width - padRight - 15, height - padBottom + 4);

    // OpenSeesPy Ground Truth Hysteresis (Amber dashed)
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = "#F59E0B";
    ctx.lineWidth = 1.25;
    ctx.setLineDash([3, 3]);
    for (let i = 0; i < uGT.length; i++) {
      const x = getX(uGT[i]);
      const y = getY(frGT[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // SeismoFNO Predicted Hysteresis (Cyan solid)
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = "#00D2FF";
    ctx.lineWidth = 1.5;
    for (let i = 0; i < uPred.length; i++) {
      const x = getX(uPred[i]);
      const y = getY(frPred[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // Synchronized Cursor Dot
    if (hoverTimeIdx !== null && hoverTimeIdx < uGT.length) {
      const xGt = getX(uGT[hoverTimeIdx]);
      const yGt = getY(frGT[hoverTimeIdx]);
      const xFno = getX(uPred[hoverTimeIdx]);
      const yFno = getY(frPred[hoverTimeIdx]);

      ctx.fillStyle = "#F59E0B";
      ctx.beginPath();
      ctx.arc(xGt, yGt, 3.5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "#00D2FF";
      ctx.beginPath();
      ctx.arc(xFno, yFno, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [data, uy, hoverTimeIdx, resizeTrigger]);

  return (
    <div className="select-none flex flex-col font-mono text-xs">
      <div className="px-6 py-2.5 border-b border-border-subtle bg-background-base flex items-center justify-between">
        <span className="text-text-primary uppercase tracking-wider font-semibold">
          FORCE–DISPLACEMENT F<sub>R</sub>(u)
        </span>
        {data && (
          <span className="text-[11px] text-text-muted font-mono-num">
            FORCE ERROR: {formatNumber(data.metrics.err_fr_rel_l2, 1)}%
          </span>
        )}
      </div>

      <div ref={containerRef} className="relative w-full h-[220px] bg-background-deep overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
};
