import React from "react";
import type { SystemHealth, SimulationResponse } from "../../types/agent";
import { Cpu, FlaskConical, ShieldCheck, Zap, Activity, AlertTriangle, CheckCircle2, Info } from "lucide-react";

interface JudgeModeViewProps {
  health: SystemHealth | null;
  simulationData: SimulationResponse | null;
  onRunHeroDemo: () => void;
  onRunOODDemo: () => void;
  isLoading: boolean;
}

const Tile: React.FC<{
  icon: React.ReactNode;
  label: string;
  title: string;
  lines: string[];
  accent?: string;
  status?: "ok" | "warn" | "offline";
}> = ({ icon, label, title, lines, accent = "border-border-subtle", status }) => (
  <div className={`bg-panel-base border ${accent} rounded-lg p-4 flex flex-col space-y-2`}>
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-2">
        <div className="text-fno">{icon}</div>
        <span className="text-[10px] font-mono uppercase tracking-widest text-text-muted">{label}</span>
      </div>
      {status && (
        <span
          className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
            status === "ok"
              ? "bg-energy/10 text-energy border border-energy/20"
              : status === "warn"
              ? "bg-fno/10 text-fno border border-fno/20"
              : "bg-text-muted/10 text-text-muted border border-border-subtle"
          }`}
        >
          {status === "ok" ? "ACTIVE" : status === "warn" ? "CONFIGURED" : "OFFLINE"}
        </span>
      )}
    </div>
    <div className="text-sm font-bold text-text-primary">{title}</div>
    <div className="space-y-0.5">
      {lines.map((l, i) => (
        <div key={i} className="text-xs text-text-secondary font-mono leading-relaxed">
          {l}
        </div>
      ))}
    </div>
  </div>
);

const MetricCard: React.FC<{
  value: string;
  label: string;
  sublabel: string;
  type: "measured" | "model" | "physics" | "derived" | "unknown";
  highlight?: boolean;
}> = ({ value, label, sublabel, type, highlight }) => {
  const colors: Record<string, string> = {
    measured: "text-energy border-energy/30 bg-energy/5",
    model: "text-fno border-fno/30 bg-fno/5",
    physics: "text-opensees border-opensees/30 bg-opensees/5",
    derived: "text-text-primary border-border-subtle bg-panel-base",
    unknown: "text-text-muted border-border-subtle bg-panel-base",
  };
  const badges: Record<string, string> = {
    measured: "MEASURED",
    model: "MODEL PREDICTION",
    physics: "PHYSICS REFERENCE",
    derived: "DERIVED",
    unknown: "NOT QUANTIFIED",
  };
  return (
    <div className={`border rounded-lg p-3 flex flex-col space-y-1 ${highlight ? colors[type] : "border-border-subtle bg-panel-base"}`}>
      <div className={`text-xl font-bold font-mono ${type === "model" ? "text-fno" : type === "physics" ? "text-opensees" : type === "measured" ? "text-energy" : "text-text-primary"}`}>
        {value}
      </div>
      <div className="text-[11px] font-semibold text-text-primary">{label}</div>
      <div className="text-[9px] font-mono text-text-muted">{sublabel}</div>
      <div className={`text-[8px] font-mono mt-1 px-1.5 py-0.5 rounded w-fit ${
        type === "model" ? "bg-fno/10 text-fno"
        : type === "physics" ? "bg-opensees/10 text-opensees"
        : type === "measured" ? "bg-energy/10 text-energy"
        : type === "derived" ? "bg-panel-hover text-text-muted"
        : "bg-danger/10 text-danger"
      }`}>
        {badges[type]}
      </div>
    </div>
  );
};

export const JudgeModeView: React.FC<JudgeModeViewProps> = ({
  health,
  simulationData,
  onRunHeroDemo,
  onRunOODDemo,
  isLoading,
}) => {
  const m = simulationData?.metrics;
  const ood = simulationData?.ood;
  const nebiusOk = health?.nebius_configured ?? false;

  const agentFlow = [
    { step: "USER", label: "Engineering question" },
    { step: "SEISMOAGENT API", label: "FastAPI gateway" },
    { step: "NVIDIA NEMOTRON", label: "ReAct reasoning & tool planning" },
    { step: "NEBIUS", label: "Token Factory inference" },
    { step: "TOOL: earthquake_analysis", label: "Signal metrics from record" },
    { step: "TOOL: fno_inference", label: "SeismoFNO forward pass (~11ms)" },
    { step: "TOOL: physics_simulation", label: "OpenSeesPy NLTHA (~54ms)" },
    { step: "TOOL: compare_fno_vs_physics", label: "Discrepancy & speedup" },
    { step: "TOOL: assess_ood", label: "Domain validity & trust" },
    { step: "NEMOTRON SYNTHESIS", label: "Evidence-backed explanation" },
    { step: "ENGINEERING REPORT", label: "Traceable audit document" },
  ];

  return (
    <div className="flex-1 overflow-y-auto bg-background-base p-6 space-y-8 font-mono">
      {/* Hero Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center space-x-3">
          <span className="w-3 h-3 rounded-sm bg-fno shadow-[0_0_12px_rgba(0,210,255,0.7)]"></span>
          <h1 className="text-2xl font-bold tracking-widest text-text-primary uppercase">SEISMOAGENT</h1>
        </div>
        <p className="text-sm text-text-secondary max-w-xl mx-auto leading-relaxed">
          Autonomous AI for Physics-Verified Seismic Engineering Analysis
        </p>
        <p className="text-xs text-text-muted max-w-2xl mx-auto">
          Ask natural-language engineering questions → Nemotron selects deterministic tools →
          FNO predicts → OpenSeesPy verifies → evidence is traceable and honest.
        </p>
      </div>

      {/* One-click hero demos */}
      <div className="flex items-center justify-center space-x-4">
        <button
          onClick={onRunHeroDemo}
          disabled={isLoading}
          className="px-6 py-2.5 rounded-lg bg-fno/10 border border-fno/40 text-fno hover:bg-fno hover:text-background-deep text-sm font-bold transition disabled:opacity-50 cursor-pointer"
        >
          {isLoading ? "⟳ Running..." : "▶ HERO EXPERIMENT  (In-Domain)"}
        </button>
        <button
          onClick={onRunOODDemo}
          disabled={isLoading}
          className="px-6 py-2.5 rounded-lg bg-danger/10 border border-danger/40 text-danger hover:bg-danger hover:text-background-deep text-sm font-bold transition disabled:opacity-50 cursor-pointer"
        >
          ▲ OOD STRESS TEST
        </button>
      </div>

      {/* 4-column product story tiles */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <Tile
          icon={<Info size={16} />}
          label="WHAT"
          title="Autonomous Engineering Agent"
          lines={[
            "Natural-language interface over",
            "deterministic scientific tools.",
            "Not a chatbot — an agent that",
            "orchestrates real computation.",
          ]}
          accent="border-fno/30"
        />
        <Tile
          icon={<Cpu size={16} />}
          label="HOW"
          title="Nemotron + Physics Tools"
          lines={[
            "NVIDIA Nemotron (340B Instruct)",
            "via Nebius Token Factory",
            "selects & sequences 8 tools.",
            "ReAct loop, bounded at 10 calls.",
          ]}
          accent="border-fno/30"
          status={nebiusOk ? "ok" : "offline"}
        />
        <Tile
          icon={<FlaskConical size={16} />}
          label="PROOF"
          title="FNO vs OpenSeesPy"
          lines={[
            "SeismoFNO: Fourier Neural Operator",
            "Physics ref: OpenSeesPy NLTHA",
            "Every result cross-validated.",
            "Discrepancy disclosed honestly.",
          ]}
          accent="border-opensees/30"
          status="ok"
        />
        <Tile
          icon={<ShieldCheck size={16} />}
          label="SAFETY"
          title="OOD + Trust Layer"
          lines={[
            "Domain bounds enforced at runtime.",
            "OOD violations flagged explicitly.",
            "Uncertainty: NOT QUANTIFIED",
            "No false confidence given.",
          ]}
          accent="border-energy/30"
          status="ok"
        />
      </div>

      {/* Infrastructure row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Tile
          icon={<Cpu size={15} />}
          label="AI ORCHESTRATION"
          title="NVIDIA Nemotron-4-340B"
          lines={[
            `Model: nvidia/nemotron-4-340b-instruct`,
            `Provider: Nebius Token Factory`,
            `Protocol: OpenAI-compatible`,
            `Status: ${nebiusOk ? "Configured" : "NEBIUS_API_KEY not set (mock mode)"}`,
          ]}
          accent={nebiusOk ? "border-energy/30" : "border-border-subtle"}
          status={nebiusOk ? "ok" : "offline"}
        />
        <Tile
          icon={<Activity size={15} />}
          label="LEARNED SURROGATE"
          title="SeismoFNO (FNO)"
          lines={[
            "Architecture: Fourier Neural Operator",
            "Device: " + (health?.device?.toUpperCase() || "MPS"),
            "Inference: ~4–11ms per record",
            "Training: Bilinear SDOF dataset",
          ]}
          accent="border-fno/30"
          status="ok"
        />
        <Tile
          icon={<Zap size={15} />}
          label="PHYSICS REFERENCE"
          title="OpenSeesPy NLTHA"
          lines={[
            "Solver: Newmark-β, Newton-Raphson",
            "Physics: ~5–54ms per record",
            "Concurrency: serialized (threading.Lock)",
            "Verified ground-truth reference.",
          ]}
          accent="border-opensees/30"
          status="ok"
        />
      </div>

      {/* Live results (if hero experiment ran) */}
      {simulationData && m ? (
        <div className="space-y-4">
          <div className="flex items-center space-x-2 text-xs uppercase tracking-widest text-text-muted">
            <CheckCircle2 size={14} className="text-energy" />
            <span>LIVE EXPERIMENT RESULTS — ALL VALUES COMPUTED DETERMINISTICALLY</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
            <MetricCard
              value={`${(m.u_max_pred_m * 1000).toFixed(1)} mm`}
              label="FNO Peak Disp."
              sublabel="SeismoFNO forward pass"
              type="model"
              highlight
            />
            <MetricCard
              value={`${(m.u_max_gt_m * 1000).toFixed(1)} mm`}
              label="Physics Peak Disp."
              sublabel="OpenSeesPy NLTHA"
              type="physics"
              highlight
            />
            <MetricCard
              value={`${m.err_u_rel_l2.toFixed(1)}%`}
              label="FNO–Physics Error"
              sublabel="Relative L2 norm (disclosed)"
              type="derived"
            />
            <MetricCard
              value={`${m.t_fno_ms.toFixed(1)} ms`}
              label="FNO Runtime"
              sublabel="time.perf_counter() measured"
              type="measured"
            />
            <MetricCard
              value={`${m.t_gt_ms.toFixed(1)} ms`}
              label="Physics Runtime"
              sublabel="time.perf_counter() measured"
              type="measured"
            />
            <MetricCard
              value={ood?.is_ood ? "⚠ OOD" : "✓ IN-DOMAIN"}
              label="Domain Status"
              sublabel={ood?.is_ood ? `${ood.domain_violations?.length || 0} violations` : "All bounds satisfied"}
              type={ood?.is_ood ? "unknown" : "measured"}
            />
          </div>

          {/* Error context banner */}
          <div className="bg-fno/5 border border-fno/20 rounded-lg p-4 text-xs">
            <div className="flex items-start space-x-2">
              <Info size={14} className="text-fno mt-0.5 shrink-0" />
              <div className="space-y-1">
                <div className="font-bold text-fno">Why is FNO–Physics error {m.err_u_rel_l2.toFixed(1)}%?</div>
                <div className="text-text-secondary leading-relaxed">
                  This is the <strong>scientific finding</strong>, not a product failure. Nonlinear hysteretic displacement prediction
                  is the research problem SeismoFNO is designed to solve. The system shows you the discrepancy precisely
                  so you can decide whether to trust the surrogate. When error is high, SeismoAgent explicitly recommends
                  physics verification and refuses to express false confidence.
                </div>
              </div>
            </div>
          </div>

          {/* OOD warning if triggered */}
          {ood?.is_ood && (
            <div className="bg-danger/5 border border-danger/30 rounded-lg p-4 text-xs">
              <div className="flex items-start space-x-2">
                <AlertTriangle size={14} className="text-danger mt-0.5 shrink-0" />
                <div className="space-y-1">
                  <div className="font-bold text-danger">⚠ OUT-OF-DISTRIBUTION INPUT DETECTED</div>
                  <div className="text-text-secondary">{ood.domain_violations?.join(" · ")}</div>
                  <div className="text-text-muted">
                    FNO reliability: NOT ESTABLISHED · Uncertainty: NOT QUANTIFIED ·
                    Recommended: Run OpenSeesPy physics verification before engineering reliance.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12 text-text-muted text-xs">
          Click <span className="text-fno font-bold">▶ HERO EXPERIMENT</span> to run a full physics-verified analysis.
        </div>
      )}

      {/* Agent workflow trace */}
      <div className="space-y-2">
        <div className="text-[10px] uppercase tracking-widest text-text-muted">AGENTIC WORKFLOW — NEMOTRON ORCHESTRATION CHAIN</div>
        <div className="grid grid-cols-1 xl:grid-cols-11 gap-1 items-center">
          {agentFlow.map((node, idx) => (
            <React.Fragment key={idx}>
              <div className="bg-panel-base border border-border-subtle rounded p-2 text-center">
                <div className="text-[9px] font-bold text-fno leading-tight">{node.step}</div>
                <div className="text-[8px] text-text-muted leading-tight mt-0.5">{node.label}</div>
              </div>
              {idx < agentFlow.length - 1 && (
                <div className="hidden xl:flex items-center justify-center text-text-muted text-xs">→</div>
              )}
            </React.Fragment>
          ))}
        </div>
        {!nebiusOk && (
          <div className="text-[10px] text-text-muted text-center">
            ⚠ NEBIUS_API_KEY not set — Nemotron step uses rule-based mock fallback. All tool steps are live.
          </div>
        )}
      </div>
    </div>
  );
};
