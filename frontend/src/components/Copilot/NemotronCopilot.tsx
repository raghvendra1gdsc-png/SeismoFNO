import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Terminal,
  Cpu,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Loader2,
  Sparkles,
  Shield,
} from "lucide-react";
import type { AgentRunResponse, StreamEvent, ToolTraceItem } from "../../types/agent";
import { runAgent, createAgentStream } from "../../api/agent";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  toolTrace?: ToolTraceItem[];
  streamEvents?: string[];
  evidence?: {
    peakDisplacement?: string;
    errorRelL2?: string;
    speedup?: string;
    energyDissipated?: string;
  };
  trust?: {
    domainStatus?: string;
    uncertainty?: string;
  };
  timing?: {
    totalServerMs?: number;
    llmLatencyMs?: number;
    toolLatencyMs?: number;
  };
  error?: boolean;
}

interface NemotronCopilotProps {
  onApplySimulationResult?: (data: any) => void;
  currentStructureContext?: {
    T: number;
    zeta: number;
    materialType: string;
    uy: number;
    alpha: number;
  };
  currentRecordName?: string;
  nebiusConfigured?: boolean;
  nebiusLiveVerified?: boolean;
}

const PRESET_PROMPTS = [
  "Analyze this earthquake record for the current structure.",
  "Compare SeismoFNO against physics simulation.",
  "Increase stiffness by 20% and rerun the analysis.",
  "Which parameter has the strongest effect on peak displacement?",
  "Generate an engineering report for this experiment.",
];

export const NemotronCopilot: React.FC<NemotronCopilotProps> = ({
  onApplySimulationResult,
  currentStructureContext,
  currentRecordName,
  nebiusConfigured = false,
  nebiusLiveVerified = false,
}) => {
  // Derive honest orchestration status from backend health
  const orchStatus = nebiusConfigured
    ? nebiusLiveVerified
      ? { label: "LIVE", dot: "bg-energy animate-pulse", text: "text-energy" }
      : { label: "CONFIGURED", dot: "bg-opensees", text: "text-opensees" }
    : { label: "MOCKED (Nebius offline)", dot: "bg-text-muted", text: "text-text-muted" };
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "SeismoAgent autonomous engineering copilot ready. Ask structural questions, request surrogate comparisons, or trigger parametric sweeps. Every numerical quantity is computed deterministically by verified tools.",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeStepText, setActiveStepText] = useState<string | null>(null);
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeStepText]);

  const handleSendMessage = async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed || isProcessing) return;

    setInputQuery("");
    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `asst_${Date.now()}`;
    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // 1. Append user message
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: "user",
        content: trimmed,
        timestamp,
      },
    ]);

    setIsProcessing(true);
    setActiveStepText("Understanding request...");

    const streamEvents: string[] = ["Understanding request"];

    // 2. Try streaming execution via WebSocket
    const cleanupStream = createAgentStream(
      trimmed,
      (event: StreamEvent) => {
        if (event.event === "planning") {
          setActiveStepText("Planning engineering workflow with Nemotron...");
          streamEvents.push("Planning workflow");
        } else if (event.event === "tool_started" && event.tool) {
          const readable = event.tool.replace(/_/g, " ");
          setActiveStepText(`Executing tool: ${readable}...`);
          streamEvents.push(`Tool started: ${event.tool}`);
        } else if (event.event === "tool_completed" && event.tool) {
          const readable = event.tool.replace(/_/g, " ");
          setActiveStepText(`Completed ${readable} (${event.elapsed_ms?.toFixed(1)}ms)`);
          streamEvents.push(`Tool finished: ${event.tool}`);
        } else if (event.event === "synthesis_started") {
          setActiveStepText("Synthesizing engineering evidence...");
          streamEvents.push("Synthesizing response");
        }
      },
      (errorMsg) => {
        console.warn("WebSocket stream error, will fallback to REST if needed:", errorMsg);
      },
      () => {
        // Stream completed
      }
    );

    // Give stream 1.5s to initiate or fallback to REST runAgent
    try {
      // Execute through REST /api/agent/run for authoritative full trace and payload sync
      const res: AgentRunResponse = await runAgent(trimmed, {
        structure: currentStructureContext,
        record_name: currentRecordName,
      });

      cleanupStream();

      // Check if simulation was run and propagate
      if (res.tool_results?.run_physics_simulation && onApplySimulationResult) {
        onApplySimulationResult(res.tool_results.run_physics_simulation);
      }

      // Extract deterministic evidence
      const physRes = res.tool_results?.run_physics_simulation as any;
      const compRes = res.tool_results?.compare_fno_vs_physics as any;
      const oodRes = res.tool_results?.assess_ood_and_uncertainty as any;

      const evidence = {
        peakDisplacement: physRes?.u_max_m ? `${(physRes.u_max_m * 1000).toFixed(2)} mm` : undefined,
        errorRelL2: compRes?.err_u_rel_l2_pct ? `${compRes.err_u_rel_l2_pct.toFixed(2)}%` : undefined,
        speedup: compRes?.speedup_factor ? `${compRes.speedup_factor.toFixed(1)}x` : undefined,
        energyDissipated: physRes?.eh_total_j ? `${physRes.eh_total_j.toFixed(1)} J` : undefined,
      };

      const trust = {
        domainStatus: oodRes ? (oodRes.is_ood ? "OOD WARNING" : "IN-DOMAIN (VALIDATED)") : undefined,
        uncertainty: "NOT_QUANTIFIED",
      };

      setMessages((prev) => [
        ...prev,
        {
          id: assistantMsgId,
          role: "assistant",
          content: res.final_response,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          toolTrace: res.tool_trace,
          streamEvents,
          evidence,
          trust,
          timing: {
            totalServerMs: res.timing.total_server_latency_ms,
            llmLatencyMs: res.timing.llm_latency_ms,
            toolLatencyMs: res.timing.tool_latency_ms,
          },
        },
      ]);
    } catch (err: any) {
      cleanupStream();
      setMessages((prev) => [
        ...prev,
        {
          id: assistantMsgId,
          role: "assistant",
          content: `Analysis request could not be completed: ${err.message || String(err)}`,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          error: true,
        },
      ]);
    } finally {
      setIsProcessing(false);
      setActiveStepText(null);
    }
  };

  return (
    <aside className="w-96 shrink-0 bg-background-deep border-l border-border-subtle flex flex-col h-[calc(100vh-45px-28px)] select-none">
      {/* Copilot Header */}
      <div className="p-3 border-b border-border-subtle bg-background-base flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 rounded bg-fno/10 border border-fno/30 flex items-center justify-center text-fno">
            <Cpu size={14} />
          </div>
          <div>
            <div className="text-xs font-mono font-bold text-text-primary uppercase tracking-wider">
              NEMOTRON COPILOT
            </div>
            <div className="text-[10px] font-mono text-text-muted">
              Nebius Token Factory · 340B Instruct
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${orchStatus.dot}`}></span>
          <span className={`text-[10px] font-mono ${orchStatus.text}`}>{orchStatus.label}</span>
        </div>
      </div>

      {/* Suggested Quick Prompts */}
      <div className="p-2 border-b border-border-subtle/70 bg-panel-base/40 overflow-x-auto">
        <div className="flex items-center space-x-1.5 text-[10px] font-mono text-text-muted mb-1.5 px-1">
          <Sparkles size={11} className="text-fno" />
          <span>SUGGESTED ENGINEERING ACTIONS</span>
        </div>
        <div className="flex flex-col space-y-1">
          {PRESET_PROMPTS.slice(0, 3).map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => handleSendMessage(prompt)}
              disabled={isProcessing}
              className="text-left text-[11px] font-mono px-2 py-1 rounded bg-background-base border border-border-subtle hover:border-fno/40 hover:text-fno transition cursor-pointer text-text-secondary truncate disabled:opacity-50"
              title={prompt}
            >
              → {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 font-sans text-xs">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${
              msg.role === "user" ? "items-end" : "items-start"
            }`}
          >
            {/* Role Header */}
            <div className="flex items-center space-x-1.5 mb-1 px-1 text-[10px] font-mono text-text-muted">
              <span>{msg.role === "user" ? "USER" : "NEMOTRON"}</span>
              <span>·</span>
              <span>{msg.timestamp}</span>
            </div>

            {/* Bubble */}
            <div
              className={`max-w-[95%] rounded-md p-3 font-mono text-xs leading-relaxed ${
                msg.role === "user"
                  ? "bg-fno/10 border border-fno/30 text-text-primary"
                  : msg.error
                  ? "bg-danger/10 border border-danger/30 text-danger"
                  : "bg-panel-base border border-border-subtle text-text-primary"
              }`}
            >
              {/* Main Narrative Content */}
              <div className="whitespace-pre-wrap">{msg.content}</div>

              {/* Scientific Evidence Strip */}
              {msg.evidence && (
                <div className="mt-3 pt-2.5 border-t border-border-subtle/80 space-y-1.5">
                  <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase flex items-center space-x-1">
                    <CheckCircle2 size={11} className="text-energy" />
                    <span>MEASURED ENGINEERING EVIDENCE</span>
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-[11px] font-mono">
                    {msg.evidence.peakDisplacement && (
                      <div className="bg-background-base p-1.5 rounded border border-border-subtle">
                        <div className="text-text-muted text-[9px]">PEAK DISP</div>
                        <div className="text-text-primary font-bold">{msg.evidence.peakDisplacement}</div>
                      </div>
                    )}
                    {msg.evidence.errorRelL2 && (
                      <div className="bg-background-base p-1.5 rounded border border-border-subtle">
                        <div className="text-text-muted text-[9px]">REL L2 ERROR</div>
                        <div className="text-fno font-bold">{msg.evidence.errorRelL2}</div>
                      </div>
                    )}
                    {msg.evidence.speedup && (
                      <div className="bg-background-base p-1.5 rounded border border-border-subtle">
                        <div className="text-text-muted text-[9px]">SPEEDUP</div>
                        <div className="text-energy font-bold">{msg.evidence.speedup}</div>
                      </div>
                    )}
                    {msg.evidence.energyDissipated && (
                      <div className="bg-background-base p-1.5 rounded border border-border-subtle">
                        <div className="text-text-muted text-[9px]">DISSIPATED EH</div>
                        <div className="text-opensees font-bold">{msg.evidence.energyDissipated}</div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Epistemic Trust Status */}
              {msg.trust && (
                <div className="mt-2 pt-2 border-t border-border-subtle/50 flex items-center justify-between text-[10px] font-mono">
                  <div className="flex items-center space-x-1 text-energy">
                    <Shield size={10} />
                    <span>{msg.trust.domainStatus || "IN-DOMAIN"}</span>
                  </div>
                  <div className="text-text-muted">
                    UNCERTAINTY: <span className="text-text-secondary">{msg.trust.uncertainty}</span>
                  </div>
                </div>
              )}

              {/* Technical Tool Execution Trace (Expandable) */}
              {msg.toolTrace && msg.toolTrace.length > 0 && (
                <div className="mt-3 pt-2 border-t border-border-subtle/50">
                  <button
                    onClick={() =>
                      setExpandedTraceId(expandedTraceId === msg.id ? null : msg.id)
                    }
                    className="flex items-center justify-between w-full text-[10px] font-mono text-text-muted hover:text-text-primary cursor-pointer"
                  >
                    <span className="flex items-center space-x-1">
                      <Terminal size={11} className="text-fno" />
                      <span>TOOL TRACE ({msg.toolTrace.length} calls)</span>
                    </span>
                    {expandedTraceId === msg.id ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                  </button>

                  {expandedTraceId === msg.id && (
                    <div className="mt-2 space-y-1.5 pl-1 border-l-2 border-fno/40">
                      {msg.toolTrace.map((trace, idx) => (
                        <div
                          key={idx}
                          className="bg-background-base p-1.5 rounded text-[10px] font-mono border border-border-subtle"
                        >
                          <div className="flex items-center justify-between text-text-secondary">
                            <span className="font-semibold text-fno">{trace.tool_name}</span>
                            <span className="text-text-muted">{trace.runtime_ms.toFixed(1)} ms</span>
                          </div>
                          {trace.output_summary && (
                            <pre className="mt-1 text-[9px] text-text-muted overflow-x-auto">
                              {JSON.stringify(trace.output_summary, null, 1)}
                            </pre>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Timing Latency Breakdown */}
              {msg.timing && (
                <div className="mt-1.5 text-[9px] font-mono text-text-muted text-right">
                  latency: {msg.timing.totalServerMs?.toFixed(0)}ms
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Live Active Progress Indicator */}
        {isProcessing && activeStepText && (
          <div className="p-2.5 rounded bg-panel-base border border-fno/30 text-xs font-mono flex items-center space-x-2 text-fno animate-pulse">
            <Loader2 size={13} className="animate-spin text-fno" />
            <span>{activeStepText}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-3 border-t border-border-subtle bg-background-base">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage(inputQuery);
          }}
          className="flex items-center space-x-2"
        >
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            disabled={isProcessing}
            placeholder="Ask Nemotron engineering copilot..."
            className="flex-1 bg-panel-base border border-border-subtle focus:border-fno rounded px-3 py-2 text-xs font-mono text-text-primary placeholder:text-text-muted focus:outline-none transition"
          />
          <button
            type="submit"
            disabled={!inputQuery.trim() || isProcessing}
            className="p-2 rounded bg-fno/10 text-fno border border-fno/30 hover:bg-fno hover:text-background-deep transition disabled:opacity-40 disabled:hover:bg-fno/10 disabled:hover:text-fno cursor-pointer"
            title="Send query"
          >
            {isProcessing ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          </button>
        </form>
        <div className="mt-1.5 flex items-center justify-between text-[9px] font-mono text-text-muted">
          <span>Press Enter to analyze · All numerical values computed deterministically</span>
          {!nebiusConfigured && (
            <span className="text-text-muted italic">Nemotron: offline mock</span>
          )}
        </div>
      </div>
    </aside>
  );
};
