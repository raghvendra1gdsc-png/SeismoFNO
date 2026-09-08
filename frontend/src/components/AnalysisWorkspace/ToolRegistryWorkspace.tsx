import React, { useEffect, useState } from "react";
import { Wrench, Shield, Check, Code2, Loader2 } from "lucide-react";
import { fetchTools } from "../../api/agent";
import type { ToolInfo } from "../../types/agent";

export const ToolRegistryWorkspace: React.FC = () => {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedTool, setSelectedTool] = useState<ToolInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetchTools();
        setTools(res.tools || []);
        if (res.tools?.length > 0) {
          setSelectedTool(res.tools[0]);
        }
      } catch (err: any) {
        setError(err.message || String(err));
      } finally {
        setIsLoading(false);
      }
    }
    load();
  }, []);

  if (isLoading) {
    return (
      <div className="p-8 text-center text-text-muted font-mono text-xs flex items-center justify-center space-x-2">
        <Loader2 size={16} className="animate-spin text-fno" />
        <span>Introspecting tool registry from SeismoAgent gateway...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-danger/10 border border-danger/30 rounded text-danger text-xs font-mono">
        Failed to load tool registry: {error}
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4 font-mono text-xs select-none">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center space-x-2">
          <Wrench size={16} className="text-fno" />
          <span className="font-bold text-text-primary uppercase tracking-wider text-sm">
            DETERMINISTIC ENGINEERING TOOL REGISTRY ({tools.length} REGISTERED)
          </span>
        </div>
        <div className="text-[11px] text-energy flex items-center space-x-1">
          <Shield size={12} />
          <span>ZERO-HALLUCINATION ENFORCEMENT</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Tool List Sidebar */}
        <div className="space-y-1 bg-panel-base border border-border-subtle rounded-md p-2">
          <div className="px-2 py-1 text-[10px] text-text-muted uppercase tracking-wider border-b border-border-subtle mb-1">
            Registered Engineering Tools
          </div>
          {tools.map((t) => (
            <button
              key={t.name}
              onClick={() => setSelectedTool(t)}
              className={`w-full text-left px-2.5 py-2 rounded text-xs transition cursor-pointer flex items-center justify-between ${
                selectedTool?.name === t.name
                  ? "bg-fno/10 text-fno font-bold border-l-2 border-fno"
                  : "text-text-secondary hover:text-text-primary hover:bg-panel-hover"
              }`}
            >
              <span className="truncate">{t.name}</span>
              {selectedTool?.name === t.name && <Check size={12} />}
            </button>
          ))}
        </div>

        {/* Tool Details & Schema Inspector */}
        <div className="md:col-span-2 bg-panel-base border border-border-subtle rounded-md p-4 space-y-3">
          {selectedTool ? (
            <>
              <div className="flex items-center justify-between border-b border-border-subtle pb-2">
                <div className="text-sm font-bold text-fno font-mono">{selectedTool.name}</div>
                <span className="text-[10px] bg-energy/10 text-energy px-2 py-0.5 rounded border border-energy/20">
                  VERIFIED DETERMINISTIC
                </span>
              </div>

              <div>
                <div className="text-[10px] uppercase text-text-muted mb-1">Operational Description</div>
                <p className="text-xs text-text-secondary leading-relaxed bg-background-base p-2.5 rounded border border-border-subtle">
                  {selectedTool.description}
                </p>
              </div>

              <div>
                <div className="text-[10px] uppercase text-text-muted mb-1 flex items-center space-x-1">
                  <Code2 size={12} />
                  <span>Input Parameter Schema (JSON Schema / Pydantic v2)</span>
                </div>
                <div className="bg-background-deep border border-border-subtle rounded p-3 overflow-x-auto max-h-72">
                  <pre className="text-[10px] text-text-primary leading-tight">
                    {JSON.stringify(selectedTool.parameters, null, 2)}
                  </pre>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center text-text-muted py-8">Select a tool to inspect.</div>
          )}
        </div>
      </div>
    </div>
  );
};
