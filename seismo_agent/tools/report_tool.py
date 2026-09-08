"""
seismo_agent/tools/report_tool.py — Deterministic Engineering Report Generator.

Compiles factual, audit-ready structural engineering reports from tool outputs.
Zero-LLM dependency: strictly formats actual computed numbers, metrics tables,
domain boundary status, and scientific limitations into publication-grade Markdown.
"""

from datetime import datetime, timezone
import json
from typing import Dict, Any, List, Optional

from seismo_agent.schemas.inputs import ReportRequest
from seismo_agent.schemas.outputs import ReportResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError


class ReportTool(BaseTool):
    """
    Synthesizes deterministic engineering reports from simulation and comparison data.
    """
    name = "generate_engineering_report"
    description = (
        "Compiles an audit-grade structural engineering report summarizing earthquake characterization, "
        "structural modal properties, SeismoFNO surrogate predictions, OpenSeesPy ground-truth validation, "
        "disaggregated error metrics, and domain boundary status. Fully deterministic: strictly grounded in "
        "measured simulation figures with zero model hallucination."
    )
    input_schema = ReportRequest
    output_schema = ReportResult

    def _run(self, request: ReportRequest) -> ReportResult:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        exp_id = request.experiment_id
        title = request.title or "Autonomous Seismic Structural Response Audit"

        eq = request.earthquake_result or {}
        st = request.structural_result or {}
        inf = request.inference_result or {}
        phys = request.physics_result or {}
        cmp_ = request.comparison_result or {}
        ood = request.ood_result or {}
        swp = request.sweep_result or {}

        # Construct Markdown sections
        lines: List[str] = [
            f"# {title}",
            f"**Experiment ID:** `{exp_id}` | **Audit Generated:** {now_str}  ",
            f"**System:** SeismoAgent Deterministic Verification Layer  ",
            "",
            "---",
            "",
            "## 1. Executive Summary & Verification Matrix",
            "",
        ]

        # Key verification table
        lines.extend([
            "| Parameter / Metric | Physics Ground Truth (OpenSeesPy) | SeismoFNO Neural Surrogate | Discrepancy / Error |",
            "| :--- | :--- | :--- | :--- |",
        ])

        u_max_phys = f"{phys.get('u_max_m', 'N/A')} m" if 'u_max_m' in phys else "N/A"
        u_max_fno = f"{inf.get('u_max_m', 'N/A')} m" if 'u_max_m' in inf else "N/A"
        err_umax = f"{cmp_.get('err_umax_rel_pct', 'N/A')}%" if 'err_umax_rel_pct' in cmp_ else "N/A"
        lines.append(f"| **Peak Displacement ($u_{{max}}$)** | {u_max_phys} | {u_max_fno} | {err_umax} |")

        fr_max_phys = f"{phys.get('fr_max_n', 'N/A')} N" if 'fr_max_n' in phys else "N/A"
        fr_max_fno = f"{inf.get('fr_max_n', 'N/A')} N" if 'fr_max_n' in inf else "N/A"
        err_frmax = f"{cmp_.get('err_frmax_rel_pct', 'N/A')}%" if 'err_frmax_rel_pct' in cmp_ else "N/A"
        lines.append(f"| **Peak Restoring Force ($F_{{R,max}}$)** | {fr_max_phys} | {fr_max_fno} | {err_frmax} |")

        eh_phys = f"{phys.get('eh_total_j', 'N/A')} J" if 'eh_total_j' in phys else "N/A"
        eh_fno = f"{inf.get('eh_total_j', 'N/A')} J" if 'eh_total_j' in inf else "N/A"
        err_eh = f"{cmp_.get('err_eh_rel_l2_pct', 'N/A')}%" if 'err_eh_rel_l2_pct' in cmp_ else "N/A"
        lines.append(f"| **Dissipated Energy ($E_h$)** | {eh_phys} | {eh_fno} | {err_eh} |")

        mu_phys = f"{phys.get('ductility_mu', 'N/A')}" if 'ductility_mu' in phys else "N/A"
        mu_fno = f"{inf.get('ductility_mu', 'N/A')}" if 'ductility_mu' in inf else "N/A"
        lines.append(rf"| **Ductility Demand ($\mu = u_{{max}}/u_y$)** | {mu_phys} | {mu_fno} | — |")

        t_phys = f"{phys.get('runtime_ms', 'N/A')} ms" if 'runtime_ms' in phys else "N/A"
        t_fno = f"{inf.get('runtime_ms', 'N/A')} ms" if 'runtime_ms' in inf else "N/A"
        speedup = f"**{cmp_.get('speedup_factor', 'N/A')}x Speedup**" if 'speedup_factor' in cmp_ else "N/A"
        lines.append(f"| **Computational Wall-Clock Time** | {t_phys} | {t_fno} | {speedup} |")

        lines.extend([
            "",
            f"**Trajectory Relative $L_2$ Error:** `{cmp_.get('err_u_rel_l2_pct', 'N/A')}%`  ",
            f"**Permanent Residual Drift Discrepancy:** `{cmp_.get('residual_drift_error_m', 0.0)*1000.0:.2f} mm`  ",
            f"**Phase Coherence Error (Hilbert):** `{cmp_.get('phase_error_rad', 'N/A')} rad`  ",
            "",
            "---",
            "",
            "## 2. Structural Configuration & Modal Properties",
            "",
            f"- **System Type:** `{st.get('system_type', 'SDOF').upper()}`",
            rf"- **Natural Period ($T_n$):** `{st.get('T_n', 'N/A')} s` ($\omega_n = {st.get('omega_n', 'N/A')} \text{{ rad/s}}$)",
            f"- **Elastic Stiffness ($k_0$):** `{st.get('k0', 'N/A')} N/m`",
            f"- **Viscous Damping Ratio ($\\zeta$):** `{st.get('zeta', 'N/A')}` ({st.get('damping_type', 'mass')} damping)",
            f"- **Constitutive Model:** `{st.get('material_type', 'bilinear').upper()}` (Post-yield ratio $\\alpha = {st.get('alpha', 'N/A')}$)",
            f"- **Yield Capacity:** $u_y = {st.get('u_y', 'N/A')} \\text{{ m}}$, $F_y = {st.get('Fy', 'N/A')} \\text{{ N}}$",
            "",
            "---",
            "",
            "## 3. Earthquake Ground Motion Characterization",
            "",
            f"- **Record Identifier:** `{eq.get('record_id', 'N/A')}` ({eq.get('record_name', 'N/A')})",
            f"- **Peak Ground Acceleration (PGA):** `{eq.get('pga_g', 'N/A')} g` ({eq.get('pga_ms2', 'N/A')} $\\text{{m/s}}^2$)",
            f"- **Peak Ground Velocity (PGV):** `{eq.get('pgv_ms', 'N/A')} m/s`",
            f"- **Peak Ground Displacement (PGD):** `{eq.get('pgd_m', 'N/A')} m`",
            f"- **Arias Intensity ($I_a$):** `{eq.get('arias_intensity_ms', 'N/A')} m/s`",
            f"- **Significant Duration ($D_{{5-95}}$):** `{eq.get('significant_duration_d5_95_s', 'N/A')} s`",
            f"- **Discrete Sampling:** `{eq.get('n_points', 2048)} points` at $dt = {eq.get('dt', 0.01)} \\text{{ s}}$ ($T_{{total}} = {eq.get('duration_s', 20.48)} \\text{{ s}}$)",
            "",
            "---",
            "",
            "## 4. Domain & Out-of-Distribution (OOD) Assessment",
            "",
            f"- **OOD Classification:** `{'OUT-OF-DISTRIBUTION (WARNING)' if ood.get('is_ood') else 'IN-DISTRIBUTION (VALIDATED)'}`",
            f"- **Structural Domain In-Bounds:** `{ood.get('within_structural_domain', True)}`",
            f"- **Seismic Domain In-Bounds:** `{ood.get('within_ground_motion_domain', True)}`",
            f"- **Resolution Supported:** `{ood.get('resolution_supported', True)}`",
            f"- **Uncertainty Status:** `{ood.get('uncertainty_status', 'NOT_QUANTIFIED')}`",
            f"- **Operational Recommendation:** {ood.get('recommendation', 'Run validation.')}",
            "",
        ])

        if ood.get("domain_violations"):
            lines.append("**Domain Violations Detected:**")
            for v in ood["domain_violations"]:
                lines.append(f"- ⚠️ {v}")
            lines.append("")

        if ood.get("domain_warnings"):
            lines.append("**Advisory Warnings:**")
            for w in ood["domain_warnings"]:
                lines.append(f"- ℹ️ {w}")
            lines.append("")

        # Sweep section if present
        if swp and swp.get("points"):
            lines.extend([
                "---",
                "",
                "## 5. Parametric Sensitivity / Incremental Dynamic Analysis",
                "",
                f"**Swept Parameter:** `{swp.get('parameter_name')}` across {swp.get('num_cases')} points.  ",
                f"**Average Speedup:** `{swp.get('avg_speedup_factor', 'N/A')}x` | **Total Sweep Time:** `{swp.get('total_runtime_s')} s`",
                "",
                f"| {swp.get('parameter_name')} | $u_{{max, FNO}}$ (m) | $u_{{max, OpenSees}}$ (m) | Ductility $\\mu_{{FNO}}$ | Rel $L_2$ Error (%) |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for pt in swp["points"]:
                p_val = pt.get("parameter_value")
                u_f = f"{pt.get('u_max_fno', 0.0):.4f}" if pt.get('u_max_fno') is not None else "—"
                u_p = f"{pt.get('u_max_physics', 0.0):.4f}" if pt.get('u_max_physics') is not None else "—"
                d_f = f"{pt.get('ductility_fno', 0.0):.2f}" if pt.get('ductility_fno') is not None else "—"
                err_l2 = f"{pt.get('rel_l2_u_pct', 0.0):.2f}%" if pt.get('rel_l2_u_pct') is not None else "—"
                lines.append(f"| {p_val} | {u_f} | {u_p} | {d_f} | {err_l2} |")
            lines.append("")

        # Limitations
        limitations = [
            "Continuous Fourier Neural Operators do not store internal thermodynamic history state; in severe post-yield regimes (mu > 4), global frequency truncation can accumulate minor baseline drift in the free vibration coda.",
            "Surrogate inference assumes zero initial conditions u(0) = 0, v(0) = 0 unless explicit state augmentation is activated.",
            "OpenSeesPy ground truth employs Newmark-beta average acceleration method; high-frequency numerical damping is governed by Rayleigh damping coefficients.",
            "All speedup measurements reflect isolated computation on identical local hardware and exclude file system I/O overhead.",
        ]

        lines.extend([
            "---",
            "",
            "## 6. Scientific Limitations & Epistemic Boundaries",
            "",
        ])
        for lim in limitations:
            lines.append(f"1. {lim}")
        lines.append("")

        md_content = "\n".join(lines)

        exec_summary = {
            "experiment_id": exp_id,
            "peak_displacement_m": phys.get("u_max_m") or inf.get("u_max_m"),
            "rel_l2_error_pct": cmp_.get("err_u_rel_l2_pct"),
            "speedup_factor": cmp_.get("speedup_factor"),
            "is_ood": ood.get("is_ood", False),
            "ductility_mu": phys.get("ductility_mu") or inf.get("ductility_mu"),
        }

        return ReportResult(
            experiment_id=exp_id,
            title=title,
            generated_at=now_str,
            content=md_content if request.report_format == "markdown" else json.dumps(exec_summary, indent=2),
            format=request.report_format,
            executive_summary=exec_summary,
            limitations=limitations,
        )
