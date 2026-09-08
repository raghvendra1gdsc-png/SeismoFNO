"""
src/demo/metrics_adapter.py — Scientific Metrics & Analysis Provider for Demo Layer.

Serves verified research metrics, experimental progression stages, out-of-distribution
comparison tables, falsification ablations, and failure analysis.
"""

from typing import Dict, Any, List


class MetricsAdapter:
    """Provides authoritative, verified empirical metrics from EXP4, EXP5, and EXP6."""

    @staticmethod
    def get_research_progression() -> List[Dict[str, Any]]:
        """The core four-step scientific research progression."""
        return [
            {
                "step": 1,
                "phase": "EXP4",
                "title": "Fixed-Grid Fourier Neural Operator",
                "representation": "Fixed 5x2048 2D Tensor (Zero-Padded Stories 4-5)",
                "result_highlight": "3-Story Rel L2 = 99.60%",
                "status": "TOPOLOGY BOUNDARY FAILURE",
                "finding": "Standard FNO assumes a regular Euclidean lattice. Zero-padding smaller structures forces non-physical spatial step discontinuities, causing high-frequency Gibbs ringing that destroys physical response prediction.",
                "color": "rose",
            },
            {
                "step": 2,
                "phase": "EXP5",
                "title": "Spatiotemporal Graph Neural Operator",
                "representation": "Topology-Native Graph G=(V, E) (0 Padding Nodes)",
                "result_highlight": "3-Story Rel L2 = 22.09% (77.51 pp drop)",
                "status": "TOPOLOGY LIMITATION RESOLVED",
                "finding": "Treating building floors as graph nodes and columns as edges eliminates artificial boundary padding. 3-story relative error drops from 99.60% to 22.09% (77.82% relative reduction).",
                "color": "emerald",
            },
            {
                "step": 3,
                "phase": "EXP6",
                "title": "Physics/Modal-Conditioned GNO",
                "representation": "Native Graph + Dual-Branch FiLM on [T1-3, omega1-3]",
                "result_highlight": "OOD-B Peak Error: 35.21% -> 13.06% (62.9% reduction)",
                "status": "MODAL ENVELOPE GENERALIZATION",
                "finding": "Conditioning spatial message-passing and temporal spectral kernels on pre-earthquake modal eigenvalue invariants rescales the response envelope, yielding a 62.9% relative reduction in peak displacement error on unseen flexible structure 5S_T120.",
                "color": "indigo",
            },
            {
                "step": 4,
                "phase": "ANALYSIS",
                "title": "Documented Scientific Boundary",
                "representation": "Global 1D Fourier Kernel Over Long Horizons (20.48s)",
                "result_highlight": "OOD-B Waveform Rel L2 > 100%, Pearson r ~ 0.05-0.09",
                "status": "PHASE EXTRAPOLATION LIMITATION",
                "finding": "While peak displacement envelopes are accurately bounded, trajectory Relative L2 remains elevated due to cumulative phase drift in static 1D Fourier bases when vibration periods extrapolate far outside training support.",
                "color": "amber",
            },
        ]

    @staticmethod
    def get_ood_matrix() -> Dict[str, Any]:
        """Verified Out-of-Distribution evaluation matrix across partitions."""
        partitions = ["ID (120)", "OOD-A (300)", "OOD-B (240)", "OOD-C (60)", "5S_T105 (60)", "5S_T140 (60)"]
        
        peak_errors = {
            "EXP5 Baseline GNO": [12.70, 15.86, 35.21, 38.66, 17.97, 48.69],
            "EXP6-B T1-GNO": [2.09, 8.81, 13.47, 14.36, 10.21, 41.84],
            "EXP6-C Multi-Modal GNO": [8.87, 7.63, 13.06, 17.01, 10.79, 31.72],
            "EXP6-D Shuffled Modal (Ablation)": [2.65, 8.94, 24.33, 36.16, 14.93, 32.62],
        }

        rel_l2_errors = {
            "EXP5 Baseline GNO": [22.09, 29.26, 115.70, 116.16, 109.12, 111.96],
            "EXP6-B T1-GNO": [5.33, 19.59, 157.64, 145.16, 123.67, 173.43],
            "EXP6-C Multi-Modal GNO": [12.57, 19.39, 124.07, 120.49, 111.26, 124.99],
            "EXP6-D Shuffled Modal (Ablation)": [6.02, 17.95, 119.54, 114.88, 109.18, 119.71],
        }

        return {
            "partitions": partitions,
            "peak_disp_error_pct": peak_errors,
            "rel_l2_u_pct": rel_l2_errors,
            "key_finding": "Multi-Modal GNO reduces peak displacement error on unseen flexible structure 5S_T120 from 35.21% down to 13.06% (62.9% relative reduction).",
        }

    @staticmethod
    def get_ablation_falsification() -> Dict[str, Any]:
        """Ablation D falsification test results and conservative scientific interpretation."""
        return {
            "hypothesis": "Does modal conditioning improve generalization via true physical correspondence, or merely by providing auxiliary scalar MLP parameter capacity?",
            "protocol": "Randomly permuting the conditioning vector across batch instances during training and evaluation breaks physical correspondence while preserving parameter capacity.",
            "metrics": [
                {
                    "partition": "OOD-B (Unseen 5S_T120, T1=1.20s)",
                    "true_multimodal_err": 13.06,
                    "shuffled_err": 24.33,
                    "delta_percentage_points": 11.27,
                    "degradation_pct": 86.3,
                },
                {
                    "partition": "OOD-C (Combined Structure & Earthquake OOD)",
                    "true_multimodal_err": 17.01,
                    "shuffled_err": 36.16,
                    "delta_percentage_points": 19.15,
                    "degradation_pct": 112.6,
                },
                {
                    "partition": "Progressive 5S_T105 (Moderate OOD, T1=1.05s)",
                    "true_multimodal_err": 10.79,
                    "shuffled_err": 14.93,
                    "delta_percentage_points": 4.14,
                    "degradation_pct": 38.4,
                },
            ],
            "interpretation": "The systematic degradation under shuffled conditioning is consistent with the hypothesis that performance gains depend on meaningful structural-modal correspondence rather than merely additional conditioning capacity.",
        }

    @staticmethod
    def get_failure_analysis() -> List[Dict[str, Any]]:
        """Four first-class documented scientific failure modes."""
        return [
            {
                "id": "failure_1",
                "title": "Fixed-Grid Spatial Discretization Failure",
                "phase": "EXP4",
                "symptom": "3-story frame Relative L2 displacement error = 99.60%",
                "mechanism": "Enforcing variable-floor buildings onto a uniform Cartesian grid requires zero-padding stories 4 and 5. The spatial discrete Fourier transform assumes periodic boundary conditions; forcing a sharp jump to zero creates non-physical Gibbs ringing across physical lower floors.",
                "resolution": "Resolved in EXP5 by adopting a topology-native graph representation with 0 zero-padding nodes (cutting 3-story error to 22.09%).",
            },
            {
                "id": "failure_2",
                "title": "Modal Extrapolation Waveform Phase Drift",
                "phase": "EXP5 & EXP6",
                "symptom": "OOD-B trajectory Relative L2 > 100%, roof Pearson r ~ 0.05-0.09",
                "mechanism": "Global 1D Fourier layers compute static frequency convolutions over the entire 20.48 s duration. When the fundamental natural period shifts from training (T1 <= 0.85 s) to OOD-B (T1 = 1.20 s), minor oscillation frequency discrepancies integrate over 1,024 steps, producing cumulative phase opposition (Rel L2 approx sqrt(2) = 141%).",
                "resolution": "Partially addressed: EXP6 scales the peak displacement envelope within 13.06%, but phase drift remains an inherent limitation of static Fourier bases.",
            },
            {
                "id": "failure_3",
                "title": "Dynamic Period Elongation During Severe Yielding",
                "phase": "EXP6",
                "symptom": "Degraded tracking under intense non-linear plastic excursions (PGA >= 0.8g, ductility mu > 4.0)",
                "mechanism": "Modal conditioning vectors [T1, omega1] are derived from initial undamped elastic matrices [K],[M]. When structural members undergo significant plastic hinging, the effective stiffness drops and vibration period lengthens dynamically. Static pre-earthquake invariants cannot track this time-varying stiffness degradation.",
                "resolution": "Open research direction: continuous-time Neural ODEs or dynamic state-space recurrence.",
            },
            {
                "id": "failure_4",
                "title": "Planar 2D Shear Frame Idealization",
                "phase": "EXP4-EXP6",
                "symptom": "Surrogate cannot yet model multi-directional torsional shaking in irregular buildings",
                "mechanism": "Present models assume planar horizontal shear building dynamics with lumped floor masses, ignoring diaphragm flexibility, P-Delta geometric non-linearities, and bidirectional torsional coupling.",
                "resolution": "Open research direction: 3D spatial graph extensions with 6 DOFs per node.",
            },
        ]

    @staticmethod
    def get_computational_benchmark() -> Dict[str, Any]:
        """Verified Apple Silicon MPS latency and wall-clock speedup benchmarks."""
        return {
            "hardware": "Apple Silicon M-series GPU (MPS backend, unified memory)",
            "benchmark_protocol": "Single-building transient simulation (T = 20.48s, 1024 steps) with pre-warmup iterations and torch.mps.synchronize()",
            "models": [
                {
                    "name": "OpenSeesPy 5-Story NLTHA (Ground Truth)",
                    "batch_size": 1,
                    "latency_ms": 54.68,
                    "throughput_sim_s": 18.29,
                    "speedup": 1.00,
                    "badge": "NUMERICAL GROUND TRUTH",
                },
                {
                    "name": "EXP4 FNO2D (Frozen Baseline)",
                    "batch_size": 1,
                    "latency_ms": 6.60,
                    "throughput_sim_s": 151.52,
                    "speedup": 8.28,
                    "badge": "FASTEST (TOPOLOGY RESTRICTED)",
                },
                {
                    "name": "EXP5 Spatiotemporal GNO (Single)",
                    "batch_size": 1,
                    "latency_ms": 16.25,
                    "throughput_sim_s": 61.52,
                    "speedup": 3.36,
                    "badge": "TOPOLOGY-NATIVE",
                },
                {
                    "name": "EXP6 T1-GNO (Single Inference)",
                    "batch_size": 1,
                    "latency_ms": 21.45,
                    "throughput_sim_s": 46.61,
                    "speedup": 2.55,
                    "badge": "PHYSICS-CONDITIONED",
                },
                {
                    "name": "EXP6 T1-GNO (Batched B=32)",
                    "batch_size": 32,
                    "latency_ms": 30.19,
                    "throughput_sim_s": 33.13,
                    "speedup": 1.81,
                    "badge": "BATCHED SERVING",
                },
            ],
            "conclusion": "EXP6 T1-GNO achieves a 2.55x wall-clock speedup over OpenSeesPy while providing topology-native flexibility and physics-informed peak response scaling.",
        }


# Convenience function aliases
get_research_progression = MetricsAdapter.get_research_progression
get_ood_matrix = MetricsAdapter.get_ood_matrix
get_ablation_falsification = MetricsAdapter.get_ablation_falsification
get_failure_analysis = MetricsAdapter.get_failure_analysis
get_computational_benchmark = MetricsAdapter.get_computational_benchmark
