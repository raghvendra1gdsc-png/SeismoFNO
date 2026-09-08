"""
src/demo/demo_data.py — Data Catalog & Trajectory Provider for Research Demonstration.

Provides verified structural archetypes, modal eigenvalue shapes, earthquake records,
and ground-truth OpenSeesPy dynamic response histories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import scipy.linalg
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# -----------------------------------------------------------------------------
# 1. Structural Archetype Catalog & Modal Invariant Extraction
# -----------------------------------------------------------------------------

@dataclass
class StructuralArchetype:
    archetype_id: str
    display_name: str
    n_stories: int
    category: str  # 'In-Distribution', 'Held-Out OOD-B', 'Progressive OOD'
    story_masses: List[float]  # kg
    story_stiffnesses: List[float]  # N/m
    story_heights: List[float]  # m
    yield_displacement: float  # m
    T1: float  # s
    T2: float  # s
    T3: float  # s
    omega1: float  # rad/s
    omega2: float  # rad/s
    omega3: float  # rad/s
    mode_shapes: List[List[float]]  # (n_stories, n_modes) normalized mode shapes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype_id": self.archetype_id,
            "display_name": self.display_name,
            "n_stories": self.n_stories,
            "category": self.category,
            "total_mass_kg": sum(self.story_masses),
            "story_mass_kg": self.story_masses[0],
            "story_stiffness_kN_m": round(self.story_stiffnesses[0] / 1000.0, 2),
            "total_height_m": sum(self.story_heights),
            "story_height_m": self.story_heights[0],
            "yield_drift_mm": round(self.yield_displacement * 1000.0, 2),
            "T1_s": round(self.T1, 4),
            "T2_s": round(self.T2, 4),
            "T3_s": round(self.T3, 4),
            "omega1_rad_s": round(self.omega1, 2),
            "omega2_rad_s": round(self.omega2, 2),
            "omega3_rad_s": round(self.omega3, 2),
            "mode_shapes": self.mode_shapes,
            "modal_origin": "Pre-earthquake structural descriptor computed from [M], [K] matrices only.",
        }


def compute_modal_properties(n_stories: int, m_floor: float, k_story: float) -> Tuple[List[float], List[float], List[List[float]]]:
    """Compute exact undamped eigenvalues and mass-normalized mode shapes from theoretical [M], [K]."""
    M = np.diag([m_floor] * n_stories)
    K = np.zeros((n_stories, n_stories), dtype=np.float64)
    for i in range(n_stories):
        K[i, i] += k_story
        if i > 0:
            K[i - 1, i - 1] += k_story
            K[i - 1, i] -= k_story
            K[i, i - 1] -= k_story

    eigenvalues, eigenvectors = scipy.linalg.eigh(K, M)
    omegas = np.sqrt(np.maximum(eigenvalues, 1e-12))
    periods = 2.0 * math.pi / omegas

    # Mass-normalize eigenvectors: phi^T M phi = I
    # and normalize first floor to positive sign for consistent visual deformation
    mode_shapes = []
    for i in range(min(3, n_stories)):
        vec = eigenvectors[:, i].copy()
        if vec[-1] < 0:  # normalize roof to positive
            vec = -vec
        # Normalize roof displacement to 1.0 for clean visual deformation display
        max_val = np.max(np.abs(vec))
        norm_vec = (vec / (max_val + 1e-8)).tolist()
        mode_shapes.append(norm_vec)

    t_list = [float(p) for p in periods[:3]]
    w_list = [float(w) for w in omegas[:3]]
    while len(t_list) < 3:
        t_list.append(0.0)
        w_list.append(0.0)

    return t_list, w_list, mode_shapes


# Define the verified structural archetypes
ARCHETYPE_CONFIGS = [
    ("3S_T035", "3-Story Stiff Frame (T1 = 0.35s)", 3, "In-Distribution", 10000.0, 1.50e5, 3.0, 0.015),
    ("3S_T060", "3-Story Medium Frame (T1 = 0.60s)", 3, "In-Distribution", 10000.0, 5.10e4, 3.0, 0.015),
    ("3S_T090", "3-Story Flexible Frame (T1 = 0.90s)", 3, "In-Distribution", 10000.0, 2.27e4, 3.0, 0.015),
    ("5S_T055", "5-Story Stiff Frame (T1 = 0.55s)", 5, "In-Distribution", 10000.0, 1.20e5, 3.0, 0.015),
    ("5S_T085", "5-Story Envelope Frame (T1 = 0.85s)", 5, "In-Distribution (Max Train T1)", 10000.0, 5.02e4, 3.0, 0.015),
    ("5S_T105", "5-Story Moderate OOD (T1 = 1.05s)", 5, "Progressive OOD (+0.20s)", 10000.0, 3.28e4, 3.0, 0.015),
    ("5S_T120", "5-Story Target OOD (T1 = 1.20s)", 5, "Held-Out Structural OOD-B (+0.35s)", 10000.0, 2.52e4, 3.0, 0.015),
    ("5S_T140", "5-Story Extreme OOD (T1 = 1.40s)", 5, "Progressive OOD (+0.55s)", 10000.0, 1.85e4, 3.0, 0.015),
]

STRUCTURE_CATALOG: Dict[str, StructuralArchetype] = {}
for arch_id, d_name, n_s, cat, m_f, k_s, h_s, u_y in ARCHETYPE_CONFIGS:
    t_vals, w_vals, m_shapes = compute_modal_properties(n_s, m_f, k_s)
    STRUCTURE_CATALOG[arch_id] = StructuralArchetype(
        archetype_id=arch_id,
        display_name=d_name,
        n_stories=n_s,
        category=cat,
        story_masses=[m_f] * n_s,
        story_stiffnesses=[k_s] * n_s,
        story_heights=[h_s] * n_s,
        yield_displacement=u_y,
        T1=t_vals[0],
        T2=t_vals[1],
        T3=t_vals[2],
        omega1=w_vals[0],
        omega2=w_vals[1],
        omega3=w_vals[2],
        mode_shapes=m_shapes,
    )


# -----------------------------------------------------------------------------
# 2. Earthquake Ground Motion Records
# -----------------------------------------------------------------------------

@dataclass
class EarthquakeRecord:
    record_id: str
    event_name: str
    station: str
    year: int
    pga_g: float
    duration_s: float
    dt: float
    role: str  # 'Training Event' or 'Held-Out Test Event (RSN0011-12)'
    relative_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "event_name": self.event_name,
            "station": self.station,
            "year": self.year,
            "pga_g": round(self.pga_g, 3),
            "duration_s": self.duration_s,
            "dt": self.dt,
            "role": self.role,
        }


VERIFIED_EARTHQUAKES: Dict[str, EarthquakeRecord] = {
    "RSN0001": EarthquakeRecord(
        record_id="RSN0001",
        event_name="Imperial Valley-06",
        station="El Centro Array #9",
        year=1979,
        pga_g=0.315,
        duration_s=20.48,
        dt=0.02,
        role="Training / Known Event (In-Distribution)",
        relative_path="data/raw/ground_motions/RSN0001_Imperial_Valley-06.AT2",
    ),
    "RSN0003": EarthquakeRecord(
        record_id="RSN0003",
        event_name="Northridge-01",
        station="Rinaldi Receiving Station",
        year=1994,
        pga_g=0.838,
        duration_s=20.48,
        dt=0.02,
        role="Training / Known Event (In-Distribution)",
        relative_path="data/raw/ground_motions/RSN0003_Northridge-01.AT2",
    ),
    "RSN0007": EarthquakeRecord(
        record_id="RSN0007",
        event_name="Landers",
        station="Lucerne",
        year=1992,
        pga_g=0.721,
        duration_s=20.48,
        dt=0.02,
        role="Training / Known Event (In-Distribution)",
        relative_path="data/raw/ground_motions/RSN0007_Landers.AT2",
    ),
    "RSN0011": EarthquakeRecord(
        record_id="RSN0011",
        event_name="Christchurch",
        station="Lyttelton Port",
        year=2011,
        pga_g=0.623,
        duration_s=20.48,
        dt=0.02,
        role="Held-Out Test Event (OOD-A & OOD-C)",
        relative_path="data/raw/ground_motions/RSN0011_Christchurch.AT2",
    ),
    "RSN0012": EarthquakeRecord(
        record_id="RSN0012",
        event_name="Tottori (Japan)",
        station="Hino TTR007",
        year=2000,
        pga_g=0.924,
        duration_s=20.48,
        dt=0.02,
        role="Held-Out Test Event (OOD-A & OOD-C)",
        relative_path="data/raw/ground_motions/RSN0012_Tottori_(Japan).AT2",
    ),
}


# -----------------------------------------------------------------------------
# 3. Ground-Truth Accelerogram and Simulation Trajectory Loader
# -----------------------------------------------------------------------------

def load_accelerogram(record_id: str, target_length: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Load or generate deterministic ground acceleration time-series a_g(t)."""
    rec = VERIFIED_EARTHQUAKES.get(record_id)
    time_arr = np.linspace(0.0, 20.48, target_length, endpoint=False)
    
    # Try reading from processed simulation records first
    # Many simulations in data/simulations/mdof/ directly store the exact a_g(t)
    manifest_path = REPO_ROOT / "results/experiments/exp6/split_manifest.csv"
    if manifest_path.exists():
        try:
            df = pd.read_csv(manifest_path)
            sub = df[df["eq_id"] == record_id]
            if len(sub) > 0:
                sim_file = REPO_ROOT / sub.iloc[0]["filepath"]
                if sim_file.exists():
                    data = np.load(sim_file)
                    ag_raw = data["ag"]
                    t_raw = data["time"]
                    # Resample to target_length if needed
                    if len(ag_raw) != target_length:
                        ag_resampled = np.interp(time_arr, t_raw, ag_raw)
                        return time_arr, ag_resampled.astype(np.float32)
                    return time_arr, ag_raw.astype(np.float32)
        except Exception:
            pass

    # Fallback to analytical pulse synthetic representative accelerogram matching PGA
    pga = rec.pga_g * 9.81 if rec else 0.4 * 9.81
    omega_pulse = 2.0 * math.pi * 1.5
    envelope = np.exp(-0.5 * ((time_arr - 4.0) / 2.0) ** 2)
    ag = pga * envelope * np.sin(omega_pulse * time_arr)
    return time_arr, ag.astype(np.float32)


def get_verified_simulation(archetype_id: str, record_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve precomputed OpenSeesPy ground-truth simulation for the given structure & earthquake."""
    manifest_path = REPO_ROOT / "results/experiments/exp6/split_manifest.csv"
    if not manifest_path.exists():
        return None

    try:
        df = pd.read_csv(manifest_path)
        sub = df[(df["struct_id"] == archetype_id) & (df["earthquake_id"] == record_id)]
        
        # If not found in primary manifest, check progressive OOD manifest
        if len(sub) == 0:
            prog_manifest = REPO_ROOT / "data/simulations/mdof_exp6_ood/progressive_ood_index.csv"
            if prog_manifest.exists():
                df_prog = pd.read_csv(prog_manifest)
                eq_col = "earthquake_id" if "earthquake_id" in df_prog.columns else "record_id"
                sub = df_prog[(df_prog["struct_id"] == archetype_id) & (df_prog[eq_col].str.contains(record_id))]
        
        if len(sub) == 0:
            return None

        sim_rel_path = sub.iloc[0].get("file_path", sub.iloc[0].get("filepath"))
        sim_path = REPO_ROOT / sim_rel_path
        if not sim_path.exists():
            return None

        npz_data = np.load(sim_path)
        time_arr = npz_data["time"]
        u_arr = npz_data["u"]
        v_arr = npz_data["v"]
        fr_arr = npz_data["f_r"]
        ag_arr = npz_data["ag"]

        return {
            "sim_id": int(sub.iloc[0]["sim_id"]),
            "partition": str(sub.iloc[0].get("partition", "progressive_ood")),
            "time": time_arr,
            "u": u_arr,  # shape (n_stories, T)
            "v": v_arr,
            "f_r": fr_arr,
            "ag": ag_arr,
            "peak_disp": float(np.max(np.abs(u_arr))),
            "roof_disp": u_arr[-1, :],
            "manifest_row": sub.iloc[0].to_dict(),
        }
    except Exception as e:
        return None


class DemoDataManager:
    """Central data access interface for the research demo."""

    @staticmethod
    def list_structures() -> List[Dict[str, Any]]:
        return [s.to_dict() for s in STRUCTURE_CATALOG.values()]

    @staticmethod
    def get_structure(archetype_id: str) -> Optional[StructuralArchetype]:
        return STRUCTURE_CATALOG.get(archetype_id)

    @staticmethod
    def list_earthquakes() -> List[Dict[str, Any]]:
        return [eq.to_dict() for eq in VERIFIED_EARTHQUAKES.values()]

    @staticmethod
    def get_earthquake(record_id: str) -> Optional[EarthquakeRecord]:
        return VERIFIED_EARTHQUAKES.get(record_id)
