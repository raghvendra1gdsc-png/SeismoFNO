"""
batch_runner.py — Batch Simulation Runner for SDOF Ground Truth Responses.

Runs OpenSeesPy across the full combination of:
  - Ground motion records from `data/simulations/dataset_manifest.csv`
  - Structural parameter grid (periods T, damping ratios zeta, elastic and bilinear ductility designs)

Outputs:
  - Time-history responses stored as individual .npz files containing:
      u: relative displacement [m]
      v: relative velocity [m/s]
      a: relative acceleration [m/s^2]
      total_accel: absolute acceleration [m/s^2]
      f_r: restoring force [N]
      e_h: cumulative hysteretic energy [J]
      e_d: viscous damping dissipation [J]
      ag: ground motion input [m/s^2]
      time: time vector [s]
  - Comprehensive index CSV: `data/simulations/simulation_index.csv`
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import os
from pathlib import Path
import math
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd

from src.ground_truth.opensees_sdof_model import (
    SDOFParams,
    SDOFResponse,
    simulate_sdof,
)
from src.data_pipeline.gm_preprocessing import G_ACCEL


# Curated structural parameter grid for SDOF database
DEFAULT_STRUCTURAL_GRID = [
    # 1. Linear-elastic structures (short, medium, long period)
    {"struct_id": "SDOF_T0p20s_Z0p05_ELAS", "T": 0.20, "zeta": 0.05, "material_type": "elastic", "u_y": None, "alpha": 0.0, "mass": 1.0},
    {"struct_id": "SDOF_T0p50s_Z0p05_ELAS", "T": 0.50, "zeta": 0.05, "material_type": "elastic", "u_y": None, "alpha": 0.0, "mass": 1.0},
    {"struct_id": "SDOF_T1p00s_Z0p05_ELAS", "T": 1.00, "zeta": 0.05, "material_type": "elastic", "u_y": None, "alpha": 0.0, "mass": 1.0},
    {"struct_id": "SDOF_T1p50s_Z0p05_ELAS", "T": 1.50, "zeta": 0.05, "material_type": "elastic", "u_y": None, "alpha": 0.0, "mass": 1.0},
    {"struct_id": "SDOF_T2p00s_Z0p05_ELAS", "T": 2.00, "zeta": 0.05, "material_type": "elastic", "u_y": None, "alpha": 0.0, "mass": 1.0},
    
    # 2. Bilinear-hysteretic structures (varying period, yield displacement, post-yield stiffness)
    {"struct_id": "SDOF_T0p20s_Z0p05_BILIN_uy0p005_a0p02", "T": 0.20, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.005, "alpha": 0.02, "mass": 1.0},
    {"struct_id": "SDOF_T0p50s_Z0p05_BILIN_uy0p010_a0p02", "T": 0.50, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.010, "alpha": 0.02, "mass": 1.0},
    {"struct_id": "SDOF_T0p50s_Z0p05_BILIN_uy0p020_a0p05", "T": 0.50, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.020, "alpha": 0.05, "mass": 1.0},
    {"struct_id": "SDOF_T1p00s_Z0p05_BILIN_uy0p020_a0p02", "T": 1.00, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.020, "alpha": 0.02, "mass": 1.0},
    {"struct_id": "SDOF_T1p00s_Z0p05_BILIN_uy0p040_a0p05", "T": 1.00, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.040, "alpha": 0.05, "mass": 1.0},
    {"struct_id": "SDOF_T1p50s_Z0p05_BILIN_uy0p030_a0p02", "T": 1.50, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.030, "alpha": 0.02, "mass": 1.0},
    {"struct_id": "SDOF_T2p00s_Z0p05_BILIN_uy0p050_a0p02", "T": 2.00, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.050, "alpha": 0.02, "mass": 1.0},
]


def _worker_single_sim(task_args: Tuple[Dict[str, Any], Dict[str, Any], str]) -> Optional[Dict[str, Any]]:
    """Worker function executing a single SDOF dynamic simulation."""
    rec_meta, struct_cfg, output_dir_str = task_args
    
    sim_id = f"{rec_meta['record_id']}__{struct_cfg['struct_id']}"
    output_path = Path(output_dir_str) / f"{sim_id}.npz"
    
    try:
        # Load ground motion array in g and convert to m/s^2 for OpenSees
        raw_arr = np.load(rec_meta["file_path"])
        dt = float(rec_meta["dt"])
        ag_ms2 = raw_arr * G_ACCEL
        
        # Build SDOF parameters
        params = SDOFParams(
            T=float(struct_cfg["T"]),
            zeta=float(struct_cfg["zeta"]),
            material_type=struct_cfg["material_type"],
            u_y=float(struct_cfg["u_y"]) if struct_cfg["u_y"] is not None else None,
            alpha=float(struct_cfg["alpha"]),
            mass=float(struct_cfg.get("mass", 1.0)),
        )
        
        # Run OpenSees dynamic simulation
        resp = simulate_sdof(params=params, ag=ag_ms2, dt=dt)
        
        # Save response array
        np.savez_compressed(
            output_path,
            time=resp.time.astype(np.float32),
            u=resp.u.astype(np.float32),
            v=resp.v.astype(np.float32),
            a=resp.a.astype(np.float32),
            total_accel=resp.total_accel.astype(np.float32),
            f_r=resp.f_r.astype(np.float32),
            e_h=resp.e_h.astype(np.float32),
            e_d=resp.e_d.astype(np.float32),
            ag=ag_ms2.astype(np.float32),
        )
        
        # Compute summary metrics
        u_max = float(np.max(np.abs(resp.u)))
        f_max = float(np.max(np.abs(resp.f_r)))
        e_h_final = float(resp.e_h[-1])
        ductility = float(u_max / struct_cfg["u_y"]) if struct_cfg["u_y"] is not None and struct_cfg["u_y"] > 0 else 1.0
        
        return {
            "sim_id": sim_id,
            "record_id": rec_meta["record_id"],
            "base_record_id": rec_meta["base_record_id"],
            "earthquake_name": rec_meta["earthquake_name"],
            "magnitude": rec_meta["magnitude"],
            "r_rup_km": rec_meta["r_rup_km"],
            "vs30_ms": rec_meta["vs30_ms"],
            "target_pga_g": rec_meta["target_pga_g"],
            "resulting_pga_g": rec_meta["resulting_pga_g"],
            "struct_id": struct_cfg["struct_id"],
            "T": struct_cfg["T"],
            "zeta": struct_cfg["zeta"],
            "material_type": struct_cfg["material_type"],
            "u_y": struct_cfg["u_y"] if struct_cfg["u_y"] is not None else np.nan,
            "alpha": struct_cfg["alpha"],
            "u_max_m": u_max,
            "f_max_n": f_max,
            "e_h_final_j": e_h_final,
            "ductility": ductility,
            "dt": dt,
            "n_steps": len(resp.u),
            "sim_file_path": str(output_path.resolve()),
        }
    except Exception as e:
        print(f"Error running simulation {sim_id}: {e}")
        return None


def run_batch_simulations(
    manifest_csv: Union[str, Path] = "data/simulations/dataset_manifest.csv",
    structural_grid: Optional[List[Dict[str, Any]]] = None,
    output_dir: Union[str, Path] = "data/simulations/sdof_responses",
    index_csv_path: Union[str, Path] = "data/simulations/simulation_index.csv",
    max_records: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Execute batch simulations across records and structural configurations.

    Parameters
    ----------
    manifest_csv : str or Path
        Path to dataset_manifest.csv.
    structural_grid : list of dict, optional
        List of structural parameter configurations.
    output_dir : str or Path
        Directory to save simulation .npz outputs.
    index_csv_path : str or Path
        Path to output simulation_index.csv.
    max_records : int, optional
        Optional cap on number of records (useful for rapid testing).
    num_workers : int, optional
        Number of parallel worker processes.

    Returns
    -------
    index_df : pd.DataFrame
        Comprehensive simulation index dataframe.
    """
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file {manifest_path} does not exist.")

    manifest_df = pd.read_csv(manifest_path)
    if max_records is not None:
        manifest_df = manifest_df.head(max_records)

    if structural_grid is None:
        structural_grid = DEFAULT_STRUCTURAL_GRID

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_file = Path(index_csv_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    records_list = manifest_df.to_dict(orient="records")
    
    # Build list of all tasks
    tasks = []
    for rec in records_list:
        for struct in structural_grid:
            tasks.append((rec, struct, str(output_path)))

    print(f"Starting batch execution of {len(tasks)} simulations across {len(records_list)} records and {len(structural_grid)} structures...")
    
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    results = []
    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for res in executor.map(_worker_single_sim, tasks, chunksize=10):
                if res is not None:
                    results.append(res)
    else:
        for task in tasks:
            res = _worker_single_sim(task)
            if res is not None:
                results.append(res)

    index_df = pd.DataFrame(results)
    index_df.to_csv(index_file, index=False)
    print(f"Batch simulations complete. {len(index_df)} / {len(tasks)} runs indexed to {index_file}")
    return index_df


if __name__ == "__main__":
    df = run_batch_simulations()
    print(df.head(5))
