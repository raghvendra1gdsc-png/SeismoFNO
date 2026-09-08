"""
mdof_batch_runner.py — High-Throughput Batch OpenSeesPy MDOF Simulation Runner.

Generates ground-truth multi-story building response simulations across:
  - Story counts N in {3, 5} (3-story and 5-story buildings)
  - Scaled ground motion time series spanning 0.05g to 1.2g PGA
  - Grids of fundamental periods T_1 and inter-story yield ductility limits
  - Both linear-elastic and nonlinear bilinear-hysteretic constitutive models

Saves simulation trajectory arrays in compressed .npz format:
  data/simulations/mdof/sim_{sim_id:06d}.npz
and builds the manifest:
  data/simulations/mdof/simulation_index.csv
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import math
import numpy as np
import pandas as pd
from src.ground_truth.opensees_mdof_model import (
    MDOFParams,
    MDOFResponse,
    simulate_mdof,
)


def generate_mdof_dataset(
    manifest_csv: str = "data/simulations/dataset_manifest.csv",
    output_dir: str = "data/simulations/mdof",
    max_records: int = 100,
    target_dt: float = 0.01,
    target_steps: int = 2048,
) -> pd.DataFrame:
    """
    Generate batch OpenSees MDOF building response simulations.

    Args:
        manifest_csv: Path to scaled ground motion manifest
        output_dir: Output directory for .npz simulation files and simulation_index.csv
        max_records: Number of scaled ground motion records to use
        target_dt: Common simulation time step (0.01 s -> 100 Hz)
        target_steps: Sequence length (2048 steps -> 20.48 s duration)

    Returns:
        DataFrame index of all completed MDOF simulations
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_csv)
    if len(manifest_df) > max_records:
        manifest_df = manifest_df.iloc[:max_records].reset_index(drop=True)

    # Define MDOF structural parameter combinations
    # (n_stories, T1_target, story_mass, story_height)
    structural_archetypes = [
        # 3-Story Configurations
        {"n_stories": 3, "T1": 0.35, "mass": 1000.0, "height": 3.0, "struct_id": "3S_T035"},
        {"n_stories": 3, "T1": 0.60, "mass": 1200.0, "height": 3.2, "struct_id": "3S_T060"},
        {"n_stories": 3, "T1": 0.90, "mass": 1500.0, "height": 3.5, "struct_id": "3S_T090"},
        # 5-Story Configurations
        {"n_stories": 5, "T1": 0.55, "mass": 1000.0, "height": 3.0, "struct_id": "5S_T055"},
        {"n_stories": 5, "T1": 0.85, "mass": 1200.0, "height": 3.2, "struct_id": "5S_T085"},
        {"n_stories": 5, "T1": 1.20, "mass": 1400.0, "height": 3.5, "struct_id": "5S_T120"},
    ]

    material_types = ["elastic", "bilinear"]
    yield_drift_ratios = [0.004, 0.007]  # 0.4% and 0.7% drift limits for bilinear

    sim_records = []
    sim_id = 0

    total_records = len(manifest_df)
    print(f"Generating MDOF simulations for {total_records} records across {len(structural_archetypes)} archetypes...")

    for r_idx, (_, gm_row) in enumerate(manifest_df.iterrows()):
        if (r_idx + 1) % 10 == 0 or r_idx == 0:
            print(f"  Processing record [{r_idx + 1:3d}/{total_records:3d}] | Total simulations generated: {sim_id}")
        record_id = gm_row["record_id"]
        earthquake_id = gm_row.get("base_record_id", gm_row.get("earthquake_name", record_id.split("_")[0]))
        pga = gm_row.get("target_pga_g", gm_row.get("resulting_pga_g", 0.1))
        scale_factor = gm_row.get("scale_factor", 1.0)
        gm_file = gm_row["file_path"]

        # Load ground motion array (.npy)
        ag_raw = np.load(gm_file)
        if isinstance(ag_raw, np.lib.npyio.NpzFile):
            ag_raw = ag_raw["ag"]

        # Resample / pad to target_steps
        if len(ag_raw) < target_steps:
            ag = np.pad(ag_raw, (0, target_steps - len(ag_raw)), mode="constant")
        else:
            ag = ag_raw[:target_steps]

        for arch in structural_archetypes:
            n_stories = arch["n_stories"]
            t1_target = arch["T1"]
            m_floor = arch["mass"]
            h_floor = arch["height"]
            struct_id = arch["struct_id"]

            # Calculate inter-story stiffness k to achieve target T1
            # For uniform shear building: omega_1 approx 2 * sqrt(k/m) * sin(pi / (2*(2N+1)))
            # k approx m * (omega_1 / (2 * sin(pi / (4N+2))))^2
            omega1_target = 2.0 * math.pi / t1_target
            k_story = m_floor * (omega1_target / (2.0 * math.sin(math.pi / (4 * n_stories + 2)))) ** 2

            masses = [m_floor] * n_stories
            heights = [h_floor] * n_stories
            stiffnesses = [k_story] * n_stories

            for mat in material_types:
                if mat == "elastic":
                    drift_list = [yield_drift_ratios[0]]
                else:
                    drift_list = yield_drift_ratios

                for drift_ratio in drift_list:
                    yield_drifts = [drift_ratio * h for h in heights]

                    params = MDOFParams(
                        n_stories=n_stories,
                        story_masses=masses,
                        story_heights=heights,
                        story_stiffnesses=stiffnesses,
                        material_type=mat,
                        yield_displacements=yield_drifts,
                        alpha=0.05,
                        zeta_1=0.05,
                        zeta_2=0.05,
                    )

                    try:
                        resp = simulate_mdof(params=params, ag=ag, dt=target_dt)
                    except Exception as e:
                        # Skip unstable or diverged nonlinear cases
                        continue

                    # Save simulation arrays
                    sim_filename = f"sim_{sim_id:06d}.npz"
                    sim_file_path = out_path / sim_filename

                    np.savez_compressed(
                        sim_file_path,
                        time=resp.time,
                        ag=ag,
                        u=resp.u,
                        v=resp.v,
                        a=resp.a,
                        f_r=resp.f_r,
                        e_h=resp.e_h,
                        modal_omegas=resp.modal_omegas,
                        modal_periods=resp.modal_periods,
                    )

                    # Compute summary metrics
                    peak_roof_disp = float(np.max(np.abs(resp.u[-1, :])))
                    peak_roof_accel = float(np.max(np.abs(resp.a[-1, :])))
                    total_hysteretic_energy = float(np.sum(resp.e_h[:, -1]))
                    max_interstory_drift = float(np.max([np.max(np.abs(resp.u[0, :])) / heights[0]] + [
                        np.max(np.abs(resp.u[i, :] - resp.u[i - 1, :])) / heights[i] for i in range(1, n_stories)
                    ]))

                    sim_records.append({
                        "sim_id": sim_id,
                        "file_path": str(sim_file_path),
                        "record_id": record_id,
                        "earthquake_id": earthquake_id,
                        "pga_g": pga,
                        "scale_factor": scale_factor,
                        "struct_id": struct_id,
                        "n_stories": n_stories,
                        "T1_s": float(resp.modal_periods[0]),
                        "T2_s": float(resp.modal_periods[1]) if n_stories > 1 else float(resp.modal_periods[0]),
                        "material_type": mat,
                        "yield_drift_ratio": drift_ratio,
                        "peak_roof_disp_m": peak_roof_disp,
                        "peak_roof_accel_g": peak_roof_accel / 9.81,
                        "max_interstory_drift_ratio": max_interstory_drift,
                        "total_hysteretic_energy_J": total_hysteretic_energy,
                    })

                    sim_id += 1

    df_index = pd.DataFrame(sim_records)
    index_csv = out_path / "simulation_index.csv"
    df_index.to_csv(index_csv, index=False)
    print(f"\nGenerated {len(df_index)} MDOF simulations. Manifest saved to: {index_csv}")
    return df_index


if __name__ == "__main__":
    generate_mdof_dataset(max_records=120)
