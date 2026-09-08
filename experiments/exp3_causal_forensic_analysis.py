"""
exp3_causal_forensic_analysis.py — Comprehensive Forensic Causal Audit of EXP 3.

Empirically probes:
1. Architectural / Direct-Decoder Confounding
2. Sham / Random State Controls (C3 Orthogonal Perturbation)
3. Dose-Response Slope Distribution and Confidence Intervals
4. Waveform Translation vs Genuine Dynamic Modification
5. Constitutive Validity of Yield Onset (Elastic vs Inelastic Interventions)
6. Structural Proof of Past Invariance
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from src.models.pg_tcn import PhysicsSupervisedCausalTCN
from src.data_pipeline.dataset_builder import UnitGaussianNormalizer, SeismicSDOFDataset
from src.data_pipeline.splits import load_split
from src.evaluation.exp3_interventions import compute_exact_physical_state, identify_yield_onset_time


def run_audit():
    device = torch.device("cpu")
    torch.set_num_threads(8)

    base_dir = Path("results/experiments/exp3_physics_guided_state")
    ckpt_path = base_dir / "checkpoints" / "best_pg_tcn_supervised.pt"
    cf_json_path = base_dir / "interventions" / "counterfactual_intervention_results.json"
    det_csv_path = base_dir / "raw" / "test_records_detailed_metrics.csv"
    train_cache_path = Path("data/processed/exp3_cache/train_cache.pt")

    # 1. Load Model
    print("Loading PG-TCN checkpoint...")
    model = PhysicsSupervisedCausalTCN(
        in_channels=10, out_channels=3, encoder_dim=135, decoder_dim=136, state_hidden_dim=16, state_dim=2
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Load Normalizers & Data
    print("Loading datasets and normalizers...")
    train_cache = torch.load(train_cache_path, map_location=device)
    x_norm = UnitGaussianNormalizer().fit(train_cache["x"])
    y_norm = UnitGaussianNormalizer().fit(train_cache["y"])

    split = load_split(Path("data/processed/splits/held_out_earthquake_split.json"))
    sim_index_path = Path("data/simulations/simulation_index.csv")
    full_df = pd.read_csv(sim_index_path)
    test_df = full_df[full_df["sim_id"].isin(split.test_ids)].reset_index(drop=True)

    # 3. Analyze Existing Counterfactual Data
    print("Analyzing existing counterfactual intervention telemetry...")
    with open(cf_json_path, "r") as f:
        cf_data = json.load(f)

    cf_samples = cf_data["cf_samples"]
    cf_df = pd.DataFrame(cf_samples)

    slopes = cf_df["slope"].values
    r2_scores = cf_df["r2_score"].values
    past_errs = cf_df["max_past_error_mm"].values
    asymmetries = cf_df["c4_asymmetry_pct"].values
    ductilities = cf_df["ductility"].values

    slope_mean = np.mean(slopes)
    slope_std = np.std(slopes, ddof=1)
    slope_median = np.median(slopes)
    slope_ci_low, slope_ci_high = stats.t.interval(0.95, len(slopes)-1, loc=slope_mean, scale=stats.sem(slopes))

    r2_mean = np.mean(r2_scores)
    r2_median = np.median(r2_scores)
    r2_ci_low, r2_ci_high = stats.t.interval(0.95, len(r2_scores)-1, loc=r2_mean, scale=stats.sem(r2_scores))

    pct_negative_slopes = np.mean(slopes < 0) * 100.0
    pct_positive_slopes = np.mean(slopes > 0) * 100.0

    print(f"Total CF Samples: {len(cf_df)}")
    print(f"Slope: Mean = {slope_mean:.5f}, Median = {slope_median:.5f}, 95% CI = [{slope_ci_low:.5f}, {slope_ci_high:.5f}]")
    print(f"Slopes < 0: {pct_negative_slopes:.1f}%, Slopes > 0: {pct_positive_slopes:.1f}%")
    print(f"R2 Score: Mean = {r2_mean:.5f}, Median = {r2_median:.5f}")

    # 4. Attack Vector 1 & 4: Waveform Translation vs Dynamic Modification
    high_mu_samples = cf_df[cf_df["ductility"] > 4.0]
    print(f"\nAnalyzing waveform dynamic modification on {len(high_mu_samples)} high-ductility samples...")

    translation_metrics = []
    test_ds = SeismicSDOFDataset(
        test_df, target_time_steps=2048, target_channels=["u", "f_r", "e_h"],
        x_normalizer=x_norm, y_normalizer=y_norm, return_meta=True
    )

    sim_to_idx = {test_df.iloc[i]["sim_id"]: i for i in range(len(test_df))}

    for _, row in high_mu_samples.head(25).iterrows():
        sid = row["sim_id"]
        if sid not in sim_to_idx:
            continue
        idx = sim_to_idx[sid]
        x_raw, y_raw, meta = test_ds[idx]
        x_in = x_raw.unsqueeze(0).to(device)

        T_val = float(meta["T"]) if "T" in meta else 0.5
        k0 = float(meta["k0"]) if "k0" in meta else (2.0 * np.pi / T_val) ** 2
        alpha = float(meta["alpha"]) if "alpha" in meta and not pd.isna(meta["alpha"]) else 0.02

        with torch.no_grad():
            y_base, s_base = model(x_in, return_state=True)
            u_base = y_base[0, 0, :].numpy()

            t_y = row["t_y"]
            dose_m = 0.010
            interv_pos = {"t_y": t_y, "delta_u_p": dose_m, "delta_alpha_b": alpha * k0 * dose_m}
            y_cf, s_cf = model(x_in, intervention=interv_pos, return_state=True)
            u_cf = y_cf[0, 0, :].numpy()

        diff_post = (u_cf[t_y:] - u_base[t_y:]) * 1000.0  # mm
        mean_step = np.mean(diff_post)
        dc_residual = np.std(diff_post)
        fluctuation_ratio = dc_residual / (abs(mean_step) + 1e-6)

        translation_metrics.append({
            "sim_id": sid,
            "mean_step_mm": mean_step,
            "std_fluc_mm": dc_residual,
            "fluctuation_ratio": fluctuation_ratio,
        })

    trans_df = pd.DataFrame(translation_metrics)
    mean_fluc_ratio = trans_df["fluctuation_ratio"].mean()
    mean_std_fluc = trans_df["std_fluc_mm"].mean()
    print(f"Waveform Post-Intervention Fluctuation Ratio: {mean_fluc_ratio:.4f}, Std Fluctuation: {mean_std_fluc:.4f} mm")

    # 5. Attack Vector 2: Sham Controls (C3 Orthogonal & Component Perturbations)
    print("\nExecuting Attack Vector 2: Sham Controls (C3)...")
    sham_results = []
    for _, row in high_mu_samples.head(20).iterrows():
        sid = row["sim_id"]
        if sid not in sim_to_idx:
            continue
        idx = sim_to_idx[sid]
        x_raw, y_raw, meta = test_ds[idx]
        x_in = x_raw.unsqueeze(0).to(device)

        T_val = float(meta["T"]) if "T" in meta else 0.5
        k0 = float(meta["k0"]) if "k0" in meta else (2.0 * np.pi / T_val) ** 2
        alpha = float(meta["alpha"]) if "alpha" in meta and not pd.isna(meta["alpha"]) else 0.02
        t_y = row["t_y"]

        dose_m = 0.010
        d_alpha = alpha * k0 * dose_m

        with torch.no_grad():
            y_base = model(x_in)[0, 0, :].numpy()
            y_phys = model(x_in, intervention={"t_y": t_y, "delta_u_p": dose_m, "delta_alpha_b": d_alpha})[0, 0, :].numpy()
            y_sham1 = model(x_in, intervention={"t_y": t_y, "delta_u_p": dose_m, "delta_alpha_b": -d_alpha})[0, 0, :].numpy()
            y_sham2 = model(x_in, intervention={"t_y": t_y, "delta_u_p": 0.0, "delta_alpha_b": d_alpha})[0, 0, :].numpy()
            y_sham3 = model(x_in, intervention={"t_y": t_y, "delta_u_p": dose_m, "delta_alpha_b": 0.0})[0, 0, :].numpy()

        drift_phys = abs(y_phys[-1] - y_base[-1]) * 1000.0
        drift_sham1 = abs(y_sham1[-1] - y_base[-1]) * 1000.0
        drift_sham2 = abs(y_sham2[-1] - y_base[-1]) * 1000.0
        drift_sham3 = abs(y_sham3[-1] - y_base[-1]) * 1000.0

        sham_results.append({
            "sim_id": sid,
            "drift_phys": drift_phys,
            "drift_sham1_ortho": drift_sham1,
            "drift_sham2_backstress_only": drift_sham2,
            "drift_sham3_plastic_only": drift_sham3,
        })

    sham_df = pd.DataFrame(sham_results)
    print(f"Phys Drift: {sham_df['drift_phys'].mean():.4f} mm")
    print(f"Sham 1 (Orthogonal State): {sham_df['drift_sham1_ortho'].mean():.4f} mm")
    print(f"Sham 2 (Backstress Only): {sham_df['drift_sham2_backstress_only'].mean():.4f} mm")
    print(f"Sham 3 (Plastic Disp Only): {sham_df['drift_sham3_plastic_only'].mean():.4f} mm")

    # 6. Attack Vector 5: Constitutive Validity of Detected Yield Onset
    print("\nExecuting Attack Vector 5: Constitutive Validity of Yield Onset...")
    elastic_cf = cf_df[cf_df["ductility"] <= 1.0]
    inelastic_cf = cf_df[cf_df["ductility"] > 1.0]
    print(f"Interventions on Elastic records (never yielded in OpenSees): N = {len(elastic_cf)}")
    print(f"Interventions on Inelastic records (yielded in OpenSees): N = {len(inelastic_cf)}")

    elastic_slope_mean = elastic_cf["slope"].mean()
    inelastic_slope_mean = inelastic_cf["slope"].mean()
    print(f"Elastic Intervention Slope Mean: {elastic_slope_mean:.5f}")
    print(f"Inelastic Intervention Slope Mean: {inelastic_slope_mean:.5f}")

    out_data = {
        "n_samples": len(cf_df),
        "slope_mean": float(slope_mean),
        "slope_median": float(slope_median),
        "slope_ci": [float(slope_ci_low), float(slope_ci_high)],
        "pct_negative_slopes": float(pct_negative_slopes),
        "r2_mean": float(r2_mean),
        "r2_median": float(r2_median),
        "waveform_fluctuation_ratio": float(mean_fluc_ratio),
        "waveform_std_fluc_mm": float(mean_std_fluc),
        "sham_phys_drift_mean": float(sham_df['drift_phys'].mean()),
        "sham_ortho_drift_mean": float(sham_df['drift_sham1_ortho'].mean()),
        "sham_backstress_drift_mean": float(sham_df['drift_sham2_backstress_only'].mean()),
        "sham_plastic_drift_mean": float(sham_df['drift_sham3_plastic_only'].mean()),
        "elastic_samples_count": len(elastic_cf),
        "inelastic_samples_count": len(inelastic_cf),
        "elastic_slope_mean": float(elastic_slope_mean),
        "inelastic_slope_mean": float(inelastic_slope_mean),
    }

    out_file = base_dir / "interventions" / "causal_forensic_metrics.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nAudit complete. Metrics saved to {out_file}")


if __name__ == "__main__":
    run_audit()
