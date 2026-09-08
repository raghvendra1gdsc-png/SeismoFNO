# SeismoFNO REST API Specification

## Base URL
```
http://localhost:8000
```

---

## 1. System & Health

### `GET /health`
Liveness and readiness check.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "timestamp": 1788693785.446,
  "version": "1.0.0",
  "model_loaded": true,
  "device": "mps"
}
```

### `GET /api/v1/system/info`
Introspects model checkpoint provenance, Fourier modes, width, and compute acceleration.

**Response `200 OK`**:
```json
{
  "status": "operational",
  "version": "1.0.0",
  "model_loaded": true,
  "device": "mps",
  "model_parameters": 1196931,
  "checkpoint_path": "experiments/phase5_c_data_energy_boundary/checkpoints/best_model.pt",
  "config_path": "configs/phase5_c_data_energy_boundary.yaml",
  "modes": 128,
  "width": 48,
  "is_mock": false,
  "scientific_integrity": "Strict — Frozen Research Core"
}
```

---

## 2. Seismic & Building Catalogs

### `GET /api/v1/earthquakes`
Lists available ground motion records from the Indian seismic catalog and PEER NGA-West2 library.

**Query Parameters**:
- `include_indian` (bool, default `true`): Include historical Indian catalog.
- `limit_peer` (int, default `50`): Maximum PEER records to return.

**Response `200 OK`**:
```json
{
  "indian_catalog": [
    {
      "id": "IND-2001-BHUJ",
      "name": "Bhuj Earthquake",
      "year": 2001,
      "magnitude": 7.7,
      "depth_km": 16.0,
      "latitude": 23.419,
      "longitude": 70.232,
      "is1893_zone": "Zone V",
      "region_state": "Gujarat (Kachchh)",
      "tectonic_regime": "Intraplate Reverse Faulting (Kachchh Rift)",
      "associated_peer_proxy": "RSN0001_Imperial_Valley-06.AT2"
    }
  ],
  "peer_records": [
    {
      "record_id": "RSN0001",
      "filename": "RSN0001_Imperial_Valley-06.AT2",
      "earthquake_name": "Imperial Valley-06",
      "year": 1979,
      "magnitude": 6.67,
      "raw_pga_g": 0.151,
      "raw_dt": 0.005
    }
  ],
  "total_indian": 12,
  "total_peer": 50
}
```

### `GET /api/v1/buildings`
Lists standard structural building archetypes (RC frame, steel MRF, shear wall, masonry, isolated).

**Response `200 OK`**:
```json
{
  "buildings": [
    {
      "id": "BLD-RC-03",
      "name": "3-Story Reinforced Concrete Moment Frame",
      "structural_system": "Special RC Moment-Resisting Frame (SMRF)",
      "stories": 3,
      "total_height_m": 9.6,
      "fundamental_period_s": 0.45,
      "damping_ratio": 0.05,
      "yield_displacement_m": 0.012,
      "post_yield_ratio": 0.05,
      "material_type": "bilinear"
    }
  ],
  "total": 5
}
```

---

## 3. Digital-Twin Simulation & Validation

### `POST /api/v1/scenario/validate`
Validates scenario physical parameters and alerts for out-of-training domain conditions.

**Request Body**:
```json
{
  "earthquake_id": "RSN0001_Imperial_Valley-06.AT2",
  "pga_g": 0.40,
  "T0": 0.50,
  "damping_ratio": 0.05,
  "yield_displacement_m": 0.010,
  "post_yield_ratio": 0.05,
  "material_type": "bilinear"
}
```

**Response `200 OK`**:
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [],
  "domain_status": "IN_DOMAIN",
  "computed_properties": {
    "omega_n_rad_s": 12.5664,
    "stiffness_k0_N_m": 157.91,
    "yield_force_N": 1.58,
    "frequency_hz": 2.0
  }
}
```

### `POST /api/v1/scenario/predict`
Executes sub-millisecond continuous Fourier Neural Operator surrogate inference, computes civil engineering metrics, and optionally executes OpenSeesPy ground truth.

**Request Body**:
```json
{
  "earthquake_id": "RSN0001_Imperial_Valley-06.AT2",
  "pga_g": 0.40,
  "T0": 0.50,
  "damping_ratio": 0.05,
  "yield_displacement_m": 0.010,
  "post_yield_ratio": 0.05,
  "material_type": "bilinear",
  "mass_kg": 1.0,
  "building_height_m": 9.6,
  "include_ground_truth": true,
  "stride": 2
}
```

**Response `200 OK`**:
```json
{
  "scenario": { ... },
  "trajectories": {
    "time": [0.0, 0.02, 0.04, ...],
    "ag": [0.0, 0.012, 0.045, ...],
    "u": [0.0, -0.0002, ...],
    "v": [0.0, -0.012, ...],
    "fr": [0.0, -0.031, ...],
    "up": [0.0, 0.0, ...],
    "eh": [0.0, 0.0, 0.012, ...],
    "u_gt": [0.0, -0.00021, ...]
  },
  "metrics": {
    "peak_displacement_m": 0.0555,
    "peak_displacement_mm": 55.51,
    "drift_ratio_percent": 0.578,
    "ductility_demand_mu": 4.63,
    "yield_time_sec": 1.45,
    "peak_restoring_force_N": 3.45,
    "total_hysteretic_energy_J": 0.35
  },
  "validation": {
    "relative_l2_u_percent": 55.18,
    "peak_u_error_percent": 4.95,
    "rmse_u_mm": 11.23,
    "speedup_factor": 2.3,
    "fno_latency_ms": 8.67,
    "opensees_latency_ms": 19.56,
    "benchmark_status": "Empirically Measured Ground Truth"
  },
  "inference_time_ms": 8.67,
  "is_mock": false,
  "uncertainty_note": "Uncertainty estimation: research module pending",
  "model_provenance": {
    "source": "SeismoFNO Continuous Spectral Operator",
    "checkpoint": "experiments/phase5_c_data_energy_boundary/checkpoints/best_model.pt",
    "device": "mps",
    "modes": 128
  }
}
```
