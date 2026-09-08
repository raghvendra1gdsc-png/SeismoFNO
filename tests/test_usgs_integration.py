"""
tests/test_usgs_integration.py — Validation Suite for USGS Live Earthquake Integration Layer.

Validates:
1. USGS GeoJSON response parsing
2. Malformed / invalid response handling
3. Network timeout handling
4. Empty feed handling
5. Event normalization & schema consistency
6. Magnitude filtering
7. Geographic filtering & Haversine distance calculations
8. Cached offline fallback resilience
9. FastAPI demo endpoints (/api/v1/demo/earthquakes/live and /{event_id})
10. Event selection & scientific waveform boundary (distinguishing metadata from waveform)
"""

import json
from unittest.mock import patch, MagicMock
import urllib.error
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.demo.usgs_client import (
    USGSClient,
    haversine_distance_km,
    parse_geojson_feature,
    usgs_client,
)
from src.demo.inference_adapter import run_demonstration_inference


@pytest.fixture
def client():
    return TestClient(app)


# Sample valid GeoJSON feature from USGS
SAMPLE_FEATURE = {
    "type": "Feature",
    "id": "us7000j197",
    "properties": {
        "mag": 7.8,
        "place": "26 km E of Nurdağı, Turkey",
        "time": 1675646254000,
        "updated": 1675650000000,
        "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000j197",
        "status": "reviewed",
        "tsunami": 0,
        "sig": 995,
        "net": "us",
        "code": "7000j197",
    },
    "geometry": {
        "type": "Point",
        "coordinates": [37.042, 37.166, 10.0],
    },
}


# 1. USGS Response Parsing
def test_usgs_response_parsing():
    """Verify that a standard USGS GeoJSON feature is parsed into normalized dictionary."""
    parsed = parse_geojson_feature(SAMPLE_FEATURE)
    assert parsed is not None
    assert parsed["event_id"] == "us7000j197"
    assert parsed["magnitude"] == 7.8
    assert parsed["location"] == "26 km E of Nurdağı, Turkey"
    assert parsed["depth_km"] == 10.0
    assert parsed["latitude"] == 37.166
    assert parsed["longitude"] == 37.042
    assert parsed["status"] == "reviewed"
    assert parsed["source"] == "USGS"
    assert parsed["data_type"] == "EARTHQUAKE CATALOG / REAL-TIME FEED"
    assert parsed["has_compatible_waveform"] is False
    assert "2023-02-06" in parsed["origin_time"]


# 2. Malformed Response Handling
def test_malformed_response_handling():
    """Verify that malformed or incomplete GeoJSON records are handled without crashing."""
    # Non-dictionary input
    assert parse_geojson_feature(None) is None
    assert parse_geojson_feature([]) is None
    assert parse_geojson_feature("invalid") is None

    # Missing ID
    assert parse_geojson_feature({"properties": {"mag": 5.0}, "geometry": {"coordinates": [0, 0, 0]}}) is None

    # Missing coordinates
    feat_missing_coords = {
        "id": "test001",
        "properties": {"mag": 5.2, "place": "Test Loc", "time": 1000000},
        "geometry": {},
    }
    parsed = parse_geojson_feature(feat_missing_coords)
    assert parsed is not None
    assert parsed["latitude"] == 0.0
    assert parsed["longitude"] == 0.0
    assert parsed["depth_km"] == 0.0

    # Non-numeric magnitude or None
    feat_none_mag = {
        "id": "test002",
        "properties": {"mag": None, "place": "Test Loc 2", "time": 1000000},
        "geometry": {"coordinates": [10.5, 20.5, 5.0]},
    }
    parsed_none = parse_geojson_feature(feat_none_mag)
    assert parsed_none is not None
    assert parsed_none["magnitude"] == 0.0


# 3. Network Timeout
def test_network_timeout():
    """Verify that remote network timeouts trigger cached offline fallback cleanly."""
    client_instance = USGSClient(timeout=0.01)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        res = client_instance.fetch_live_feed()
        assert res is not None
        assert res["is_fallback"] is True
        assert "USGS LIVE FEED UNAVAILABLE" in res["status_message"]
        assert res["count"] > 0
        assert len(res["events"]) > 0


# 4. Empty Feed
def test_empty_feed():
    """Verify that empty GeoJSON response falls back to cached events."""
    client_instance = USGSClient()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"type": "FeatureCollection", "features": []}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client_instance.fetch_live_feed()
        assert res is not None
        assert res["is_fallback"] is True
        assert "EMPTY" in res["status_message"]
        assert res["count"] > 0


# 5. Event Normalization & Schema Consistency
def test_event_normalization():
    """Verify that every event adheres strictly to the required scientific schema."""
    fallback_data = usgs_client.load_cached_fallback()
    events = fallback_data.get("events", [])
    assert len(events) >= 10

    required_keys = {
        "event_id",
        "magnitude",
        "location",
        "depth_km",
        "latitude",
        "longitude",
        "origin_time",
        "time_epoch_ms",
        "status",
        "source",
        "data_type",
        "has_compatible_waveform",
        "url",
    }

    for ev in events:
        for k in required_keys:
            assert k in ev, f"Missing required key '{k}' in event {ev}"
        assert ev["source"] == "USGS"
        assert ev["data_type"] == "EARTHQUAKE CATALOG / REAL-TIME FEED"
        assert ev["has_compatible_waveform"] is False
        assert isinstance(ev["magnitude"], (int, float))
        assert isinstance(ev["depth_km"], (int, float))


# 6. Magnitude Filtering
def test_magnitude_filtering():
    """Verify magnitude threshold filtering."""
    client_instance = USGSClient()
    res_m7 = client_instance.fetch_live_feed(min_magnitude=7.0)
    for ev in res_m7["events"]:
        assert ev["magnitude"] >= 7.0, f"Event {ev['event_id']} magnitude {ev['magnitude']} < 7.0"

    res_m6 = client_instance.fetch_live_feed(min_magnitude=6.5)
    for ev in res_m6["events"]:
        assert ev["magnitude"] >= 6.5


# 7. Geographic Filtering & Haversine Distance
def test_geographic_filtering():
    """Verify great-circle distance calculation and radius filtering."""
    # San Francisco (37.7749, -122.4194) to Los Angeles (34.0522, -118.2437) ~ 559 km
    dist_sf_la = haversine_distance_km(37.7749, -122.4194, 34.0522, -118.2437)
    assert 550.0 <= dist_sf_la <= 570.0

    # New Delhi (28.6139, 77.2090) to Mumbai (19.0760, 72.8777) ~ 1150 km
    dist_delhi_mumbai = haversine_distance_km(28.6139, 77.2090, 19.0760, 72.8777)
    assert 1100.0 <= dist_delhi_mumbai <= 1200.0

    # Test radius filtering from Los Angeles (34.0522, -118.2437)
    client_instance = USGSClient()
    res_radius = client_instance.fetch_live_feed(
        scenario_lat=34.0522,
        scenario_lon=-118.2437,
        radius_km=1000.0,
    )
    for ev in res_radius["events"]:
        assert ev["distance_km"] is not None
        assert ev["distance_km"] <= 1000.0, f"Event {ev['location']} distance {ev['distance_km']} > 1000 km"


# 8. Cached Fallback Resilience
def test_cached_fallback():
    """Verify that cached data is valid and non-empty even when network calls are blocked."""
    cached = usgs_client.load_cached_fallback()
    assert cached["is_fallback"] is True
    assert cached["count"] >= 10
    event_ids = [e["event_id"] for e in cached["events"]]
    # Verify well-known historical reference events exist in cache
    assert "us7000j197" in event_ids  # 2023 Turkey
    assert "ci3144585" in event_ids   # 1994 Northridge
    assert "nc216859" in event_ids    # 1989 Loma Prieta
    assert "usp000a876" in event_ids  # 2001 Bhuj


# 9. FastAPI API Endpoints
def test_api_endpoints(client):
    """Verify GET /api/v1/demo/earthquakes/live and GET /api/v1/demo/earthquakes/{event_id}."""
    # Test /live endpoint
    resp_live = client.get("/api/v1/demo/earthquakes/live?limit=5")
    assert resp_live.status_code == 200
    data = resp_live.json()
    assert "events" in data
    assert "source" in data
    assert data["source"] == "USGS"
    assert data["data_type"] == "EARTHQUAKE CATALOG / REAL-TIME FEED"
    assert len(data["events"]) <= 5

    # Test /live with magnitude filter
    resp_mag = client.get("/api/v1/demo/earthquakes/live?minmagnitude=7.0")
    assert resp_mag.status_code == 200
    data_mag = resp_mag.json()
    for ev in data_mag["events"]:
        assert ev["magnitude"] >= 7.0

    # Test /{event_id} detail endpoint for existing event
    first_id = data["events"][0]["event_id"]
    resp_detail = client.get(f"/api/v1/demo/earthquakes/{first_id}")
    assert resp_detail.status_code == 200
    detail = resp_detail.json()
    assert detail["event_id"] == first_id
    assert "location" in detail
    assert "magnitude" in detail

    # Test 404 for unknown event
    resp_404 = client.get("/api/v1/demo/earthquakes/non_existent_event_99999")
    assert resp_404.status_code == 404


# 10. Frontend Event Selection & Waveform Provenance Boundary
def test_frontend_event_selection_and_waveform_boundary():
    """Verify that selecting a USGS event requires a verified research waveform

    to execute SeismoFNO, preserving the scientific provenance chain.
    """
    cached = usgs_client.load_cached_fallback()
    selected_event = cached["events"][0]

    # Explicit scientific boundary check:
    # 1. Event metadata does NOT contain compatible acceleration waveform
    assert selected_event["has_compatible_waveform"] is False

    # 2. SeismoFNO cannot infer response from magnitude/location alone;
    # it requires a verified acceleration record (e.g. RSN0001) + structural model (5S_T055)
    result = run_demonstration_inference(
        archetype_id="5S_T055",
        record_id="RSN0001",
        model_id="exp6_multimodal_gno",
        selected_floor=5,
    )

    assert result is not None
    assert len(result["time"]) > 0
    assert len(result["u_pred"]) == len(result["time"])
    assert result["metrics"]["peak_disp_err_pct"] is not None
    assert "LIVE_COMPUTED" in result["data_provenance"]["prediction"]
    assert "GROUND_TRUTH" in result["data_provenance"]["ground_truth"]
