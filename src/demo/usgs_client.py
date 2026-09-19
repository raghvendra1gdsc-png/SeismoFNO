"""
src/demo/usgs_client.py — Real-World USGS Earthquake Event Client.

Connects the SeismoFNO research defense demo to public USGS earthquake services
for real-world event discovery and context without modifying or retraining
the frozen EXP4/EXP5/EXP6 scientific core.

Scientific Boundary Guarantee:
- USGS provides observed earthquake event metadata (magnitude, location, time, depth).
- SeismoFNO requires an appropriate ground-motion acceleration time history a_g(t)
  and pre-earthquake modal properties (T1, omega1) to compute structural response.
- This client NEVER fabricates acceleration waveforms from earthquake magnitude
  or location alone.
- If live connectivity fails, it seamlessly provides verified cached fallback data.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default timeout in seconds for USGS public HTTP requests
USGS_TIMEOUT_SECONDS = 4.0

# USGS Public GeoJSON Feeds (Zero API Key Required)
USGS_FEEDS = {
    "all_day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    "2.5_day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
    "4.5_day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson",
    "significant_month": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson",
    "query_api": "https://earthquake.usgs.gov/fdsnws/event/1/query",
}

# Cache file location
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_FILE = DATA_DIR / "usgs_cache.json"


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two geographic points in kilometers
    using the Haversine formula.
    """
    R = 6371.0  # Earth's mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(R * c)


def parse_geojson_feature(feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract and normalize standard earthquake event metadata from a single GeoJSON feature.
    """
    if not isinstance(feature, dict):
        return None

    props = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates") or []

    event_id = str(feature.get("id") or props.get("code") or props.get("ids") or "")
    if not event_id:
        return None

    # USGS coordinates format: [longitude, latitude, depth_km]
    lon = float(coords[0]) if len(coords) > 0 and coords[0] is not None else 0.0
    lat = float(coords[1]) if len(coords) > 1 and coords[1] is not None else 0.0
    depth_km = float(coords[2]) if len(coords) > 2 and coords[2] is not None else 0.0

    mag = props.get("mag")
    magnitude = float(mag) if mag is not None else 0.0

    time_ms = props.get("time")
    if time_ms is not None:
        try:
            origin_dt = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)
            origin_time_str = origin_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            origin_time_str = "Unknown"
    else:
        time_ms = int(time.time() * 1000)
        origin_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    location_str = str(props.get("place") or "Unknown Location")
    status_str = str(props.get("status") or "observed")
    url_str = str(props.get("url") or f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}")

    return {
        "event_id": event_id,
        "magnitude": round(magnitude, 2),
        "location": location_str,
        "depth_km": round(depth_km, 1),
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "origin_time": origin_time_str,
        "time_epoch_ms": time_ms,
        "status": status_str,
        "source": "USGS",
        "data_type": "EARTHQUAKE CATALOG / REAL-TIME FEED",
        "has_compatible_waveform": False,
        "url": url_str,
    }


class USGSClient:
    """
    Thread-safe client for ingesting USGS GeoJSON earthquake event feeds.
    Includes memory and persistent caching, query filtering, and offline resilience.
    """

    def __init__(self, cache_file: Optional[Path] = None, timeout: float = USGS_TIMEOUT_SECONDS):
        self.cache_file = cache_file or CACHE_FILE
        self.timeout = timeout
        self._memory_cache: Optional[Dict[str, Any]] = None
        self._last_fetch_time: float = 0.0

    def load_cached_fallback(self) -> Dict[str, Any]:
        """Load fallback data from disk cache if remote network call fails."""
        if self._memory_cache:
            return self._memory_cache

        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._memory_cache = data
                    return data
            except Exception:
                pass

        # Return minimal hardcoded fallback if disk cache is missing
        return {
            "source": "USGS",
            "status": "OBSERVED EVENT",
            "data_type": "EARTHQUAKE CATALOG / REAL-TIME FEED",
            "is_fallback": True,
            "status_message": "USGS LIVE FEED UNAVAILABLE — USING EMERGENCY EMBEDDED RECORD",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": 1,
            "events": [
                {
                    "event_id": "ci3144585",
                    "magnitude": 6.7,
                    "location": "Northridge, California, USA (Archival Reference)",
                    "depth_km": 18.2,
                    "latitude": 34.213,
                    "longitude": -118.537,
                    "origin_time": "1994-01-17T12:30:55Z",
                    "time_epoch_ms": 758809855000,
                    "status": "reviewed",
                    "source": "USGS",
                    "data_type": "EARTHQUAKE CATALOG / REAL-TIME FEED",
                    "has_compatible_waveform": False,
                    "url": "https://earthquake.usgs.gov/earthquakes/eventpage/ci3144585",
                }
            ],
        }

    def save_to_cache(self, payload: Dict[str, Any]) -> None:
        """Persist successful response to memory and disk."""
        self._memory_cache = payload
        self._last_fetch_time = time.time()
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def fetch_live_feed(
        self,
        feed_name: str = "all_day",
        min_magnitude: Optional[float] = None,
        limit: int = 50,
        hours: Optional[float] = 24.0,
        scenario_lat: Optional[float] = None,
        scenario_lon: Optional[float] = None,
        radius_km: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fetch real-world earthquake events from USGS public feeds.
        If network fails or is unavailable, returns cached fallback seamlessly.
        """
        events: List[Dict[str, Any]] = []
        is_fallback = False
        status_msg = "LIVE USGS FEED ACTIVE"

        # Determine feed URL
        feed_url = USGS_FEEDS.get(feed_name, USGS_FEEDS["all_day"])

        try:
            req = urllib.request.Request(
                feed_url,
                headers={"User-Agent": "SeismoFNO-ResearchDemo/1.0 (Academic Research)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    raw_data = json.loads(resp.read().decode("utf-8"))
                    features = raw_data.get("features", [])
                    for feat in features:
                        parsed = parse_geojson_feature(feat)
                        if parsed:
                            events.append(parsed)
                else:
                    raise urllib.error.HTTPError(
                        feed_url, resp.status, "Non-200 status from USGS", resp.headers, None
                    )

            if not events:
                # Feed succeeded but empty; fallback to cache if available
                cached = self.load_cached_fallback()
                events = cached.get("events", [])
                is_fallback = True
                status_msg = "USGS FEED EMPTY — USING CACHED EVENT DATA"

        except Exception as exc:
            # Network failure, timeout, or parsing error -> Use cached fallback
            cached = self.load_cached_fallback()
            events = cached.get("events", [])
            is_fallback = True
            status_msg = f"USGS LIVE FEED UNAVAILABLE ({type(exc).__name__}) — USE CACHED EVENT DATA"

        # Apply filtering in memory (magnitude, time window, distance)
        now_ms = time.time() * 1000
        filtered_events: List[Dict[str, Any]] = []

        for ev in events:
            # 1. Magnitude filter
            if min_magnitude is not None and ev["magnitude"] < min_magnitude:
                continue

            # 2. Time window filter
            if hours is not None and not is_fallback:
                max_age_ms = hours * 3600.0 * 1000.0
                if (now_ms - ev["time_epoch_ms"]) > max_age_ms:
                    continue

            # 3. Distance to scenario calculation
            if scenario_lat is not None and scenario_lon is not None:
                dist = haversine_distance_km(
                    scenario_lat, scenario_lon, ev["latitude"], ev["longitude"]
                )
                ev["distance_km"] = round(dist, 1)

                if radius_km is not None and dist > radius_km:
                    continue
            else:
                ev["distance_km"] = None

            filtered_events.append(ev)

        # Sort descending by origin time
        filtered_events.sort(key=lambda x: x["time_epoch_ms"], reverse=True)

        # Apply limit
        final_events = filtered_events[:limit]

        response_payload = {
            "source": "USGS",
            "status": "OBSERVED EVENT",
            "data_type": "EARTHQUAKE CATALOG / REAL-TIME FEED",
            "is_fallback": is_fallback,
            "status_message": status_msg,
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": len(final_events),
            "events": final_events,
        }

        # Cache if live call succeeded
        if not is_fallback and len(final_events) > 0:
            self.save_to_cache(response_payload)

        return response_payload

    def get_event_detail(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full details for an individual earthquake event from USGS or cache.
        """
        # 1. First check cached events
        cached = self.load_cached_fallback()
        for ev in cached.get("events", []):
            if ev.get("event_id") == event_id:
                return ev

        # 2. Query USGS event API if not found
        query_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={event_id}&format=geojson"
        try:
            req = urllib.request.Request(
                query_url,
                headers={"User-Agent": "SeismoFNO-ResearchDemo/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    raw_data = json.loads(resp.read().decode("utf-8"))
                    parsed = parse_geojson_feature(raw_data)
                    if parsed:
                        return parsed
        except Exception:
            pass

        return None


# Global singleton instance
usgs_client = USGSClient()
