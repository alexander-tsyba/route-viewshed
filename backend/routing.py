import httpx
import math
from typing import List, Tuple

OSRM_BASE = "https://router.project-osrm.org"


async def get_route(start: Tuple[float, float], end: Tuple[float, float]) -> dict:
    """Get route from OSRM. Coordinates are (lon, lat)."""
    url = (
        f"{OSRM_BASE}/route/v1/driving/"
        f"{start[0]},{start[1]};{end[0]},{end[1]}"
        f"?overview=full&geometries=geojson&steps=false"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    if data["code"] != "Ok":
        raise ValueError(f"OSRM error: {data.get('message', data['code'])}")

    route = data["routes"][0]
    coords = route["geometry"]["coordinates"]  # list of [lon, lat]
    distance_m = route["distance"]
    duration_s = route["duration"]

    return {
        "coordinates": coords,
        "distance_m": distance_m,
        "duration_s": duration_s,
    }


def sample_route_points(
    coords: List[List[float]], route_distance_m: float
) -> List[dict]:
    """Sample points along the route at adaptive intervals.

    Returns list of {lon, lat, bearing} dicts.
    bearing is the direction of travel in degrees (0=north, 90=east).
    """
    if route_distance_m <= 50_000:
        interval_m = 300
    elif route_distance_m <= 200_000:
        interval_m = 500
    elif route_distance_m <= 1_000_000:
        interval_m = 1000
    else:
        interval_m = 2000

    max_points = 3000
    if route_distance_m / interval_m > max_points:
        interval_m = route_distance_m / max_points

    sampled = []
    accumulated = 0.0

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        seg_dist = _haversine(lat1, lon1, lat2, lon2)
        bearing = _bearing(lat1, lon1, lat2, lon2)

        while accumulated <= seg_dist:
            frac = accumulated / seg_dist if seg_dist > 0 else 0
            lon = lon1 + frac * (lon2 - lon1)
            lat = lat1 + frac * (lat2 - lat1)
            sampled.append({"lon": lon, "lat": lat, "bearing": bearing})
            accumulated += interval_m

        accumulated -= seg_dist

    return sampled


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
