import asyncio
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from routing import get_route, sample_route_points
from elevation import ElevationProvider
from landcover import LandCoverProvider
from viewshed import compute_viewshed_for_route, ATMOSPHERIC_MAX
from geocode import geocode

app = FastAPI(title="Route Viewshed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

elevation_provider = ElevationProvider()
landcover_provider = LandCoverProvider()

# Fixed buffer for data downloads — 30km matches atmospheric max
ELEVATION_BUFFER_KM = ATMOSPHERIC_MAX / 1000  # 30
LANDCOVER_BUFFER_KM = 5  # forests/buildings beyond 5km negligible


class RouteRequest(BaseModel):
    start: str  # place name or "lat,lon"
    end: str
    include_landcover: Optional[bool] = True


class RouteResponse(BaseModel):
    route_coords: list  # [[lon, lat], ...]
    sampled_points: list  # [{lon, lat, bearing}, ...]
    viewshed_polygons: list  # [[[lon, lat], ...], ...]
    stats: dict


async def _parse_location(loc: str):
    """Parse 'lat,lon' or geocode a place name."""
    loc = loc.strip()
    parts = loc.split(",")
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lon, lat)
        except ValueError:
            pass
    return await geocode(loc)


@app.post("/api/viewshed", response_model=RouteResponse)
async def compute_route_viewshed(req: RouteRequest):
    t0 = time.time()

    # 1. Geocode start/end
    try:
        start_coord = await _parse_location(req.start)
        end_coord = await _parse_location(req.end)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 2. Get route
    try:
        route = await get_route(start_coord, end_coord)
    except Exception as e:
        raise HTTPException(400, f"Routing failed: {e}")

    coords = route["coordinates"]
    distance_m = route["distance_m"]
    t_route = time.time()

    # 3. Sample points
    sampled = sample_route_points(coords, distance_m)
    t_sample = time.time()

    # 4. Download elevation data (30km buffer for atmospheric max)
    await elevation_provider.preload_tiles_for_route(coords, buffer_km=ELEVATION_BUFFER_KM)
    t_elev = time.time()

    # 5. Load land cover (forests/buildings) — 5km buffer
    if req.include_landcover:
        await landcover_provider.load_obstacles_for_route(coords, buffer_km=LANDCOVER_BUFFER_KM)
    t_lc = time.time()

    # 6. Compute viewshed
    viewshed_polygons = await asyncio.to_thread(
        compute_viewshed_for_route,
        sampled,
        elevation_provider,
        landcover_provider,
    )
    t_viewshed = time.time()

    return RouteResponse(
        route_coords=coords,
        sampled_points=sampled,
        viewshed_polygons=viewshed_polygons,
        stats={
            "route_distance_km": round(distance_m / 1000, 1),
            "route_duration_min": round(route["duration_s"] / 60, 1),
            "sampled_points": len(sampled),
            "timing": {
                "routing_s": round(t_route - t0, 2),
                "sampling_s": round(t_sample - t_route, 2),
                "elevation_download_s": round(t_elev - t_sample, 2),
                "landcover_download_s": round(t_lc - t_elev, 2),
                "viewshed_compute_s": round(t_viewshed - t_lc, 2),
                "total_s": round(t_viewshed - t0, 2),
            },
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}
