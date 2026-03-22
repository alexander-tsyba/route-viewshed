import math
import numpy as np
from typing import List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from elevation import ElevationProvider
from landcover import LandCoverProvider, OBSTACLE_HEIGHT

EARTH_RADIUS = 6_371_000  # meters
OBSERVER_HEIGHT = 1.5  # eye height in car
# Geometric horizon at sea level: sqrt(2 * R * h) ≈ 4.4 km for h=1.5m
# But from elevated positions it can be much more.
# We cap at a practical maximum.
MAX_VIEW_DISTANCE = 30_000  # 30 km absolute max
RAY_ANGULAR_STEP = 2.0  # degrees between rays
RAY_DISTANCE_STEP = 100  # meters between samples along each ray


def compute_viewshed_for_route(
    sampled_points: List[dict],
    elevation: ElevationProvider,
    landcover: LandCoverProvider,
    max_distance: float = MAX_VIEW_DISTANCE,
    ray_step_deg: float = RAY_ANGULAR_STEP,
    distance_step: float = RAY_DISTANCE_STEP,
    progress_callback=None,
) -> List[List[List[float]]]:
    """Compute visible area polygons for all sampled route points.

    Returns a list of polygon rings (each ring = list of [lon, lat]).
    """
    prepared_obstacles = landcover.get_prepared()
    all_visible_points = []

    for idx, pt in enumerate(sampled_points):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx, len(sampled_points))

        visible = _compute_single_viewshed(
            pt["lon"], pt["lat"], pt["bearing"],
            elevation, prepared_obstacles,
            max_distance, ray_step_deg, distance_step,
        )
        all_visible_points.extend(visible)

    if not all_visible_points:
        return []

    # Build polygon from all visible points using convex hull of clusters
    # We use a buffered point union approach for efficiency
    return _build_visibility_polygon(all_visible_points, distance_step)


def _compute_single_viewshed(
    lon: float, lat: float, bearing: float,
    elevation: ElevationProvider,
    prepared_obstacles,
    max_distance: float,
    ray_step_deg: float,
    distance_step: float,
) -> List[List[float]]:
    """Cast rays from a single point and return visible [lon, lat] points."""

    observer_elev = elevation.get_elevation(lat, lon) + OBSERVER_HEIGHT

    # Geometric horizon distance from this elevation
    horizon_dist = min(
        max_distance,
        math.sqrt(2 * EARTH_RADIUS * max(observer_elev, OBSERVER_HEIGHT)) + 1000
    )

    # 180 degree arc centered on bearing (direction of travel)
    start_angle = (bearing - 90) % 360
    end_angle = (bearing + 90) % 360

    visible_points = []
    angle = start_angle
    n_rays = int(180 / ray_step_deg)

    for i in range(n_rays + 1):
        angle = (start_angle + i * ray_step_deg) % 360
        ray_points = _cast_ray(
            lon, lat, observer_elev, angle,
            horizon_dist, distance_step,
            elevation, prepared_obstacles,
        )
        visible_points.extend(ray_points)

    return visible_points


def _cast_ray(
    origin_lon: float, origin_lat: float,
    observer_elev: float,
    angle_deg: float,
    max_dist: float,
    step: float,
    elevation: ElevationProvider,
    prepared_obstacles,
) -> List[List[float]]:
    """Cast a single ray and return visible points along it."""

    visible = []
    max_tan_angle = float("-inf")  # track maximum tangent angle seen so far

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    # Convert step to approximate degree offsets
    lat_deg_per_m = 1 / 111_320
    lon_deg_per_m = 1 / (111_320 * max(math.cos(math.radians(origin_lat)), 0.001))

    dist = step
    while dist <= max_dist:
        # Destination point
        dlat = dist * cos_angle * lat_deg_per_m
        dlon = dist * sin_angle * lon_deg_per_m
        target_lat = origin_lat + dlat
        target_lon = origin_lon + dlon

        # Get terrain elevation at target
        terrain_elev = elevation.get_elevation(target_lat, target_lon)

        # Add obstacle height if in forest/building
        effective_elev = terrain_elev
        if prepared_obstacles is not None:
            try:
                if prepared_obstacles.contains(Point(target_lon, target_lat)):
                    effective_elev += OBSTACLE_HEIGHT
            except Exception:
                pass

        # Earth curvature correction: at distance d, apparent drop = d²/(2R)
        curvature_drop = (dist * dist) / (2 * EARTH_RADIUS)

        # Tangent angle from observer to this point (accounting for curvature)
        apparent_elev = effective_elev - curvature_drop
        delta_elev = apparent_elev - observer_elev
        tan_angle = delta_elev / dist

        if tan_angle >= max_tan_angle:
            # This point is visible — the line of sight clears all previous terrain
            visible.append([target_lon, target_lat])
            max_tan_angle = tan_angle
        else:
            # Hidden behind previous terrain — update max if this blocks further
            max_tan_angle = max(max_tan_angle, tan_angle)

        dist += step

    return visible


def _build_visibility_polygon(
    visible_points: List[List[float]],
    resolution: float,
) -> List[List[List[float]]]:
    """Convert visible points into polygon rings for display.

    Uses buffered point union — each visible point becomes a small circle,
    then we union them all and simplify.
    """
    if not visible_points:
        return []

    # Buffer radius: use 1.5x the ray distance step to ensure overlap between
    # adjacent ray sample points, producing a connected polygon
    buf_deg = (resolution * 1.5) / 111_320

    # Process in chunks for memory efficiency
    chunk_size = 20_000
    polys = []

    for i in range(0, len(visible_points), chunk_size):
        chunk = visible_points[i:i + chunk_size]
        points = [Point(p[0], p[1]).buffer(buf_deg, resolution=4) for p in chunk]
        if points:
            merged = unary_union(points)
            polys.append(merged)

    if not polys:
        return []

    total = unary_union(polys)
    # Aggressively simplify to reduce polygon vertex count for frontend
    simplified = total.simplify(buf_deg * 2)

    rings = []
    if isinstance(simplified, Polygon):
        rings.append([[c[0], c[1]] for c in simplified.exterior.coords])
    elif isinstance(simplified, MultiPolygon):
        for poly in simplified.geoms:
            rings.append([[c[0], c[1]] for c in poly.exterior.coords])

    return rings
