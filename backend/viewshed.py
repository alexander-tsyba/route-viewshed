import math
import numpy as np
from typing import List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from elevation import ElevationProvider
from landcover import LandCoverProvider, OBSTACLE_HEIGHT

EARTH_RADIUS = 6_371_000  # meters
OBSERVER_HEIGHT = 1.5  # eye height in car
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
    fan_polys = []

    for idx, pt in enumerate(sampled_points):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx, len(sampled_points))

        fan = _compute_single_viewshed_fan(
            pt["lon"], pt["lat"], pt["bearing"],
            elevation, prepared_obstacles,
            max_distance, ray_step_deg, distance_step,
        )
        if fan is not None:
            fan_polys.append(fan)

    if not fan_polys:
        return []

    # Union all fan polygons and simplify
    merged = unary_union(fan_polys)
    # Simplify: ~50m tolerance in degrees
    tol = 50 / 111_320
    simplified = merged.simplify(tol)

    rings = []
    if isinstance(simplified, Polygon):
        if not simplified.is_empty:
            rings.append([[c[0], c[1]] for c in simplified.exterior.coords])
    elif isinstance(simplified, MultiPolygon):
        for poly in simplified.geoms:
            if not poly.is_empty:
                rings.append([[c[0], c[1]] for c in poly.exterior.coords])

    return rings


def _compute_single_viewshed_fan(
    lon: float, lat: float, bearing: float,
    elevation: ElevationProvider,
    prepared_obstacles,
    max_distance: float,
    ray_step_deg: float,
    distance_step: float,
) -> Optional[Polygon]:
    """Cast rays from a single point, return a fan-shaped polygon of visible area."""

    observer_elev = elevation.get_elevation(lat, lon) + OBSERVER_HEIGHT

    # Geometric horizon distance from this elevation
    horizon_dist = min(
        max_distance,
        math.sqrt(2 * EARTH_RADIUS * max(observer_elev, OBSERVER_HEIGHT)) + 1000
    )

    # 180 degree arc centered on bearing (direction of travel)
    start_angle = (bearing - 90) % 360
    n_rays = int(180 / ray_step_deg)

    lat_deg_per_m = 1 / 111_320
    lon_deg_per_m = 1 / (111_320 * max(math.cos(math.radians(lat)), 0.001))

    # For each ray, find the maximum visible distance
    boundary_points = []

    for i in range(n_rays + 1):
        angle = (start_angle + i * ray_step_deg) % 360
        max_visible_dist = _cast_ray_max_distance(
            lon, lat, observer_elev, angle,
            horizon_dist, distance_step,
            elevation, prepared_obstacles,
            lat_deg_per_m, lon_deg_per_m,
        )

        # Convert max visible distance to a point
        angle_rad = math.radians(angle)
        end_lon = lon + max_visible_dist * math.sin(angle_rad) * lon_deg_per_m
        end_lat = lat + max_visible_dist * math.cos(angle_rad) * lat_deg_per_m
        boundary_points.append((end_lon, end_lat))

    # Build fan polygon: origin + boundary arc + back to origin
    coords = [(lon, lat)] + boundary_points + [(lon, lat)]

    try:
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 0:
            return poly
        # Try to fix
        poly = poly.buffer(0)
        if isinstance(poly, Polygon) and poly.area > 0:
            return poly
        if isinstance(poly, MultiPolygon):
            biggest = max(poly.geoms, key=lambda g: g.area)
            return biggest
    except Exception:
        pass

    return None


def _cast_ray_max_distance(
    origin_lon: float, origin_lat: float,
    observer_elev: float,
    angle_deg: float,
    max_dist: float,
    step: float,
    elevation: ElevationProvider,
    prepared_obstacles,
    lat_deg_per_m: float,
    lon_deg_per_m: float,
) -> float:
    """Cast a single ray and return the maximum visible distance along it.

    Uses the "maximum angle" algorithm: a point is visible if the angle from
    the observer to it (accounting for curvature) exceeds all previous angles.
    Returns the distance of the farthest visible point.
    """
    max_tan_angle = float("-inf")
    max_visible_dist = step  # at minimum, can see the first step

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    dist = step
    blocked = False

    while dist <= max_dist:
        dlat = dist * cos_angle * lat_deg_per_m
        dlon = dist * sin_angle * lon_deg_per_m
        target_lat = origin_lat + dlat
        target_lon = origin_lon + dlon

        terrain_elev = elevation.get_elevation(target_lat, target_lon)

        effective_elev = terrain_elev
        if prepared_obstacles is not None:
            try:
                if prepared_obstacles.contains(Point(target_lon, target_lat)):
                    effective_elev += OBSTACLE_HEIGHT
            except Exception:
                pass

        # Earth curvature correction
        curvature_drop = (dist * dist) / (2 * EARTH_RADIUS)

        apparent_elev = effective_elev - curvature_drop
        delta_elev = apparent_elev - observer_elev
        tan_angle = delta_elev / dist

        if tan_angle >= max_tan_angle:
            # Visible — update max visible distance
            max_visible_dist = dist
            max_tan_angle = tan_angle
            blocked = False
        else:
            # Hidden — but keep going, might see over a ridge
            if not blocked and (effective_elev + OBSTACLE_HEIGHT - curvature_drop - observer_elev) / dist > max_tan_angle + 0.05:
                # Solid tall obstacle — stop
                break
            blocked = True

        dist += step

    return max_visible_dist
