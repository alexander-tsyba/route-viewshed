import math
import numpy as np
from typing import List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from elevation import ElevationProvider
from landcover import LandCoverProvider, OBSTACLE_HEIGHT

EARTH_RADIUS = 6_371_000  # meters
OBSERVER_HEIGHT = 1.5  # eye height in car
TARGET_HEIGHT = 2.0  # consider a point "seen" if a 2m feature there is visible
RAY_ANGULAR_STEP = 2.0  # degrees between rays

# Ray step sizes: finer near observer (catches buildings), coarser far away
RAY_STEP_NEAR = 30  # meters — within NEAR_DISTANCE (matches SRTM pixel)
RAY_STEP_FAR = 120  # meters — beyond NEAR_DISTANCE
NEAR_DISTANCE = 2000  # meters — fine step within this radius

# Minimum polygon area to keep (km²)
MIN_POLYGON_AREA_KM2 = 0.05


def _atmospheric_max(observer_elev_asl: float) -> float:
    """Elevation-dependent atmospheric visibility limit.

    At sea level: ~30km. At 2000m: ~60km. Capped at 80km.
    """
    base = 30_000
    bonus = min(observer_elev_asl, 3000) * 16
    return min(base + bonus, 80_000)


def compute_viewshed_for_route(
    sampled_points: List[dict],
    elevation: ElevationProvider,
    landcover: LandCoverProvider,
    ray_step_deg: float = RAY_ANGULAR_STEP,
    progress_callback=None,
) -> List[List[List[float]]]:
    """Compute visible area polygons for all sampled route points."""
    prepared_obstacles = landcover.get_prepared()
    fan_polys = []

    for idx, pt in enumerate(sampled_points):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx, len(sampled_points))

        fan = _compute_single_viewshed_fan(
            pt["lon"], pt["lat"], pt["bearing"],
            elevation, prepared_obstacles,
            ray_step_deg,
        )
        if fan is not None:
            fan_polys.append(fan)

    if not fan_polys:
        return []

    merged = unary_union(fan_polys)

    # Adaptive morphological smoothing
    area_deg2 = merged.area
    area_km2_approx = area_deg2 * (111.32 ** 2)

    if area_km2_approx > 50:
        erode_m = 150
    elif area_km2_approx > 5:
        erode_m = 80
    else:
        erode_m = 30

    erode_deg = erode_m / 111_320

    opened = merged.buffer(-erode_deg).buffer(erode_deg)
    if opened.is_empty:
        opened = merged

    close_deg = erode_deg * 0.7
    closed = opened.buffer(close_deg).buffer(-close_deg)
    if closed.is_empty:
        closed = opened

    tol = max(30, erode_m * 0.5) / 111_320
    simplified = closed.simplify(tol)

    min_area_deg2 = MIN_POLYGON_AREA_KM2 / (111.32 ** 2)
    return _extract_rings(simplified, min_area_deg2)


def _extract_rings(geom, min_area: float = 0) -> List[List[List[float]]]:
    """Extract polygon exterior rings, filtering tiny fragments."""
    rings = []
    if isinstance(geom, Polygon):
        if not geom.is_empty and geom.area >= min_area:
            rings.append([[c[0], c[1]] for c in geom.exterior.coords])
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            if not poly.is_empty and poly.area >= min_area:
                rings.append([[c[0], c[1]] for c in poly.exterior.coords])
    return rings


def _compute_single_viewshed_fan(
    lon: float, lat: float, bearing: float,
    elevation: ElevationProvider,
    prepared_obstacles,
    ray_step_deg: float,
) -> Optional[Polygon]:
    """Cast rays from a single point, return a fan-shaped polygon."""

    observer_ground = elevation.get_elevation(lat, lon)
    observer_elev = observer_ground + OBSERVER_HEIGHT

    h_above_ground = OBSERVER_HEIGHT
    geometric_horizon = (
        math.sqrt(2 * EARTH_RADIUS * h_above_ground)
        + math.sqrt(2 * EARTH_RADIUS * TARGET_HEIGHT)
    )
    atmo_max = _atmospheric_max(observer_ground)
    max_ray_dist = min(atmo_max, geometric_horizon * 3)

    start_angle = (bearing - 90) % 360
    n_rays = int(180 / ray_step_deg)

    lat_deg_per_m = 1 / 111_320
    lon_deg_per_m = 1 / (111_320 * max(math.cos(math.radians(lat)), 0.001))

    ray_angles = []
    ray_dists = []

    for i in range(n_rays + 1):
        angle = (start_angle + i * ray_step_deg) % 360
        max_visible_dist = _cast_ray_adaptive(
            lon, lat, observer_elev, angle,
            max_ray_dist,
            elevation, prepared_obstacles,
            lat_deg_per_m, lon_deg_per_m,
        )
        ray_angles.append(angle)
        ray_dists.append(max_visible_dist)

    smoothed = _smooth_ray_distances(ray_dists, window=5)

    boundary_points = []
    for angle, dist in zip(ray_angles, smoothed):
        angle_rad = math.radians(angle)
        end_lon = lon + dist * math.sin(angle_rad) * lon_deg_per_m
        end_lat = lat + dist * math.cos(angle_rad) * lat_deg_per_m
        boundary_points.append((end_lon, end_lat))

    coords = [(lon, lat)] + boundary_points + [(lon, lat)]

    try:
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 0:
            return poly
        poly = poly.buffer(0)
        if isinstance(poly, Polygon) and poly.area > 0:
            return poly
        if isinstance(poly, MultiPolygon):
            biggest = max(poly.geoms, key=lambda g: g.area)
            return biggest
    except Exception:
        pass

    return None


def _smooth_ray_distances(dists: List[float], window: int = 5) -> List[float]:
    """Median filter to remove noise spikes."""
    n = len(dists)
    if n <= window:
        return dists

    half = window // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(float(np.median(dists[lo:hi])))
    return result


def _cast_ray_adaptive(
    origin_lon: float, origin_lat: float,
    observer_elev: float,
    angle_deg: float,
    max_dist: float,
    elevation: ElevationProvider,
    prepared_obstacles,
    lat_deg_per_m: float,
    lon_deg_per_m: float,
) -> float:
    """Cast a ray with adaptive step size.

    Near the observer (< NEAR_DISTANCE): use fine 30m steps to detect
    buildings and forest edges accurately.
    Far from observer: use coarse 120m steps for performance.

    When hitting an obstacle, the ray is blocked — obstacles are opaque
    walls that terminate visibility in that direction.
    """
    max_tan_angle = float("-inf")
    max_visible_dist = RAY_STEP_NEAR  # minimum visibility

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    consecutive_blocked = 0
    BLOCKED_THRESHOLD = 15
    obstacle_consecutive = 0

    dist = RAY_STEP_NEAR
    while dist <= max_dist:
        # Adaptive step: fine near, coarse far
        step = RAY_STEP_NEAR if dist <= NEAR_DISTANCE else RAY_STEP_FAR

        dlat = dist * cos_angle * lat_deg_per_m
        dlon = dist * sin_angle * lon_deg_per_m
        target_lat = origin_lat + dlat
        target_lon = origin_lon + dlon

        terrain_elev = elevation.get_elevation(target_lat, target_lon)

        # Check obstacle
        in_obstacle = False
        if prepared_obstacles is not None:
            try:
                in_obstacle = prepared_obstacles.contains(Point(target_lon, target_lat))
            except Exception:
                pass

        if in_obstacle:
            # Forest or building — opaque obstacle
            effective_elev = terrain_elev + OBSTACLE_HEIGHT
            curvature_drop = (dist * dist) / (2 * EARTH_RADIUS)
            surface_tan = (effective_elev - curvature_drop - observer_elev) / dist

            if surface_tan > max_tan_angle:
                max_tan_angle = surface_tan

            obstacle_consecutive += 1
            consecutive_blocked += 1

            # After 3 consecutive obstacle hits (90m at 30m step), view is blocked
            if obstacle_consecutive >= 3:
                consecutive_blocked = BLOCKED_THRESHOLD
                break

            dist += step
            continue

        obstacle_consecutive = 0

        effective_elev = terrain_elev
        curvature_drop = (dist * dist) / (2 * EARTH_RADIUS)

        surface_apparent = effective_elev - curvature_drop
        surface_tan = (surface_apparent - observer_elev) / dist

        target_apparent = (terrain_elev + TARGET_HEIGHT) - curvature_drop
        target_tan = (target_apparent - observer_elev) / dist

        noise_tolerance = 3.0 / dist

        if target_tan >= max_tan_angle - noise_tolerance:
            max_visible_dist = dist
            consecutive_blocked = 0

        if surface_tan > max_tan_angle:
            max_tan_angle = surface_tan

        if target_tan < max_tan_angle - noise_tolerance:
            consecutive_blocked += 1
            if consecutive_blocked >= BLOCKED_THRESHOLD:
                break
        else:
            consecutive_blocked = 0

        dist += step

    return max_visible_dist
