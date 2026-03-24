import math
import numpy as np
from typing import List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from elevation import ElevationProvider
from landcover import LandCoverProvider, OBSTACLE_HEIGHT

EARTH_RADIUS = 6_371_000  # meters
OBSERVER_HEIGHT = 1.5  # eye height in car
# Atmospheric visibility limit on a clear day.
# Geometric horizon is calculated per-point from actual elevation.
ATMOSPHERIC_MAX = 30_000  # 30 km — beyond this, haze hides everything
TARGET_HEIGHT = 2.0  # consider a point "seen" if a 2m feature there is visible
RAY_ANGULAR_STEP = 2.0  # degrees between rays
RAY_DISTANCE_STEP = 90  # meters between samples (aligned with SRTM ~30m × 3)


def compute_viewshed_for_route(
    sampled_points: List[dict],
    elevation: ElevationProvider,
    landcover: LandCoverProvider,
    ray_step_deg: float = RAY_ANGULAR_STEP,
    distance_step: float = RAY_DISTANCE_STEP,
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
            ray_step_deg, distance_step,
        )
        if fan is not None:
            fan_polys.append(fan)

    if not fan_polys:
        return []

    # Union all fan polygons
    merged = unary_union(fan_polys)

    # Morphological smoothing to remove thin spikes and round edges.
    #
    # Strategy: adaptive erosion based on viewshed size.
    # In cities (small viewshed), use gentle smoothing.
    # In open terrain (large viewshed), use aggressive spike removal.
    area_deg2 = merged.area
    area_km2_approx = area_deg2 * (111.32 ** 2)

    if area_km2_approx > 50:
        erode_m = 150  # open terrain: remove spikes < 300m wide
    elif area_km2_approx > 5:
        erode_m = 80   # suburban: remove spikes < 160m wide
    else:
        erode_m = 30   # dense city: only remove < 60m noise

    erode_deg = erode_m / 111_320

    # Opening: erode then dilate — removes thin spikes
    opened = merged.buffer(-erode_deg).buffer(erode_deg)
    if opened.is_empty:
        opened = merged

    # Closing: dilate then erode — fills small holes and rounds edges
    close_deg = erode_deg * 0.7
    closed = opened.buffer(close_deg).buffer(-close_deg)
    if closed.is_empty:
        closed = opened

    # Simplify to reduce vertex count
    tol = max(30, erode_m * 0.5) / 111_320
    simplified = closed.simplify(tol)

    return _extract_rings(simplified)


def _extract_rings(geom) -> List[List[List[float]]]:
    """Extract polygon exterior rings from a Shapely geometry."""
    rings = []
    if isinstance(geom, Polygon):
        if not geom.is_empty:
            rings.append([[c[0], c[1]] for c in geom.exterior.coords])
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            if not poly.is_empty:
                rings.append([[c[0], c[1]] for c in poly.exterior.coords])
    return rings


def _compute_single_viewshed_fan(
    lon: float, lat: float, bearing: float,
    elevation: ElevationProvider,
    prepared_obstacles,
    ray_step_deg: float,
    distance_step: float,
) -> Optional[Polygon]:
    """Cast rays from a single point, return a fan-shaped polygon of visible area."""

    observer_ground = elevation.get_elevation(lat, lon)
    observer_elev = observer_ground + OBSERVER_HEIGHT

    # Geometric horizon for this observer: how far can we see a TARGET_HEIGHT
    # object on flat ground?
    # d = sqrt(2*R*h_obs) + sqrt(2*R*h_target)
    h_above_ground = OBSERVER_HEIGHT  # height above local terrain
    geometric_horizon = (
        math.sqrt(2 * EARTH_RADIUS * h_above_ground)
        + math.sqrt(2 * EARTH_RADIUS * TARGET_HEIGHT)
    )
    # On higher terrain, line of sight extends further to lower areas
    # Add bonus for elevated position (above regional average)
    max_ray_dist = min(ATMOSPHERIC_MAX, geometric_horizon * 3)

    start_angle = (bearing - 90) % 360
    n_rays = int(180 / ray_step_deg)

    lat_deg_per_m = 1 / 111_320
    lon_deg_per_m = 1 / (111_320 * max(math.cos(math.radians(lat)), 0.001))

    # Cast all rays and collect distances
    ray_angles = []
    ray_dists = []

    for i in range(n_rays + 1):
        angle = (start_angle + i * ray_step_deg) % 360
        max_visible_dist = _cast_ray_max_distance(
            lon, lat, observer_elev, angle,
            max_ray_dist, distance_step,
            elevation, prepared_obstacles,
            lat_deg_per_m, lon_deg_per_m,
        )
        ray_angles.append(angle)
        ray_dists.append(max_visible_dist)

    # Smooth ray distances: median filter (window=5) removes SRTM noise spikes
    # without erasing real terrain features
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
    """Smooth ray distances with a median filter to remove SRTM noise spikes.

    Uses median (not mean) to preserve real terrain edges: if 4 out of 5
    adjacent rays see 10km but one sees 2km due to a DEM artifact, the
    median keeps 10km. But if 3 out of 5 are blocked, the median correctly
    reflects that.
    """
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
    """Cast a single ray and return the max visible distance.

    A point at distance d is "visible" if a TARGET_HEIGHT feature at that
    location would be seen from the observer (i.e. the angle from observer
    to the top of the feature exceeds all prior terrain angles).

    This is the standard "max angle" viewshed algorithm with two key
    improvements:
    1. We check visibility of TARGET_HEIGHT above ground, not ground level.
       This prevents the geometric horizon from cutting off flat terrain
       unrealistically early.
    2. We tolerate small SRTM noise (±3m) to avoid false occlusion from
       DEM artifacts on flat ground.
    """
    max_tan_angle = float("-inf")
    max_visible_dist = step

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    consecutive_blocked = 0
    BLOCKED_THRESHOLD = 20  # stop ray after 20 consecutive fully-blocked samples

    dist = step
    while dist <= max_dist:
        dlat = dist * cos_angle * lat_deg_per_m
        dlon = dist * sin_angle * lon_deg_per_m
        target_lat = origin_lat + dlat
        target_lon = origin_lon + dlon

        terrain_elev = elevation.get_elevation(target_lat, target_lon)

        # Check for obstacle (forest/building)
        obstacle_height = 0.0
        if prepared_obstacles is not None:
            try:
                if prepared_obstacles.contains(Point(target_lon, target_lat)):
                    obstacle_height = OBSTACLE_HEIGHT
            except Exception:
                pass

        effective_elev = terrain_elev + obstacle_height

        # Earth curvature correction
        curvature_drop = (dist * dist) / (2 * EARTH_RADIUS)

        # Angle from observer to terrain surface (including obstacles)
        surface_apparent = effective_elev - curvature_drop
        surface_tan = (surface_apparent - observer_elev) / dist

        # Angle from observer to top of a TARGET_HEIGHT feature at this point
        # (this is what we actually check for "can you see this area")
        target_apparent = (terrain_elev + TARGET_HEIGHT) - curvature_drop
        target_tan = (target_apparent - observer_elev) / dist

        # Tolerance: ignore SRTM noise (±3m elevation jitter)
        noise_tolerance = 3.0 / dist  # 3m error at distance d

        if target_tan >= max_tan_angle - noise_tolerance:
            # A 2m feature at this point is visible
            max_visible_dist = dist
            consecutive_blocked = 0

        # Update max angle using the terrain surface (not target height)
        # — terrain blocks what's behind it regardless of what we're looking for
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
