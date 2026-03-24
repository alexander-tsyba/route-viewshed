import math
import numpy as np
from typing import List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiPoint
from shapely.ops import unary_union

from elevation import ElevationProvider
from landcover import LandCoverProvider, OBSTACLE_HEIGHT

EARTH_RADIUS = 6_371_000  # meters
OBSERVER_HEIGHT = 1.5  # eye height in car
TARGET_HEIGHT = 2.0  # consider a point "seen" if a 2m feature there is visible
RAY_ANGULAR_STEP = 2.0  # degrees between rays
RAY_DISTANCE_STEP = 120  # meters for terrain-only ray casting (far field)

# Forest interior visibility: when inside a forest, you can see ~80m
FOREST_INTERIOR_VISIBILITY = 80  # meters

# Minimum polygon area to keep (km²)
MIN_POLYGON_AREA_KM2 = 0.01  # ~100m × 100m — reduced to keep city viewsheds


def _atmospheric_max(observer_elev_asl: float) -> float:
    """Elevation-dependent atmospheric visibility limit."""
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
    obstacle_tree = landcover.get_tree()
    obstacle_list = landcover.get_obstacle_list()
    prepared_obstacles = landcover.get_prepared()
    fan_polys = []

    for idx, pt in enumerate(sampled_points):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx, len(sampled_points))

        fan = _compute_single_viewshed_fan(
            pt["lon"], pt["lat"], pt["bearing"],
            elevation,
            obstacle_tree, obstacle_list, prepared_obstacles,
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
        erode_m = 120
    elif area_km2_approx > 5:
        erode_m = 60
    else:
        erode_m = 20  # gentle for city viewsheds

    erode_deg = erode_m / 111_320

    opened = merged.buffer(-erode_deg).buffer(erode_deg)
    if opened.is_empty:
        opened = merged

    close_deg = erode_deg * 0.7
    closed = opened.buffer(close_deg).buffer(-close_deg)
    if closed.is_empty:
        closed = opened

    tol = max(20, erode_m * 0.4) / 111_320
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
    obstacle_tree, obstacle_list, prepared_obstacles,
    ray_step_deg: float,
) -> Optional[Polygon]:
    """Compute visibility fan from a single point.

    Two-phase approach:
    1. Ray-obstacle intersection: find nearest obstacle along each ray direction
    2. Terrain viewshed: for each ray, determine how far elevation allows seeing
    3. Visible distance = min(obstacle_dist, terrain_dist)
    """
    observer_ground = elevation.get_elevation(lat, lon)
    observer_elev = observer_ground + OBSERVER_HEIGHT

    h_above_ground = OBSERVER_HEIGHT
    geometric_horizon = (
        math.sqrt(2 * EARTH_RADIUS * h_above_ground)
        + math.sqrt(2 * EARTH_RADIUS * TARGET_HEIGHT)
    )
    atmo_max = _atmospheric_max(observer_ground)
    max_ray_dist = min(atmo_max, geometric_horizon * 3)

    # Check if observer is inside an obstacle (e.g. driving through forest)
    observer_in_obstacle = False
    if prepared_obstacles is not None:
        try:
            observer_in_obstacle = prepared_obstacles.contains(Point(lon, lat))
        except Exception:
            pass

    start_angle = (bearing - 90) % 360
    n_rays = int(180 / ray_step_deg)

    lat_deg_per_m = 1 / 111_320
    lon_deg_per_m = 1 / (111_320 * max(math.cos(math.radians(lat)), 0.001))

    ray_angles = []
    ray_dists = []

    for i in range(n_rays + 1):
        angle = (start_angle + i * ray_step_deg) % 360

        if observer_in_obstacle:
            # Inside a forest: visibility limited to interior distance
            visible_dist = FOREST_INTERIOR_VISIBILITY
        else:
            # Phase 1: obstacle distance via ray intersection
            obstacle_dist = _ray_obstacle_distance(
                lon, lat, angle, max_ray_dist,
                obstacle_tree, obstacle_list,
                lat_deg_per_m, lon_deg_per_m,
            )

            # Phase 2: terrain distance (only if no close obstacle)
            if obstacle_dist > 500:
                # No nearby obstacle — check terrain
                terrain_dist = _cast_terrain_ray(
                    lon, lat, observer_elev, angle,
                    min(max_ray_dist, obstacle_dist),
                    elevation, lat_deg_per_m, lon_deg_per_m,
                )
                visible_dist = min(obstacle_dist, terrain_dist)
            else:
                visible_dist = obstacle_dist

        ray_angles.append(angle)
        ray_dists.append(visible_dist)

    # Smooth ray distances (reduces DEM noise on far-field rays)
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


def _ray_obstacle_distance(
    lon: float, lat: float,
    angle_deg: float,
    max_dist: float,
    obstacle_tree, obstacle_list,
    lat_deg_per_m: float,
    lon_deg_per_m: float,
) -> float:
    """Find the distance to the nearest obstacle along a ray direction.

    Uses STRtree spatial index for efficient obstacle lookup.
    Returns max_dist if no obstacle found.
    """
    if obstacle_tree is None or not obstacle_list:
        return max_dist

    angle_rad = math.radians(angle_deg)
    end_lon = lon + max_dist * math.sin(angle_rad) * lon_deg_per_m
    end_lat = lat + max_dist * math.cos(angle_rad) * lat_deg_per_m

    ray = LineString([(lon, lat), (end_lon, end_lat)])
    observer = Point(lon, lat)

    # Query spatial index for candidate obstacles
    try:
        candidate_indices = obstacle_tree.query(ray)
    except Exception:
        return max_dist

    min_dist_m = max_dist

    for idx in candidate_indices:
        obstacle = obstacle_list[idx]
        try:
            if not ray.intersects(obstacle):
                continue

            # Intersect ray with obstacle boundary (exterior ring)
            inter = ray.intersection(obstacle.boundary)
            if inter.is_empty:
                continue

            # Find nearest intersection point to observer
            nearest_dist_deg = observer.distance(inter)
            # Convert from degrees to meters (approximate)
            dist_m = nearest_dist_deg / max(lat_deg_per_m, lon_deg_per_m)

            if dist_m < min_dist_m:
                min_dist_m = dist_m

        except Exception:
            continue

    return max(min_dist_m, RAY_DISTANCE_STEP)  # minimum = one step


def _cast_terrain_ray(
    origin_lon: float, origin_lat: float,
    observer_elev: float,
    angle_deg: float,
    max_dist: float,
    elevation: ElevationProvider,
    lat_deg_per_m: float,
    lon_deg_per_m: float,
) -> float:
    """Pure terrain-based ray casting (no obstacle checks).

    Used for far-field visibility beyond nearby obstacles.
    """
    max_tan_angle = float("-inf")
    max_visible_dist = RAY_DISTANCE_STEP

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    consecutive_blocked = 0

    dist = RAY_DISTANCE_STEP
    while dist <= max_dist:
        dlat = dist * cos_angle * lat_deg_per_m
        dlon = dist * sin_angle * lon_deg_per_m
        target_lat = origin_lat + dlat
        target_lon = origin_lon + dlon

        terrain_elev = elevation.get_elevation(target_lat, target_lon)
        curvature_drop = (dist * dist) / (2 * EARTH_RADIUS)

        surface_apparent = terrain_elev - curvature_drop
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
            if consecutive_blocked >= 15:
                break
        else:
            consecutive_blocked = 0

        dist += RAY_DISTANCE_STEP

    return max_visible_dist


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
