import httpx
import math
from typing import List, Tuple
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OBSTACLE_HEIGHT = 20.0  # meters for forests and buildings


class LandCoverProvider:
    """Fetches forest and building polygons from OpenStreetMap via Overpass API."""

    def __init__(self):
        self._obstacles: MultiPolygon | Polygon | None = None
        self._tree: STRtree | None = None
        self._obstacle_list: list = []
        self._bbox: Tuple[float, float, float, float] | None = None

    async def load_obstacles_for_route(
        self, coords: List[List[float]], buffer_km: float = 15
    ):
        """Load forest and building areas along the route corridor."""
        lats = [c[1] for c in coords]
        lons = [c[0] for c in coords]

        km_to_deg_lat = 1 / 111.0
        avg_lat = sum(lats) / len(lats)
        km_to_deg_lon = 1 / (111.0 * max(math.cos(math.radians(avg_lat)), 0.01))

        min_lat = min(lats) - buffer_km * km_to_deg_lat
        max_lat = max(lats) + buffer_km * km_to_deg_lat
        min_lon = min(lons) - buffer_km * km_to_deg_lon
        max_lon = max(lons) + buffer_km * km_to_deg_lon

        self._bbox = (min_lat, min_lon, max_lat, max_lon)

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        max_bbox_area = 4.0
        bbox_area = lat_range * lon_range

        if bbox_area > max_bbox_area * 10:
            self._obstacles = MultiPolygon()
            self._obstacle_list = []
            self._tree = None
            return

        all_polys = []
        if bbox_area > max_bbox_area:
            chunks = self._split_bbox(min_lat, min_lon, max_lat, max_lon, max_bbox_area)
        else:
            chunks = [(min_lat, min_lon, max_lat, max_lon)]

        for chunk_bbox in chunks:
            polys = await self._query_overpass(chunk_bbox)
            all_polys.extend(polys)

        if all_polys:
            self._obstacles = unary_union(all_polys)
        else:
            self._obstacles = MultiPolygon()

        # Build spatial index for ray intersection queries
        if isinstance(self._obstacles, MultiPolygon):
            self._obstacle_list = list(self._obstacles.geoms)
        elif isinstance(self._obstacles, Polygon) and not self._obstacles.is_empty:
            self._obstacle_list = [self._obstacles]
        else:
            self._obstacle_list = []

        if self._obstacle_list:
            self._tree = STRtree(self._obstacle_list)
        else:
            self._tree = None

    def _split_bbox(
        self,
        min_lat: float, min_lon: float,
        max_lat: float, max_lon: float,
        max_area: float
    ) -> List[Tuple[float, float, float, float]]:
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        n_lat = max(1, int(math.ceil(math.sqrt(lat_range * lon_range / max_area) * lat_range / lon_range)))
        n_lon = max(1, int(math.ceil(math.sqrt(lat_range * lon_range / max_area) * lon_range / lat_range)))

        dlat = lat_range / n_lat
        dlon = lon_range / n_lon

        chunks = []
        for i in range(n_lat):
            for j in range(n_lon):
                chunks.append((
                    min_lat + i * dlat,
                    min_lon + j * dlon,
                    min_lat + (i + 1) * dlat,
                    min_lon + (j + 1) * dlon,
                ))
        return chunks

    async def _query_overpass(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[Polygon]:
        """Query Overpass API for forest and building polygons.

        Uses 'out geom;' to get resolved geometries — this correctly handles
        multipolygon relations (most large forests in OSM).
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        query = f"""
        [out:json][timeout:60];
        (
          way["natural"="wood"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["landuse"="forest"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["natural"="wood"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["landuse"="forest"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
        """

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    OVERPASS_URL,
                    data={"data": query},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return []

        polys = []

        for el in data.get("elements", []):
            if el["type"] == "way":
                geom = el.get("geometry", [])
                coords = [(p["lon"], p["lat"]) for p in geom]
                if len(coords) >= 4:
                    try:
                        p = Polygon(coords)
                        if p.is_valid and p.area > 0:
                            polys.append(p)
                        else:
                            p = p.buffer(0)
                            if not p.is_empty and p.area > 0:
                                if isinstance(p, MultiPolygon):
                                    polys.extend(g for g in p.geoms if g.area > 0)
                                else:
                                    polys.append(p)
                    except Exception:
                        pass

            elif el["type"] == "relation":
                # Multipolygon relation — assemble outer member ways
                outer_segments = []
                for member in el.get("members", []):
                    if member.get("role") == "outer" and "geometry" in member:
                        coords = [(p["lon"], p["lat"]) for p in member["geometry"]]
                        if len(coords) >= 2:
                            outer_segments.append(coords)

                # Merge segments into closed rings
                rings = _merge_segments_to_rings(outer_segments)
                for ring in rings:
                    if len(ring) >= 4:
                        try:
                            p = Polygon(ring)
                            if p.is_valid and p.area > 0:
                                polys.append(p)
                            else:
                                p = p.buffer(0)
                                if not p.is_empty and p.area > 0:
                                    if isinstance(p, MultiPolygon):
                                        polys.extend(g for g in p.geoms if g.area > 0)
                                    else:
                                        polys.append(p)
                        except Exception:
                            pass

        return polys

    def get_obstacles(self):
        """Return the raw obstacle geometry."""
        if self._obstacles is None or self._obstacles.is_empty:
            return None
        return self._obstacles

    def get_obstacle_list(self) -> list:
        """Return list of individual obstacle polygons for STRtree queries."""
        return self._obstacle_list

    def get_tree(self) -> STRtree | None:
        """Return spatial index for fast ray-obstacle intersection."""
        return self._tree

    def get_prepared(self):
        """Return a prepared geometry for fast containment checks."""
        if self._obstacles is None or self._obstacles.is_empty:
            return None
        return prep(self._obstacles)


def _merge_segments_to_rings(segments: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    """Merge way segments end-to-end into closed rings.

    OSM multipolygon relations have outer ways that need to be joined
    sequentially to form a closed polygon ring.
    """
    if not segments:
        return []

    # If any single segment is already closed, use it directly
    rings = []
    remaining = []
    for seg in segments:
        if len(seg) >= 4 and seg[0] == seg[-1]:
            rings.append(seg)
        else:
            remaining.append(list(seg))

    if not remaining:
        return rings

    # Try to merge remaining segments end-to-end
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for i, seg in enumerate(remaining):
                if not seg:
                    continue
                # Try to join: current end → seg start
                if _points_close(current[-1], seg[0]):
                    current.extend(seg[1:])
                    remaining.pop(i)
                    changed = True
                    break
                # Try to join: current end → seg end (reverse seg)
                elif _points_close(current[-1], seg[-1]):
                    current.extend(reversed(seg[:-1]))
                    remaining.pop(i)
                    changed = True
                    break
                # Try to join: seg end → current start
                elif _points_close(seg[-1], current[0]):
                    current = seg + current[1:]
                    remaining.pop(i)
                    changed = True
                    break
                # Try to join: seg start → current start (reverse seg)
                elif _points_close(seg[0], current[0]):
                    current = list(reversed(seg)) + current[1:]
                    remaining.pop(i)
                    changed = True
                    break

        # Close the ring if endpoints are close
        if len(current) >= 4 and _points_close(current[0], current[-1]):
            current[-1] = current[0]  # exact close
            rings.append(current)
        elif len(current) >= 4:
            # Force close
            current.append(current[0])
            rings.append(current)

    return rings


def _points_close(a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-6) -> bool:
    """Check if two coordinate points are approximately equal."""
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
