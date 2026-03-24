import httpx
import math
from typing import List, Tuple, Set
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OBSTACLE_HEIGHT = 20.0  # meters for forests and buildings


class LandCoverProvider:
    """Fetches forest and building polygons from OpenStreetMap via Overpass API."""

    def __init__(self):
        self._obstacles: MultiPolygon | Polygon | None = None
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

        # Split into chunks if bbox is too large (Overpass limits)
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        # For very large routes, we sample representative sections
        # rather than querying the entire corridor
        max_bbox_area = 4.0  # degrees^2 — Overpass can handle this
        bbox_area = lat_range * lon_range

        if bbox_area > max_bbox_area * 10:
            # Too large — skip land cover for very long routes
            # and rely only on elevation
            self._obstacles = MultiPolygon()
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
        out body;
        >;
        out skel qt;
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

        nodes = {}
        ways = {}
        polys = []

        for el in data.get("elements", []):
            if el["type"] == "node":
                nodes[el["id"]] = (el["lon"], el["lat"])
            elif el["type"] == "way":
                ways[el["id"]] = el.get("nodes", [])

        for way_id, node_ids in ways.items():
            coords = [nodes[nid] for nid in node_ids if nid in nodes]
            if len(coords) >= 4:
                try:
                    p = Polygon(coords)
                    if p.is_valid and p.area > 0:
                        polys.append(p)
                except Exception:
                    pass

        return polys

    def is_in_obstacle(self, lon: float, lat: float) -> bool:
        """Check if a point falls within a forest or building polygon."""
        if self._obstacles is None or self._obstacles.is_empty:
            return False
        from shapely.geometry import Point
        return self._obstacles.contains(Point(lon, lat))

    def get_obstacles(self):
        """Return the raw obstacle geometry for intersection checks."""
        if self._obstacles is None or self._obstacles.is_empty:
            return None
        return self._obstacles

    def get_prepared(self):
        """Return a prepared geometry for fast repeated containment checks."""
        if self._obstacles is None or self._obstacles.is_empty:
            return None
        return prep(self._obstacles)
