import math
import struct
import os
import httpx
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

SRTM_CACHE_DIR = Path(os.environ.get("SRTM_CACHE_DIR", "/tmp/srtm_cache"))
SRTM_BASE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"


class ElevationProvider:
    """Provides elevation data from SRTM HGT files (1 arc-second, ~30m)."""

    def __init__(self):
        SRTM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._tiles: dict[str, Optional[np.ndarray]] = {}

    def _tile_name(self, lat: float, lon: float) -> str:
        lat_int = int(math.floor(lat))
        lon_int = int(math.floor(lon))
        ns = "N" if lat_int >= 0 else "S"
        ew = "E" if lon_int >= 0 else "W"
        return f"{ns}{abs(lat_int):02d}{ew}{abs(lon_int):03d}"

    def _tile_path(self, name: str) -> Path:
        return SRTM_CACHE_DIR / f"{name}.hgt"

    async def _ensure_tile(self, name: str) -> Optional[Path]:
        path = self._tile_path(name)
        if path.exists():
            return path

        folder = name[:3]  # e.g. N47
        url = f"{SRTM_BASE_URL}/{folder}/{name}.hgt.gz"

        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()

            import gzip
            gz_path = SRTM_CACHE_DIR / f"{name}.hgt.gz"
            gz_path.write_bytes(resp.content)

            with gzip.open(gz_path, "rb") as f:
                path.write_bytes(f.read())
            gz_path.unlink()

            return path
        except Exception:
            return None

    def _load_tile(self, name: str) -> Optional[np.ndarray]:
        if name in self._tiles:
            return self._tiles[name]

        path = self._tile_path(name)
        if not path.exists():
            self._tiles[name] = None
            return None

        size = path.stat().st_size
        if size == 1201 * 1201 * 2:
            samples = 1201  # 3 arc-second
        elif size == 3601 * 3601 * 2:
            samples = 3601  # 1 arc-second
        else:
            self._tiles[name] = None
            return None

        data = np.frombuffer(path.read_bytes(), dtype=">i2").reshape((samples, samples))
        self._tiles[name] = data
        return data

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation in meters. Returns 0 if data unavailable."""
        name = self._tile_name(lat, lon)
        data = self._load_tile(name)
        if data is None:
            return 0.0

        samples = data.shape[0]
        lat_frac = lat - math.floor(lat)
        lon_frac = lon - math.floor(lon)

        row = int((1 - lat_frac) * (samples - 1))
        col = int(lon_frac * (samples - 1))

        row = max(0, min(samples - 1, row))
        col = max(0, min(samples - 1, col))

        elev = int(data[row, col])
        if elev == -32768:  # void
            return 0.0
        return float(elev)

    def get_elevations_batch(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Get elevations for multiple (lat, lon) points efficiently."""
        result = np.zeros(len(points))
        for i, (lat, lon) in enumerate(points):
            result[i] = self.get_elevation(lat, lon)
        return result

    async def preload_tiles_for_route(self, coords: List[List[float]], buffer_km: float = 15):
        """Download all SRTM tiles needed for the route + buffer."""
        needed = set()
        for lon, lat in coords:
            lat_min = int(math.floor(lat - buffer_km / 111))
            lat_max = int(math.floor(lat + buffer_km / 111))
            lon_min = int(math.floor(lon - buffer_km / (111 * max(math.cos(math.radians(lat)), 0.01))))
            lon_max = int(math.floor(lon + buffer_km / (111 * max(math.cos(math.radians(lat)), 0.01))))

            for la in range(lat_min, lat_max + 1):
                for lo in range(lon_min, lon_max + 1):
                    name = self._tile_name(la + 0.5, lo + 0.5)
                    needed.add(name)

        for name in needed:
            if not self._tile_path(name).exists():
                await self._ensure_tile(name)
            self._load_tile(name)
