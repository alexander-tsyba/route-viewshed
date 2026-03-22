import httpx
from typing import Tuple

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


async def geocode(query: str) -> Tuple[float, float]:
    """Geocode a place name to (lon, lat) using Nominatim."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "RouteViewshed/1.0"},
        )
        resp.raise_for_status()
        results = resp.json()

    if not results:
        raise ValueError(f"Could not geocode: {query}")

    return float(results[0]["lon"]), float(results[0]["lat"])
