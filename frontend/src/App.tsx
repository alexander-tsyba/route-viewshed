import { useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Polygon,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface ViewshedResult {
  route_coords: number[][];
  sampled_points: { lon: number; lat: number; bearing: number }[];
  viewshed_polygons: number[][][];
  stats: {
    route_distance_km: number;
    route_duration_min: number;
    sampled_points: number;
    timing: Record<string, number>;
  };
}

function FitBounds({ coords }: { coords: number[][] }) {
  const map = useMap();
  if (coords.length > 0) {
    const bounds = L.latLngBounds(
      coords.map(([lon, lat]) => [lat, lon] as [number, number])
    );
    map.fitBounds(bounds, { padding: [40, 40] });
  }
  return null;
}

function App() {
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [maxDist, setMaxDist] = useState(15);
  const [includeLandcover, setIncludeLandcover] = useState(true);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("");
  const [result, setResult] = useState<ViewshedResult | null>(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!start.trim() || !end.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);
    setProgress("Calculating route and viewshed...");

    try {
      const resp = await fetch(`${API_BASE}/api/viewshed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start: start.trim(),
          end: end.trim(),
          max_view_distance_km: maxDist,
          include_landcover: includeLandcover,
        }),
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${resp.status}`);
      }

      const data: ViewshedResult = await resp.json();
      setResult(data);
      setProgress("");
    } catch (err: any) {
      setError(err.message || "Request failed");
      setProgress("");
    } finally {
      setLoading(false);
    }
  };

  const routeLatLngs: [number, number][] =
    result?.route_coords.map(([lon, lat]) => [lat, lon]) || [];

  const viewshedLatLngs: [number, number][][] =
    result?.viewshed_polygons.map((ring) =>
      ring.map(([lon, lat]) => [lat, lon] as [number, number])
    ) || [];

  return (
    <div className="app">
      <div className="sidebar">
        <h1>Route Viewshed</h1>
        <p className="subtitle">See what you'd see driving from A to B</p>

        <form onSubmit={handleSubmit}>
          <label>
            Start
            <input
              type="text"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              placeholder="e.g. Berlin or 52.52,13.405"
              disabled={loading}
            />
          </label>

          <label>
            End
            <input
              type="text"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              placeholder="e.g. Munich or 48.137,11.576"
              disabled={loading}
            />
          </label>

          <label>
            Max view distance: {maxDist} km
            <input
              type="range"
              min={1}
              max={30}
              value={maxDist}
              onChange={(e) => setMaxDist(Number(e.target.value))}
              disabled={loading}
            />
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={includeLandcover}
              onChange={(e) => setIncludeLandcover(e.target.checked)}
              disabled={loading}
            />
            Include forests & buildings
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Computing..." : "Calculate Viewshed"}
          </button>
        </form>

        {progress && <p className="progress">{progress}</p>}
        {error && <p className="error">{error}</p>}

        {result && (
          <div className="stats">
            <h3>Results</h3>
            <p>Route: {result.stats.route_distance_km} km</p>
            <p>Drive time: {result.stats.route_duration_min} min</p>
            <p>Sample points: {result.stats.sampled_points}</p>
            <p>Viewshed areas: {result.viewshed_polygons.length}</p>
            <h4>Timing</h4>
            <ul>
              {Object.entries(result.stats.timing).map(([k, v]) => (
                <li key={k}>
                  {k.replace(/_/g, " ")}: {v}s
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="map-container">
        <MapContainer
          center={[50, 10]}
          zoom={5}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {result && <FitBounds coords={result.route_coords} />}

          {routeLatLngs.length > 0 && (
            <Polyline
              positions={routeLatLngs}
              color="#2563eb"
              weight={3}
              opacity={0.9}
            />
          )}

          {viewshedLatLngs.map((ring, i) => (
            <Polygon
              key={i}
              positions={ring}
              pathOptions={{
                color: "#16a34a",
                fillColor: "#22c55e",
                fillOpacity: 0.35,
                weight: 2,
              }}
            />
          ))}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
