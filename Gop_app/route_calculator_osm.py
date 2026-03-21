"""
route_calculator_osm.py
=======================
Tính đường đi thực tế (theo đường bộ) từ điểm A → B sử dụng
OpenStreetMap + osmnx + networkx.

Trả về danh sách waypoints [(lat, lon), ...] theo đường thực.

Dùng từ priority_api.py hoặc script khác:
    from route_calculator_osm import road_route_waypoints
    pts = road_route_waypoints(lat1, lon1, lat2, lon2)

Nếu tải bản đồ thất bại (offline / vùng thiếu dữ liệu),
fallback về interpolation thẳng.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# osmnx cache: lưu graph vào bộ nhớ tránh tải lại mỗi lần.
_graph_cache: dict = {}


def _get_graph(lat: float, lon: float, radius_m: int = 3000):
    """Tải graph đường bộ bán kính radius_m mét xung quanh điểm trung tâm."""
    key = (round(lat, 3), round(lon, 3), radius_m)
    if key in _graph_cache:
        return _graph_cache[key]

    try:
        import osmnx as ox
        ox.settings.log_console = False
        ox.settings.use_cache   = True

        G = ox.graph_from_point(
            (lat, lon),
            dist=radius_m,
            network_type='drive',
            retain_all=False,
        )
        _graph_cache[key] = G
        return G
    except Exception as e:
        logger.warning(f"[osmnx] Không tải được graph: {e}")
        return None


def _linear_interpolation(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    n: int = 10,
) -> list[tuple[float, float]]:
    """Fallback: chia đều đường thẳng thành n waypoints."""
    return [
        (lat1 + (lat2 - lat1) * i / n, lon1 + (lon2 - lon1) * i / n)
        for i in range(n + 1)
    ]


def road_route_waypoints(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    simplify_tolerance: float = 0.0001,
) -> list[tuple[float, float]]:
    """
    Trả về danh sách (lat, lon) của waypoints theo đường bộ OSM từ
    (lat1, lon1) → (lat2, lon2).

    Tự động fallback về đường thẳng nếu:
    - osmnx không cài / không online
    - Vùng quá thiếu dữ liệu OSM
    - Khoảng cách quá ngắn (< 50 m)

    Parameters
    ----------
    lat1, lon1 : Tọa độ xuất phát (điểm cứu hộ / người dùng)
    lat2, lon2 : Tọa độ đích
    simplify_tolerance : Bỏ qua các waypoint quá gần nhau (độ thập phân)

    Returns
    -------
    List of (lat, lon) tuples
    """
    import math

    # Ước lượng khoảng cách.
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dist_approx_m = math.sqrt(dlat**2 + dlon**2) * 111_000

    if dist_approx_m < 50:
        # Quá gần, trả về 2 điểm.
        return [(lat1, lon1), (lat2, lon2)]

    # Bán kính graph: ít nhất 2 km, nhiều nhất 8 km.
    radius_m = max(2000, min(8000, int(dist_approx_m * 0.6)))

    # Tâm graph = midpoint.
    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2

    G = _get_graph(center_lat, center_lon, radius_m)

    if G is None:
        logger.info("[osmnx] Dùng fallback linear interpolation.")
        return _linear_interpolation(lat1, lon1, lat2, lon2)

    try:
        import osmnx as ox
        import networkx as nx

        orig_node = ox.nearest_nodes(G, X=lon1, Y=lat1)
        dest_node = ox.nearest_nodes(G, X=lon2, Y=lat2)

        route_nodes = nx.shortest_path(G, orig_node, dest_node, weight='length')

        waypoints: list[tuple[float, float]] = []
        prev = None
        for node in route_nodes:
            data = G.nodes[node]
            pt = (data['y'], data['x'])  # (lat, lon)
            if prev is not None:
                # Loại bỏ các điểm quá gần (simplify).
                if abs(pt[0] - prev[0]) < simplify_tolerance and \
                   abs(pt[1] - prev[1]) < simplify_tolerance:
                    continue
            waypoints.append(pt)
            prev = pt

        if len(waypoints) < 2:
            return _linear_interpolation(lat1, lon1, lat2, lon2)

        return waypoints

    except Exception as e:
        logger.warning(f"[osmnx] Tính đường thất bại: {e} — fallback linear.")
        return _linear_interpolation(lat1, lon1, lat2, lon2)


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test: từ Hạt CHCN Hoàng Mai → UBND Thị xã Hoàng Mai
    pts = road_route_waypoints(
        lat1=19.3500, lon1=105.7020,
        lat2=19.3400, lon2=105.7100,
    )
    print(f"Số waypoints: {len(pts)}")
    for p in pts[:5]:
        print(f"  {p[0]:.6f}, {p[1]:.6f}")
    if len(pts) > 5:
        print(f"  ... ({len(pts) - 5} điểm nữa)")
