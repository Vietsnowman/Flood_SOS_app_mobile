import sys
import json
import networkx as nx
import osmnx as ox

def calculate_route(start_lat, start_lon, end_lat, end_lon):
    # Path to the cached graphml created by the Streamlit app
    GRAPH_CACHE = "realtime_outputs/osm_graph.graphml"
    
    try:
        # Load the graph
        G = ox.load_graphml(GRAPH_CACHE)
        
        # Find nearest nodes
        orig = ox.distance.nearest_nodes(G, start_lon, start_lat)
        dest = ox.distance.nearest_nodes(G, end_lon, end_lat)
        
        # Calculate shortest path
        route = nx.shortest_path(G, orig, dest, weight="length")
        
        # Extract coordinates [lon, lat] format for standard GeoJSON/Leaflet compatibility
        coords = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in route]
        
        # Calculate distance
        dist_m = nx.shortest_path_length(G, orig, dest, weight="length")
        
        return {
            "success": True,
            "distance_m": float(dist_m),
            "coordinates": coords
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(json.dumps({"success": False, "error": "Missing arguments. Usage: python route_calculator.py <start_lat> <start_lon> <end_lat> <end_lon>"}))
        sys.exit(1)
        
    try:
        start_lat = float(sys.argv[1])
        start_lon = float(sys.argv[2])
        end_lat = float(sys.argv[3])
        end_lon = float(sys.argv[4])
        
        result = calculate_route(start_lat, start_lon, end_lat, end_lon)
        print(json.dumps(result))
    except ValueError:
        print(json.dumps({"success": False, "error": "Invalid coordinates format. Must be floating point numbers."}))
        sys.exit(1)
