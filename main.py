import networkx as nx
import matplotlib.pyplot as plt

### Task 1 ###
G = nx.Graph()

cities = [
    "Dnipro", "Kyiv", "Kharkiv", "Lviv", "Odesa", "Zhytomyr", 
    "Vinnytsia", "Rivne", "Donetsk", "Luhansk", "Simferopol"
]

G.add_nodes_from(cities)

weighted_edges = [
    ("Kyiv", "Dnipro", 495), ("Kyiv", "Kharkiv", 410), ("Kyiv", "Odesa", 441), 
    ("Kyiv", "Zhytomyr", 139), ("Lviv", "Rivne", 210), ("Rivne", "Zhytomyr", 189),
    ("Vinnytsia", "Zhytomyr", 129), ("Vinnytsia", "Dnipro", 573), ("Lviv", "Vinnytsia", 364),
    ("Kharkiv", "Luhansk", 338), ("Luhansk", "Donetsk", 155), ("Dnipro", "Donetsk", 258),
    ("Dnipro", "Odesa", 455), ("Dnipro", "Simferopol", 539),
]

G.add_weighted_edges_from(weighted_edges)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos, with_labels=True, node_size=700, font_size=10, font_weight="bold")
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("Ukraine Cities")
plt.show()

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degrees = dict(G.degree())

print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print(f"Nodes degrees: {degrees}")

### Task 2 ###
from collections import deque

def bfs(graph, start, end):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        current, path = queue.popleft()
        visited.add(current)

        if current == end:
            return path
        for neighbor in graph.neighbors(current):
            if neighbor not in visited and neighbor not in [p[0] for p in queue]:
                queue.append((neighbor, path + [neighbor]))
    return None

def dfs(graph, start, end, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path.append(start)
    visited.add(start)

    if start == end:
        return path

    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            result = dfs(graph, neighbor, end, path.copy(), visited.copy())
            if result:
                return result
    return None

start = "Kyiv"
end = "Lviv"

dfs_path_to_lviv = dfs(G, start, end)
bfs_path_to_lviv = bfs(G, start, end)

print(dfs_path_to_lviv)
print(bfs_path_to_lviv)

## Task 3 ##
import heapq

def dijkstra(graph, start):
    distances = {vertex: float("infinity") for vertex in graph.nodes}
    distances[start] = 0
    priority_queue = [(0, start)]
    visited = set()

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_vertex in visited:
            continue
        visited.add(current_vertex)

        for neighbor in graph.neighbors(current_vertex):
            weight = graph[current_vertex][neighbor].get("weight", 1)
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

for node in G.nodes:
    print(f"Shortest path from '{node}' {dijkstra(G, node)}")
