import numpy as np
from multi_drone import MultiDrone
import heapq
import time

# Initialize the MultiDrone environment
sim = MultiDrone(num_drones=2, environment_file="environment.yaml")
# Q2.4
# sim = MultiDrone(num_drones=2, environment_file="environment_1.yaml")
# Q2.5
# sim = MultiDrone(num_drones=2, environment_file="environment_2.yaml") # 2 drones
# sim = MultiDrone(num_drones=4, environment_file="environment_2.yaml") # 4 drones
# sim = MultiDrone(num_drones=8, environment_file="environment_2.yaml") # 8 drones
# sim = MultiDrone(num_drones=12, environment_file="environment_2.yaml") # 12 drones


# ---------- PRM Implementation ----------
t0 = time.time()
t1 = 0
edge_checks = 0
node_expanded = 0

# Calculate the Euclidean distance between two configurations
def euclidean_dist(q1, q2):
    return np.linalg.norm(q1.flatten() - q2.flatten())

# Given a node q, find its k nearest neighbors
def k_nearest_neighbors(nodes, q, k=10):
    dists = [(euclidean_dist(n, q), i) for i, n in enumerate(nodes)]
    dists.sort(key=lambda x: x[0])
    return [nodes[i] for _, i in dists[1:k + 1]]

# After the search is completed, backtrack the path; came_from stores the predecessor of each node
def reconstruct_path(came_from, current):
    path = [current]
    while tuple(current.flatten()) in came_from:
        current = came_from[tuple(current.flatten())]
        path.append(current)
    return path[::-1]


def my_planner(sim, N=300, k=10):
    """
    - Sample N valid configurations
    - Connect k-nearest neighbors
    - A* search with on-demand edge validation
    """
    global node_expanded, edge_checks, t0, t1

    start = sim.initial_configuration  # Start configurations
    goal = sim.goal_positions  # goal configurations
    # Node set and edge dictionary
    nodes = [start, goal]
    nbs = {tuple(start.flatten()): [], tuple(goal.flatten()): []}


    # Sampling phase: uniform sampling within workspace bounds; keep only valid configurations
    while len(nodes) < N:
        q = np.random.uniform(sim._bounds[:, 0], sim._bounds[:, 1], size=(sim.N, 3))
        if sim.is_valid(q):
            nodes.append(q)
            nbs[tuple(q.flatten())] = []

    # Connection phase: store k-NN neighbors but don't validate edges yet
    for q in nodes:
        for nb in k_nearest_neighbors(nodes, q, k):
            nbs[tuple(q.flatten())].append(nb)

    # A* search on the roadmap
    open_set = [(0, start)] #(f_score, node)，f = g + h
    came_from = {}
    g_score = {tuple(start.flatten()): 0} # g(n): the known shortest cost from the start to node n

    while open_set:
        # Extract the current node with the smallest estimated f value
        _, current = heapq.heappop(open_set)
        node_expanded += 1

        # Termination condition: if the goal has been reached
        if sim.is_goal(current):
            path = reconstruct_path(came_from, current)
            t1 = time.time() - t0
            return path

        ctuple = tuple(current.flatten())
        for nb in nbs[tuple(current.flatten())]:
            ntuple = tuple(nb.flatten())
            tentative_g = g_score[ctuple] + euclidean_dist(current, nb)

            # If this path to nb is better, record it
            if ntuple not in g_score or tentative_g < g_score[ntuple]:
                edge_checks += 1
                # Lazy validation: perform collision checking only when the edge is likely to be adopted
                if sim.motion_valid(current, nb):
                    came_from[ntuple] = current
                    g_score[ntuple] = tentative_g
                    f_score = tentative_g + euclidean_dist(nb, goal) #In A*: f =  g + h
                    heapq.heappush(open_set, (f_score, nb))

    #If the open_set is exhausted and the goal has not been reached, consider the problem unsolvable                
    return None

def path_length(path):
    if not path or len(path) < 2: 
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += np.linalg.norm(path[i].flatten() - path[i+1].flatten())
    return float(total)


# ---------- Run the planner ----------
solution_path = my_planner(sim, N=500, k=20)
# If a solution path is found, visualize it
if solution_path:
    print(f"Found path with {len(solution_path)} waypoints.")
    print(f"Path length: {path_length(solution_path)}")
    print(f"Time taken: {t1:.2f} seconds")
    print(f"Edge checks: {edge_checks}, Nodes expanded: {node_expanded}")
    # Visualize the path in the simulation environment
    sim.visualize_paths(solution_path)
else:
    print("No path found")
