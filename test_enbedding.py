import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx

def initialize_positions(n_nodes, dim_max, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.uniform(low=-1.0, high=1.0, size=(n_nodes, dim_max)))

def sparse_compute_stress(positions, row, col, data):
    diff = positions[row] - positions[col]  # Shape: (N_nonzero, dim_max)
    current_distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # Shape: (N_nonzero,)
    stress = jnp.sum((current_distances - data) ** 2)
    return stress

@jax.jit
def update_positions(positions, row, col, data, learning_rate=0.01):
    stress_grad = jax.grad(sparse_compute_stress, argnums=0)(positions, row, col, data)
    new_positions = positions - learning_rate * stress_grad
    return new_positions

def run_embedding(dist_matrix, dim_max=2, n_steps=100, learning_rate=0.01, initial_positions=None):
    n_nodes = dist_matrix.shape[0]
    row, col = np.nonzero(dist_matrix)
    data = dist_matrix[row, col]
    row = jnp.array(row)
    col = jnp.array(col)
    data = jnp.array(data)
    
    if initial_positions is None:
        positions = initialize_positions(n_nodes, dim_max)
    else:
        positions = initial_positions
    
    for step in range(n_steps):
        positions = update_positions(positions, row, col, data, learning_rate)
        if step % 100 == 0:
            stress_value = sparse_compute_stress(positions, row, col, data)
            print(f"Step {step}, Stress: {stress_value:.6f}")
    
    return positions

def get_ordered_nodes(dist_matrix, start_node=0):
    G = nx.Graph()

    # Add edges to the graph with distances as weights
    n_nodes = dist_matrix.shape[0]
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if dist_matrix[i, j] > 0:  # Only consider non-zero distances
                G.add_edge(i, j, weight=dist_matrix[i, j])

    # Get nodes in increasing order of distance from the start_node using Dijkstra's algorithm
    ordered_nodes = list(nx.single_source_dijkstra_path_length(G, start_node).keys())
    
    return ordered_nodes

def run_embedding_iterative(dist_matrix, dim_max=2, n_steps=100, learning_rate=0.01, N_start=3, start_node=0):
    ordered_nodes = get_ordered_nodes(dist_matrix, start_node=start_node)
    n_nodes = len(ordered_nodes)
    
    if N_start > n_nodes:
        raise ValueError("N_start cannot be larger than the number of nodes in the distance matrix")
    
    # Start with the first N_start nodes
    initial_nodes = ordered_nodes[:N_start]
    initial_indices = np.ix_(initial_nodes, initial_nodes)
    initial_dist_matrix = dist_matrix[initial_indices]
    positions = run_embedding(initial_dist_matrix, dim_max, n_steps, learning_rate)
    
    # Iteratively add remaining nodes
    for i in range(N_start, n_nodes):
        print(i)
        new_node = ordered_nodes[i]
        current_nodes = ordered_nodes[:i+1]
        current_indices = np.ix_(current_nodes, current_nodes)
        current_dist_matrix = dist_matrix[current_indices]
        
        # Update positions with new node added
        new_position = initialize_positions(1, dim_max)
        positions = jnp.concatenate([positions, new_position], axis=0)
        
        # Run embedding with the updated positions
        positions = run_embedding(current_dist_matrix, dim_max, n_steps, learning_rate, initial_positions=positions)
        
    return positions



# Example usage:
if __name__ == "__main__":
    def generate_connected_graph(n_nodes, max_connections=4):
        G = nx.Graph()
        
        # Ensure the graph is connected by creating a simple path first
        for i in range(n_nodes - 1):
            G.add_edge(i, i + 1, weight=np.random.uniform(1, 10))
        
        # Add additional edges to each node, ensuring max_connections constraint
        for i in range(n_nodes):

            current_connections = list(G.neighbors(i))
            while len(current_connections) < max_connections:
                potential_node = np.random.randint(0, n_nodes)
                if potential_node != i and potential_node not in current_connections:
                    G.add_edge(i, potential_node, weight=np.random.uniform(1, 10))
                    current_connections.append(potential_node)
                if len(G.edges) >= n_nodes * max_connections // 2:
                    break

        # Convert the graph to a distance matrix
        dist_matrix = np.zeros((n_nodes, n_nodes))
        for (i, j, weight) in G.edges(data=True):
            dist_matrix[i, j] = weight['weight']
            dist_matrix[j, i] = weight['weight']
        
        return dist_matrix

    # Generate a 10x10 connected distance matrix with each node having no more than 4 connections
    dist_matrix = generate_connected_graph(10, max_connections=4)
    print(dist_matrix)

    final_positions_iterative = run_embedding_iterative(dist_matrix, dim_max=3, n_steps=1000, learning_rate=0.1, N_start=3, start_node=0)
    print("Final positions (iterative):\n", final_positions_iterative)