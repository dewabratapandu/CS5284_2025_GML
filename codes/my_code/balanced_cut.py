import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def balanced_cut_partition(graph_matrix):
    """
    Finds a good balanced cut for a graph using the Kernighan-Lin heuristic.
    This partitions the graph into two equal-sized sets.

    Args:
        graph_matrix: A NumPy adjacency matrix of the graph.

    Returns:
        A tuple containing:
        - The partitions (two sets of node indices).
        - The final cut size.
    """
    num_nodes = graph_matrix.shape[0]
    
    # 1. Start with an initial balanced (equal-sized) random partition
    nodes = np.arange(num_nodes)
    np.random.shuffle(nodes)
    partition_a = set(nodes[:num_nodes // 2])
    partition_b = set(nodes[num_nodes // 2:])

    while True:
        # Calculate D-values for each node: external_cost - internal_cost
        d_values = {}
        for node in nodes:
            cost_internal = 0
            cost_external = 0
            current_partition = partition_a if node in partition_a else partition_b
            other_partition = partition_b if node in partition_a else partition_a
            
            for neighbor in range(num_nodes):
                if graph_matrix[node, neighbor] > 0:
                    if neighbor in current_partition:
                        cost_internal += graph_matrix[node, neighbor]
                    else:
                        cost_external += graph_matrix[node, neighbor]
            d_values[node] = cost_external - cost_internal

        total_gains = []
        swapped_pairs = []
        
        temp_a = partition_a.copy()
        temp_b = partition_b.copy()

        # Iteratively find the best pair to swap from the remaining nodes
        for _ in range(num_nodes // 2):
            max_gain = -np.inf
            best_pair = (None, None)

            for a_node in temp_a:
                for b_node in temp_b:
                    edge_weight = graph_matrix[a_node, b_node]
                    gain = d_values[a_node] + d_values[b_node] - 2 * edge_weight
                    if gain > max_gain:
                        max_gain = gain
                        best_pair = (a_node, b_node)
            
            if best_pair == (None, None):
                break

            a_star, b_star = best_pair
            
            # Temporarily swap the best pair and "lock" them
            temp_a.remove(a_star)
            temp_b.remove(b_star)
            swapped_pairs.append((a_star, b_star))
            total_gains.append(max_gain)

            # Update D-values for the remaining unlocked nodes
            for node in temp_a:
                d_values[node] += 2 * graph_matrix[node, a_star] - 2 * graph_matrix[node, b_star]
            for node in temp_b:
                d_values[node] += 2 * graph_matrix[node, b_star] - 2 * graph_matrix[node, a_star]

        # Find the sequence of swaps that yielded the maximum improvement
        cumulative_gains = np.cumsum(total_gains)
        max_cumulative_gain_idx = np.argmax(cumulative_gains)
        max_cumulative_gain = cumulative_gains[max_cumulative_gain_idx]

        # If an improvement was found, make the swaps permanent
        if max_cumulative_gain > 0:
            for i in range(max_cumulative_gain_idx + 1):
                a_swap, b_swap = swapped_pairs[i]
                partition_a.remove(a_swap)
                partition_a.add(b_swap)
                partition_b.remove(b_swap)
                partition_b.add(a_swap)
        else:
            # If no improvement is possible, the algorithm has converged
            break

    # Calculate the final cut size
    cut_size = 0
    for i in partition_a:
        for j in partition_b:
            cut_size += graph_matrix[i, j]
            
    return (partition_a, partition_b), cut_size

# --- Example Usage ---

# 1. Create a sample graph (e.g., Zachary's Karate Club)
G = nx.karate_club_graph()
adj_matrix = nx.to_numpy_array(G)

# 2. Run the balanced cut algorithm
(part_a, part_b), cut = balanced_cut_partition(adj_matrix)

print(f"Partition A (size {len(part_a)}): {sorted(list(part_a))}")
print(f"Partition B (size {len(part_b)}): {sorted(list(part_b))}")
print(f"Final Balanced Cut Size: {cut}")

# 3. Visualize the result
pos = nx.spring_layout(G, seed=42)
node_colors = ['#1f78b4' if n in part_a else '#33a02c' for n in G.nodes()]

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_color='white')
plt.title(f'Balanced Cut Partition (Cut Size: {cut})')
plt.show()