import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_layer_positions(nodes):

    layer_nodes = defaultdict(list)

    # Group nodes by their layer
    for node in nodes:
        if node.type == 'input':
            layer_nodes[0].append(node)
        elif node.type == 'output':
            layer_nodes['output'].append(node)
        else:
            layer_nodes[node.type].append(node)

    # Determine number of hidden layers and output layer x-position
    hidden_layers = sorted([k for k in layer_nodes.keys() if isinstance(k, int)])
    max_hidden_layer = max(hidden_layers) if hidden_layers else 0
    output_layer_x = max_hidden_layer + 1

    # Assign x-values
    layer_x = {}
    layer_x[0] = 0  # Input layer
    for i, layer in enumerate(hidden_layers):
        layer_x[layer] = i + 1
    layer_x['output'] = output_layer_x

    # Assign y-positions
    positions = {}
    for layer, nodes_in_layer in layer_nodes.items():
        x = layer_x[layer]
        num_nodes = len(nodes_in_layer)
        y_spacing = 1
        y_start = -((num_nodes - 1) / 2) * y_spacing
        for i, node in enumerate(nodes_in_layer):
            y = y_start + i * y_spacing
            positions[node.id] = (x, y)

    return positions

def plot_chromosome(chromosome):
    # Create adjacency matrix and neuron-to-index mapping
    adj_matrix, neuron_to_index = chromosome.create_adjacency_matrix()

    # Create a directed graph from the adjacency matrix
    gA = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())

    # Generate positions for nodes
    layer_positions = generate_layer_positions(chromosome.nodes)

    # Extract edge weights
    edges = gA.edges()
    weights = [gA[u][v]['weight'] for u, v in edges]
    scaled_weights = [5 * abs(w) for w in weights]  # Scale edge widths

    # Draw the graph
    nx.draw(gA, pos=layer_positions, with_labels=True,
            node_color='lightblue', node_size=1500,
            arrowsize=20, width=scaled_weights)
    plt.show()