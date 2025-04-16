import networkx as nx
import matplotlib.pyplot as plt
from chromosome import *
import os

def create_directed_graph(c:Chromosome):
    g = nx.DiGraph()
    for node in c.nodes:
        g.add_node(node.id, layer=node.layer)
    for edge in c.edges:
        g.add_edge(edge.node, edge.out_edge_to, weight=edge.weight)
    return g


def plot_chromosome(c:Chromosome, width_scale:float=3.0, min_width:float=0.5):
    print("Making graph")
    g = create_directed_graph(c)
    pos = nx.multipartite_layout(g, subset_key='layer', align='vertical', scale=1, center=None)
    edge_weights = [g[u][v]['weight'] for u, v in g.edges()]
    edge_widths = [max(min_width, abs(weight) * width_scale) for weight in edge_weights]
    print("Drawing graph")
    nx.draw_networkx_nodes(g, pos, node_size=400)
    nx.draw_networkx_labels(g, pos, font_size=10)
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(), width=edge_widths)
    print("Saving graph")
    path_name = "figs"
    fig_name = f"chromosome-{c.id}.svg"
    full_path = os.path.join(path_name, fig_name)
    plt.savefig(full_path, format="svg", dpi=1200)
    plt.close()
    print(f"Saved as chromosome-{c.id}.svg in the figs folder.")