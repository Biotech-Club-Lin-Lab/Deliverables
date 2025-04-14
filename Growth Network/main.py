from genes import NodeGene, EdgeGene
from chromosome import Chromosome
from network import Network
from visualization import plot_chromosome

# Example: Creating C0, C1, the network, growing it, and showing/plotting
network_id = 1
num_inputs = 3
num_outputs = 2
num_hidden = 4

C0 = Chromosome(chromosome_id=0, inputs=num_inputs, outputs=num_outputs, hidden=num_hidden)

# Define reg_nodes and reg_edges for C1 (as in the notebook)
reg_nodes = [NodeGene(i, 1) for i in range(3)]
reg_edges = [
    EdgeGene(0, 0, 1, 0.7),
    EdgeGene(1, 1, 2, 0.8),
    EdgeGene(2, 0, 2, 0.5)
]
C1 = Chromosome(chromosome_id=1, edges=reg_edges, nodes=reg_nodes)

my_network = Network(network_id=1, chromosomes=[C0, C1], inputs=num_inputs, outputs=num_outputs)

# print("Growing network...")
# my_network.grow(iterations=3)
#
# print("\nFinal C0 structure:")
# my_network.genotype[0].show()

print("\nPlotting final C0:")
plot_chromosome(my_network.genotype[0])

print("\nScript finished.")

if __name__ == "__main__":
    print("Hello World")
