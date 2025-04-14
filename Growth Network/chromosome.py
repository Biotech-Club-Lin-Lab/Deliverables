from genes import *
import random
import numpy as np

class Chromosome:
    def __init__(self, c_id:int, nodes:list[NodeGene]=None, edges:list[EdgeGene]=None):
        self.c_id = c_id
        self.nodes = nodes
        self.edges = edges


    #def __init__(self, c_id:int, inputs:int, outputs:int, hidden:int):
    #    """
    #    :param c_id: id of chromosome
    #    :param inputs: number of inputs
    #    :param hidden: number of hidden nodes
    #    :param outputs: number of outputs
    #    """

    #    connectivity_ratio = 0.75 #ratio of connections to nodes
    #    nodes = []
    #    edges = []
    #    #Create iterables
    #    input_ids = range(inputs)
    #    hidden_ids = range(inputs, inputs + hidden)
    #    output_ids = range(inputs + hidden, inputs + hidden + outputs)

    #    #Create nodes per the number of inputs, outputs and hidden nodes
    #    for _ in range(inputs): nodes.append(NodeGene(len(nodes), 'input'))
    #    for _ in range(hidden): nodes.append(NodeGene(len(nodes), 1))
    #    for _ in range(outputs): nodes.append(NodeGene(len(nodes), 'output'))

    #    # Create edges between input and hidden nodes
    #    for i in input_ids:
    #        connectable_hidden = random.sample(hidden_ids, random.randint(int(hidden*connectivity_ratio), hidden)) #chooslower #bound
    #        for h in connectable_hidden:
    #            weight = np.random.uniform(-1, 1)
    #            edges.append(EdgeGene(len(edges), i, h, weight))

    #    # Create edges between hidden and output nodes
    #        for h in hidden_ids:
    #            connectable_output = random.sample(output_ids, random.randint(int(outputs*connectivity_ratio), outputs)) #choa #lower bound
    #            for o in connectable_output:
    #                    weight = np.random.uniform(-1, 1)
    #                    edges.append(EdgeGene(len(edges), h, o, weight))

#=====================================================================================================================#

    #def add(self, edges):
    #    """
    #    Add an edge to the chromosome.

    #    :param edges: Gene object to be added
    #    """
    #    self.edges.extend(edges.tolist())  # extended the gene to the list of genes

    #def show(self):
    #    output=pd.DataFrame()
    #    for edge in self.edges:
    #        df = edge.show()
    #        output = pd.concat([output, df], ignore_index=True)
    #    print(output) #jupyter will automatically display

    #def create_adjacency_matrix(self):
    #    """
    #    Builds and returns the adjacency matrix from the gene connections.
    #    Nodes are auto-discovered from genes.
    #    """
    #    # Find unique neuron IDs
    #    neuron_ids = sorted(set(e.IN for e in self.edges) | set(e.OUT for e in self.edges))
    #    neuron_to_index = {nid: idx for idx, nid in enumerate(neuron_ids)}

    #    # Initialize matrix
    #    n = len(neuron_ids)
    #    matrix = np.zeros((n, n))

    #    # Fill matrix using genes
    #    for edge in self.edges:
    #        if edge.enabled:
    #            i = neuron_to_index[edge.IN]
    #            j = neuron_to_index[edge.OUT]
    #            matrix[i, j] = edge.weight  # directed from IN to OUT


    #    return matrix, neuron_to_index
