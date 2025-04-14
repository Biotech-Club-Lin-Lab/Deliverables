import pandas as pd

class NodeGene:

    def __init__(self, node_id, layer, activation_function='ReLU'):
        """
    Node class represents the genetic encoding for a single neuron in a network.

    :param node_id: Unique identifier for the node
    :param layer: Type of node (e.g., 'input', 'output' or a number for hidden layers)
    :param activation_function: Activation function for the node (default is 'ReLU')
    """

        self.id = node_id  # Unique identifier for the node
        self.type = layer  # Type of node (e.g., 'input', 'output' or a number for hidden layers)
        self.act_func = activation_function  # define code later


class EdgeGene:
    def __init__(self, edge_id, IN, OUT, weight, enabled=True):
        """
        Edge class is the genetic encoding for a single connection in a network.
        Each edge_gene defines a connection between two neurons with a weight.

        :param edge_id: Unique identifier for the gene
        :param IN: Input neuron (source node)
        :param OUT: Output neuron (target node)
        :param weight: Weight of the connection (can be positive or negative)
        :param enabled: Boolean indicating if the gene is enabled or disabled
        """
        self.id = edge_id  # Unique identifier for the gene
        self.IN = IN  # Input neuron
        self.OUT = OUT  # Output neuron
        self.weight = weight  # weight
        self.enabled = enabled  # Boolean indicating if the gene is enabled or disabled

    def show(self, display=False):
        """
        Print gene information. currently uses pandas
        """

        # Create a DataFrame to display gene information
        df = pd.DataFrame({
            'Gene ID': [self.id],
            'Input Neuron': [self.IN],
            'Output Neuron': [self.OUT],
            'Weight': [self.weight],
            'Enabled': [self.enabled]
        })
        # if display:
        #     display(df.T)
        return df