from chromosome import *

class Network:  # represents an induvidual in the population
    def __init__(self, network_id, chromosomes=None, inputs=None, outputs=None):
        """
        Network class represents a neural network built from a collection of chromosome.
        It can be used for forward propagation and other operations.

        :param network_id: Unique identifier for the network
        :param inputs: num input nodes
        :param outputs: num output nodes
        :param chromosomes: List of chromosome objects
        """
        # random intialization of the network
        if chromosomes is None:
            num_hidden_nodes = random.randint(1, 5)
            # run func to generate random chromosomes bases on num inputs and num outputs
            self.genotype = []
            self.genotype[0] = Chromosome(0, inputs=inputs, outputs=outputs,
                                          hidden=num_hidden_nodes)  # embryonic enc choromosome (C0)

        self.id = network_id
        self.genotype = chromosomes  # Chromosome object

    def grow(self, iterations=1):
        """
        Grows the network by adding genes to C0 based on regulatory chromosomes C1-Cn.
        This iterative process simulates developmental growth of the neural network.

        :param iterations: Number of growth iterations to perform
        """
        if len(self.genotype) <= 1:  # If we only have C0 or no chromosomes
            print("No regulatory chromosomes (C1-Cn) available for growth")
            return

        C0 = self.genotype[0]  # The main neural network chromosome

        for iteration in range(iterations):
            print(f"Growth iteration {iteration + 1}/{iterations}")

            # Get the current state of C0
            current_nodes = len(C0.nodes)
            current_edges = len(C0.edges)

            # Process each regulatory chromosome (C1-Cn)
            for chrom_idx in range(1, len(self.genotype)):
                regulatory_chrom = self.genotype[chrom_idx]

                # Create a graph representation of the regulatory chromosome
                reg_matrix, reg_neuron_map = regulatory_chrom.create_adjacency_matrix()

                # Determine which genes to add based on the regulatory network
                new_edges = self._evaluate_regulatory_network(
                    regulatory_chrom,
                    reg_matrix,
                    reg_neuron_map,
                    iteration
                )

                # Add the new edges to C0
                if new_edges:
                    C0.add(new_edges)
                    print(f"Added {len(new_edges)} new connections from regulatory chromosome {chrom_idx}")

        print(f"Growth complete. C0 now has {len(C0.nodes)} nodes and {len(C0.edges)} connections.")

    def _evaluate_regulatory_network(self, reg_chrom, reg_matrix, reg_neuron_map, current_age):
        """
        Evaluates a regulatory chromosome to determine which genes to add to C0.

        :param reg_chrom: The regulatory chromosome being evaluated
        :param reg_matrix: Adjacency matrix of the regulatory network
        :param reg_neuron_map: Mapping of neuron IDs to matrix indices
        :param current_age: Current growth iteration/age
        :return: List of new EdgeGene objects to add to C0
        """
        # This is a simplified implementation - you would expand this based on your specific needs

        # Get the basic properties of C0
        C0 = self.genotype[0]
        C0_node_ids = [node.id for node in C0.nodes]

        new_edges = []

        # Simple rule: For each active edge in the regulatory network, potentially add a new edge to C0
        # The activation of regulatory edges could depend on the current age/iteration
        activation_threshold = 0.5 - (0.1 * current_age)  # Threshold decreases with age

        # Get dimensions of the regulatory matrix
        reg_size = reg_matrix.shape[0]

        # Analyze the regulatory network to determine new connections
        for i in range(reg_size):
            for j in range(reg_size):
                if reg_matrix[i, j] > activation_threshold:
                    # This regulatory connection is active - use it to guide growth

                    # Find potential source and target nodes in C0
                    # (This is a simplified approach - you could use more complex mapping)
                    potential_sources = [n for n in C0_node_ids if n % (reg_size + 1) == i]
                    potential_targets = [n for n in C0_node_ids if n % (reg_size + 1) == j]

                    # If we have both source and target candidates, create a new connection
                    if potential_sources and potential_targets:
                        source = random.choice(potential_sources)
                        target = random.choice(potential_targets)

                        # Check if this connection already exists
                        if not any(e.IN == source and e.OUT == target for e in C0.edges):
                            # Create a new edge with weight influenced by the regulatory network
                            weight = np.random.uniform(-1, 1) * reg_matrix[i, j]
                            new_edge = EdgeGene(len(C0.edges), source, target, weight)
                            new_edges.append(new_edge)

        # Convert to numpy array if needed for the add method
        if new_edges:
            return np.array(new_edges)
        return []
