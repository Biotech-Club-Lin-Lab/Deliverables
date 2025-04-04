import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import matplotlib.pyplot as plt

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, growth_interval=10):
        super(DynamicNeuralNetwork, self).__init__()
        
        # Initial Embryonic Network
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.growth_threshold = 0.5  # Probability threshold for neuron division
        self.max_new_neurons = 2  # Max neurons to add per expansion
        self.growth_interval = growth_interval  # How often to expand (every N steps)
        self.step_counter = 0  # Track training steps
        self.growth_log = []  # Stores neuron expansion events

    def forward(self, x):
        activations = []
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            activations.append(x.clone().detach())  # Store activations for neuron state
        return self.output_layer(x), activations

    def expand_network(self, state_vectors):
        """
        Expands the network dynamically by evaluating growth_function on each neuron.
        Logs expansion events for analysis.
        """
        new_neurons = []
        initial_neurons = [layer.out_features for layer in self.hidden_layers]  # Track before-expansion size

        expansion_details = {"step": self.step_counter, "expanded_neurons": []}

        for i, layer in enumerate(list(self.hidden_layers)):  # Iterate over a copy to avoid runtime errors
            for j in range(layer.out_features):
                neuron_state = state_vectors[i][j]  # Extract neuron state
                growth_prob = self.growth_function(neuron_state)  # Compute growth probability

                if growth_prob > self.growth_threshold and len(new_neurons) < self.max_new_neurons:
                    if random.random() > 0.5:
                        self._parallel_division(layer, j)
                        division_type = "Parallel"
                    else:
                        self._serial_division(i, j)
                        division_type = "Serial"

                    new_neurons.append(j)
                    expansion_details["expanded_neurons"].append(
                        {"layer": i, "neuron_idx": j, "probability": growth_prob, "type": division_type}
                    )

        # Log expansion details
        expansion_details["neurons_before"] = initial_neurons
        expansion_details["neurons_after"] = [layer.out_features for layer in self.hidden_layers]
        self.growth_log.append(expansion_details)

        # Dynamic adjustment of output layer - THIS IS THE KEY FIX
        # Create a new output layer that matches dimensions with the last hidden layer
        last_hidden_size = self.hidden_layers[-1].out_features
        old_output = self.output_layer
        new_output = nn.Linear(last_hidden_size, old_output.out_features)
        
        # Initialize new output layer with random weights
        nn.init.xavier_uniform_(new_output.weight)
        nn.init.zeros_(new_output.bias)
        
        # Replace the output layer
        self.output_layer = new_output

    def _parallel_division(self, layer, neuron_idx):
        """Duplicates a neuron and slightly mutates weights."""
        # Get current weight and bias
        old_weight = layer.weight.data
        old_bias = layer.bias.data
        
        # Create a duplicate of the neuron with small perturbation
        new_neuron_weight = old_weight[neuron_idx:neuron_idx+1].clone()
        new_neuron_weight += torch.randn_like(new_neuron_weight) * 0.05
        
        # Create new weight matrix with the extra neuron
        new_weight = torch.cat([old_weight, new_neuron_weight], dim=0)
        new_bias = torch.cat([old_bias, old_bias[neuron_idx:neuron_idx+1].clone()], dim=0)
        
        # Create new layer with extra neuron
        new_layer = nn.Linear(layer.in_features, layer.out_features + 1)
        new_layer.weight.data = new_weight
        new_layer.bias.data = new_bias
        
        # Replace the layer in hidden_layers
        for i, l in enumerate(self.hidden_layers):
            if l is layer:
                self.hidden_layers[i] = new_layer
                break

    def _serial_division(self, layer_idx, neuron_idx):
        """Inserts a new neuron between an existing neuron and its outputs."""
        # If this is the last layer, we need to handle it differently
        if layer_idx == len(self.hidden_layers) - 1:
            # Just use parallel division for the last layer as it's simpler
            self._parallel_division(self.hidden_layers[layer_idx], neuron_idx)
            return
            
        # For intermediate layers, insert a new layer after the current one
        current_layer = self.hidden_layers[layer_idx]
        current_out_features = current_layer.out_features
        
        # Create a new intermediary layer with 1 neuron
        new_layer = nn.Linear(current_out_features, 1)
        nn.init.xavier_uniform_(new_layer.weight)
        nn.init.zeros_(new_layer.bias)
        
        # Insert the new layer
        self.hidden_layers.insert(layer_idx + 1, new_layer)
        
        # Adjust the input size of the following layer
        if layer_idx + 2 < len(self.hidden_layers):
            next_layer = self.hidden_layers[layer_idx + 2]
            in_features = next_layer.in_features
            out_features = next_layer.out_features
            
            # Create a layer that can accept inputs from both the new layer and the existing layer
            adjusted_layer = nn.Linear(in_features + 1, out_features)
            
            # Copy existing weights for the original connections
            with torch.no_grad():
                adjusted_layer.weight.data[:, :in_features] = next_layer.weight.data
                adjusted_layer.bias.data = next_layer.bias.data
            
            # Initialize the new connections
            with torch.no_grad():
                # Initialize the new input connections with small random values
                adjusted_layer.weight.data[:, in_features:] = torch.randn(out_features, 1) * 0.01
            
            # Replace the layer
            self.hidden_layers[layer_idx + 2] = adjusted_layer

    def growth_function(self, neuron_state):
        """
        Defines the probability of neuron division based on its state vector.
        """
        # Convert neuron_state to tensor if it's not already
        if not isinstance(neuron_state, torch.Tensor):
            neuron_state = torch.tensor(neuron_state, dtype=torch.float32)
            
        probability = torch.sigmoid(neuron_state)  # Sigmoid to normalize probability
        return probability.item()

    def train_step(self, x, y, optimizer, criterion):
        """
        Integrates expansion dynamically within the training process.
        """
        optimizer.zero_grad()
        output, activations = self.forward(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Increment step counter
        self.step_counter += 1
        
        # Only expand every N steps
        if self.step_counter % self.growth_interval == 0:
            with torch.no_grad():
                # Extract state vectors (weights + activations)
                state_vectors = [
                    [(layer.weight[j].abs().sum().item() + activations[i][:, j].mean().item()) / 2
                     for j in range(layer.out_features)]
                    for i, layer in enumerate(self.hidden_layers)
                ]
                
                # Expand the network
                self.expand_network(state_vectors)
                
                # After expansion, we need to recreate the optimizer since parameters have changed
                optimizer = optim.Adam(self.parameters(), lr=0.01)
        
        return loss.item()

    def print_growth_log(self):
        """Prints the neuron growth log for analysis."""
        print("\nNeuron Growth Log:")
        for entry in self.growth_log:
            print(f"Step {entry['step']}: Neurons Before {entry['neurons_before']} → After {entry['neurons_after']}")
            for neuron in entry["expanded_neurons"]:
                print(f"  - Layer {neuron['layer']}, Neuron {neuron['neuron_idx']}, Growth Prob: {neuron['probability']:.4f}, Type: {neuron['type']}")

# Run the model
def run_dynamic_network():
    # Generate sample dataset (10 input features, 1 output)
    num_samples = 100
    X_train = torch.rand((num_samples, 10))  # Random input
    y_train = torch.randint(0, 2, (num_samples, 1)).float()  # Binary labels

    # Initialize Dynamic Network
    model = DynamicNeuralNetwork(input_size=10, output_size=1, hidden_size=5, growth_interval=5)
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model and log growth over 50 steps
    num_steps = 50
    neurons_per_layer = []
    losses = []

    for step in range(1, num_steps + 1):
        # Select random batch
        idx = random.sample(range(num_samples), 32)
        x_batch = X_train[idx]
        y_batch = y_train[idx]

        # Train step
        loss = model.train_step(x_batch, y_batch, optimizer, criterion)
        losses.append(loss)

        # Log neuron count per layer
        neurons_per_layer.append([layer.out_features for layer in model.hidden_layers])

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    # Print growth log
    model.print_growth_log()

    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Network growth
    plt.subplot(2, 1, 1)
    growth_steps = [entry["step"] for entry in model.growth_log]
    neurons_per_layer_over_time = [sum(entry["neurons_after"]) for entry in model.growth_log]
    
    plt.plot(growth_steps, neurons_per_layer_over_time, marker='o', linestyle='-', color='b', label="Total Neurons")
    plt.xlabel("Training Step")
    plt.ylabel("Total Neurons in Hidden Layers")
    plt.title("Neuron Growth Over Training")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Loss over time
    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_steps + 1), losses, color='r')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss Over Training")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model

if __name__ == "__main__":
    run_dynamic_network()