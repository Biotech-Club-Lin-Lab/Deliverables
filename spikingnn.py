import torch
import torch.nn as nn
import torch.optim as optim

def spike_function(potential, threshold = 1.0):
    return (potential >= threshold).float()

class LIFNeuron(nn.Module):
    def __init__(self, decay = 0.9, threshold = 1.0):
        super(LIFNeuron, self).__init__()
        self.decay = decay
        self.threshold = threshold
        self.potential = 0.0 #membrane potential

    def forward(self, input_current):
        self.potential = self.decay * self.potential + input_current
        spikes = spike_function(self.potential, self.threshold)
        self.potential = self.potential * (1 - spikes) #Reset after spike
        return spikes
    
class SpikingANDGate(nn.Module):
    def __init__(self):
        super(SpikingANDGate, self).__init__()
        self.neuron1 = LIFNeuron()
        self.neuron2 = LIFNeuron()
        self.output_neuron = LIFNeuron()

    def forward(self, x1, x2):
        spike1 = self.neuron1(x1)
        spike2 = self.neuron2(x2)
        output_spike = self.output_neuron(spike1*spike2) #AND logic
        return output_spike
    
    #Simulating the AND gate over 10 time steps
and_gate = SpikingANDGate()
x_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

time_steps = 10
for x1, x2 in x_inputs:
    spikes = []
    for t in range(time_steps):
        x1_tensor = torch.tensor([x1], dtype = torch.float32)
        x2_tensor = torch.tensor([x2], dtype = torch.float32)
        spikes.append(and_gate(x1_tensor, x2_tensor).item())
    print(f"Input ({x1},{x2}) -> Output: {int(any(spikes))}")

# Spiking XOR Gate - New class added
class SpikingXORGate(nn.Module):
    def __init__(self):
        super(SpikingXORGate, self).__init__()
        self.and_gate = SpikingANDGate()
        self.or_gate = SpikingANDGate()  # We'll use an OR gate logic
        self.not_gate = LIFNeuron()     # Not gate (inhibitory effect)

    def forward(self, x1, x2):
        # Using AND gate logic for both the OR and NOT gates
        and_output = self.and_gate(x1, x2)  # AND logic for (x1 and x2)
        not_output = self.not_gate(1 - and_output)  # Inhibit the AND result (NOT)

        # OR gate logic (using AND gates with both neurons activated)
        or_output = self.or_gate(x1, x2) + not_output  # XOR = (x1 OR x2) AND (NOT x1)

        # The output is the combination of these neurons
        return or_output

# Initialize the XOR gate
xor_gate = SpikingXORGate()

# Test inputs
x_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

time_steps = 10
for x1, x2 in x_inputs:
    spikes = []
    for t in range(time_steps):
        x1_tensor = torch.tensor([x1], dtype=torch.float32)
        x2_tensor = torch.tensor([x2], dtype=torch.float32)
        spikes.append(xor_gate(x1_tensor, x2_tensor).item())
    print(f"Input ({x1},{x2}) -> Output: {int(any(spikes))}")