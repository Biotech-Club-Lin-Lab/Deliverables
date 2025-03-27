import torch
import torch.nn as nn

def spike_function(potential, threshold=1.0):
    return (potential >= threshold).float()

class LIFNeuron(nn.Module):
    def __init__(self, decay=0.9, threshold=1.0):
        super(LIFNeuron, self).__init__()
        self.decay = decay
        self.threshold = threshold
        self.potential = 0.0

    def reset(self):
        self.potential = 0.0

    def forward(self, input_current):
        # Update potential with decay
        self.potential = self.decay * self.potential + input_current
        
        # Generate spikes
        spikes = spike_function(self.potential, self.threshold)
        
        # Reset potential if a spike occurs
        self.potential = self.potential * (1 - spikes)
        
        return spikes

class SpikingNOTGate(nn.Module):
    def __init__(self):
        super(SpikingNOTGate, self).__init__()
        self.neuron = LIFNeuron(threshold=0.5)  # Lower threshold for NOT gate

    def forward(self, x):
        # Invert the input by using a lower threshold
        return 1 - self.neuron(x)

class SpikingXORGate(nn.Module):
    def __init__(self):
        super(SpikingXORGate, self).__init__()
        # Create neurons with different characteristics
        self.neuron1 = LIFNeuron(threshold=0.7)
        self.neuron2 = LIFNeuron(threshold=0.7)
        self.not1 = SpikingNOTGate()
        self.not2 = SpikingNOTGate()

    def forward(self, x1, x2):
        # XOR logic: (x1 AND NOT x2) OR (NOT x1 AND x2)
        not_x1 = self.not1(x1)
        not_x2 = self.not2(x2)
        
        # First term: x1 AND (NOT x2)
        term1 = self.neuron1(x1 * not_x2)
        
        # Second term: (NOT x1) AND x2
        term2 = self.neuron2(not_x1 * x2)
        
        # Combine terms
        return term1 + term2

# Test the XOR gate
xor_gate = SpikingXORGate()

# Test inputs
x_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

time_steps = 10
print("Spiking XOR Gate Results:")
for x1, x2 in x_inputs:
    spikes = []
    for t in range(time_steps):
        x1_tensor = torch.tensor([x1], dtype=torch.float32)
        x2_tensor = torch.tensor([x2], dtype=torch.float32)
        spikes.append(xor_gate(x1_tensor, x2_tensor).item())
    print(f"Input ({x1},{x2}) -> Output: {int(any(spikes))}")