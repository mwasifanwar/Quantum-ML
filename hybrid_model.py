import torch
import torch.nn as nn
import pennylane as qml
from quantum_circuit import create_quantum_circuit

class HybridQuantumClassical(nn.Module):
    def __init__(self, n_qubits, n_layers, classical_dim, output_dim):
        super(HybridQuantumClassical, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qcircuit = create_quantum_circuit(n_qubits, n_layers)
        self.pre_net = nn.Linear(classical_dim, n_qubits)
        self.post_net = nn.Linear(n_qubits, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.tanh(self.pre_net(x))
        quantum_out = torch.tensor([self.qcircuit(x_i, self.quantum_weights) for x_i in x]).float()
        return self.softmax(self.post_net(quantum_out))
    
    def init_quantum_weights(self):
        self.quantum_weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 2))