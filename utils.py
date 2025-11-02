import numpy as np
import torch

def encode_classical_to_quantum(data):
    return np.arctan(data) * 2

def measure_entanglement(state):
    state_tensor = torch.tensor(state)
    density_matrix = torch.outer(state_tensor, state_tensor.conj())
    squared_density = torch.matrix_power(density_matrix, 2)
    purity = torch.trace(squared_density).real
    return 1 - purity

def quantum_accuracy(predictions, targets):
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == targets).float()
    return correct.mean().item()