import pennylane as qml
from pennylane import numpy as np

def create_quantum_circuit(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return quantum_circuit

def init_weights(n_qubits, n_layers):
    return np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 2))