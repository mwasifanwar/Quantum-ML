QUANTUM_CONFIG = {
    "n_qubits": 4,
    "n_layers": 2,
    "shots": 1000,
    "backend": "default.qubit"
}

CLASSICAL_CONFIG = {
    "hidden_dim": 64,
    "learning_rate": 0.01,
    "epochs": 100
}

HYBRID_CONFIG = {
    "quantum_dim": 4,
    "classical_dim": 4,
    "output_dim": 3
}