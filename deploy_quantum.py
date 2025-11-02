import pennylane as qml
from quantum_circuit import create_quantum_circuit

def run_on_real_qpu():
    ibmq_device = qml.device("qiskit.ibmq", wires=4, backend="ibmq_quito", shots=1000)
    
    @qml.qnode(ibmq_device)
    def real_qpu_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(4))
        for layer in range(2):
            for i in range(4):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(3):
                qml.CZ(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    
    weights = np.random.uniform(0, 2*np.pi, (2, 4, 2))
    result = real_qpu_circuit([0.1, 0.2, 0.3, 0.4], weights)
    print("QPU Result:", result)

if __name__ == "__main__":
    run_on_real_qpu()