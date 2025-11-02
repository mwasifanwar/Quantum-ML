import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hybrid_model import HybridQuantumClassical

def load_data():
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    model = HybridQuantumClassical(n_qubits=4, n_layers=2, classical_dim=4, output_dim=3)
    model.init_quantum_weights()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).long()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).long()
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}")

if __name__ == "__main__":
    train_model()