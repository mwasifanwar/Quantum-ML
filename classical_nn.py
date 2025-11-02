import torch
import torch.nn as nn

class ClassicalNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassicalNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.softmax(self.output(x))
        return x

def create_classical_model(input_dim, hidden_dim, output_dim):
    return ClassicalNN(input_dim, hidden_dim, output_dim)