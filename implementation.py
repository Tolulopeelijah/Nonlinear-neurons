import torch
import torch.nn as nn
import torch.optim as optim

# Custom quadratic neuron
class QuadraticNeuron(nn.Module):
    def __init__(self):
        super(QuadraticNeuron, self).__init__()
        # Learnable parameters
        self.w1 = nn.Parameter(torch.randn(1))  # x1^2
        self.w2 = nn.Parameter(torch.randn(1))  # x2^2
        self.w3 = nn.Parameter(torch.randn(1))  # x1*x2
        self.w4 = nn.Parameter(torch.randn(1))  # x1
        self.w5 = nn.Parameter(torch.randn(1))  # x2
        self.b = nn.Parameter(torch.randn(1))  # bias

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        z = self.w1 * x1**2 + self.w2 * x2**2 + self.w3 * x1 * x2 + self.w4 * x1 + self.w5 * x2 + self.b
        return z

# XOR Dataset
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0.0, 1.0, 1.0, 0.0]).unsqueeze(1)  # Target outputs

# Model definition
class QuadraticNetwork(nn.Module):
    def __init__(self):
        super(QuadraticNetwork, self).__init__()
        self.neuron = QuadraticNeuron()
        self.activation = nn.Sigmoid()  # Add non-linearity

    def forward(self, x):
        z = self.neuron(x)
        return self.activation(z)

# Initialize model, loss, and optimizer
model = QuadraticNetwork()
criterion = nn.BCELoss()  # Binary Cross-Entropy for XOR
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X).unsqueeze(1)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test the network
with torch.no_grad():
    test_output = model(X)
    preds = [round(i) for i in test_output.numpy()]
    print("Predictions:", preds)
