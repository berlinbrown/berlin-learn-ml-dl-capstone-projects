#!/bin/python

# Basic pytorch
# Test performance

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to test performance
def test_performance(device):
    model = SimpleNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Generate some random data
    inputs = torch.randn(10000, 1024).to(device)
    targets = torch.randn(10000, 10).to(device)

    # Training loop
    start_time = time.time()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    print(f"Training time on {device}: {end_time - start_time:.2f} seconds")

# Test on CPU
test_performance("cpu")

# Test on GPU (if available)
if torch.cuda.is_available():
    test_performance("cuda")
else:
    print("CUDA is not available.")

