from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#create known parameters
weight = 0.7
bias = 0.3

#create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

#the input and output
X[:10], y[:10]

len(X), len(y)

#create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#create linear regresson model class
class LinearRegressionModel(nn.Module): #almost everything in pytorch inherits from nn.Module
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    #forward() defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: #"x" is the inpt data
        return self.weights * x + self.bias # this is the linear regression formula



load_model = LinearRegressionModel()


load_model.load_state_dict(torch.load(f='models/pytorch_linearRegrression_model.pth'))

load_model.eval()

with torch.inference_mode():
    load_model_preds = load_model(X_test)
    
print(load_model_preds)